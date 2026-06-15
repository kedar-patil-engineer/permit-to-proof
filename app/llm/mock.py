"""The Mock backend: the default, offline, fully deterministic extractor.

It needs no API key and no internet, so the app and the whole test suite run on
a clean machine out of the box (Part B7, B10). It works in two parts:

  1. A real, deterministic regex pass over whatever PDF was supplied. Every
     obligation it produces quotes the permit text verbatim, so it is honestly
     grounded. This pass also serves as the simple pattern matching baseline
     the paper compares the LLM against (Part A5.3).

  2. For the bundled sample permit only (recognized by its permit number), a
     small set of planted demonstration cases derived from the real ones, each
     corrupted in exactly one way so the UI shows the full spectrum of
     verification outcomes, including a hallucinated quote that fails grounding.
     These are clearly labeled and never injected into a user's own upload.
"""

from __future__ import annotations

import re
from typing import List, Optional

from app.core.schema import Obligation, Operator, Segment

# Marker baked into the bundled synthetic permit. Its presence is how the Mock
# backend knows it may add the planted demonstration cases.
SAMPLE_PERMIT_MARKER = "PTP-2026-0001"


# ---------------------------------------------------------------------------
# Patterns for the deterministic regex pass
# ---------------------------------------------------------------------------

# Surface forms ordered so the more specific names win (PM10 before PM).
_PARAM_SURFACE = [
    ("NOx", r"\bNOx\b|\boxides of nitrogen\b|\bnitrogen oxides\b"),
    ("SO2", r"\bSO2\b|\bsulfur dioxide\b"),
    ("PM10", r"\bPM-?10\b"),
    ("PM2.5", r"\bPM-?2\.5\b"),
    ("VOC", r"\bVOC\b|\bvolatile organic compounds?\b"),
    ("Opacity", r"\bopacity\b"),
    ("CO", r"\bcarbon monoxide\b|\bCO\b"),
    ("PM", r"\bparticulate matter\b|\btotal particulate\b|\bPM\b"),
    ("pH", r"\bpH\b"),
    ("BOD5", r"\bBOD5\b|\bBOD\b|\bbiochemical oxygen demand\b"),
    ("TSS", r"\bTSS\b|\btotal suspended solids\b"),
    ("Temperature", r"\btemperature\b"),
    ("Fecal Coliform", r"\bfecal coliform\b"),
    ("Flow", r"\b(?:effluent |discharge )?flow\b"),
]
_PARAM_SURFACE = [(name, re.compile(rx, re.IGNORECASE)) for name, rx in _PARAM_SURFACE]

_UNIT_RX = (
    r"ppmvd|ppm|mg/l|mg/m3|lb/mmbtu|lb/hr|tons?/(?:yr|year)|ug/l|"
    r"s\.u\.|mgd|cfu/100\s*ml|gr/dscf|%|deg\s*c|°\s*c"
)
_VALUE_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(" + _UNIT_RX + r")", re.IGNORECASE)
_PH_RANGE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:and|to|-|–|through)\s*(\d+(?:\.\d+)?)")

_CITATION = re.compile(
    r"(Condition\s+[A-Za-z0-9.\-]+|40\s*CFR\s*[\d.\-()a-z]+"
    r"|Section\s+[A-Za-z0-9.\-]+|Part\s+\d+)",
    re.IGNORECASE,
)
_FREQUENCY = re.compile(
    r"\b(continuous(?:ly)?|hourly|daily|weekly|monthly|quarterly"
    r"|semi-?annual(?:ly)?|annual(?:ly)?)\b",
    re.IGNORECASE,
)
_DEADLINE = re.compile(
    r"(within\s+\d+\s+days[^.;,]*|no later than[^.;,]*"
    r"|by\s+(?:January|February|March|April|May|June|July|August|September"
    r"|October|November|December)\b[^.;,]*)",
    re.IGNORECASE,
)
_REPORTING_CUE = re.compile(
    r"\bshall (?:submit|report|maintain|keep|record|monitor|notify)\b"
    r"|\bmonitoring\b|\brecordkeeping\b|\breport\b",
    re.IGNORECASE,
)

_LE_CUES = ("shall not exceed", "not to exceed", "may not exceed", "no more than",
            "not more than", "maximum", "not exceed")
_GE_CUES = ("at least", "no less than", "not less than", "minimum", "at or above",
            "shall not be less than")


def _infer_operator(text: str) -> Optional[Operator]:
    t = text.lower()
    if "between" in t or re.search(r"\d\s*(?:-|to|–|through)\s*\d", t):
        return Operator.RANGE
    if any(c in t for c in _LE_CUES):
        return Operator.LE
    if any(c in t for c in _GE_CUES):
        return Operator.GE
    return None


def _first(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def _detect_parameter(text: str) -> Optional[str]:
    for name, rx in _PARAM_SURFACE:
        if rx.search(text):
            return name
    return None


def _normalize_pdf_unit(raw_unit: str) -> str:
    u = raw_unit.strip()
    u = re.sub(r"\s+", " ", u)
    return u


_HEADER = re.compile(r"^\s*(?:SECTION|PART|CHAPTER|ARTICLE|APPENDIX)\b", re.IGNORECASE)


def _extract_from_segment(seg: Segment, ob_index: int) -> Optional[Obligation]:
    """Build at most one obligation from a segment, quoting it verbatim."""
    text = seg.text
    if _HEADER.match(text):
        # Section headers are structure, not obligations.
        return None
    parameter = _detect_parameter(text)

    # pH is special: usually a range with no conventional unit.
    if parameter == "pH":
        m = _PH_RANGE.search(text)
        if m:
            low = float(m.group(1))
            return _build(
                seg, ob_index, parameter="pH",
                limit_value=low, limit_unit="s.u.", operator=Operator.RANGE,
                description="pH of the discharge must remain within %s to %s standard units."
                % (m.group(1), m.group(2)),
            )

    vu = _VALUE_UNIT.search(text)
    if parameter and vu:
        value = float(vu.group(1))
        unit = _normalize_pdf_unit(vu.group(2))
        operator = _infer_operator(text) or Operator.LE
        return _build(
            seg, ob_index, parameter=parameter,
            limit_value=value, limit_unit=unit, operator=operator,
            description="%s is limited to %g %s." % (parameter, value, unit),
        )

    # Narrative reporting / monitoring obligation with no numeric limit.
    if _REPORTING_CUE.search(text):
        return _build(
            seg, ob_index, parameter=parameter,
            description=_summarize(text),
        )

    return None


def _summarize(text: str) -> str:
    # Drop a leading condition/section label ("Condition 5.1.", "Section 4 -")
    # so the description is the actual requirement sentence, not the label.
    t = re.sub(
        r"^(?:Condition|Section|Part|Article|Paragraph)\s+[A-Za-z0-9.\-]+\.?\s*[-–]?\s*",
        "", text.strip(), flags=re.IGNORECASE,
    )
    sentence = re.split(r"(?<=[.;])\s+", t)[0]
    return sentence if len(sentence) <= 220 else sentence[:217] + "..."


def _build(seg: Segment, ob_index: int, *, parameter=None, limit_value=None,
           limit_unit=None, operator=None, description="", confidence=0.9) -> Obligation:
    return Obligation(
        obligation_id="OB%04d" % ob_index,
        description=description or _summarize(seg.text),
        parameter=parameter,
        limit_value=limit_value,
        limit_unit=limit_unit,
        operator=operator,
        frequency=_first(_FREQUENCY, seg.text),
        deadline=_first(_DEADLINE, seg.text),
        citation=_first(_CITATION, seg.text),
        source_segment_id=seg.segment_id,
        source_quote=seg.text,
        model_confidence=confidence,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Planted demonstration cases (bundled sample permit only)
# ---------------------------------------------------------------------------


def _is_sample_permit(segments: List[Segment]) -> bool:
    return any(SAMPLE_PERMIT_MARKER in s.text for s in segments)


def _find(obs: List[Obligation], predicate) -> Optional[Obligation]:
    for ob in obs:
        if predicate(ob):
            return ob
    return None


def _planted_cases(real: List[Obligation], start_index: int) -> List[Obligation]:
    """Derive deliberately flawed obligations from the real grounded ones.

    Each case is corrupted in exactly one way so the demo surfaces every kind
    of verification outcome. Grounded quotes are kept real (copied from actual
    segments); only the structured fields or, for the hallucination case, the
    quote itself are altered.
    """
    cases: List[Obligation] = []
    idx = start_index

    air = _find(real, lambda o: o.parameter in {"NOx", "SO2", "CO", "PM", "PM10"}
                and o.limit_value is not None)
    numeric = _find(real, lambda o: o.limit_value is not None)

    # 1. Hallucination: invented quote that is not in the cited segment.
    if numeric is not None:
        cases.append(numeric.model_copy(update={
            "obligation_id": "OB%04d" % idx,
            "parameter": "NOx",
            "limit_value": 250.0,
            "limit_unit": "ppm",
            "operator": Operator.LE,
            "description": "NOx control efficiency requirement (planted hallucination).",
            "source_quote": "The permittee shall achieve 250 ppm NOx using a "
                            "proprietary catalytic process described in Appendix Z.",
            "model_confidence": 0.88,
        }))
        idx += 1

    # 2. Missing unit on a numeric limit -> schema_complete error.
    if numeric is not None:
        cases.append(numeric.model_copy(update={
            "obligation_id": "OB%04d" % idx,
            "limit_unit": None,
            "description": numeric.description + " (planted: unit dropped)",
            "model_confidence": 0.7,
        }))
        idx += 1

    # 3. Wrong unit for the parameter (water unit on an air pollutant).
    if air is not None:
        cases.append(air.model_copy(update={
            "obligation_id": "OB%04d" % idx,
            "limit_unit": "mg/L",
            "description": air.description + " (planted: wrong unit mg/L on an air pollutant)",
            "model_confidence": 0.75,
        }))
        idx += 1

    # 4. Out of range value -> range_plausible warning.
    if numeric is not None:
        cases.append(numeric.model_copy(update={
            "obligation_id": "OB%04d" % idx,
            "limit_value": 99999.0,
            "description": numeric.description + " (planted: implausible value)",
            "model_confidence": 0.6,
        }))
        idx += 1

    # 5. Operator contradicts the wording -> operator_consistent warning.
    le_like = _find(real, lambda o: o.operator == Operator.LE and o.limit_value is not None)
    if le_like is not None:
        cases.append(le_like.model_copy(update={
            "obligation_id": "OB%04d" % idx,
            "operator": Operator.GE,
            "description": le_like.description + " (planted: operator flipped to >=)",
            "model_confidence": 0.65,
        }))
        idx += 1

    # 6. Exact duplicate of a real obligation -> no_duplicate info.
    if numeric is not None:
        cases.append(numeric.model_copy(update={
            "obligation_id": "OB%04d" % idx,
            "description": numeric.description + " (planted: duplicate)",
            "model_confidence": 0.9,
        }))
        idx += 1

    return cases


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


class MockBackend:
    """Deterministic, offline extractor. Default backend."""

    name = "Mock"

    def extract_obligations(self, segments: List[Segment]) -> List[Obligation]:
        real: List[Obligation] = []
        index = 0
        for seg in segments:
            ob = _extract_from_segment(seg, index)
            if ob is not None:
                real.append(ob)
                index += 1

        obligations = list(real)
        if _is_sample_permit(segments):
            obligations.extend(_planted_cases(real, index))
        return obligations
