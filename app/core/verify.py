"""The deterministic verification layer. The most important file in the project.

Pure, deterministic Python. No model calls, no internet. The layer takes one
candidate obligation plus the text of the segment it was extracted from, and
returns the same obligation with its checks filled in and match_type set. The
separation from the language model is the entire scientific point: the model
proposes, this layer disposes, and nothing is ever marked Verified on the
model's say so alone (Part B2, B5).

Seven checks run on every obligation, in a fixed order so results are stable
and testable:

    schema_complete     error    required fields present and well typed
    grounded            error    source_quote really appears in the segment
    citation_present    warning  a permit section/reference is recorded
    unit_valid          warning  the unit fits the parameter (air vs water)
    range_plausible     warning  the value sits in a sensible envelope
    operator_consistent warning  the operator matches the wording
    no_duplicate        info     not a repeat of an earlier obligation

The three domain checks (unit_valid, range_plausible, operator_consistent) are
the differentiator from general purpose verifiers. They encode environmental
compliance semantics and are real, not placeholders (Part A3, B5).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .schema import Check, MatchType, Obligation, Operator, Severity

# A normalized quote whose best aligned window scores at or above this ratio is
# treated as fuzzily grounded. High enough that invented text does not slip
# through, low enough to tolerate spacing, case, and punctuation drift (B5).
FUZZY_THRESHOLD = 0.85

# A quote thinner than this is too weak to count as grounding evidence: it would
# earn full grounding credit on a near empty string. Such quotes are routed to a
# human rather than auto trusted.
MIN_GROUND_TOKENS = 2

# Words that change compliance meaning. A fuzzy match is allowed to absorb
# spacing, case, punctuation, and within-word typos, but it must NOT absorb a
# change to any of these tokens. This is what stops a fabricated value
# ("30 ppm" -> "90 ppm"), a unit swap (mg/l -> ug/l), a negation flip
# ("shall not exceed" -> "shall exceed"), or an averaging-period swap
# (weekly -> monthly) from sliding through as a fuzzy match (B5).
_POLARITY_WORDS = {
    "not", "no", "never", "nor", "exceed", "exceeds", "exceeding",
    "below", "above", "less", "greater", "minimum", "maximum", "least",
    "under", "over", "within", "without",
}
_FREQUENCY_WORDS = {
    "hourly", "daily", "weekly", "monthly", "quarterly", "biweekly",
    "semiannual", "semiannually", "annual", "annually", "annum",
    "continuous", "continuously", "rolling", "instantaneous",
    "day", "days", "hour", "hours", "week", "weeks", "month", "months",
    "quarter", "year", "years",
}
# Spelled out numbers carry the same compliance weight as digits. Permits often
# write limits in words ("not to exceed twenty percent", "thirty-day average"),
# so a fabricated word-number must be caught just like a fabricated digit.
_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    "hundred", "thousand", "million", "billion",
    "once", "twice", "thrice",
}


# ---------------------------------------------------------------------------
# Text normalization and grounding
# ---------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_text(s: str) -> str:
    """Lowercase and collapse every run of non alphanumeric characters to a
    single space. Applied symmetrically to both the quote and the segment so
    spacing, case, and punctuation differences cannot, on their own, break a
    match, while numbers and words are preserved.
    """
    if not s:
        return ""
    return _NON_ALNUM.sub(" ", s.lower()).strip()


def _is_significant(token: str) -> bool:
    """True if changing this token would change compliance meaning."""
    return (
        any(ch.isdigit() for ch in token)
        or token in _SIGNIFICANT_WORDS
    )


def _significant_sequence(text_norm: str) -> List[str]:
    """The compliance critical tokens of a normalized string, in order."""
    return [t for t in text_norm.split() if _is_significant(t)]


def _matched_span(quote_norm: str, seg_norm: str) -> Tuple[str, float]:
    """Locate where the quote maps onto the segment and score the fit.

    Returns the contiguous slice of the segment spanned by the quote's matching
    blocks, and a span ratio = 2 * matched_chars / (len(quote) + span_length).
    Because the score divides by the span actually covered (not the whole
    segment), a quote spliced from distant parts of the segment scores low even
    though each piece exists somewhere: the span stretches and the ratio falls.
    """
    if not quote_norm or not seg_norm:
        return "", 0.0
    sm = difflib.SequenceMatcher(None, quote_norm, seg_norm, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return "", 0.0
    matched = sum(b.size for b in blocks)
    start = blocks[0].b
    end = blocks[-1].b + blocks[-1].size
    span_text = seg_norm[start:end]
    denom = len(quote_norm) + (end - start)
    ratio = (2.0 * matched / denom) if denom else 0.0
    return span_text, ratio


def grounding_match(quote: str, segment_text: str) -> Tuple[MatchType, float]:
    """Decide whether quote is grounded in segment_text.

    Returns the match type and a strength in [0, 1]. An exact normalized
    substring scores 1.0. Otherwise the best length aligned window in the
    segment is compared to the quote with difflib; a fuzzy match is accepted
    only when the character ratio clears FUZZY_THRESHOLD AND no compliance
    critical token differs (so invented values, units, or negations cannot slip
    through). A quote too thin to be evidence grounds to none.
    """
    q = normalize_text(quote)
    s = normalize_text(segment_text)
    if not q or not s:
        return MatchType.NONE, 0.0

    q_tokens = q.split()
    significant = sum(1 for t in q_tokens if _is_significant(t))
    too_thin = len(q_tokens) < MIN_GROUND_TOKENS or (
        len(q_tokens) < 4 and significant == 0
    )
    if too_thin:
        return MatchType.NONE, 0.0

    if q in s:
        return MatchType.EXACT, 1.0

    # A fuzzy match requires both: the quote maps to one contiguous span of the
    # segment (span ratio clears the threshold), AND no compliance critical
    # token in that span differs from the quote (no swapped value, unit,
    # negation, frequency, or spelled out number).
    span_text, ratio = _matched_span(q, s)
    if ratio >= FUZZY_THRESHOLD and _significant_sequence(span_text) == _significant_sequence(q):
        return MatchType.FUZZY, ratio
    return MatchType.NONE, ratio


# ---------------------------------------------------------------------------
# Domain knowledge: which units and value ranges are plausible per parameter
# ---------------------------------------------------------------------------


def normalize_unit(unit: Optional[str]) -> str:
    """Map a free text unit to a canonical normalized form."""
    if not unit:
        return ""
    u = unit.strip().lower()
    u = u.replace("µ", "u").replace("μ", "u").replace(".", "")
    u = re.sub(r"\s+", " ", u).strip()
    return _UNIT_ALIASES.get(u, u)


_UNIT_ALIASES: Dict[str, str] = {
    # air concentrations
    "ppm": "ppm", "ppmv": "ppm", "parts per million": "ppm",
    "ppmvd": "ppmvd", "ppmd": "ppmvd",
    "mg/m3": "mg/m3", "mg/m^3": "mg/m3", "mg/nm3": "mg/m3",
    # air mass rates
    "lb/hr": "lb/hr", "lbs/hr": "lb/hr", "lb/hour": "lb/hr", "pounds per hour": "lb/hr",
    "lb/mmbtu": "lb/mmbtu", "lb/mm btu": "lb/mmbtu", "lb/mbtu": "lb/mmbtu",
    "ton/yr": "tons/yr", "tons/yr": "tons/yr", "tpy": "tons/yr",
    "tons per year": "tons/yr", "tons/year": "tons/yr",
    "gr/dscf": "gr/dscf", "grains/dscf": "gr/dscf",
    "%": "%", "percent": "%", "pct": "%",
    # water
    "mg/l": "mg/l", "milligrams per liter": "mg/l",
    "ug/l": "ug/l", "ug/ l": "ug/l", "micrograms per liter": "ug/l",
    "lb/day": "lb/day", "lbs/day": "lb/day", "pounds per day": "lb/day",
    "su": "su", "s.u.": "su", "s u": "su", "standard units": "su",
    "ph units": "su", "ph": "su",
    "deg c": "deg c", "degc": "deg c", "c": "deg c",
    "degrees c": "deg c", "celsius": "deg c", "deg celsius": "deg c",
    "deg f": "deg f", "degf": "deg f", "f": "deg f", "fahrenheit": "deg f",
    "mgd": "mgd", "million gallons per day": "mgd",
    "cfu/100ml": "cfu/100ml", "cfu/100 ml": "cfu/100ml",
    "#/100ml": "cfu/100ml", "mpn/100ml": "cfu/100ml", "col/100ml": "cfu/100ml",
}

# Unit names broken into the word tokens grounding will see, e.g. "mg/l" -> mg, l.
# A swapped unit (mg/l -> ug/l) therefore registers as a significant token change.
_UNIT_TOKENS = set()
for _u in set(_UNIT_ALIASES) | set(_UNIT_ALIASES.values()):
    _UNIT_TOKENS.update(normalize_text(_u).split())

# Every token whose change alters compliance meaning (see grounding_match).
_SIGNIFICANT_WORDS = _POLARITY_WORDS | _FREQUENCY_WORDS | _NUMBER_WORDS | _UNIT_TOKENS


@dataclass(frozen=True)
class ParameterSpec:
    """What is known about a regulated parameter for the domain checks."""

    canonical: str
    media: str  # "air" or "water"
    allowed_units: frozenset  # normalized units valid for this parameter
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    aliases: Tuple[str, ...] = ()


def _p(canonical, media, units, ranges, *aliases) -> ParameterSpec:
    return ParameterSpec(
        canonical=canonical,
        media=media,
        allowed_units=frozenset(units),
        ranges=dict(ranges),
        aliases=tuple(aliases),
    )


# Curated from Title V air and NPDES water permit practice. Ranges are
# deliberately generous plausibility envelopes: they exist to catch obvious
# nonsense (negatives, order of magnitude errors, mismatched media) without
# flagging legitimate permit limits.
_PARAMETERS: Tuple[ParameterSpec, ...] = (
    _p("NOx", "air",
       {"ppm", "ppmvd", "mg/m3", "lb/hr", "lb/mmbtu", "tons/yr"},
       {"ppm": (0, 2000), "ppmvd": (0, 2000), "mg/m3": (0, 5000),
        "lb/hr": (0, 10000), "lb/mmbtu": (0, 5), "tons/yr": (0, 100000)},
       "nox", "nitrogen oxides", "oxides of nitrogen", "no2", "nitrogen dioxide"),
    _p("SO2", "air",
       {"ppm", "ppmvd", "mg/m3", "lb/hr", "lb/mmbtu", "tons/yr"},
       {"ppm": (0, 5000), "ppmvd": (0, 5000), "mg/m3": (0, 10000),
        "lb/hr": (0, 10000), "lb/mmbtu": (0, 10), "tons/yr": (0, 100000)},
       "so2", "sulfur dioxide", "sulphur dioxide", "sox"),
    _p("CO", "air",
       {"ppm", "ppmvd", "mg/m3", "lb/hr", "tons/yr"},
       {"ppm": (0, 5000), "ppmvd": (0, 5000), "mg/m3": (0, 20000),
        "lb/hr": (0, 20000), "tons/yr": (0, 100000)},
       "co", "carbon monoxide"),
    _p("PM", "air",
       {"gr/dscf", "lb/hr", "lb/mmbtu", "mg/m3", "tons/yr"},
       {"gr/dscf": (0, 5), "lb/hr": (0, 5000), "lb/mmbtu": (0, 5),
        "mg/m3": (0, 5000), "tons/yr": (0, 50000)},
       "pm", "pm10", "pm2.5", "particulate matter", "total particulate",
       "particulates"),
    _p("VOC", "air",
       {"ppm", "lb/hr", "tons/yr"},
       {"ppm": (0, 5000), "lb/hr": (0, 5000), "tons/yr": (0, 100000)},
       "voc", "volatile organic compounds", "volatile organic compound"),
    _p("Opacity", "air",
       {"%"},
       {"%": (0, 100)},
       "opacity"),
    _p("pH", "water",
       {"su", ""},
       {"su": (0, 14), "": (0, 14)},
       "ph"),
    _p("BOD", "water",
       {"mg/l", "lb/day"},
       {"mg/l": (0, 10000), "lb/day": (0, 1000000)},
       "bod", "bod5", "cbod", "cbod5", "biochemical oxygen demand",
       "carbonaceous biochemical oxygen demand"),
    _p("TSS", "water",
       {"mg/l", "lb/day"},
       {"mg/l": (0, 10000), "lb/day": (0, 1000000)},
       "tss", "total suspended solids", "suspended solids"),
    _p("Dissolved Oxygen", "water",
       {"mg/l"},
       {"mg/l": (0, 20)},
       "do", "dissolved oxygen"),
    _p("Temperature", "water",
       {"deg c", "deg f"},
       {"deg c": (0, 40), "deg f": (32, 110)},
       "temperature", "temp", "thermal"),
    _p("Flow", "water",
       {"mgd"},
       {"mgd": (0, 1000)},
       "flow", "discharge flow", "effluent flow"),
    _p("Fecal Coliform", "water",
       {"cfu/100ml"},
       {"cfu/100ml": (0, 10000000)},
       "fecal coliform", "e. coli", "e.coli", "total coliform", "enterococci"),
    _p("Ammonia", "water",
       {"mg/l"},
       {"mg/l": (0, 10000)},
       "ammonia", "nh3", "ammonia nitrogen", "total nitrogen", "tkn"),
    _p("Total Phosphorus", "water",
       {"mg/l"},
       {"mg/l": (0, 10000)},
       "phosphorus", "total phosphorus", "tp", "orthophosphate"),
    _p("Oil and Grease", "water",
       {"mg/l"},
       {"mg/l": (0, 10000)},
       "oil and grease", "o&g", "fog"),
)

# Build a lookup from every normalized alias/canonical to its spec.
_PARAM_LOOKUP: Dict[str, ParameterSpec] = {}
for _spec in _PARAMETERS:
    for _key in (_spec.canonical, *_spec.aliases):
        _PARAM_LOOKUP[_key.lower()] = _spec


def lookup_parameter(parameter: Optional[str]) -> Optional[ParameterSpec]:
    """Resolve a free text parameter name to a ParameterSpec, or None."""
    if not parameter:
        return None
    key = parameter.strip().lower()
    if key in _PARAM_LOOKUP:
        return _PARAM_LOOKUP[key]
    # tolerate trailing descriptors, e.g. "NOx emissions" or "PM10 (filterable)"
    for alias, spec in _PARAM_LOOKUP.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", key):
            return spec
    return None


# ---------------------------------------------------------------------------
# Operator wording cues
# ---------------------------------------------------------------------------

# Note: bare "below"/"above" are deliberately NOT cues. "shall not fall below"
# is a minimum (>=), so a bare "below" cue would collide with it and wrongly
# mark a genuinely inverted operator as ambiguous. We rely on anchored phrases.
_LE_CUES = (
    "shall not exceed", "not to exceed", "may not exceed", "must not exceed",
    "no more than", "not more than", "shall not be greater than", "at or below",
    "less than or equal", "maximum", "not exceed", "up to",
)
_GE_CUES = (
    "shall not be less than", "no less than", "not less than", "at least",
    "at or above", "greater than or equal", "minimum", "not fall below",
    "shall not fall below", "fall below", "shall be maintained at or above",
)
_EQ_CUES = ("equal to", "exactly", "equals", "shall equal")
_RANGE_CUES = ("between", "ranging from", "range of", "in the range")
_RANGE_NUMERIC = re.compile(r"\d+(?:\.\d+)?\s*(?:-|to|through|–)\s*\d+(?:\.\d+)?")

_FAMILY_OPERATORS = {
    "le": {Operator.LE, Operator.LT},
    "ge": {Operator.GE, Operator.GT},
    "eq": {Operator.EQ},
    "range": {Operator.RANGE},
}


def _implied_operator_families(text: str) -> Set[str]:
    """Which operator families the wording implies. Empty when unclear."""
    t = " " + text.lower() + " "
    families: Set[str] = set()
    if any(cue in t for cue in _GE_CUES):
        families.add("ge")
    if any(cue in t for cue in _LE_CUES):
        families.add("le")
    if any(cue in t for cue in _EQ_CUES):
        families.add("eq")
    if _RANGE_NUMERIC.search(t) or any(cue in t for cue in _RANGE_CUES):
        families.add("range")
    return families


# ---------------------------------------------------------------------------
# The individual checks
# ---------------------------------------------------------------------------


def check_schema_complete(ob: Obligation) -> Check:
    missing: List[str] = []
    if not ob.description or not ob.description.strip():
        missing.append("description")
    if ob.has_numeric_limit():
        if not ob.limit_unit or not str(ob.limit_unit).strip():
            missing.append("limit_unit")
        if ob.operator is None:
            missing.append("operator")
    if missing:
        return Check(
            name="schema_complete",
            passed=False,
            severity=Severity.ERROR,
            message="Missing required field(s): " + ", ".join(missing)
            + (". A numeric limit must carry a value, unit, and operator."
               if ob.has_numeric_limit() else "."),
        )
    return Check(
        name="schema_complete",
        passed=True,
        severity=Severity.ERROR,
        message="All required fields are present and well typed.",
    )


def check_grounded(ob: Obligation, segment_text: str) -> Check:
    """Confirm the source_quote genuinely appears in the cited segment.

    This is the anti hallucination core. It also sets ob.match_type, because
    match_type is recorded for the calibration analysis in the paper (B4.2).
    """
    if not ob.source_quote or not ob.source_quote.strip():
        ob.match_type = MatchType.NONE
        return Check(
            name="grounded",
            passed=False,
            severity=Severity.ERROR,
            message="No source_quote was provided, so grounding cannot be verified.",
        )
    if not segment_text:
        ob.match_type = MatchType.NONE
        return Check(
            name="grounded",
            passed=False,
            severity=Severity.ERROR,
            message="Cited segment '%s' was not found in the document." % ob.source_segment_id,
        )

    match_type, strength = grounding_match(ob.source_quote, segment_text)
    ob.match_type = match_type
    if match_type == MatchType.EXACT:
        return Check(
            name="grounded", passed=True, severity=Severity.ERROR,
            message="Source quote found verbatim in the cited segment (exact match).",
        )
    if match_type == MatchType.FUZZY:
        return Check(
            name="grounded", passed=True, severity=Severity.ERROR,
            message="Source quote found with minor formatting differences "
                    "(fuzzy match, %.0f%%)." % (strength * 100),
        )
    return Check(
        name="grounded", passed=False, severity=Severity.ERROR,
        message="Source quote does NOT appear in the cited segment "
                "(no match, best %.0f%%). Likely a hallucination." % (strength * 100),
    )


def check_citation_present(ob: Obligation) -> Check:
    if ob.citation and str(ob.citation).strip():
        return Check(
            name="citation_present", passed=True, severity=Severity.WARNING,
            message="Permit citation recorded: %s" % ob.citation,
        )
    return Check(
        name="citation_present", passed=False, severity=Severity.WARNING,
        message="No permit section or regulatory citation was recorded for this obligation.",
    )


def check_unit_valid(ob: Obligation) -> Check:
    spec = lookup_parameter(ob.parameter)
    if spec is None:
        return Check(
            name="unit_valid", passed=True, severity=Severity.WARNING,
            message="Parameter not in the domain table; unit not checked.",
        )
    if not ob.limit_unit or not str(ob.limit_unit).strip():
        # A missing unit on a numeric limit is reported by schema_complete.
        return Check(
            name="unit_valid", passed=True, severity=Severity.WARNING,
            message="No unit to validate for parameter %s." % spec.canonical,
        )
    unit = normalize_unit(ob.limit_unit)
    if unit in spec.allowed_units:
        return Check(
            name="unit_valid", passed=True, severity=Severity.WARNING,
            message="Unit '%s' is valid for %s (%s parameter)."
                    % (ob.limit_unit, spec.canonical, spec.media),
        )
    return Check(
        name="unit_valid", passed=False, severity=Severity.WARNING,
        message="Unit '%s' is not valid for %s, which is an %s parameter."
                % (ob.limit_unit, spec.canonical, spec.media),
    )


def check_range_plausible(ob: Obligation) -> Check:
    spec = lookup_parameter(ob.parameter)
    if spec is None or ob.limit_value is None:
        return Check(
            name="range_plausible", passed=True, severity=Severity.WARNING,
            message="No parameter/value pair to range check.",
        )
    unit = normalize_unit(ob.limit_unit)
    if unit not in spec.ranges:
        return Check(
            name="range_plausible", passed=True, severity=Severity.WARNING,
            message="No plausibility range defined for %s in '%s'; not checked."
                    % (spec.canonical, ob.limit_unit or "(no unit)"),
        )
    low, high = spec.ranges[unit]
    if low <= ob.limit_value <= high:
        return Check(
            name="range_plausible", passed=True, severity=Severity.WARNING,
            message="Value %g %s is within the plausible range [%g, %g] for %s."
                    % (ob.limit_value, ob.limit_unit, low, high, spec.canonical),
        )
    return Check(
        name="range_plausible", passed=False, severity=Severity.WARNING,
        message="Value %g %s is outside the plausible range [%g, %g] for %s."
                % (ob.limit_value, ob.limit_unit, low, high, spec.canonical),
    )


def check_operator_consistent(ob: Obligation) -> Check:
    text = " ".join(filter(None, [ob.description, ob.source_quote]))
    families = _implied_operator_families(text)
    if ob.operator is None or not families:
        return Check(
            name="operator_consistent", passed=True, severity=Severity.WARNING,
            message="Operator wording is not decisive; not checked.",
        )
    if len(families) > 1:
        return Check(
            name="operator_consistent", passed=True, severity=Severity.WARNING,
            message="Wording implies more than one operator; treated as ambiguous.",
        )
    family = next(iter(families))
    expected = _FAMILY_OPERATORS[family]
    if ob.operator in expected:
        return Check(
            name="operator_consistent", passed=True, severity=Severity.WARNING,
            message="Operator '%s' is consistent with the obligation wording."
                    % ob.operator.value,
        )
    expected_str = ", ".join(sorted(o.value for o in expected))
    return Check(
        name="operator_consistent", passed=False, severity=Severity.WARNING,
        message="Operator '%s' contradicts the wording, which implies one of: %s."
                % (ob.operator.value, expected_str),
    )


def _signature(ob: Obligation) -> str:
    """A normalized key used to detect duplicate obligations."""
    if ob.parameter or ob.limit_value is not None or ob.limit_unit:
        return "|".join([
            (ob.parameter or "").strip().lower(),
            "" if ob.limit_value is None else repr(float(ob.limit_value)),
            normalize_unit(ob.limit_unit),
            "" if ob.operator is None else ob.operator.value,
        ])
    return "desc:" + normalize_text(ob.description)


def check_no_duplicate(ob: Obligation, seen: Set[str]) -> Check:
    sig = _signature(ob)
    if sig in seen:
        return Check(
            name="no_duplicate", passed=False, severity=Severity.INFO,
            message="Appears to duplicate an earlier obligation.",
        )
    seen.add(sig)
    return Check(
        name="no_duplicate", passed=True, severity=Severity.INFO,
        message="No duplicate of an earlier obligation detected.",
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

# The fixed order in which checks are emitted on every obligation.
CHECK_ORDER = (
    "schema_complete", "grounded", "citation_present",
    "unit_valid", "range_plausible", "operator_consistent", "no_duplicate",
)


def verify_obligation(
    ob: Obligation,
    segment_text: str,
    seen_signatures: Optional[Set[str]] = None,
) -> Obligation:
    """Run all checks against one obligation and return it with checks filled in.

    segment_text is the text of the segment named by ob.source_segment_id.
    seen_signatures lets the caller detect duplicates across a batch; when not
    supplied a fresh set is used so a single obligation never flags itself.
    This function mutates and returns the same obligation. It never sets the
    final status (that is the score stage's job) but it does set match_type.
    """
    if seen_signatures is None:
        seen_signatures = set()

    ob.checks = [
        check_schema_complete(ob),
        check_grounded(ob, segment_text),
        check_citation_present(ob),
        check_unit_valid(ob),
        check_range_plausible(ob),
        check_operator_consistent(ob),
        check_no_duplicate(ob, seen_signatures),
    ]
    return ob


def verify_all(
    obligations: List[Obligation],
    segments_by_id: Dict[str, str],
) -> List[Obligation]:
    """Verify a batch of obligations against a map of segment_id -> text."""
    seen: Set[str] = set()
    for ob in obligations:
        segment_text = segments_by_id.get(ob.source_segment_id, "")
        verify_obligation(ob, segment_text, seen)
    return obligations
