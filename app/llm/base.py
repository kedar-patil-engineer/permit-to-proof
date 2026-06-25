"""The backend interface plus helpers shared by the real backends.

Every backend turns a list of Segments into candidate Obligations with
source_segment_id and source_quote filled in, leaving status PENDING.
Verification is emphatically not the backend's job (Part B7).

The pipeline depends only on the LLMBackend protocol, so backends are fully
interchangeable and the UI can pick one at runtime.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

from app.core.schema import Obligation, Operator, Segment

# Operator strings the model is allowed to emit, mapped to the enum.
_OPERATOR_LOOKUP = {op.value: op for op in Operator}
_OPERATOR_LOOKUP.update({
    "<=": Operator.LE, "=<": Operator.LE, "leq": Operator.LE,
    ">=": Operator.GE, "=>": Operator.GE, "geq": Operator.GE,
    "<": Operator.LT, ">": Operator.GT, "=": Operator.EQ, "==": Operator.EQ,
    "range": Operator.RANGE,
})


@runtime_checkable
class LLMBackend(Protocol):
    """The one interface the pipeline talks to."""

    name: str

    def extract_obligations(self, segments: List[Segment]) -> List[Obligation]:
        """Return candidate obligations with source_segment_id and
        source_quote filled in. Leave status = PENDING. Verification is NOT
        this layer's job.
        """
        ...


# ---------------------------------------------------------------------------
# Prompting (shared by the OpenAI and Ollama backends)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a meticulous environmental compliance analyst. You extract "
    "compliance obligations from environmental permits. You never invent text. "
    "For every obligation you MUST copy the exact supporting sentence from the "
    "permit verbatim into 'source_quote' so it can be verified against the "
    "source. If you are unsure, omit the obligation rather than guess."
)

_SCHEMA_HINT = (
    '{\n'
    '  "obligations": [\n'
    '    {\n'
    '      "description": "plain-language statement of what is required",\n'
    '      "parameter": "pollutant or parameter, e.g. NOx or pH, or null",\n'
    '      "limit_value": 30.0,\n'
    '      "limit_unit": "ppm",\n'
    '      "operator": "one of <=, <, >=, >, =, range, or null",\n'
    '      "frequency": "how often to monitor/report, or null",\n'
    '      "deadline": "reporting deadline/schedule, or null",\n'
    '      "citation": "permit section or regulatory reference, or null",\n'
    '      "source_segment_id": "the id of the segment this came from",\n'
    '      "source_quote": "the EXACT supporting text, copied verbatim",\n'
    '      "confidence": 0.0\n'
    '    }\n'
    '  ]\n'
    '}'
)


def build_user_prompt(segments: List[Segment]) -> str:
    """Build the extraction prompt from numbered permit segments."""
    lines = [
        "Extract every compliance obligation from the permit segments below.",
        "Return ONLY JSON matching this shape, with no commentary:",
        _SCHEMA_HINT,
        "",
        "Rules:",
        "- Copy 'source_quote' verbatim from the segment text; do not paraphrase.",
        "- Set 'source_segment_id' to the id of the segment you took it from.",
        "- Use null for any field that does not apply.",
        "- Do not output anything that is not in the permit text.",
        "",
        "PERMIT SEGMENTS:",
    ]
    for seg in segments:
        lines.append("[%s | page %d] %s" % (seg.segment_id, seg.page, seg.text))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Defensive parsing of model output
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
    """Pull the first JSON value out of a model response, tolerating fences and
    surrounding prose. Usually a dict, but a top level array is also returned as
    is (parse_obligations_payload handles both). None if nothing parseable.
    """
    if not text:
        return None
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} span.
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = cleaned.find("{", start + 1)
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None  # reject NaN / Infinity
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if m:
            try:
                f = float(m.group())
            except ValueError:
                return None
            return f if math.isfinite(f) else None
    return None


def _coerce_operator(value: Any) -> Optional[Operator]:
    if value is None:
        return None
    key = str(value).strip().lower()
    return _OPERATOR_LOOKUP.get(key)


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def parse_obligation(raw: Dict[str, Any], obligation_id: str) -> Obligation:
    """Build an Obligation from one raw model dict, defensively.

    Never raises on a messy dict: unknown operators become None, non numeric
    limits become None, and a missing or unknown source_segment_id is kept as
    given so that grounding can fail honestly on it later (segment id validity
    is deliberately checked by the verify stage, not here).
    """
    description = _coerce_str(raw.get("description")) or ""
    source_segment_id = _coerce_str(raw.get("source_segment_id")) or ""
    confidence = _coerce_float(raw.get("confidence"))
    if confidence is None:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return Obligation(
        obligation_id=obligation_id,
        description=description,
        parameter=_coerce_str(raw.get("parameter")),
        limit_value=_coerce_float(raw.get("limit_value")),
        limit_unit=_coerce_str(raw.get("limit_unit")),
        operator=_coerce_operator(raw.get("operator")),
        frequency=_coerce_str(raw.get("frequency")),
        deadline=_coerce_str(raw.get("deadline")),
        citation=_coerce_str(raw.get("citation")),
        source_segment_id=source_segment_id,
        source_quote=_coerce_str(raw.get("source_quote")) or "",
        model_confidence=confidence,
        confidence=confidence,
    )


def parse_obligations_payload(
    payload: Any,
    segments: List[Segment],
    id_prefix: str = "OB",
    start_index: int = 0,
) -> List[Obligation]:
    """Turn a parsed model payload into a list of Obligations.

    Accepts either a dict with an 'obligations' list or a bare list. Anything
    unparseable yields an empty list rather than an exception, so a bad model
    response never crashes the pipeline (B7). start_index keeps obligation ids
    unique when extraction is run in batches.
    """
    if isinstance(payload, dict):
        items = payload.get("obligations", [])
    elif isinstance(payload, list):
        items = payload
    else:
        items = []
    if not isinstance(items, list):
        items = []  # tolerate {"obligations": null} and other malformed shapes

    out: List[Obligation] = []
    n = start_index
    for raw in items:
        if not isinstance(raw, dict):
            continue
        out.append(parse_obligation(raw, "%s%04d" % (id_prefix, n)))
        n += 1
    return out


# Real permits can be hundreds of pages, far more than fits in one prompt. The
# real backends therefore extract in batches of segments and accumulate the
# results, keeping obligation ids unique across batches.
DEFAULT_BATCH_SIZE = 25


def run_batched_extraction(segments, call, id_prefix, batch_size=DEFAULT_BATCH_SIZE):
    """Extract obligations from segments in batches.

    `call` is a function that takes a list of segments and returns the model's
    raw text response for that batch. Each batch is parsed defensively and the
    obligations are concatenated. A batch that fails to parse contributes
    nothing rather than crashing the whole run.
    """
    obligations: List[Obligation] = []
    size = max(1, int(batch_size))
    for start in range(0, len(segments), size):
        batch = segments[start : start + size]
        content = call(batch) or ""
        payload = extract_json_object(content)
        obligations.extend(
            parse_obligations_payload(
                payload, batch, id_prefix=id_prefix, start_index=len(obligations)
            )
        )
    return obligations
