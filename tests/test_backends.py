"""Backend behavior: the Mock backend is deterministic, and the OpenAI/Ollama
backends import safely and are skipped (not failed) without credentials
(Part B11). The defensive model-output parser is also covered here."""

import os

import pytest

from app.core.schema import Obligation, Segment
from app.llm.base import (
    extract_json_object,
    parse_obligation,
    parse_obligations_payload,
)
from app.llm.mock import MockBackend
from app.llm.ollama_backend import OllamaBackend
from app.llm.openai_backend import OpenAIBackend

# Live backend calls are opt-in so the default suite is hermetically offline,
# even on a machine that happens to be running an Ollama server (Part B11/B12).
LIVE_BACKENDS = os.environ.get("PTP_RUN_LIVE_BACKENDS") == "1"


def _segments():
    return [
        Segment(segment_id="S0001",
                text="Condition 3.1. NOx shall not exceed 30 ppm. (40 CFR 60.44c)",
                page=1, start_char=0, end_char=58),
        Segment(segment_id="S0002",
                text="Condition 4.1. pH shall be maintained between 6.0 and 9.0 "
                     "standard units.",
                page=1, start_char=59, end_char=130),
    ]


# --- Mock determinism -----------------------------------------------------

def test_mock_is_deterministic():
    segs = _segments()
    a = MockBackend().extract_obligations(segs)
    b = MockBackend().extract_obligations(segs)
    assert [o.model_dump(mode="json") for o in a] == [o.model_dump(mode="json") for o in b]


def test_mock_quotes_are_grounded_in_segments():
    segs = _segments()
    obs = MockBackend().extract_obligations(segs)
    assert obs
    by_id = {s.segment_id: s.text for s in segs}
    for ob in obs:
        # regex-extracted obligations quote the segment verbatim
        assert ob.source_quote in by_id.get(ob.source_segment_id, "")


def test_mock_leaves_status_pending():
    obs = MockBackend().extract_obligations(_segments())
    assert all(o.status.value == "PENDING" for o in obs)


# --- defensive parsing of model output ------------------------------------

def test_extract_json_object_strips_code_fences():
    text = '```json\n{"obligations": []}\n```'
    assert extract_json_object(text) == {"obligations": []}


def test_extract_json_object_finds_embedded_object():
    text = 'Sure! Here is the result:\n{"obligations": [{"description": "x"}]} Thanks.'
    payload = extract_json_object(text)
    assert payload["obligations"][0]["description"] == "x"


def test_extract_json_object_returns_none_on_garbage():
    assert extract_json_object("not json at all") is None


def test_parse_payload_tolerates_bad_records():
    segs = _segments()
    payload = {
        "obligations": [
            {"description": "good", "source_segment_id": "S0001",
             "source_quote": "NOx shall not exceed 30 ppm", "operator": "<=",
             "limit_value": 30, "limit_unit": "ppm", "parameter": "NOx"},
            {"description": "messy", "operator": "definitely-not-an-operator",
             "limit_value": "thirty-ish"},
            "this is not even a dict",
        ]
    }
    obs = parse_obligations_payload(payload, segs)
    assert len(obs) == 2  # the string is dropped
    assert isinstance(obs[0], Obligation)
    assert obs[1].operator is None       # unknown operator coerced to None
    assert obs[1].limit_value is None    # unparseable number coerced to None


def test_parse_payload_empty_on_non_payload():
    assert parse_obligations_payload(None, _segments()) == []
    assert parse_obligations_payload(42, _segments()) == []


def test_parse_payload_tolerates_null_obligations():
    # {"obligations": null} is a plausible bad model response; must not raise.
    assert parse_obligations_payload({"obligations": None}, _segments()) == []
    assert parse_obligations_payload({"obligations": 7}, _segments()) == []


def test_coerce_float_rejects_non_finite():
    inf_ob = parse_obligation({"description": "d", "limit_value": float("inf")}, "OB0")
    nan_ob = parse_obligation({"description": "d", "limit_value": float("nan")}, "OB1")
    assert inf_ob.limit_value is None
    assert nan_ob.limit_value is None


# --- real backends import safely and skip without credentials -------------

def test_openai_backend_imports_and_reports_availability(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert OpenAIBackend.is_available() is False
    backend = OpenAIBackend()  # constructs without raising
    assert backend.name == "OpenAI"


@pytest.mark.skipif(not LIVE_BACKENDS, reason="Set PTP_RUN_LIVE_BACKENDS=1 to run live backend calls.")
def test_openai_real_call_optin():
    if not OpenAIBackend.is_available():
        pytest.skip("No OPENAI_API_KEY configured.")
    obs = OpenAIBackend().extract_obligations(_segments())
    assert isinstance(obs, list)


def test_ollama_backend_imports_and_reports_availability():
    assert isinstance(OllamaBackend.is_available(), bool)
    backend = OllamaBackend()
    assert backend.name == "Ollama"


@pytest.mark.skipif(not LIVE_BACKENDS, reason="Set PTP_RUN_LIVE_BACKENDS=1 to run live backend calls.")
def test_ollama_real_call_optin():
    # Opt-in only: a stray local Ollama server must never pull the default
    # suite onto the network. Short timeout so a stuck server cannot hang it.
    if not OllamaBackend.is_available():
        pytest.skip("No Ollama server reachable.")
    obs = OllamaBackend(timeout=10).extract_obligations(_segments())
    assert isinstance(obs, list)
