"""The schema accepts valid records and rejects malformed ones (Part B11)."""

import pytest
from pydantic import ValidationError

from app.core.schema import (
    Check,
    MatchType,
    Obligation,
    Operator,
    Segment,
    Severity,
    Status,
)


def test_segment_valid():
    s = Segment(segment_id="S0001", text="hello", page=1, start_char=0, end_char=5)
    assert s.page == 1


def test_segment_rejects_zero_page():
    with pytest.raises(ValidationError):
        Segment(segment_id="S0001", text="hi", page=0, start_char=0, end_char=2)


def test_check_valid():
    c = Check(name="grounded", passed=True, severity=Severity.ERROR, message="ok")
    assert c.severity is Severity.ERROR


def test_check_rejects_bad_severity():
    with pytest.raises(ValidationError):
        Check(name="grounded", passed=True, severity="catastrophic", message="x")


def test_obligation_minimal_defaults():
    ob = Obligation(obligation_id="OB0001", description="do a thing")
    assert ob.status is Status.PENDING
    assert ob.match_type is MatchType.NONE
    assert ob.confidence == 0.0
    assert ob.checks == []


def test_obligation_coerces_operator_symbol():
    ob = Obligation(obligation_id="OB1", description="d", operator="<=")
    assert ob.operator is Operator.LE


def test_obligation_rejects_unknown_operator():
    with pytest.raises(ValidationError):
        Obligation(obligation_id="OB1", description="d", operator="much-less-than")


def test_obligation_rejects_non_numeric_limit():
    with pytest.raises(ValidationError):
        Obligation(obligation_id="OB1", description="d", limit_value="not a number")


def test_obligation_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        Obligation(obligation_id="OB1", description="d", confidence=1.5)


def test_model_confidence_optional_and_bounded():
    ob = Obligation(obligation_id="OB1", description="d")
    assert ob.model_confidence is None
    with pytest.raises(ValidationError):
        Obligation(obligation_id="OB1", description="d", model_confidence=2.0)


def test_obligation_forbids_extra_fields():
    with pytest.raises(ValidationError):
        Obligation(obligation_id="OB1", description="d", surprise="field")


def test_has_numeric_limit_helper():
    assert Obligation(obligation_id="o", description="d", limit_value=3.0).has_numeric_limit()
    assert not Obligation(obligation_id="o", description="d").has_numeric_limit()
