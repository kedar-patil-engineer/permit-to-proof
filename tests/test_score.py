"""The confidence formula and routing must be deterministic: fixed inputs give
fixed outputs (Part B6)."""

import pytest

from app.core.schema import Check, MatchType, Obligation, Severity, Status
from app.core.score import (
    apply_threshold,
    checks_score,
    compute_confidence,
    route_status,
    score_and_route,
)


def _passing_checks():
    return [
        Check(name="schema_complete", passed=True, severity=Severity.ERROR, message=""),
        Check(name="grounded", passed=True, severity=Severity.ERROR, message=""),
        Check(name="citation_present", passed=True, severity=Severity.WARNING, message=""),
        Check(name="unit_valid", passed=True, severity=Severity.WARNING, message=""),
        Check(name="range_plausible", passed=True, severity=Severity.WARNING, message=""),
        Check(name="operator_consistent", passed=True, severity=Severity.WARNING, message=""),
        Check(name="no_duplicate", passed=True, severity=Severity.INFO, message=""),
    ]


def _ob(**kw):
    base = dict(obligation_id="OB1", description="d", match_type=MatchType.EXACT)
    base.update(kw)
    ob = Obligation(**{k: v for k, v in base.items() if k != "checks"})
    ob.checks = base.get("checks", _passing_checks())
    return ob


def test_confidence_is_deterministic_and_exact():
    ob = _ob()
    # 0.25*0.9(model) + 0.40*1.0(exact) + 0.35*1.0(all checks pass)
    assert compute_confidence(ob, 0.9) == pytest.approx(0.975)


def test_confidence_repeatable():
    ob = _ob()
    assert compute_confidence(ob, 0.42) == compute_confidence(ob, 0.42)


def test_checks_score_full_when_all_pass():
    assert checks_score(_ob()) == pytest.approx(1.0)


def test_checks_score_drops_with_error_failure():
    failing = _passing_checks()
    failing[0] = Check(name="schema_complete", passed=False,
                       severity=Severity.ERROR, message="")
    ob = _ob(checks=failing)
    assert checks_score(ob) < 1.0


def test_grounding_none_lowers_confidence():
    exact = compute_confidence(_ob(match_type=MatchType.EXACT), 0.5)
    none = compute_confidence(_ob(match_type=MatchType.NONE), 0.5)
    assert none < exact


def test_route_flagged_on_error():
    failing = _passing_checks()
    failing[1] = Check(name="grounded", passed=False, severity=Severity.ERROR, message="")
    ob = _ob(checks=failing, confidence=0.99)
    assert route_status(ob, 0.6) is Status.FLAGGED


def test_route_needs_review_on_warning():
    warned = _passing_checks()
    warned[3] = Check(name="unit_valid", passed=False, severity=Severity.WARNING, message="")
    ob = _ob(checks=warned, confidence=0.99)
    assert route_status(ob, 0.6) is Status.NEEDS_REVIEW


def test_route_needs_review_on_low_confidence():
    ob = _ob(confidence=0.50)
    assert route_status(ob, 0.6) is Status.NEEDS_REVIEW


def test_route_verified_when_clean_and_confident():
    ob = _ob(confidence=0.90)
    assert route_status(ob, 0.6) is Status.VERIFIED


def test_route_preserves_user_override():
    ob = _ob(confidence=0.0, status=Status.USER_ACCEPTED)
    assert route_status(ob, 0.6) is Status.USER_ACCEPTED


def test_apply_threshold_reroutes_without_touching_confidence():
    ob = _ob(confidence=0.70)
    apply_threshold([ob], 0.6)
    assert ob.status is Status.VERIFIED
    apply_threshold([ob], 0.8)
    assert ob.status is Status.NEEDS_REVIEW
    assert ob.confidence == 0.70  # unchanged


def test_score_and_route_consumes_model_confidence_once():
    ob = _ob(confidence=0.9)
    score_and_route([ob], 0.6)
    first = ob.confidence
    # Running apply_threshold (not score) must not recompute confidence.
    apply_threshold([ob], 0.6)
    assert ob.confidence == first


def test_score_and_route_is_idempotent_on_double_call():
    # B6: model confidence is consumed once; re-scoring must not drift it.
    ob = _ob(confidence=0.9)
    score_and_route([ob], 0.6)
    first = ob.confidence
    score_and_route([ob], 0.6)
    score_and_route([ob], 0.6)
    assert ob.confidence == first
    assert ob.model_confidence == 0.9
