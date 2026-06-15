"""End-to-end pipeline on the sample permit with the Mock backend, in both
verification ON and OFF modes (Part B11). Everything here runs offline."""

from app.core.pipeline import (
    error_detection_lift,
    run_pipeline,
    summarize,
)
from app.core.schema import Status
from app.llm.mock import MockBackend


def _dump(obs):
    return [o.model_dump(mode="json") for o in obs]


def test_pipeline_on_produces_mixed_statuses(sample_pdf_path):
    res = run_pipeline(sample_pdf_path, MockBackend(), threshold=0.6,
                       verification_enabled=True)
    statuses = {o.status for o in res.obligations}
    assert len(res.obligations) > 10
    assert Status.VERIFIED in statuses
    assert Status.FLAGGED in statuses          # at least one hallucination / error
    assert Status.NEEDS_REVIEW in statuses     # at least one domain warning


def test_pipeline_is_deterministic(sample_pdf_path):
    a = run_pipeline(sample_pdf_path, MockBackend(), verification_enabled=True)
    b = run_pipeline(sample_pdf_path, MockBackend(), verification_enabled=True)
    assert _dump(a.obligations) == _dump(b.obligations)


def test_pipeline_off_trusts_everything(sample_pdf_path):
    off = run_pipeline(sample_pdf_path, MockBackend(), verification_enabled=False)
    assert all(o.status is Status.VERIFIED for o in off.obligations)
    assert all(o.checks == [] for o in off.obligations)


def test_grounding_flags_the_planted_hallucination(sample_pdf_path):
    res = run_pipeline(sample_pdf_path, MockBackend(), verification_enabled=True)
    ungrounded = [o for o in res.obligations
                  if o.match_type.value == "none"]
    assert ungrounded, "the sample must contain at least one ungrounded obligation"
    assert all(o.status is Status.FLAGGED for o in ungrounded)


def test_error_detection_lift_is_positive(sample_pdf_path):
    on = run_pipeline(sample_pdf_path, MockBackend(), verification_enabled=True)
    off = run_pipeline(sample_pdf_path, MockBackend(), verification_enabled=False)
    lift = error_detection_lift(on, off)
    assert lift["issues_caught_off"] == 0
    assert lift["lift"] > 0


def test_summarize_shape(sample_pdf_path):
    res = run_pipeline(sample_pdf_path, MockBackend(), verification_enabled=True)
    m = summarize(res.obligations)
    for key in ("total", "status_counts", "verified_rate", "issues", "flag_reasons"):
        assert key in m
    assert 0.0 <= m["verified_rate"] <= 1.0


def test_char_offset_invariant(sample_pdf_path):
    res = run_pipeline(sample_pdf_path, MockBackend(), verification_enabled=True)
    for s in res.segments:
        assert res.full_text[s.start_char:s.end_char] == s.text
