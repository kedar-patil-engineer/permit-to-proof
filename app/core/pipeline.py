"""The pipeline: run the five stages in order, with a verification ON/OFF switch.

    PDF
      -> ingest()          List[Segment]    (text, page, char positions)
      -> extract()         List[Obligation] (raw, status = PENDING)
      -> verify()          List[Obligation] (checks[] + match_type attached)
      -> score_and_route() List[Obligation] (confidence, status)

The ON/OFF switch is the heart of the paper's headline result (Part A5.3, B7):
with verification ON the deterministic layer catches problems; with it OFF the
raw model output is trusted as is. The difference is the error detection lift.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .ingest import PdfSource, ingest_pdf, segments_to_text_map
from .schema import MatchType, Obligation, Segment, Severity, Status
from .score import DEFAULT_THRESHOLD, score_and_route
from .verify import verify_all


@dataclass
class PipelineResult:
    """Everything one run produces, ready for the UI or for metrics."""

    obligations: List[Obligation]
    segments: List[Segment]
    full_text: str
    verification_enabled: bool
    threshold: float
    backend_name: str = ""
    source_name: str = ""

    @property
    def page_count(self) -> int:
        return max((s.page for s in self.segments), default=0)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def run_pipeline(
    source: PdfSource,
    backend,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    verification_enabled: bool = True,
    backend_name: str = "",
    source_name: str = "",
) -> PipelineResult:
    """Ingest, extract, and (optionally) verify and score a permit."""
    full_text, segments = ingest_pdf(source)
    obligations = backend.extract_obligations(segments)

    if verification_enabled:
        text_map = segments_to_text_map(segments)
        verify_all(obligations, text_map)
        score_and_route(obligations, threshold)
    else:
        # Raw mode: trust the model as is. No grounding, no checks, everything
        # marked Verified. This is the baseline the ON mode is measured against.
        for ob in obligations:
            ob.checks = []
            ob.match_type = MatchType.NONE
            raw = ob.model_confidence if ob.model_confidence is not None else ob.confidence
            ob.confidence = _clamp(raw)
            ob.status = Status.VERIFIED

    return PipelineResult(
        obligations=obligations,
        segments=segments,
        full_text=full_text,
        verification_enabled=verification_enabled,
        threshold=threshold,
        backend_name=backend_name or getattr(backend, "name", ""),
        source_name=source_name,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

_ISSUE_STATUSES = (Status.FLAGGED, Status.NEEDS_REVIEW)


def summarize(obligations: List[Obligation]) -> Dict:
    """Counts, verified rate, and a flag reason breakdown for a result set."""
    total = len(obligations)
    status_counts = Counter(ob.status.value for ob in obligations)
    verified = status_counts.get(Status.VERIFIED.value, 0)
    accepted = status_counts.get(Status.USER_ACCEPTED.value, 0)

    reason_counts: Counter = Counter()
    for ob in obligations:
        for check in ob.failed_checks():
            reason_counts[check.name] += 1

    issues = sum(1 for ob in obligations if ob.status in _ISSUE_STATUSES)

    return {
        "total": total,
        "status_counts": dict(status_counts),
        "verified": verified,
        "verified_rate": (verified / total) if total else 0.0,
        "trusted_rate": ((verified + accepted) / total) if total else 0.0,
        "issues": issues,
        "flag_reasons": dict(reason_counts.most_common()),
    }


def error_detection_lift(on: PipelineResult, off: PipelineResult) -> Dict:
    """How many obligations verification surfaces versus trusting raw output.

    The in app proxy for the paper's headline result: the count of obligations
    the verification layer routes to a human (FLAGGED or NEEDS_REVIEW) that the
    raw OFF pipeline would have passed silently as Verified.
    """
    on_issues = sum(1 for ob in on.obligations if ob.status in _ISSUE_STATUSES)
    off_issues = sum(1 for ob in off.obligations if ob.status in _ISSUE_STATUSES)
    # Partition the surfaced issues the same way route_status does: a failed
    # ERROR check (grounding/schema) -> FLAGGED, a warning or low confidence
    # -> NEEDS_REVIEW. The paper can cite hard errors caught separately from
    # warnings routed for review.
    on_flagged = sum(1 for ob in on.obligations if ob.status is Status.FLAGGED)
    on_review = sum(1 for ob in on.obligations if ob.status is Status.NEEDS_REVIEW)
    total = len(on.obligations)
    return {
        "issues_caught_on": on_issues,
        "issues_caught_off": off_issues,
        "lift": on_issues - off_issues,
        "lift_rate": ((on_issues - off_issues) / total) if total else 0.0,
        "lift_flagged": on_flagged,        # hard errors caught (grounding/schema)
        "lift_needs_review": on_review,    # warnings / low confidence routed
        "total": total,
    }
