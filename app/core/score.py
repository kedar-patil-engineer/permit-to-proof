"""Confidence scoring and status routing.

Confidence is a single number in [0, 1] combining three signals (Part B6):

    * model_confidence  the backend's own self reported confidence, if any
    * grounding         how strongly the source_quote matched (exact > fuzzy > none)
    * checks_score      the share of checks passed, weighted by severity

The formula is a fixed weighted average, kept here, documented, and unit
tested so that identical inputs always yield identical outputs.

Status is then routed deterministically from the checks and the confidence,
against a user controlled threshold (Part B5.1):

    any failed error check                    -> FLAGGED
    a failed warning check, or low confidence -> NEEDS_REVIEW
    everything passes and confidence is high  -> VERIFIED

User overrides (USER_ACCEPTED / USER_REJECTED) are never silently changed.
"""

from __future__ import annotations

from typing import List

from .schema import MatchType, Obligation, Severity, Status

# Weights for the three confidence signals. They sum to 1.0.
W_MODEL = 0.25
W_GROUNDING = 0.40
W_CHECKS = 0.35

# Grounding strength mapped to a [0, 1] contribution.
_GROUNDING_STRENGTH = {
    MatchType.EXACT: 1.0,
    MatchType.FUZZY: 0.6,
    MatchType.NONE: 0.0,
}

# Per severity weight used when computing the share of checks passed. Errors
# dominate, warnings matter, info barely moves the needle.
_SEVERITY_WEIGHT = {
    Severity.ERROR: 3.0,
    Severity.WARNING: 1.0,
    Severity.INFO: 0.25,
}

# Default routing threshold. Exposed as a slider in the UI; this slider is the
# automation versus review tradeoff curve the paper reports (B6).
DEFAULT_THRESHOLD = 0.60

# Statuses set by a human that the router must never overwrite.
_OVERRIDE_STATES = (Status.USER_ACCEPTED, Status.USER_REJECTED)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def checks_score(ob: Obligation) -> float:
    """Severity weighted fraction of checks that passed. 1.0 if no checks."""
    if not ob.checks:
        return 1.0
    total = 0.0
    earned = 0.0
    for c in ob.checks:
        w = _SEVERITY_WEIGHT[c.severity]
        total += w
        if c.passed:
            earned += w
    return earned / total if total else 1.0


def compute_confidence(ob: Obligation, model_confidence: float) -> float:
    """Combine the three signals into a confidence in [0, 1].

    model_confidence is supplied explicitly (the backend stores its self
    reported value in ob.confidence before scoring, and the pipeline passes it
    in here) so the function is pure with respect to its arguments.
    """
    grounding = _GROUNDING_STRENGTH.get(ob.match_type, 0.0)
    score = (
        W_MODEL * _clamp(model_confidence)
        + W_GROUNDING * grounding
        + W_CHECKS * checks_score(ob)
    )
    return _clamp(score)


def route_status(ob: Obligation, threshold: float) -> Status:
    """Decide the obligation's status from its checks and confidence."""
    if ob.status in _OVERRIDE_STATES:
        return ob.status
    if ob.failed_checks(Severity.ERROR):
        return Status.FLAGGED
    if ob.failed_checks(Severity.WARNING) or ob.confidence < threshold:
        return Status.NEEDS_REVIEW
    return Status.VERIFIED


def score_and_route(
    obligations: List[Obligation],
    threshold: float = DEFAULT_THRESHOLD,
) -> List[Obligation]:
    """Set confidence and status on every obligation. Mutates in place.

    The backend's self reported confidence is captured once into
    model_confidence (from ob.model_confidence, or ob.confidence the first
    time) and then never reconsumed, so calling this twice on the same objects
    is idempotent: the final confidence does not drift.
    """
    for ob in obligations:
        if ob.model_confidence is None:
            ob.model_confidence = ob.confidence
        ob.confidence = compute_confidence(ob, ob.model_confidence)
        ob.status = route_status(ob, threshold)
    return obligations


def apply_threshold(
    obligations: List[Obligation],
    threshold: float,
) -> List[Obligation]:
    """Re-route status against a new threshold without recomputing confidence.

    Used when the UI threshold slider moves: confidence is already final, so we
    only revisit the routing decision. User overrides are preserved because
    route_status leaves USER_ACCEPTED / USER_REJECTED untouched.
    """
    for ob in obligations:
        ob.status = route_status(ob, threshold)
    return obligations
