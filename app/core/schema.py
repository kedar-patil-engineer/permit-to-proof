"""Data contracts shared by every stage of the Permit-to-Proof pipeline.

These Pydantic models are the agreements that ingest, extract, verify, score,
and the user interface all rely on. The field names defined here are part of
the contract and must not drift. See Part B4 of the master specification.

The design rule the whole project rests on: the language model proposes
candidate obligations, and a separate deterministic layer disposes. Nothing in
this file performs verification; it only describes the shape of the data that
flows between stages.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Operator(str, Enum):
    """Comparison operator that relates a measured value to its limit."""

    LE = "<="
    LT = "<"
    GE = ">="
    GT = ">"
    EQ = "="
    RANGE = "range"


class MatchType(str, Enum):
    """How well an obligation's source_quote was found in the cited segment.

    Recorded by the verification layer. It is deliberately kept on every
    obligation because the calibration analysis in the paper depends on it
    (see Part A5 and B4.2): grounding strength is one of the signals that
    feeds the confidence score, and exact beats fuzzy beats none.
    """

    EXACT = "exact"
    FUZZY = "fuzzy"
    NONE = "none"


class Severity(str, Enum):
    """How seriously a failed verification check should be treated."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Status(str, Enum):
    """Lifecycle state of a single obligation as it moves through the system."""

    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    FLAGGED = "FLAGGED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    USER_ACCEPTED = "USER_ACCEPTED"
    USER_REJECTED = "USER_REJECTED"


class Segment(BaseModel):
    """A chunk of permit text with enough provenance to locate it again.

    Produced by the ingest stage. start_char and end_char are positions in the
    full concatenated document text so a quote can always be traced back to the
    page it came from.
    """

    model_config = ConfigDict(extra="forbid")

    segment_id: str
    text: str
    page: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)


class Check(BaseModel):
    """One verification result attached to an obligation by the verify stage."""

    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    severity: Severity
    message: str


class Obligation(BaseModel):
    """One compliance obligation extracted from a permit.

    A backend fills in everything down to source_quote and leaves status
    PENDING. The verify stage attaches checks and sets match_type. The score
    stage sets confidence and the final status. The fields below are the
    required contract from Part B4.2.
    """

    model_config = ConfigDict(extra="forbid")

    obligation_id: str
    description: str
    parameter: Optional[str] = None
    limit_value: Optional[float] = None
    limit_unit: Optional[str] = None
    operator: Optional[Operator] = None
    frequency: Optional[str] = None
    deadline: Optional[str] = None
    citation: Optional[str] = None

    source_segment_id: str = ""
    source_quote: str = ""

    match_type: MatchType = MatchType.NONE
    # The backend's own self reported confidence, kept separate from the final
    # combined confidence so re-scoring is idempotent and so the paper can study
    # raw vs calibrated confidence. None until a backend supplies one.
    model_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: Status = Status.PENDING
    checks: List[Check] = Field(default_factory=list)

    def has_numeric_limit(self) -> bool:
        """True when this obligation asserts a numeric limit value."""
        return self.limit_value is not None

    def failed_checks(self, severity: Optional[Severity] = None) -> List[Check]:
        """Return checks that did not pass, optionally filtered by severity."""
        out = [c for c in self.checks if not c.passed]
        if severity is not None:
            out = [c for c in out if c.severity == severity]
        return out
