"""The gold answer key: trusted true obligations to grade extraction against.

A gold set is one JSON file per permit, validated on load. The field names
mirror the scored subset of the Obligation schema so the matcher compares like
with like and can reuse the verifier's normalizers.

Honesty note (Part A5.4): the bundled gold set is ILLUSTRATIVE author-known
truth on a synthetic permit, used to exercise this harness. It is NOT the real
expert answer key, which must be built by a domain expert on real permits. The
label_provenance field records which kind a gold set is, and the report and UI
surface it on every figure so the two are never confused.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from app.core.schema import Operator


class LabelProvenance(str, Enum):
    ILLUSTRATIVE_AUTHOR_KNOWN = "ILLUSTRATIVE_AUTHOR_KNOWN"
    EXPERT_SINGLE = "EXPERT_SINGLE"
    EXPERT_ADJUDICATED = "EXPERT_ADJUDICATED"


class GoldObligation(BaseModel):
    """One true obligation. Mirrors the scored fields of schema.Obligation."""

    model_config = ConfigDict(extra="forbid")

    gold_id: str
    description: str
    parameter: Optional[str] = None
    limit_value: Optional[float] = None
    limit_unit: Optional[str] = None
    operator: Optional[Operator] = None
    frequency: Optional[str] = None
    deadline: Optional[str] = None
    citation: Optional[str] = None
    source_segment_id: Optional[str] = None
    is_obligation: bool = True


class GoldSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    permit_id: str
    source_pdf: Optional[str] = None
    label_provenance: LabelProvenance
    labeler: str = ""
    notes: str = ""
    obligations: List[GoldObligation]

    @property
    def is_illustrative(self) -> bool:
        return self.label_provenance == LabelProvenance.ILLUSTRATIVE_AUTHOR_KNOWN


def load_gold(path: str) -> GoldSet:
    """Load and validate a gold set, failing loud on a malformed key."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return GoldSet.model_validate(data)


def discover_gold(pdf_path: str) -> Optional[str]:
    """Find the gold file paired with a permit PDF by file stem."""
    p = Path(pdf_path)
    candidate = p.parent / "gold" / (p.stem + ".json")
    return str(candidate) if candidate.exists() else None
