"""The evaluation metrics (Part A5). Pure functions, JSON-serializable output.

All four metrics reuse the verifier's own normalizers (verify.normalize_unit,
verify.lookup_parameter, verify.normalize_text) so the grader and the verifier
agree on what counts as the same parameter or unit.

  1. Extraction precision/recall/F1  - match_extractions + extraction_prf
  2. True verification lift vs gold   - verification_lift
  3. Confidence calibration (ECE)     - calibration
  4. Selective-prediction trade-off   - selective_curve
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.core.schema import Obligation, Operator, Status
from app.core.score import apply_threshold
from app.core.verify import lookup_parameter, normalize_text, normalize_unit
from app.eval.gold import GoldObligation, GoldSet

# Numeric match tolerance: 1% relative (plus a tiny absolute floor) absorbs
# 30 vs 30.0 while still failing a fabricated 250 vs 30 or 99999 vs 30.
ABS_TOL = 1e-9
REL_TOL = 0.01
# Description similarity needed to match two narrative (no-parameter) obligations.
DESC_SIM = 0.60

_OPERATOR_FAMILY = {
    Operator.LE: "le", Operator.LT: "le",
    Operator.GE: "ge", Operator.GT: "ge",
    Operator.EQ: "eq", Operator.RANGE: "range",
}

_ISSUE_STATUSES = (Status.FLAGGED, Status.NEEDS_REVIEW)


def _op_family(op: Optional[Operator]) -> Optional[str]:
    return _OPERATOR_FAMILY.get(op) if op is not None else None


def _norm_param(p: Optional[str]) -> str:
    spec = lookup_parameter(p)
    if spec is not None:
        return spec.canonical.lower()
    return (p or "").strip().lower()


def _value_match(ev: Optional[float], gv: Optional[float]) -> bool:
    if ev is None and gv is None:
        return True
    if ev is None or gv is None:
        return False
    return abs(ev - gv) <= max(ABS_TOL, REL_TOL * abs(gv))


def is_match(ext: Obligation, gold: GoldObligation) -> bool:
    """True if an extracted obligation is the same compliance fact as a gold one."""
    ep, gp = _norm_param(ext.parameter), _norm_param(gold.parameter)
    if not ep and not gp:
        # Narrative obligations carry no parameter; compare descriptions.
        sim = difflib.SequenceMatcher(
            None, normalize_text(ext.description), normalize_text(gold.description)
        ).ratio()
        if sim < DESC_SIM:
            return False
    elif ep != gp:
        return False

    if not _value_match(ext.limit_value, gold.limit_value):
        return False
    if gold.limit_value is not None:
        if normalize_unit(ext.limit_unit) != normalize_unit(gold.limit_unit):
            return False
    if gold.operator is not None:
        if _op_family(ext.operator) != _op_family(gold.operator):
            return False
    return True


# ---------------------------------------------------------------------------
# 1. Extraction matching and precision/recall/F1
# ---------------------------------------------------------------------------

@dataclass
class MatchRecord:
    ext_id: Optional[str]
    gold_id: Optional[str]
    outcome: str  # "TP" | "FP" | "FN"


@dataclass
class MatchResult:
    records: List[MatchRecord]
    tp: int
    fp: int
    fn: int
    matched_ext_ids: set


def match_extractions(extracted: List[Obligation], gold: GoldSet) -> MatchResult:
    """Greedy one-to-one alignment of extractions to gold obligations.

    Deterministic: extractions are visited in obligation_id order and claim the
    first still-unclaimed gold they match. A second extraction of the same fact
    (a duplicate) therefore cannot claim the same gold and is counted as a false
    positive, which is correct.
    """
    golds = list(gold.obligations)
    claimed = set()
    records: List[MatchRecord] = []
    matched_ext: set = set()

    for ext in sorted(extracted, key=lambda o: o.obligation_id):
        hit = None
        for g in golds:
            if g.gold_id in claimed:
                continue
            if is_match(ext, g):
                hit = g
                break
        if hit is not None:
            claimed.add(hit.gold_id)
            matched_ext.add(ext.obligation_id)
            records.append(MatchRecord(ext.obligation_id, hit.gold_id, "TP"))
        else:
            records.append(MatchRecord(ext.obligation_id, None, "FP"))

    for g in golds:
        if g.gold_id not in claimed:
            records.append(MatchRecord(None, g.gold_id, "FN"))

    tp = len(claimed)
    fp = sum(1 for r in records if r.outcome == "FP")
    fn = sum(1 for r in records if r.outcome == "FN")
    return MatchResult(records, tp, fp, fn, matched_ext)


def near_miss_analysis(extracted: List[Obligation], gold: GoldSet) -> Dict:
    """Explain WHY each gold obligation did not get a full match.

    Beyond the TP/FP/FN counts, this categorizes every unmatched gold as found
    with a wrong operator, found with a different unit, or not extracted at all,
    so the paper's error analysis is precise and honest rather than lumping all
    misses together.
    """
    mr = match_extractions(extracted, gold)
    matched = {r.gold_id for r in mr.records if r.outcome == "TP"}
    summary = {"matched": 0, "operator_mismatch": 0, "unit_mismatch": 0,
               "not_extracted": 0}
    details = []
    for g in gold.obligations:
        if g.gold_id in matched:
            summary["matched"] += 1
            details.append({"gold_id": g.gold_id, "category": "matched"})
            continue
        cand = None
        if g.limit_value is not None:
            for o in extracted:
                if o.limit_value is None:
                    continue
                if _norm_param(o.parameter) == _norm_param(g.parameter) and \
                        abs(o.limit_value - g.limit_value) <= max(1e-9, 0.03 * abs(g.limit_value)):
                    cand = o
                    break
        if cand is None:
            category = "not_extracted"
        elif normalize_unit(cand.limit_unit) != normalize_unit(g.limit_unit):
            category = "unit_mismatch"
        elif g.operator is not None and _op_family(cand.operator) != _op_family(g.operator):
            category = "operator_mismatch"
        else:
            category = "not_extracted"
        summary[category] += 1
        rec = {"gold_id": g.gold_id, "category": category}
        if cand is not None:
            rec["model"] = {"parameter": cand.parameter, "limit_value": cand.limit_value,
                            "limit_unit": cand.limit_unit,
                            "operator": cand.operator.value if cand.operator else None,
                            "status": cand.status.value}
        details.append(rec)
    return {"summary": summary, "details": details}


def _param_compatible(a: Optional[str], b: Optional[str]) -> bool:
    """True if two parameter labels plausibly name the same quantity.

    Canonical equality first (the verifier's synonym map), then a fuzzy fallback
    on the raw labels for descriptive synonyms the map does not know. Used only
    after value and unit already match.

    The fuzzy fallback is deliberately restricted to longer descriptive labels.
    Short formula-like labels (NOx, CO, CO2, SO2, SO3, BOD, COD, TSS, TDS) name
    distinct pollutants that differ by a single character or digit, so substring
    or character-ratio comparison would wrongly equate them; for these only
    canonical equality counts.
    """
    if _norm_param(a) == _norm_param(b):
        return True
    na, nb = normalize_text(a or ""), normalize_text(b or "")
    if not na or not nb:
        return False
    # Short, formula-like labels must match canonically, never fuzzily.
    if len(na.replace(" ", "")) <= 4 or len(nb.replace(" ", "")) <= 4:
        return False
    if na in nb or nb in na:
        return True
    if difflib.SequenceMatcher(None, na, nb).ratio() >= 0.6:
        return True
    ta, tb = set(na.split()), set(nb.split())
    shared = {t for t in (ta & tb) if len(t) >= 4}  # a substantive shared word
    return bool(shared) and len(ta & tb) / len(ta | tb) >= 0.4


def limit_detection_metrics(extracted: List[Obligation], gold: GoldSet) -> Dict:
    """The IE-standard decomposition for numeric limits: detection vs attribute
    accuracy. Detection counts a numeric limit as found when an extraction has
    the same value (within tolerance), the same unit, and a compatible parameter
    label, regardless of operator. Of those detected, we then report how often
    the operator is also correct. This separates "did we find the limit" from
    "did we get every attribute exactly right", which the strict 4-way F1 folds
    together.
    """
    gnum = [g for g in gold.obligations if g.limit_value is not None]
    enum = sorted([e for e in extracted if e.limit_value is not None],
                  key=lambda o: o.obligation_id)
    used = set()
    detected = 0
    op_correct = 0
    for g in gnum:
        hit = None
        for i, e in enumerate(enum):
            if i in used:
                continue
            if not _value_match(e.limit_value, g.limit_value):
                continue
            if normalize_unit(e.limit_unit) != normalize_unit(g.limit_unit):
                continue
            if _param_compatible(e.parameter, g.parameter):
                hit = i
                break
        if hit is not None:
            used.add(hit)
            detected += 1
            e = enum[hit]
            if g.operator is None or _op_family(e.operator) == _op_family(g.operator):
                op_correct += 1
    n = len(gnum)
    return {
        "n_numeric_gold": n,
        "n_numeric_extracted": len(enum),
        "detected": detected,
        "detection_recall": (detected / n) if n else 0.0,
        "operator_correct_given_detected": (op_correct / detected) if detected else 0.0,
    }


def extraction_prf(mr: MatchResult) -> Dict:
    p = mr.tp / (mr.tp + mr.fp) if (mr.tp + mr.fp) else 0.0
    r = mr.tp / (mr.tp + mr.fn) if (mr.tp + mr.fn) else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return {
        "precision": p, "recall": r, "f1": f1,
        "tp": mr.tp, "fp": mr.fp, "fn": mr.fn,
        "matches": [vars(rec) for rec in mr.records],
    }


# ---------------------------------------------------------------------------
# 2. True verification lift vs gold
# ---------------------------------------------------------------------------

def verification_lift(on, off, gold: GoldSet) -> Dict:
    """Error-detection performance of the verification layer measured vs gold.

    An error is an extraction that matches no gold obligation (a hallucinated,
    mislabeled, mis-valued, wrong-unit, flipped-operator, or duplicate record).
    The layer "catches" an error when it routes it to a human (FLAGGED or
    NEEDS_REVIEW). Lift is the error-detection recall ON minus OFF; since the
    OFF baseline trusts everything, OFF recall is 0, so lift = recall(ON).
    """
    mr = match_extractions(on.obligations, gold)
    error_ids = {r.ext_id for r in mr.records if r.outcome == "FP"}

    def confusion(obligations):
        tp = fn = fp = tn = 0
        for o in obligations:
            err = o.obligation_id in error_ids
            flagged = o.status in _ISSUE_STATUSES
            if err and flagged:
                tp += 1
            elif err and not flagged:
                fn += 1
            elif (not err) and flagged:
                fp += 1
            else:
                tn += 1
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                "recall": recall, "precision": precision}

    con_on = confusion(on.obligations)
    con_off = confusion(off.obligations)
    return {
        "on": con_on,
        "off": con_off,
        "lift": con_on["recall"] - con_off["recall"],
        "lift_recall": con_on["recall"],
        "errors_caught_on": con_on["tp"],
        "errors_caught_off": con_off["tp"],
        "n_true_errors": len(error_ids),
    }


# ---------------------------------------------------------------------------
# 3. Confidence calibration (ECE + reliability diagram)
# ---------------------------------------------------------------------------

def calibration(extracted: List[Obligation], gold: GoldSet,
                n_bins: int = 10, use_model_confidence: bool = False) -> Dict:
    """Expected Calibration Error and reliability bins.

    Correctness is whether the obligation matched gold (1) or not (0). The
    probability is the final calibrated confidence, or the backend's raw
    self-reported confidence when use_model_confidence is set, so the paper can
    contrast the two.
    """
    correct = match_extractions(extracted, gold).matched_ext_ids
    points = []  # (confidence, correctness)
    for o in extracted:
        if use_model_confidence and o.model_confidence is not None:
            conf = o.model_confidence
        else:
            conf = o.confidence
        points.append((conf, 1 if o.obligation_id in correct else 0))

    n = len(points)
    bins = []
    ece = 0.0
    mce = 0.0
    for m in range(1, n_bins + 1):
        lo = (m - 1) / n_bins
        hi = m / n_bins
        if m == 1:
            members = [pt for pt in points if lo <= pt[0] <= hi]
        else:
            members = [pt for pt in points if lo < pt[0] <= hi]
        cnt = len(members)
        if cnt:
            accuracy = sum(c for _, c in members) / cnt
            mean_conf = sum(c for c, _ in members) / cnt
            gap = accuracy - mean_conf
            ece += (cnt / n) * abs(gap) if n else 0.0
            mce = max(mce, abs(gap))
        else:
            accuracy = mean_conf = gap = 0.0
        bins.append({
            "bin_index": m, "lo": lo, "hi": hi, "count": cnt,
            "mean_confidence": mean_conf, "accuracy": accuracy, "gap": gap,
        })
    return {
        "ece": ece, "mce": mce, "n_bins": n_bins, "n": n,
        "variant": "raw" if use_model_confidence else "calibrated",
        "bins": bins,
    }


# ---------------------------------------------------------------------------
# 4. Selective-prediction trade-off curve
# ---------------------------------------------------------------------------

def selective_curve(on_result, gold: GoldSet,
                    thresholds: Optional[List[float]] = None,
                    target_accuracy: float = 0.95) -> Dict:
    """Sweep the routing threshold; report automation vs accuracy at each.

    For each threshold, the auto-accepted set is the obligations that route to
    VERIFIED (no human). automation_rate is their share; auto_accept_accuracy is
    how many of them actually match gold. The operating point is the most
    automation achievable while keeping auto-accept accuracy at or above target.
    Statuses are restored afterward so the sweep has no side effects.
    """
    obs = on_result.obligations
    correct = match_extractions(obs, gold).matched_ext_ids
    n = len(obs)
    saved = [o.status for o in obs]

    if thresholds is None:
        grid = {round(i / 100.0, 4) for i in range(0, 101)}
        grid |= {o.confidence for o in obs} | {0.0, 1.0}
        thresholds = sorted(grid)

    points = []
    for t in thresholds:
        apply_threshold(obs, t)
        auto = [o for o in obs if o.status is Status.VERIFIED]
        na = len(auto)
        ar = na / n if n else 0.0
        acc = (sum(1 for o in auto if o.obligation_id in correct) / na) if na else 1.0
        points.append({
            "threshold": t, "automation_rate": ar, "human_review_rate": 1 - ar,
            "auto_accept_accuracy": acc, "n_auto": na,
        })

    for o, s in zip(obs, saved):
        o.status = s  # restore

    feasible = [p for p in points if p["n_auto"] > 0 and p["auto_accept_accuracy"] >= target_accuracy]
    operating_point = max(feasible, key=lambda p: p["automation_rate"]) if feasible else None
    return {"points": points, "target_accuracy": target_accuracy,
            "operating_point": operating_point}


# ---------------------------------------------------------------------------
# Bundle everything into one report object
# ---------------------------------------------------------------------------

def evaluate_all(on, off, gold: GoldSet, *, n_bins: int = 10,
                 threshold: float = 0.60, target_accuracy: float = 0.95,
                 backend: str = "") -> Dict:
    mr = match_extractions(on.obligations, gold)
    return {
        "permit_id": gold.permit_id,
        "label_provenance": gold.label_provenance.value,
        "labeler": gold.labeler,
        "notes": gold.notes,
        "backend": backend,
        "threshold": threshold,
        "n_obligations": len(on.obligations),
        "n_gold": len(gold.obligations),
        "extraction": extraction_prf(mr),
        "limit_detection": limit_detection_metrics(on.obligations, gold),
        "near_miss": near_miss_analysis(on.obligations, gold),
        "verification_lift": verification_lift(on, off, gold),
        "calibration": calibration(on.obligations, gold, n_bins=n_bins),
        "calibration_raw": calibration(on.obligations, gold, n_bins=n_bins,
                                       use_model_confidence=True),
        "selective": selective_curve(on, gold, target_accuracy=target_accuracy),
    }
