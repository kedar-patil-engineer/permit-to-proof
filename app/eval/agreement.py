"""Inter-annotator agreement for two independently labeled gold keys.

When two domain experts label the same permit, this module aligns their keys
with the same matching rule the evaluation uses and reports how much they agree:
span-level overlap (a pairwise F1), per-field exact agreement on the matched
obligations, and Cohen's kappa (chance-corrected) on the operator family and the
parameter. Reporting these strengthens the paper's evaluation (master document
A5.4). It never creates labels; it only compares two human-made keys.

CLI:
  python -m app.eval.agreement --a annotatorA.json --b annotatorB.json --out iaa.json
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
from typing import Dict, List, Optional, Tuple

from app.core.verify import normalize_text, normalize_unit
from app.eval import metrics as M
from app.eval.gold import GoldObligation, GoldSet, load_gold


def _match(a: GoldObligation, b: GoldObligation) -> bool:
    ap, bp = M._norm_param(a.parameter), M._norm_param(b.parameter)
    if not ap and not bp:
        sim = difflib.SequenceMatcher(
            None, normalize_text(a.description), normalize_text(b.description)
        ).ratio()
        if sim < M.DESC_SIM:
            return False
    elif ap != bp:
        return False
    if not M._value_match(a.limit_value, b.limit_value):
        return False
    if a.limit_value is not None:  # both have a value (else _value_match failed)
        if normalize_unit(a.limit_unit) != normalize_unit(b.limit_unit):
            return False
    if a.operator is not None and b.operator is not None:
        if M._op_family(a.operator) != M._op_family(b.operator):
            return False
    return True


def _align(a_list: List[GoldObligation], b_list: List[GoldObligation]):
    claimed = set()
    pairs: List[Tuple[GoldObligation, GoldObligation]] = []
    a_only: List[GoldObligation] = []
    for a in a_list:
        hit = None
        for j, b in enumerate(b_list):
            if j in claimed:
                continue
            if _match(a, b):
                hit = j
                break
        if hit is not None:
            claimed.add(hit)
            pairs.append((a, b_list[hit]))
        else:
            a_only.append(a)
    b_only = [b for j, b in enumerate(b_list) if j not in claimed]
    return pairs, a_only, b_only


def cohen_kappa(labels_a: List[str], labels_b: List[str]) -> Optional[float]:
    n = len(labels_a)
    if n == 0:
        return None
    cats = set(labels_a) | set(labels_b)
    po = sum(1 for x, y in zip(labels_a, labels_b) if x == y) / n
    ra = collections.Counter(labels_a)
    rb = collections.Counter(labels_b)
    pe = sum((ra[c] / n) * (rb[c] / n) for c in cats)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _op_label(o: GoldObligation) -> str:
    fam = M._op_family(o.operator)
    return fam if fam is not None else "none"


def inter_annotator_agreement(gold_a: GoldSet, gold_b: GoldSet) -> Dict:
    pairs, a_only, b_only = _align(list(gold_a.obligations), list(gold_b.obligations))
    matched = len(pairs)
    denom = 2 * matched + len(a_only) + len(b_only)
    agreement_f1 = (2 * matched / denom) if denom else 1.0

    # per-field exact agreement over matched pairs
    field_hits = {"parameter": 0, "value": 0, "unit": 0, "operator": 0, "frequency": 0}
    for a, b in pairs:
        if M._norm_param(a.parameter) == M._norm_param(b.parameter):
            field_hits["parameter"] += 1
        if M._value_match(a.limit_value, b.limit_value):
            field_hits["value"] += 1
        if normalize_unit(a.limit_unit) == normalize_unit(b.limit_unit):
            field_hits["unit"] += 1
        if M._op_family(a.operator) == M._op_family(b.operator):
            field_hits["operator"] += 1
        if normalize_text(a.frequency or "") == normalize_text(b.frequency or ""):
            field_hits["frequency"] += 1
    field_agreement = {k: (v / matched if matched else None) for k, v in field_hits.items()}

    kappa = {
        "operator_family": cohen_kappa([_op_label(a) for a, _ in pairs],
                                       [_op_label(b) for _, b in pairs]),
        "parameter": cohen_kappa([M._norm_param(a.parameter) for a, _ in pairs],
                                 [M._norm_param(b.parameter) for _, b in pairs]),
    }
    return {
        "permit_id": gold_a.permit_id,
        "n_annotator_a": len(gold_a.obligations),
        "n_annotator_b": len(gold_b.obligations),
        "matched": matched,
        "a_only": len(a_only),
        "b_only": len(b_only),
        "agreement_f1": agreement_f1,
        "field_agreement": field_agreement,
        "cohen_kappa": kappa,
        "a_only_ids": [o.gold_id for o in a_only],
        "b_only_ids": [o.gold_id for o in b_only],
        "matched_pairs": [{"a": a.gold_id, "b": b.gold_id} for a, b in pairs],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inter-annotator agreement for two gold keys")
    ap.add_argument("--a", required=True, help="annotator A gold JSON")
    ap.add_argument("--b", required=True, help="annotator B gold JSON")
    ap.add_argument("--out", default=None, help="optional iaa.json output path")
    args = ap.parse_args(argv)

    result = inter_annotator_agreement(load_gold(args.a), load_gold(args.b))
    print("=" * 56)
    print("Inter-annotator agreement: %s" % result["permit_id"])
    print("Obligations:  A=%d  B=%d  matched=%d  A-only=%d  B-only=%d"
          % (result["n_annotator_a"], result["n_annotator_b"], result["matched"],
             result["a_only"], result["b_only"]))
    print("Span-level agreement F1: %.3f" % result["agreement_f1"])
    fa = result["field_agreement"]
    print("Field agreement (matched): " + ", ".join(
        "%s=%.2f" % (k, v) for k, v in fa.items() if v is not None))
    k = result["cohen_kappa"]
    for name, val in k.items():
        print("Cohen's kappa (%s): %s" % (name, "n/a" if val is None else "%.3f" % val))
    print("=" * 56)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
