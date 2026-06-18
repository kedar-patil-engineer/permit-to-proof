"""Turn a filled annotation template (CSV) into the validated gold JSON the
evaluation harness reads, and back again for review.

The CSV is the human-friendly form a domain expert fills while reading a permit
(see annotation/PROTOCOL.md and annotation/gold_template.csv). This module never
invents labels; it only reshapes what a person wrote into the GoldSet schema and
validates it, failing loud on a malformed row.

CLI:
  python -m app.eval.annotate --csv annotation/my_permit.csv \
      --permit-id PTP-XXX --source-pdf sample_data/permits/my_permit.pdf \
      --provenance EXPERT_SINGLE --labeler "Jane Doe" \
      --out sample_data/gold/my_permit.json

  python -m app.eval.annotate --from-gold sample_data/gold/sample_permit.json \
      --to-csv review.csv          # export an existing key back to the template
"""

from __future__ import annotations

import argparse
import csv
import sys
from typing import Dict, List

from app.eval.gold import GoldObligation, GoldSet, LabelProvenance

CSV_COLUMNS = [
    "gold_id", "parameter", "limit_value", "limit_unit", "operator",
    "frequency", "deadline", "citation", "source_segment_id",
    "is_obligation", "description",
]

_EXAMPLE_MARKER = "example row"


def _clean(value):
    if value is None:
        return None
    s = str(value).strip()
    return s or None


def _parse_bool(value, default=True):
    s = (str(value).strip().lower() if value is not None else "")
    if s in ("true", "yes", "y", "1"):
        return True
    if s in ("false", "no", "n", "0"):
        return False
    return default


def _parse_float(value):
    s = _clean(value)
    if s is None:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError as exc:
        raise ValueError("limit_value '%s' is not a number" % value) from exc


def rows_to_obligations(rows: List[Dict]) -> List[GoldObligation]:
    """Validate annotation rows into GoldObligation objects.

    Rows that are blank or still carry the EXAMPLE marker are skipped so a
    template left with its examples does not pollute the key.
    """
    obligations: List[GoldObligation] = []
    for i, row in enumerate(rows, start=2):  # row 1 is the header
        gold_id = _clean(row.get("gold_id"))
        description = _clean(row.get("description"))
        if not gold_id and not description:
            continue
        if description and _EXAMPLE_MARKER in description.lower():
            continue
        if not gold_id:
            raise ValueError("Row %d has a description but no gold_id." % i)
        try:
            ob = GoldObligation(
                gold_id=gold_id,
                description=description or "",
                parameter=_clean(row.get("parameter")),
                limit_value=_parse_float(row.get("limit_value")),
                limit_unit=_clean(row.get("limit_unit")),
                operator=_clean(row.get("operator")),
                frequency=_clean(row.get("frequency")),
                deadline=_clean(row.get("deadline")),
                citation=_clean(row.get("citation")),
                source_segment_id=_clean(row.get("source_segment_id")),
                is_obligation=_parse_bool(row.get("is_obligation")),
            )
        except Exception as exc:
            raise ValueError("Row %d (%s): %s" % (i, gold_id, exc)) from exc
        obligations.append(ob)

    ids = [o.gold_id for o in obligations]
    dupes = sorted({x for x in ids if ids.count(x) > 1})
    if dupes:
        raise ValueError("Duplicate gold_id(s): %s" % ", ".join(dupes))
    return obligations


def csv_to_goldset(csv_path: str, *, permit_id: str, source_pdf: str = "",
                   provenance: str = "EXPERT_SINGLE", labeler: str = "",
                   notes: str = "") -> GoldSet:
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    obligations = rows_to_obligations(rows)
    if not obligations:
        raise ValueError("No obligations found in %s (only blank/example rows?)." % csv_path)
    return GoldSet(
        permit_id=permit_id,
        source_pdf=source_pdf or None,
        label_provenance=LabelProvenance(provenance),
        labeler=labeler,
        notes=notes,
        obligations=obligations,
    )


def goldset_to_rows(gs: GoldSet) -> List[Dict]:
    rows = []
    for o in gs.obligations:
        d = o.model_dump(mode="json")
        d["is_obligation"] = "TRUE" if o.is_obligation else "FALSE"
        rows.append({c: ("" if d.get(c) is None else d.get(c)) for c in CSV_COLUMNS})
    return rows


def write_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Annotation template <-> gold JSON")
    ap.add_argument("--csv", help="filled annotation CSV to convert to gold JSON")
    ap.add_argument("--permit-id", default="")
    ap.add_argument("--source-pdf", default="")
    ap.add_argument("--provenance", default="EXPERT_SINGLE",
                    choices=[p.value for p in LabelProvenance])
    ap.add_argument("--labeler", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--out", help="output gold JSON path")
    ap.add_argument("--from-gold", help="export an existing gold JSON ...")
    ap.add_argument("--to-csv", help="... to this annotation CSV path")
    args = ap.parse_args(argv)

    if args.from_gold and args.to_csv:
        from app.eval.gold import load_gold
        gs = load_gold(args.from_gold)
        write_csv(goldset_to_rows(gs), args.to_csv)
        print("Wrote %d rows to %s" % (len(gs.obligations), args.to_csv))
        return 0

    if not args.csv or not args.out:
        ap.error("provide --csv and --out (or --from-gold and --to-csv)")
    if not args.permit_id:
        ap.error("--permit-id is required when building a gold key")
    try:
        gs = csv_to_goldset(args.csv, permit_id=args.permit_id,
                            source_pdf=args.source_pdf, provenance=args.provenance,
                            labeler=args.labeler, notes=args.notes)
    except Exception as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(gs.model_dump_json(indent=2))
    print("Wrote %d obligations to %s (provenance=%s)"
          % (len(gs.obligations), args.out, gs.label_provenance.value))
    if gs.label_provenance == LabelProvenance.ILLUSTRATIVE_AUTHOR_KNOWN:
        print("NOTE: ILLUSTRATIVE provenance is for the synthetic self-test only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
