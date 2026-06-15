"""Command-line evaluation harness (Part A5).

Runs the pipeline with verification ON and OFF over a permit, grades the
extraction against a gold answer key, and writes the four paper metrics plus
figures and a run manifest.

    python evaluate.py                       # bundled synthetic permit + illustrative gold
    python evaluate.py --pdf my.pdf --gold my.json --backend OpenAI --out eval_out

Exits non-zero if the gold set is missing or malformed.
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.core.pipeline import run_pipeline
from app.eval import metrics as M
from app.eval import report as R
from app.eval.gold import discover_gold, load_gold
from app.llm import make_backend


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Permit-to-Proof evaluation harness")
    ap.add_argument("--pdf", default=os.path.join("sample_data", "sample_permit.pdf"))
    ap.add_argument("--gold", default=None, help="gold JSON (default: auto-discover by stem)")
    ap.add_argument("--backend", default="Mock", choices=["Mock", "OpenAI", "Ollama"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--target-accuracy", type=float, default=0.95)
    ap.add_argument("--out", default="eval_out")
    args = ap.parse_args(argv)

    if not os.path.exists(args.pdf):
        print("ERROR: permit PDF not found: %s" % args.pdf, file=sys.stderr)
        return 2

    gold_path = args.gold or discover_gold(args.pdf)
    if not gold_path or not os.path.exists(gold_path):
        print("ERROR: gold answer key not found for %s. Pass --gold or place it "
              "at sample_data/gold/<stem>.json." % args.pdf, file=sys.stderr)
        return 2
    try:
        gold = load_gold(gold_path)
    except Exception as exc:
        print("ERROR: gold set is malformed: %s" % exc, file=sys.stderr)
        return 2

    backend = make_backend(args.backend, model=args.model)
    on = run_pipeline(args.pdf, backend, threshold=args.threshold,
                      verification_enabled=True, backend_name=args.backend)
    off = run_pipeline(args.pdf, backend, threshold=args.threshold,
                       verification_enabled=False, backend_name=args.backend)

    results = M.evaluate_all(on, off, gold, n_bins=args.bins,
                             threshold=args.threshold,
                             target_accuracy=args.target_accuracy,
                             backend=args.backend)
    paths = R.write_report(results, args.out)
    figs = R.render_figures(results, args.out)

    ex = results["extraction"]
    vl = results["verification_lift"]
    cal = results["calibration"]
    op = results["selective"]["operating_point"]
    print("=" * 60)
    if gold.is_illustrative:
        print("ILLUSTRATIVE gold (synthetic permit) - NOT the expert key (A5.4)")
    print("Extraction:  P=%.3f  R=%.3f  F1=%.3f  (TP=%d FP=%d FN=%d)" % (
        ex["precision"], ex["recall"], ex["f1"], ex["tp"], ex["fp"], ex["fn"]))
    print("Verif. lift: error-detection recall ON=%.3f vs OFF=%.3f  (+%.3f), "
          "%d/%d errors caught" % (vl["on"]["recall"], vl["off"]["recall"],
                                   vl["lift"], vl["errors_caught_on"], vl["n_true_errors"]))
    print("Calibration: ECE=%.3f  MCE=%.3f" % (cal["ece"], cal["mce"]))
    if op:
        print("Selective:   %.0f%% automation at >=%.0f%% accuracy (%.0f%% to human)" % (
            op["automation_rate"] * 100, results["selective"]["target_accuracy"] * 100,
            op["human_review_rate"] * 100))
    print("Wrote: %s, %s%s" % (paths["metrics"], paths["report"],
                               ", figures" if figs else " (figures skipped)"))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
