"""Render an evaluation result to metrics.json, a markdown summary, and figures.

matplotlib is imported lazily so the metrics still write even if it is absent
(the figures are then skipped with a note). Every figure and table carries the
gold set's provenance so illustrative numbers are never mistaken for the
paper's measured results.
"""

from __future__ import annotations

import json
import os
from typing import Dict


def write_report(results: Dict, outdir: str) -> Dict[str, str]:
    """Write metrics.json and report.md. Returns the paths written."""
    os.makedirs(outdir, exist_ok=True)
    metrics_path = os.path.join(outdir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    md_path = os.path.join(outdir, "report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(_markdown(results))
    return {"metrics": metrics_path, "report": md_path}


def _markdown(r: Dict) -> str:
    ex = r["extraction"]
    vl = r["verification_lift"]
    cal = r["calibration"]
    sel = r["selective"]
    op = sel.get("operating_point")
    prov = r["label_provenance"]
    illustrative = prov == "ILLUSTRATIVE_AUTHOR_KNOWN"

    lines = []
    lines.append("# Permit-to-Proof evaluation report\n")
    if illustrative:
        lines.append("> **ILLUSTRATIVE gold set on a synthetic permit. NOT the "
                     "expert answer key (Part A5.4). These numbers self-test the "
                     "harness; they are not the paper's measured results.**\n")
    lines.append("- Permit: `%s`  |  backend: `%s`  |  threshold: %.2f  |  "
                 "gold provenance: `%s`\n" % (
                     r["permit_id"], r["backend"], r["threshold"], prov))
    lines.append("- Obligations extracted: %d  |  gold obligations: %d\n"
                 % (r["n_obligations"], r["n_gold"]))

    lines.append("\n## 1. Extraction quality (vs gold)\n")
    lines.append("| Precision | Recall | F1 | TP | FP | FN |\n|---|---|---|---|---|---|\n")
    lines.append("| %.3f | %.3f | %.3f | %d | %d | %d |\n" % (
        ex["precision"], ex["recall"], ex["f1"], ex["tp"], ex["fp"], ex["fn"]))

    nm = r.get("near_miss")
    if nm:
        s = nm["summary"]
        lines.append("\n## 1b. Near-miss breakdown (why gold limits missed)\n")
        lines.append("| Matched | Operator mismatch | Unit mismatch | Not extracted |\n"
                     "|---|---|---|---|\n")
        lines.append("| %d | %d | %d | %d |\n" % (
            s["matched"], s["operator_mismatch"], s["unit_mismatch"],
            s["not_extracted"]))
        lines.append("\nA limit counted as a *miss* in section 1 is broken out here: "
                     "found with the wrong operator, found with a different but "
                     "possibly equivalent unit, or genuinely not extracted. Only the "
                     "last is a recall gap; the first two are scoring-strictness "
                     "effects worth reporting separately.\n")

    lines.append("\n## 2. Verification lift (errors caught ON vs OFF)\n")
    lines.append("- True errors in the extraction set: **%d**\n" % vl["n_true_errors"])
    lines.append("- Error-detection recall: **ON %.3f** vs OFF %.3f  ->  "
                 "lift **+%.3f**\n" % (vl["on"]["recall"], vl["off"]["recall"], vl["lift"]))
    lines.append("- Error-detection precision (ON): %.3f  |  errors caught: "
                 "ON %d, OFF %d\n" % (vl["on"]["precision"], vl["errors_caught_on"],
                                      vl["errors_caught_off"]))

    lines.append("\n## 3. Confidence calibration\n")
    lines.append("- ECE (calibrated): **%.3f**  |  MCE: %.3f  (over %d bins, "
                 "n=%d)\n" % (cal["ece"], cal["mce"], cal["n_bins"], cal["n"]))
    raw = r.get("calibration_raw")
    if raw:
        lines.append("- ECE (raw model confidence): %.3f\n" % raw["ece"])

    lines.append("\n## 4. Selective-prediction trade-off\n")
    if op:
        lines.append("- At target auto-accept accuracy >= %.2f: automation "
                     "**%.1f%%**, human review **%.1f%%** (threshold %.2f).\n" % (
                         sel["target_accuracy"], op["automation_rate"] * 100,
                         op["human_review_rate"] * 100, op["threshold"]))
    else:
        lines.append("- No threshold reached target auto-accept accuracy %.2f.\n"
                     % sel["target_accuracy"])
    lines.append("\nFigures: reliability_diagram.png, selective_tradeoff.png, "
                 "risk_coverage.png, lift_bar.png\n")
    if illustrative:
        lines.append("\n_%s_\n" % r.get("notes", ""))
    return "".join(lines)


def render_figures(results: Dict, outdir: str) -> bool:
    """Render the four paper figures. Returns False (with a note) if matplotlib
    is unavailable; the JSON/markdown are unaffected."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on environment
        with open(os.path.join(outdir, "FIGURES_SKIPPED.txt"), "w", encoding="utf-8") as fh:
            fh.write("matplotlib unavailable, figures skipped: %s\n" % exc)
        return False

    os.makedirs(outdir, exist_ok=True)
    prov = results["label_provenance"]
    tag = "ILLUSTRATIVE (synthetic permit)" if prov == "ILLUSTRATIVE_AUTHOR_KNOWN" else prov

    # Reliability diagram
    cal = results["calibration"]
    used = [b for b in cal["bins"] if b["count"] > 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    if used:
        ax.scatter([b["mean_confidence"] for b in used],
                   [b["accuracy"] for b in used],
                   s=[20 + 12 * b["count"] for b in used], color="#1f77b4", zorder=3)
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Reliability diagram (ECE=%.3f)\n%s" % (cal["ece"], tag))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "reliability_diagram.png"), dpi=130)
    plt.close(fig)

    # Selective trade-off: accuracy vs automation
    pts = results["selective"]["points"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([p["automation_rate"] for p in pts],
            [p["auto_accept_accuracy"] for p in pts], color="#2ca02c")
    ax.set_xlabel("automation rate (fraction auto-accepted)")
    ax.set_ylabel("auto-accept accuracy")
    ax.set_title("Automation vs accuracy\n%s" % tag)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "selective_tradeoff.png"), dpi=130)
    plt.close(fig)

    # Risk-coverage
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([p["automation_rate"] for p in pts],
            [1 - p["auto_accept_accuracy"] for p in pts], color="#d62728")
    ax.set_xlabel("coverage (automation rate)")
    ax.set_ylabel("risk (1 - accuracy)")
    ax.set_title("Risk-coverage curve\n%s" % tag)
    ax.set_xlim(0, 1); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "risk_coverage.png"), dpi=130)
    plt.close(fig)

    # Lift bar
    vl = results["verification_lift"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["caught ON", "caught OFF", "silent ON"],
           [vl["errors_caught_on"], vl["errors_caught_off"], vl["on"]["fn"]],
           color=["#2ca02c", "#7f7f7f", "#d62728"])
    ax.set_ylabel("count of true errors (n=%d)" % vl["n_true_errors"])
    ax.set_title("Error detection: verification ON vs OFF\n%s" % tag)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "lift_bar.png"), dpi=130)
    plt.close(fig)
    return True
