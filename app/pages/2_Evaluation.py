"""Evaluation page (Part A5): extraction P/R/F1, verification lift, calibration,
and the selective-prediction trade-off, graded against a gold answer key.

Presentation only. All computation is in app/eval. Every panel surfaces the
gold set's provenance so illustrative numbers are never mistaken for the
paper's measured results.
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import streamlit as st

from app.core.pipeline import run_pipeline
from app.eval import metrics as M
from app.eval.gold import discover_gold, load_gold
from app.llm import make_backend
from app.ui_theme import hero, inject_theme

st.set_page_config(page_title="Permit-to-Proof — Evaluation", page_icon="📊", layout="wide")
inject_theme()
hero("Evaluation harness &nbsp;//&nbsp; extraction P/R/F1 · verification lift · "
     "calibration · automation trade-off")

SAMPLE_PDF = os.path.join(_ROOT, "sample_data", "sample_permit.pdf")


@st.cache_data(show_spinner=False)
def _evaluate(pdf_path: str, gold_path: str, backend_name: str, threshold: float):
    gold = load_gold(gold_path)
    backend = make_backend(backend_name)
    on = run_pipeline(pdf_path, backend, threshold=threshold, verification_enabled=True)
    off = run_pipeline(pdf_path, backend, threshold=threshold, verification_enabled=False)
    results = M.evaluate_all(on, off, gold, threshold=threshold, backend=backend_name)
    return results, gold.model_dump(mode="json")


with st.sidebar:
    st.header("Evaluation settings")
    backend_name = st.selectbox("Backend", ["Mock", "OpenAI", "Ollama"])
    threshold = st.slider("Routing threshold", 0.0, 1.0, 0.60, 0.05)
    st.caption("Grades the run against sample_data/gold/sample_permit.json.")

gold_path = discover_gold(SAMPLE_PDF)
if not gold_path:
    st.error("No gold answer key found at sample_data/gold/sample_permit.json.")
    st.stop()

results, gold_meta = _evaluate(SAMPLE_PDF, gold_path, backend_name, threshold)

# Provenance banner — always visible.
if results["label_provenance"] == "ILLUSTRATIVE_AUTHOR_KNOWN":
    st.warning(
        "**Illustrative gold set on a synthetic permit — NOT the expert answer "
        "key (Part A5.4).** These numbers self-test the harness end to end. The "
        "paper's measured results require an expert-labeled key built on real "
        "permits. The same harness produces the real numbers when given that key.",
        icon="⚠️",
    )

ex = results["extraction"]
vl = results["verification_lift"]
cal = results["calibration"]
cal_raw = results["calibration_raw"]
sel = results["selective"]

# --- 1. Extraction quality ------------------------------------------------
st.subheader("1 · Extraction quality (vs gold)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision", "%.3f" % ex["precision"])
c2.metric("Recall", "%.3f" % ex["recall"])
c3.metric("F1", "%.3f" % ex["f1"])
c4.metric("TP / FP / FN", "%d / %d / %d" % (ex["tp"], ex["fp"], ex["fn"]))

# --- 2. Verification lift -------------------------------------------------
st.subheader("2 · Verification lift (errors caught ON vs OFF)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("True errors", vl["n_true_errors"])
c2.metric("Error-detection recall (ON)", "%.3f" % vl["on"]["recall"])
c3.metric("Recall (OFF baseline)", "%.3f" % vl["off"]["recall"])
c4.metric("Lift", "+%.3f" % vl["lift"],
          help="Error-detection recall ON minus OFF. OFF trusts everything, so "
               "its recall is 0 by construction.")
st.caption("Errors caught: **ON %d** vs OFF %d  ·  silent errors (ON): %d  ·  "
           "error-detection precision (ON): %.3f"
           % (vl["errors_caught_on"], vl["errors_caught_off"], vl["on"]["fn"],
              vl["on"]["precision"]))

# --- 3. Calibration -------------------------------------------------------
st.subheader("3 · Confidence calibration")
c1, c2, c3 = st.columns(3)
c1.metric("ECE (calibrated)", "%.3f" % cal["ece"])
c2.metric("MCE", "%.3f" % cal["mce"])
c3.metric("ECE (raw model conf.)", "%.3f" % cal_raw["ece"])
rel = pd.DataFrame([
    {"mean_confidence": b["mean_confidence"], "accuracy": b["accuracy"],
     "perfect": b["mean_confidence"], "count": b["count"]}
    for b in cal["bins"] if b["count"] > 0
])
if not rel.empty:
    st.caption("Reliability: empirical accuracy vs mean predicted confidence "
               "(the closer to the dashed perfect line, the better calibrated).")
    st.line_chart(rel.set_index("mean_confidence")[["accuracy", "perfect"]])

# --- 4. Selective prediction ---------------------------------------------
st.subheader("4 · Automation vs human-review trade-off")
op = sel["operating_point"]
if op:
    c1, c2, c3 = st.columns(3)
    c1.metric("Automation @ target", "%.0f%%" % (op["automation_rate"] * 100))
    c2.metric("Human review @ target", "%.0f%%" % (op["human_review_rate"] * 100))
    c3.metric("At accuracy", "≥ %.0f%%" % (sel["target_accuracy"] * 100))
curve = pd.DataFrame([
    {"automation_rate": p["automation_rate"],
     "auto_accept_accuracy": p["auto_accept_accuracy"]}
    for p in sel["points"]
]).sort_values("automation_rate")
st.caption("Auto-accept accuracy as a function of how much is automated. Move "
           "the threshold slider to change the operating point.")
st.line_chart(curve.set_index("automation_rate"))

with st.expander("Per-obligation match records & raw metrics JSON"):
    st.dataframe(pd.DataFrame(ex["matches"]), use_container_width=True)
    st.download_button("Download metrics.json",
                       data=json.dumps(results, indent=2),
                       file_name="metrics.json", mime="application/json")

st.caption("Regenerate figures and a full report from the command line: "
           "`python evaluate.py --out eval_out`")
