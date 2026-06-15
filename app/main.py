"""Permit-to-Proof: the browser interface (Streamlit).

This file is presentation only. Every piece of logic lives in app/core and
app/llm; here we only collect settings, call the pipeline, and render results
(Part B8). All state is kept in Streamlit session state, never in browser
storage, and exports are real file downloads.
"""

from __future__ import annotations

import json
import os
import sys

# Make the project root importable when launched via `streamlit run app/main.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app.core.pipeline import (
    error_detection_lift,
    run_pipeline,
    summarize,
)
from app.core.schema import MatchType, Obligation, Status
from app.core.score import apply_threshold
from app.llm import make_backend
from app.llm.ollama_backend import OllamaBackend
from app.llm.openai_backend import OpenAIBackend

load_dotenv()

SAMPLE_PDF = os.path.join(_ROOT, "sample_data", "sample_permit.pdf")

st.set_page_config(page_title="Permit-to-Proof", page_icon="✅", layout="wide")

from app.ui_theme import hero, inject_theme, processing_banner, status_chip

inject_theme()

# Status -> (label, colored badge markup) for the table.
_STATUS_STYLE = {
    Status.VERIFIED: ("Verified", "#1a7f37", "✅"),
    Status.NEEDS_REVIEW: ("Needs review", "#9a6700", "🟡"),
    Status.FLAGGED: ("Flagged", "#cf222e", "🚩"),
    Status.USER_ACCEPTED: ("Accepted by user", "#1a7f37", "👤✅"),
    Status.USER_REJECTED: ("Rejected by user", "#cf222e", "👤🚫"),
    Status.PENDING: ("Pending", "#57606a", "⏳"),
}

_MATCH_STYLE = {
    MatchType.EXACT: ("#1a7f37", "exact"),
    MatchType.FUZZY: ("#9a6700", "fuzzy"),
    MatchType.NONE: ("#cf222e", "none (ungrounded)"),
}


def _badge(text: str, color: str) -> str:
    return (
        f"<span style='background:{color};color:white;padding:2px 8px;"
        f"border-radius:10px;font-size:0.8rem;white-space:nowrap'>{text}</span>"
    )


# ---------------------------------------------------------------------------
# Sidebar: all the controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    backend_name = st.selectbox(
        "LLM backend",
        ["Mock", "OpenAI", "Ollama"],
        help="Mock runs offline with no key. OpenAI and Ollama use a real model.",
    )

    model_name = ""
    if backend_name == "OpenAI":
        model_name = st.text_input("OpenAI model", value=OpenAIBackend.DEFAULT_MODEL)
        if OpenAIBackend.is_available():
            st.caption("✅ OPENAI_API_KEY detected.")
        else:
            st.caption("⚠️ No OPENAI_API_KEY found; set it in .env to use this backend.")
    elif backend_name == "Ollama":
        model_name = st.text_input("Ollama model", value=OllamaBackend.DEFAULT_MODEL)
        if OllamaBackend.is_available():
            st.caption("✅ Ollama server reachable.")
        else:
            st.caption("⚠️ No Ollama server reachable at OLLAMA_HOST.")

    verification_on = st.checkbox(
        "Verification layer ON",
        value=True,
        help="OFF shows the raw model output trusted as is. The difference "
             "between ON and OFF is an in-app proxy for the error-detection "
             "lift the paper measures against the gold answer key.",
    )

    threshold = st.slider(
        "Routing threshold",
        min_value=0.0, max_value=1.0, value=0.60, step=0.05,
        help="Obligations below this confidence are routed to human review. "
             "Sliding this trades automation against review effort.",
    )

    st.divider()
    use_sample = st.checkbox("Use bundled sample permit", value=True)
    uploaded = None
    if not use_sample:
        uploaded = st.file_uploader("Upload a permit PDF", type=["pdf"])
        if uploaded is not None:
            # Show file name and page count right after ingest (B8), caching the
            # peek per file so it is computed once.
            peek_key = "pages_%s_%d" % (uploaded.name, len(uploaded.getvalue()))
            if peek_key not in st.session_state:
                try:
                    from app.core.ingest import ingest_pdf
                    _, _segs = ingest_pdf(uploaded.getvalue())
                    st.session_state[peek_key] = max((s.page for s in _segs), default=0)
                except Exception:
                    st.session_state[peek_key] = None
            pages = st.session_state.get(peek_key)
            st.caption("**%s** — %s" % (
                uploaded.name,
                ("%d pages" % pages) if pages else "could not read as PDF"))

    run_clicked = st.button("Run extraction", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------

def _run(source, source_name: str) -> None:
    backend = make_backend(backend_name, model=model_name or None)
    banner = st.empty()
    banner.markdown(processing_banner(), unsafe_allow_html=True)
    try:
        on = run_pipeline(
            source, backend, threshold=threshold, verification_enabled=True,
            backend_name=backend_name, source_name=source_name,
        )
        off = run_pipeline(
            source, backend, threshold=threshold, verification_enabled=False,
            backend_name=backend_name, source_name=source_name,
        )
    except Exception as exc:  # never crash the UI; surface the problem
        banner.empty()
        st.session_state["error"] = "%s: %s" % (type(exc).__name__, exc)
        st.session_state["ran"] = False
        return
    banner.empty()
    st.session_state["error"] = None
    st.session_state["ran"] = True
    st.session_state["obs_on"] = on.obligations
    st.session_state["obs_off"] = off.obligations
    st.session_state["meta"] = {
        "backend": backend_name,
        "source": source_name,
        "pages": on.page_count,
        "segments": len(on.segments),
    }


if run_clicked:
    if use_sample:
        if not os.path.exists(SAMPLE_PDF):
            st.session_state["error"] = "Sample permit not found. Run sample_data/make_sample_permit.py."
            st.session_state["ran"] = False
        else:
            with open(SAMPLE_PDF, "rb") as fh:
                _run(fh.read(), "sample_permit.pdf")
    elif uploaded is not None:
        _run(uploaded.getvalue(), uploaded.name)
    else:
        st.session_state["error"] = "Upload a PDF or check 'Use bundled sample permit'."
        st.session_state["ran"] = False


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

hero("Deterministic compliance verification engine &nbsp;//&nbsp; "
     "the AI proposes &nbsp;&rsaquo;&nbsp; the checker disposes")

if st.session_state.get("error"):
    st.error(st.session_state["error"])

if not st.session_state.get("ran"):
    st.info("Choose a backend and click **Run extraction** in the sidebar. "
            "The bundled sample runs instantly, offline, with no API key.")
    st.stop()


# ---------------------------------------------------------------------------
# Apply the current threshold live (re-route only; confidence is unchanged)
# ---------------------------------------------------------------------------

obs_on = st.session_state["obs_on"]
obs_off = st.session_state["obs_off"]
apply_threshold(obs_on, threshold)  # respects user overrides

displayed = obs_on if verification_on else obs_off
meta = st.session_state["meta"]


# ---------------------------------------------------------------------------
# Metrics panel
# ---------------------------------------------------------------------------

st.subheader("Summary")
m = summarize(displayed)
lift = error_detection_lift(
    type("R", (), {"obligations": obs_on})(),
    type("R", (), {"obligations": obs_off})(),
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Obligations", m["total"])
c2.metric("Verified rate", "%.0f%%" % (m["verified_rate"] * 100))
c3.metric("Routed to human", m["issues"])
c4.metric("Threshold", "%.2f" % threshold)
c5.metric(
    "Issues surfaced (vs raw)", "+%d" % lift["lift"],
    help=("In-app proxy for the paper's error-detection lift: obligations the "
          "verification layer routes to a human that the raw (OFF) pipeline "
          "would trust silently. Of these, %d are hard errors flagged "
          "(grounding/schema) and %d are warnings or low confidence sent to "
          "review. On the Mock sample the OFF baseline trusts everything by "
          "construction, so this demonstrates the mechanism; the measured lift "
          "in the paper requires the expert gold answer key."
          % (lift["lift_flagged"], lift["lift_needs_review"])))

st.caption(
    "Source: **%s** · backend: **%s** · %d pages · %d segments · verification **%s**"
    % (meta["source"], meta["backend"], meta["pages"], meta["segments"],
       "ON" if verification_on else "OFF")
)

if m["flag_reasons"]:
    reason_df = pd.DataFrame(
        sorted(m["flag_reasons"].items(), key=lambda kv: -kv[1]),
        columns=["Failed check", "Count"],
    )
    with st.expander("Flag-reason breakdown", expanded=False):
        st.bar_chart(reason_df.set_index("Failed check"))

st.divider()


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

fcol1, fcol2 = st.columns([1, 2])
status_filter = fcol1.selectbox(
    "Filter by status", ["All", "Verified", "Needs review", "Flagged", "User overridden"]
)
search = fcol2.text_input("Search text", placeholder="parameter, citation, or words in the quote")


def _matches_filter(ob: Obligation) -> bool:
    if status_filter == "Verified" and ob.status not in (Status.VERIFIED, Status.USER_ACCEPTED):
        return False
    if status_filter == "Needs review" and ob.status != Status.NEEDS_REVIEW:
        return False
    if status_filter == "Flagged" and ob.status not in (Status.FLAGGED, Status.USER_REJECTED):
        return False
    if status_filter == "User overridden" and ob.status not in (Status.USER_ACCEPTED, Status.USER_REJECTED):
        return False
    if search:
        hay = " ".join(filter(None, [
            ob.description, ob.parameter, ob.citation, ob.source_quote,
            ob.limit_unit,
        ])).lower()
        if search.lower() not in hay:
            return False
    return True


rows = [ob for ob in displayed if _matches_filter(ob)]
st.write("Showing **%d** of **%d** obligations." % (len(rows), len(displayed)))


# ---------------------------------------------------------------------------
# Results table (one expander per obligation)
# ---------------------------------------------------------------------------

def _limit_str(ob: Obligation) -> str:
    if ob.limit_value is None:
        return "—"
    op = ob.operator.value if ob.operator else ""
    return "%s %g %s" % (op, ob.limit_value, ob.limit_unit or "")


def _override(ob_id: str, status: Status) -> None:
    # Apply to both the ON and OFF lists so the override is reflected whichever
    # view is shown (route_status / apply_threshold preserve USER_* statuses).
    for list_key in ("obs_on", "obs_off"):
        for ob in st.session_state.get(list_key, []):
            if ob.obligation_id == ob_id:
                ob.status = status


for ob in rows:
    label, color, icon = _STATUS_STYLE[ob.status]
    failed = ob.failed_checks()
    reasons = ", ".join(c.name for c in failed) if failed else "all checks passed"
    header = "%s  **%s**  ·  %s  ·  %s  ·  conf %.2f  ·  %s" % (
        icon,
        ob.parameter or "(narrative)",
        _limit_str(ob),
        label,
        ob.confidence,
        reasons,
    )
    with st.expander(header):
        mcolor2, mlabel2 = _MATCH_STYLE[ob.match_type]
        st.markdown(
            "%s &nbsp; %s &nbsp; <span style='color:#6f8aa6'>CONF</span> "
            "<b style='color:#00e5ff'>%.2f</b>"
            % (status_chip(ob.status, label), _badge("grounding: " + mlabel2, mcolor2),
               ob.confidence),
            unsafe_allow_html=True,
        )
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Description:** %s" % ob.description)
            st.markdown("**Frequency:** %s  ·  **Deadline:** %s" % (
                ob.frequency or "—", ob.deadline or "—"))
            st.markdown("**Citation:** %s" % (ob.citation or "—"))
            mcolor, mlabel = _MATCH_STYLE[ob.match_type]
            st.markdown(
                "**Source (page from segment %s)** %s"
                % (ob.source_segment_id, _badge(mlabel, mcolor)),
                unsafe_allow_html=True,
            )
            if ob.match_type == MatchType.NONE and verification_on:
                st.error("This quote was not found in the cited segment. "
                         "Likely a hallucination.")
            st.markdown("> %s" % (ob.source_quote or "_(no quote provided)_"))
        with right:
            st.markdown("**Verification checks**")
            if not ob.checks:
                st.caption("Verification was OFF for this view.")
            for c in ob.checks:
                mark = "✅" if c.passed else ("🚩" if c.severity.value == "error" else "⚠️")
                st.markdown("%s **%s** — %s" % (mark, c.name, c.message))

            b1, b2 = st.columns(2)
            b1.button("Accept", key="acc_%s" % ob.obligation_id,
                      on_click=_override, args=(ob.obligation_id, Status.USER_ACCEPTED),
                      use_container_width=True)
            b2.button("Reject", key="rej_%s" % ob.obligation_id,
                      on_click=_override, args=(ob.obligation_id, Status.USER_REJECTED),
                      use_container_width=True)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Export")


def _to_records(obs):
    records = []
    for ob in obs:
        d = ob.model_dump(mode="json")
        d["failed_checks"] = ";".join(c.name for c in ob.failed_checks())
        d["checks_detail"] = " | ".join(
            "%s=%s" % (c.name, "pass" if c.passed else "FAIL") for c in ob.checks
        )
        d.pop("checks", None)
        records.append(d)
    return records


records = _to_records(displayed)
json_bytes = json.dumps(
    [ob.model_dump(mode="json") for ob in displayed], indent=2
).encode("utf-8")
csv_bytes = pd.DataFrame(records).to_csv(index=False).encode("utf-8")

e1, e2 = st.columns(2)
e1.download_button("Download JSON", json_bytes, file_name="permit_obligations.json",
                   mime="application/json", use_container_width=True)
e2.download_button("Download CSV", csv_bytes, file_name="permit_obligations.csv",
                   mime="text/csv", use_container_width=True)
