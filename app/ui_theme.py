"""Futuristic command-console theme for the Permit-to-Proof UI.

Pure presentation: this module only injects CSS and small HTML fragments. It
holds no application logic, so the separation from app/core and app/llm is
preserved. The look is a dark heads-up display with neon accents, an animated
scanline, glowing metrics, pulsing alert chips, and a live status banner.
"""

from __future__ import annotations

import streamlit as st

from app.core.schema import Status

# Continuously running animations (scanline, grid drift, pulses, flicker,
# sweep, flow) give the "live" feel. Pure CSS keyframes, no JavaScript.
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap');

:root{
  --ptp-cyan:#00e5ff; --ptp-green:#27ff9d; --ptp-amber:#ffb84d;
  --ptp-red:#ff3b6b; --ptp-text:#c9e3f5; --ptp-dim:#6f8aa6;
}

/* ---- base canvas ---- */
.stApp{
  background:
    radial-gradient(circle at 18% -10%, rgba(0,229,255,.07), transparent 45%),
    radial-gradient(circle at 95% 110%, rgba(39,255,157,.06), transparent 45%),
    linear-gradient(180deg,#070b14,#04070d) !important;
  color:var(--ptp-text);
}
/* drifting grid behind everything */
.stApp::before{
  content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:
    linear-gradient(rgba(0,229,255,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,.045) 1px, transparent 1px);
  background-size:44px 44px;
  animation:ptp-grid 9s linear infinite;
  -webkit-mask-image:radial-gradient(circle at 50% 25%, #000, transparent 78%);
  mask-image:radial-gradient(circle at 50% 25%, #000, transparent 78%);
}
@keyframes ptp-grid{from{background-position:0 0}to{background-position:44px 44px}}

/* sweeping scanline */
.ptp-scan{
  position:fixed; left:0; right:0; top:0; height:140px;
  pointer-events:none; z-index:9998;
  background:linear-gradient(180deg, transparent, rgba(0,229,255,.05) 55%, rgba(0,229,255,.16));
  animation:ptp-scanline 7s linear infinite;
}
@keyframes ptp-scanline{0%{transform:translateY(-150px)}100%{transform:translateY(100vh)}}

/* keep content above the grid */
.block-container{position:relative; z-index:1;}

/* ---- typography ---- */
html, body, .stApp, .stMarkdown, p, span, label, div[data-testid="stMetricLabel"]{
  font-family:'Share Tech Mono','JetBrains Mono',monospace;
}
h1,h2,h3,h4{
  font-family:'Orbitron','Share Tech Mono',monospace !important;
  letter-spacing:1.5px; color:#e2f6ff !important;
  text-shadow:0 0 14px rgba(0,229,255,.30);
}

/* ---- hero banner ---- */
.ptp-hero{
  position:relative; z-index:1; overflow:hidden;
  border:1px solid rgba(0,229,255,.35); border-radius:7px;
  background:linear-gradient(120deg, rgba(0,229,255,.07), rgba(8,18,32,.35));
  padding:18px 24px; margin:2px 0 18px;
  box-shadow:inset 0 0 26px rgba(0,229,255,.10), 0 0 22px rgba(0,0,0,.55);
}
.ptp-hero::after{
  content:""; position:absolute; top:0; left:-45%; width:45%; height:100%;
  background:linear-gradient(90deg, transparent, rgba(0,229,255,.14), transparent);
  animation:ptp-sweep 5.5s linear infinite;
}
@keyframes ptp-sweep{0%{left:-45%}100%{left:145%}}
.ptp-hero-top{display:flex; align-items:center; justify-content:space-between; gap:16px}
.ptp-title{
  font-family:'Orbitron'; font-weight:900; font-size:2.05rem; letter-spacing:5px;
  color:#eafcff; margin:0;
  text-shadow:0 0 12px rgba(0,229,255,.85), 0 0 34px rgba(0,229,255,.35);
  animation:ptp-flicker 5s infinite;
}
.ptp-title b{color:var(--ptp-cyan)}
@keyframes ptp-flicker{0%,17%,21%,100%{opacity:1}19%{opacity:.72}82%{opacity:.9}}
.ptp-sub{color:var(--ptp-dim); font-size:.82rem; letter-spacing:2px; text-transform:uppercase; margin-top:8px}
.ptp-live{display:inline-flex; align-items:center; gap:9px; white-space:nowrap;
  font-size:.78rem; letter-spacing:2px; text-transform:uppercase; color:var(--ptp-green)}
.ptp-dot{width:10px; height:10px; border-radius:50%; background:var(--ptp-green);
  box-shadow:0 0 12px var(--ptp-green); animation:ptp-blink 1.5s infinite}
@keyframes ptp-blink{0%,55%{opacity:1}58%,100%{opacity:.18}}
.ptp-flow{height:2px; margin-top:14px;
  background:linear-gradient(90deg, transparent, var(--ptp-cyan), var(--ptp-green), transparent);
  background-size:200% 100%; animation:ptp-flow 3s linear infinite}
@keyframes ptp-flow{0%{background-position:0 0}100%{background-position:200% 0}}

/* ---- buttons ---- */
.stButton>button, [data-testid="stDownloadButton"]>button{
  background:linear-gradient(180deg, rgba(0,229,255,.10), rgba(0,229,255,.02)) !important;
  color:#bff4ff !important; border:1px solid rgba(0,229,255,.45) !important;
  border-radius:3px !important; font-family:'Share Tech Mono',monospace !important;
  text-transform:uppercase; letter-spacing:1.5px; transition:.18s ease;
}
.stButton>button:hover, [data-testid="stDownloadButton"]>button:hover{
  border-color:var(--ptp-cyan) !important; color:#fff !important;
  box-shadow:0 0 16px rgba(0,229,255,.6); transform:translateY(-1px);
}
.stButton>button[kind="primary"], [data-testid="stBaseButton-primary"]{
  background:linear-gradient(180deg, rgba(39,255,157,.20), rgba(0,229,255,.06)) !important;
  border:1px solid var(--ptp-green) !important; color:#eafff5 !important;
  animation:ptp-pulse 2.3s infinite;
}
@keyframes ptp-pulse{0%,100%{box-shadow:0 0 10px rgba(39,255,157,.35)}50%{box-shadow:0 0 24px rgba(39,255,157,.8)}}

/* ---- sidebar ---- */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#070d18,#05090f) !important;
  border-right:1px solid rgba(0,229,255,.28);
  box-shadow:5px 0 28px rgba(0,0,0,.55);
}
[data-testid="stSidebar"] h2{font-size:1.05rem}

/* ---- metrics ---- */
[data-testid="stMetric"]{
  background:linear-gradient(180deg, rgba(11,19,34,.95), rgba(6,11,20,.95));
  border:1px solid rgba(0,229,255,.22); border-radius:5px;
  padding:14px 14px 10px; position:relative; overflow:hidden;
}
[data-testid="stMetric"]::before{
  content:""; position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg, transparent, var(--ptp-cyan), transparent);
  background-size:200% 100%; animation:ptp-flow 3.2s linear infinite;
}
[data-testid="stMetricValue"]{
  font-family:'Orbitron' !important; color:var(--ptp-cyan) !important;
  text-shadow:0 0 14px rgba(0,229,255,.55);
}
[data-testid="stMetricLabel"]{text-transform:uppercase; letter-spacing:1px; color:var(--ptp-dim) !important}

/* ---- expanders (obligation rows) ---- */
[data-testid="stExpander"]{
  border:1px solid rgba(0,229,255,.16) !important;
  border-left:3px solid rgba(0,229,255,.55) !important;
  border-radius:3px !important; background:rgba(8,14,24,.55) !important;
  margin-bottom:7px; transition:.18s ease;
}
[data-testid="stExpander"]:hover{
  border-left-color:var(--ptp-cyan) !important;
  box-shadow:0 0 16px rgba(0,229,255,.18);
}
[data-testid="stExpander"] summary{font-family:'Share Tech Mono',monospace !important}
[data-testid="stExpander"] summary:hover{color:var(--ptp-cyan) !important}

/* ---- inputs / selects ---- */
[data-baseweb="select"]>div, .stTextInput input, [data-baseweb="input"]{
  background:rgba(8,16,28,.85) !important; border:1px solid rgba(0,229,255,.30) !important;
  color:var(--ptp-text) !important; font-family:'Share Tech Mono',monospace !important;
}
[data-testid="stSlider"] [role="slider"]{box-shadow:0 0 12px var(--ptp-cyan)}

/* ---- alerts / blockquotes ---- */
blockquote{
  border-left:3px solid var(--ptp-cyan) !important;
  background:rgba(0,229,255,.05); color:#d7ecff;
}

/* ---- status chips (rendered in row bodies) ---- */
.ptp-chip{
  display:inline-block; padding:3px 13px; border-radius:2px; border:1px solid currentColor;
  font-family:'Share Tech Mono',monospace; font-size:.76rem; letter-spacing:2px; text-transform:uppercase;
}
.ptp-chip.v{color:var(--ptp-green); background:rgba(39,255,157,.08); box-shadow:0 0 12px rgba(39,255,157,.30)}
.ptp-chip.w{color:var(--ptp-amber); background:rgba(255,184,77,.08); box-shadow:0 0 12px rgba(255,184,77,.25)}
.ptp-chip.r{color:var(--ptp-red); background:rgba(255,59,107,.08); animation:ptp-alert 1.1s infinite}
.ptp-chip.n{color:var(--ptp-dim); background:rgba(111,138,166,.08)}
@keyframes ptp-alert{0%,100%{box-shadow:0 0 7px rgba(255,59,107,.45)}50%{box-shadow:0 0 20px rgba(255,59,107,.95)}}

/* ---- processing banner ---- */
.ptp-proc{
  position:relative; overflow:hidden; border:1px solid rgba(0,229,255,.35);
  border-radius:5px; background:rgba(6,12,22,.85); color:var(--ptp-cyan);
  padding:12px 18px; letter-spacing:3px; text-transform:uppercase; font-size:.85rem;
}
.ptp-proc::after{
  content:""; position:absolute; left:0; bottom:0; height:3px; width:38%;
  background:linear-gradient(90deg, transparent, var(--ptp-cyan), var(--ptp-green), transparent);
  animation:ptp-bar 1.1s linear infinite;
}
@keyframes ptp-bar{0%{left:-38%}100%{left:100%}}

/* trim default streamlit chrome */
[data-testid="stHeader"]{background:transparent}
#MainMenu, footer{visibility:hidden}
"""

_CHIP_CLASS = {
    Status.VERIFIED: "v", Status.USER_ACCEPTED: "v",
    Status.NEEDS_REVIEW: "w",
    Status.FLAGGED: "r", Status.USER_REJECTED: "r",
    Status.PENDING: "n",
}


def inject_theme() -> None:
    """Inject the theme CSS and the animated scanline overlay (call once/run)."""
    st.markdown("<style>%s</style>" % _CSS, unsafe_allow_html=True)
    st.markdown("<div class='ptp-scan'></div>", unsafe_allow_html=True)


def hero(subtitle: str) -> None:
    """Render the animated command-console banner."""
    html = (
        "<div class='ptp-hero'>"
        "<div class='ptp-hero-top'>"
        "<div class='ptp-title'>PERMIT<b>//</b>TO<b>//</b>PROOF</div>"
        "<div class='ptp-live'><span class='ptp-dot'></span>System Online</div>"
        "</div>"
        "<div class='ptp-sub'>%s</div>"
        "<div class='ptp-flow'></div>"
        "</div>"
    ) % subtitle
    st.markdown(html, unsafe_allow_html=True)


def status_chip(status: Status, label: str) -> str:
    """HTML for a glowing status chip (flagged pulses as an alert)."""
    cls = _CHIP_CLASS.get(status, "n")
    return "<span class='ptp-chip %s'>%s</span>" % (cls, label)


def processing_banner(text: str = "Analyzing permit // grounding obligations // verifying") -> str:
    """HTML for the animated extraction banner."""
    return "<div class='ptp-proc'>&#9670;&#9670; %s &#9670;&#9670;</div>" % text
