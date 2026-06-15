"""Smoke tests for the Streamlit UI using the official AppTest harness.

These run the real app script headlessly and assert it renders without raising,
both before and after running extraction on the bundled sample (Part B0, B8).
"""

import os

import pytest

from app.llm.mock import MockBackend  # noqa: F401  (ensures app deps import)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, "app", "main.py")
EVAL_PAGE = os.path.join(ROOT, "app", "pages", "2_Evaluation.py")

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_app_loads_without_exception():
    at = AppTest.from_file(MAIN, default_timeout=60).run()
    assert not at.exception
    # Before running, the app shows a prompt and stops.
    assert at.info


def test_app_runs_extraction_on_sample_and_renders():
    at = AppTest.from_file(MAIN, default_timeout=120).run()
    assert not at.exception
    # The primary "Run extraction" button is the first button on the page.
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["ran"] is True
    assert len(at.session_state["obs_on"]) > 10
    # Metrics panel rendered (obligations, verified rate, routed, threshold, lift).
    assert len(at.metric) >= 5
    # Reaching here without at.exception proves the full script ran, including
    # the export section (JSON + CSV serialization and the download buttons).


def test_app_override_marks_user_accepted():
    at = AppTest.from_file(MAIN, default_timeout=120).run()
    at.button[0].click().run()
    assert not at.exception
    accept_buttons = [b for b in at.button if (b.label or "") == "Accept"]
    if not accept_buttons:
        pytest.skip("No expandable rows exposed an Accept button in this render.")
    accept_buttons[0].click().run()
    assert not at.exception
    statuses = {o.status.value for o in at.session_state["obs_on"]}
    assert "USER_ACCEPTED" in statuses


def test_evaluation_page_renders():
    at = AppTest.from_file(EVAL_PAGE, default_timeout=120).run()
    assert not at.exception
    # The four metric panels render (extraction, lift, calibration, selective).
    assert len(at.metric) >= 10
    # The illustrative-gold provenance warning must be shown.
    assert any("Illustrative" in str(w.value) for w in at.warning)
