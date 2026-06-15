"""Tests for the evaluation harness (Part A5 metrics). Offline, deterministic."""

import os

import pytest
from pydantic import ValidationError

from app.core.pipeline import run_pipeline
from app.eval import metrics as M
from app.eval.gold import GoldSet, load_gold
from app.llm.mock import MockBackend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_PATH = os.path.join(ROOT, "sample_data", "gold", "sample_permit.json")


@pytest.fixture(scope="module")
def gold():
    return load_gold(GOLD_PATH)


@pytest.fixture(scope="module")
def runs(sample_pdf_path):
    on = run_pipeline(sample_pdf_path, MockBackend(), threshold=0.6,
                      verification_enabled=True)
    off = run_pipeline(sample_pdf_path, MockBackend(), threshold=0.6,
                       verification_enabled=False)
    return on, off


# --- gold loading & validation -------------------------------------------

def test_gold_loads_and_validates(gold):
    assert gold.permit_id == "PTP-2026-0001"
    assert gold.is_illustrative
    assert len(gold.obligations) == 15


def test_gold_rejects_unknown_field():
    with pytest.raises(ValidationError):
        GoldSet.model_validate({
            "permit_id": "x", "label_provenance": "ILLUSTRATIVE_AUTHOR_KNOWN",
            "obligations": [], "surprise": 1,
        })


def test_gold_rejects_bad_provenance():
    with pytest.raises(ValidationError):
        GoldSet.model_validate({
            "permit_id": "x", "label_provenance": "made-up", "obligations": [],
        })


# --- extraction matching / P / R / F1 ------------------------------------

def test_extraction_matches_all_true_obligations(runs, gold):
    on, _ = runs
    mr = M.match_extractions(on.obligations, gold)
    # 15 genuine obligations found, 6 planted corruptions are false positives.
    assert mr.tp == 15
    assert mr.fn == 0
    assert mr.fp == 6


def test_extraction_prf_values(runs, gold):
    on, _ = runs
    prf = M.extraction_prf(M.match_extractions(on.obligations, gold))
    assert prf["recall"] == pytest.approx(1.0)
    assert prf["precision"] == pytest.approx(15 / 21)
    assert 0.0 < prf["f1"] <= 1.0


def test_fabricated_value_is_a_false_positive(runs, gold):
    on, _ = runs
    mr = M.match_extractions(on.obligations, gold)
    fp_ids = {r.ext_id for r in mr.records if r.outcome == "FP"}
    # the planted hallucination (250 ppm) and wrong-unit/out-of-range/operator
    # cases must not match any gold obligation
    assert len(fp_ids) == 6


# --- verification lift vs gold -------------------------------------------

def test_verification_lift_counts(runs, gold):
    on, off = runs
    vl = M.verification_lift(on, off, gold)
    assert vl["n_true_errors"] == 6
    assert vl["errors_caught_off"] == 0          # OFF trusts everything
    assert vl["off"]["recall"] == 0.0
    assert vl["errors_caught_on"] >= 5           # at least 5 of 6 routed to a human
    assert vl["lift"] == pytest.approx(vl["on"]["recall"])


# --- calibration ----------------------------------------------------------

def test_calibration_bounds_and_binning(runs, gold):
    on, _ = runs
    cal = M.calibration(on.obligations, gold, n_bins=10)
    assert 0.0 <= cal["ece"] <= 1.0
    assert 0.0 <= cal["mce"] <= 1.0
    assert sum(b["count"] for b in cal["bins"]) == len(on.obligations)
    assert len(cal["bins"]) == 10


# --- selective-prediction curve ------------------------------------------

def test_selective_curve_shape_and_no_side_effects(runs, gold):
    on, _ = runs
    before = [o.status for o in on.obligations]
    sel = M.selective_curve(on, gold, target_accuracy=0.95)
    after = [o.status for o in on.obligations]
    assert before == after  # statuses restored after the sweep
    for p in sel["points"]:
        assert 0.0 <= p["automation_rate"] <= 1.0
        assert 0.0 <= p["auto_accept_accuracy"] <= 1.0
    # raising the threshold cannot increase automation
    pts = sorted(sel["points"], key=lambda p: p["threshold"])
    autos = [p["automation_rate"] for p in pts]
    assert all(autos[i] >= autos[i + 1] - 1e-9 for i in range(len(autos) - 1))


def test_operating_point_exists(runs, gold):
    on, _ = runs
    sel = M.selective_curve(on, gold, target_accuracy=0.95)
    assert sel["operating_point"] is not None
    assert sel["operating_point"]["auto_accept_accuracy"] >= 0.95


# --- end-to-end bundle ----------------------------------------------------

def test_evaluate_all_keys_and_determinism(runs, gold):
    on, off = runs
    a = M.evaluate_all(on, off, gold)
    for key in ("extraction", "verification_lift", "calibration", "selective",
                "label_provenance", "n_obligations"):
        assert key in a
    on2 = run_pipeline(os.path.join(ROOT, "sample_data", "sample_permit.pdf"),
                       MockBackend(), threshold=0.6, verification_enabled=True)
    off2 = run_pipeline(os.path.join(ROOT, "sample_data", "sample_permit.pdf"),
                        MockBackend(), threshold=0.6, verification_enabled=False)
    b = M.evaluate_all(on2, off2, gold)
    assert a["extraction"]["f1"] == b["extraction"]["f1"]
    assert a["verification_lift"]["lift"] == b["verification_lift"]["lift"]
