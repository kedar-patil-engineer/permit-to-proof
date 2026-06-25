"""Tests for the annotation toolkit: template <-> gold conversion and IAA."""

import os

import pytest

from app.eval.agreement import cohen_kappa, inter_annotator_agreement
from app.eval.annotate import (
    csv_to_goldset,
    goldset_from_expert_xlsx,
    goldset_to_rows,
    rows_to_obligations,
    write_csv,
)
from app.eval.gold import GoldSet, load_gold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_GOLD = os.path.join(ROOT, "sample_data", "gold", "sample_permit.json")


# --- CSV -> gold conversion ----------------------------------------------

def test_rows_to_obligations_parses_fields():
    rows = [
        {"gold_id": "G001", "parameter": "NOx", "limit_value": "30",
         "limit_unit": "ppm", "operator": "<=", "frequency": "continuous",
         "deadline": "", "citation": "40 CFR 60.44c", "source_segment_id": "",
         "is_obligation": "TRUE", "description": "NOx shall not exceed 30 ppm."},
        {"gold_id": "G013", "parameter": "", "limit_value": "", "limit_unit": "",
         "operator": "", "frequency": "", "deadline": "within 28 days",
         "citation": "40 CFR 122.41", "source_segment_id": "", "is_obligation": "",
         "description": "Submit DMRs within 28 days."},
    ]
    obs = rows_to_obligations(rows)
    assert len(obs) == 2
    assert obs[0].limit_value == 30.0
    assert obs[0].operator.value == "<="
    assert obs[1].parameter is None
    assert obs[1].is_obligation is True  # blank defaults to TRUE


def test_example_rows_are_skipped():
    rows = [
        {"gold_id": "G001", "description": "EXAMPLE ROW - delete me. blah", "operator": ""},
        {"gold_id": "", "description": "", "operator": ""},
    ]
    assert rows_to_obligations(rows) == []


def test_duplicate_gold_id_raises():
    rows = [
        {"gold_id": "G1", "description": "a", "operator": ""},
        {"gold_id": "G1", "description": "b", "operator": ""},
    ]
    with pytest.raises(ValueError):
        rows_to_obligations(rows)


def test_bad_limit_value_raises_with_row_number():
    rows = [{"gold_id": "G1", "description": "x", "limit_value": "thirty", "operator": ""}]
    with pytest.raises(ValueError):
        rows_to_obligations(rows)


def test_csv_roundtrip_preserves_gold(tmp_path):
    gs = load_gold(SAMPLE_GOLD)
    csv_path = tmp_path / "rt.csv"
    write_csv(goldset_to_rows(gs), str(csv_path))
    back = csv_to_goldset(str(csv_path), permit_id=gs.permit_id,
                          provenance="ILLUSTRATIVE_AUTHOR_KNOWN")
    assert [o.gold_id for o in back.obligations] == [o.gold_id for o in gs.obligations]
    assert len(back.obligations) == 15
    # numeric and operator fields survive the round trip
    nox = next(o for o in back.obligations if o.parameter == "NOx")
    assert nox.limit_value == 30.0 and nox.operator.value == "<="


def test_template_file_has_only_example_rows():
    template = os.path.join(ROOT, "annotation", "gold_template.csv")
    with pytest.raises(ValueError):  # all rows are EXAMPLE -> nothing to label
        csv_to_goldset(template, permit_id="X")


# --- expert .xlsx workbook adapter ---------------------------------------

def test_expert_xlsx_adapter_maps_columns(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    ws_path = tmp_path / "expert.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Permit"
    # Note: the "Applies to (unit / outfall / area)" header also contains the
    # word "unit" — the adapter must still map the real "Unit" column.
    ws.append(["Obligation ID", "Permit ID", "Source quote (verbatim)", "Citation",
               "Obligation type", "Applies to (unit / outfall / area)",
               "Parameter (pollutant)", "Operator", "Limit value", "Unit",
               "Averaging period", "Frequency / deadline", "Notes"])
    ws.append(["P1-001", "X1", "NOx shall not exceed 30 ppm", "p.10", "Emission limit",
               "Boiler B-1", "NOx", "≤", "30", "ppm", "30-day", "continuous", ""])
    ws.append(["P1-002", "X1", "pH maintained between 6.0 and 9.0", "Part I", "Emission limit",
               "Outfall 001", "pH", "range", "6.0–9.0", "s.u.", "instant", "", ""])
    ws.append(["P1-003", "X1", "Submit DMR each month", "Part II", "Reporting",
               "Outfall 001", "", "n/a", "", "", "", "Monthly DMR", ""])
    wb.save(str(ws_path))

    gs = goldset_from_expert_xlsx(str(ws_path), "Permit", permit_id="X1",
                                  provenance="EXPERT_SINGLE", labeler="Test")
    assert gs.label_provenance.value == "EXPERT_SINGLE"
    assert len(gs.obligations) == 3
    a, b, c = gs.obligations
    assert a.parameter == "NOx" and a.limit_value == 30.0
    assert a.limit_unit == "ppm"          # the real Unit column, not "Boiler B-1"
    assert a.operator.value == "<="
    assert b.operator.value == "range" and b.limit_value == 6.0  # lower bound of a range
    assert c.parameter is None and c.limit_value is None and "DMR" in c.description


# --- inter-annotator agreement -------------------------------------------

def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)
    assert cohen_kappa([], []) is None


def test_iaa_identical_keys_is_perfect():
    gs = load_gold(SAMPLE_GOLD)
    r = inter_annotator_agreement(gs, gs)
    assert r["matched"] == 15
    assert r["a_only"] == 0 and r["b_only"] == 0
    assert r["agreement_f1"] == pytest.approx(1.0)
    assert r["cohen_kappa"]["operator_family"] == pytest.approx(1.0)


def test_iaa_detects_disagreement():
    gs = load_gold(SAMPLE_GOLD)
    other = gs.model_copy(deep=True)
    # annotator B records a different frequency on a matched obligation and
    # misses one obligation entirely.
    other.obligations[0].frequency = "weekly"  # was continuous; does not break the match
    other.obligations.pop()                     # one obligation B did not label
    r = inter_annotator_agreement(gs, other)
    assert r["b_only"] == 0
    assert r["a_only"] == 1
    assert r["matched"] == 14
    assert r["agreement_f1"] < 1.0
    assert r["field_agreement"]["frequency"] < 1.0
