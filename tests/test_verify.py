"""The verification layer is the scientific core, so this is the heaviest test
file (Part B11). It feeds crafted obligations and asserts the exact checks,
match_type, and resulting status.
"""

import pytest

from app.core.schema import MatchType, Obligation, Operator, Severity, Status
from app.core.score import DEFAULT_THRESHOLD, score_and_route
from app.core.verify import (
    grounding_match,
    normalize_text,
    normalize_unit,
    verify_all,
    verify_obligation,
)

SEG = ("Condition 3.1. Nitrogen oxides (NOx) emissions from Boiler Unit B-1 "
       "shall not exceed 30 ppm, corrected to 15 percent oxygen, monitored "
       "continuously by a certified CEMS.")


def make_ob(**kw) -> Obligation:
    base = dict(
        obligation_id="OB1",
        description="NOx shall not exceed 30 ppm.",
        parameter="NOx",
        limit_value=30.0,
        limit_unit="ppm",
        operator=Operator.LE,
        citation="Condition 3.1",
        source_segment_id="S0001",
        source_quote="shall not exceed 30 ppm",
    )
    base.update(kw)
    return Obligation(**base)


def checks_by_name(ob):
    return {c.name: c for c in ob.checks}


# --- normalization & grounding primitives ---------------------------------

def test_normalize_text_collapses_punctuation_and_case():
    assert normalize_text("Shall NOT  exceed, 30 ppm!") == "shall not exceed 30 ppm"


def test_normalize_unit_handles_periods_and_aliases():
    assert normalize_unit("S.U.") == "su"
    assert normalize_unit("mg/L") == "mg/l"
    assert normalize_unit("CFU/100 mL") == "cfu/100ml"
    assert normalize_unit("tons per year") == "tons/yr"


def test_normalize_unit_strips_reference_conditions():
    # the oxygen-correction basis is not part of the unit
    assert normalize_unit("ppmvd @3% O2") == "ppmvd"
    assert normalize_unit("ppmvd @ 3% O2") == "ppmvd"
    assert normalize_unit("ppm corrected to 7% O2") == "ppm"
    # equivalent flow units
    assert normalize_unit("gal/min") == "gpm"
    assert normalize_unit("gallons per minute") == "gpm"


def test_grounding_exact():
    mt, strength = grounding_match("30 ppm", "the limit is 30 ppm here")
    assert mt is MatchType.EXACT
    assert strength == 1.0


def test_grounding_fuzzy_tolerates_small_difference():
    mt, strength = grounding_match(
        "the quick brown fox jumps over", "the quik brown fox jumps over"
    )
    assert mt is MatchType.FUZZY
    assert 0.85 <= strength < 1.0


def test_grounding_none_for_invented_text():
    mt, strength = grounding_match(
        "the permittee shall deploy a unicorn powered scrubber in appendix z",
        SEG,
    )
    assert mt is MatchType.NONE
    assert strength < 0.85


# --- adversarial grounding: meaning-changing edits must NOT pass as fuzzy ---

SEG_W = ("Five-day biochemical oxygen demand (BOD) shall not exceed 30 mg/L as a "
         "monthly average, monitored weekly by composite sample.")


def test_grounding_rejects_fabricated_value():
    # 30 -> 90 is a single-character edit but changes the compliance limit.
    mt, _ = grounding_match(
        "biochemical oxygen demand shall not exceed 90 mg/L as a monthly average",
        SEG_W)
    assert mt is MatchType.NONE


def test_grounding_rejects_unit_swap():
    # mg/L -> ug/L is a 1000x error that must not be absorbed as fuzzy.
    mt, _ = grounding_match(
        "biochemical oxygen demand shall not exceed 30 ug/L as a monthly average",
        SEG_W)
    assert mt is MatchType.NONE


def test_grounding_rejects_negation_flip():
    # dropping "not" inverts the obligation.
    mt, _ = grounding_match(
        "biochemical oxygen demand shall exceed 30 mg/L as a monthly average",
        SEG_W)
    assert mt is MatchType.NONE


def test_grounding_rejects_frequency_swap():
    mt, _ = grounding_match(
        "biochemical oxygen demand shall not exceed 30 mg/L as a monthly average "
        "monitored daily", SEG_W)
    assert mt is MatchType.NONE


def test_grounding_allows_formatting_typo_as_fuzzy():
    # A within-word OCR typo on a non-significant word, no meaning change.
    seg = "the discharge shall be collected by a representative grab sample each day"
    mt, strength = grounding_match(
        "the discharge shall be collected by a representative grab sampple each day",
        seg)
    assert mt is MatchType.FUZZY
    assert strength >= 0.85


def test_grounding_rejects_too_thin_quote():
    assert grounding_match("ppm", "the limit is 30 ppm")[0] is MatchType.NONE
    assert grounding_match("the limit", "the limit is 30 ppm")[0] is MatchType.NONE


def test_grounding_rejects_spelled_out_number_fabrication():
    # Permits often write limits in words; a word-number swap must be caught
    # just like a digit swap.
    assert grounding_match(
        "copper shall not exceed seven micrograms per liter as a daily maximum",
        "copper shall not exceed eleven micrograms per liter as a daily maximum",
    )[0] is MatchType.NONE
    assert grounding_match(
        "removal efficiency of at least ninety-five percent",
        "removal efficiency of at least fifty percent",
    )[0] is MatchType.NONE
    assert grounding_match(
        "BOD shall not exceed 30 mg/L as a thirty-day rolling average",
        "BOD shall not exceed 30 mg/L as a seven-day rolling average",
    )[0] is MatchType.NONE


def test_grounding_rejects_spliced_quote_value_present_elsewhere():
    # The fabricated quote attributes CO's 90 ppm limit to NOx. The text
    # "shall not exceed 90 ppm" exists, but not as one span next to "NOx".
    mt, strength = grounding_match(
        "NOx shall not exceed 90 ppm",
        "NOx shall not exceed 30 ppm and CO shall not exceed 90 ppm",
    )
    assert mt is MatchType.NONE


# --- the seven checks on a clean obligation -------------------------------

def test_clean_obligation_passes_all_checks_exact():
    ob = verify_obligation(make_ob(), SEG)
    by = checks_by_name(ob)
    assert set(by) == {
        "schema_complete", "grounded", "citation_present",
        "unit_valid", "range_plausible", "operator_consistent", "no_duplicate",
    }
    assert all(c.passed for c in ob.checks)
    assert ob.match_type is MatchType.EXACT


def test_check_order_is_stable():
    ob = verify_obligation(make_ob(), SEG)
    assert [c.name for c in ob.checks] == [
        "schema_complete", "grounded", "citation_present",
        "unit_valid", "range_plausible", "operator_consistent", "no_duplicate",
    ]


# --- schema_complete ------------------------------------------------------

def test_schema_complete_fails_on_missing_unit():
    ob = verify_obligation(make_ob(limit_unit=None), SEG)
    c = checks_by_name(ob)["schema_complete"]
    assert not c.passed
    assert c.severity is Severity.ERROR
    assert "limit_unit" in c.message


def test_schema_complete_fails_on_missing_operator():
    ob = verify_obligation(make_ob(operator=None), SEG)
    assert not checks_by_name(ob)["schema_complete"].passed


def test_narrative_obligation_without_limit_is_schema_complete():
    ob = make_ob(limit_value=None, limit_unit=None, operator=None, parameter=None,
                 description="The permittee shall submit reports.",
                 source_quote="The permittee shall submit reports.")
    verify_obligation(ob, "The permittee shall submit reports.")
    assert checks_by_name(ob)["schema_complete"].passed


# --- grounded (the anti-hallucination core) -------------------------------

def test_grounded_fails_on_ungrounded_quote():
    ob = make_ob(source_quote="emissions shall not exceed 9000 gigawatts of plasma")
    verify_obligation(ob, SEG)
    c = checks_by_name(ob)["grounded"]
    assert not c.passed
    assert c.severity is Severity.ERROR
    assert ob.match_type is MatchType.NONE


def test_grounded_fails_when_segment_missing():
    ob = make_ob()
    verify_obligation(ob, "")  # cited segment not found
    assert not checks_by_name(ob)["grounded"].passed
    assert ob.match_type is MatchType.NONE


def test_grounded_fails_on_empty_quote():
    ob = make_ob(source_quote="")
    verify_obligation(ob, SEG)
    assert not checks_by_name(ob)["grounded"].passed


# --- domain checks: unit_valid --------------------------------------------

def test_unit_valid_fails_for_water_unit_on_air_pollutant():
    ob = make_ob(limit_unit="mg/L")  # mg/L is a water unit; NOx is air
    verify_obligation(ob, SEG)
    c = checks_by_name(ob)["unit_valid"]
    assert not c.passed
    assert c.severity is Severity.WARNING


def test_unit_valid_passes_for_correct_unit():
    ob = verify_obligation(make_ob(limit_unit="ppm"), SEG)
    assert checks_by_name(ob)["unit_valid"].passed


def test_unit_valid_skipped_for_unknown_parameter():
    ob = make_ob(parameter="Dilithium", limit_unit="ppm")
    verify_obligation(ob, SEG)
    assert checks_by_name(ob)["unit_valid"].passed  # not in domain table => not judged


# --- domain checks: range_plausible ---------------------------------------

def test_range_plausible_fails_for_absurd_value():
    ob = make_ob(limit_value=99999.0)  # NOx in ppm tops out around 2000
    verify_obligation(ob, SEG)
    c = checks_by_name(ob)["range_plausible"]
    assert not c.passed
    assert c.severity is Severity.WARNING


def test_range_plausible_passes_for_normal_value():
    ob = verify_obligation(make_ob(limit_value=30.0), SEG)
    assert checks_by_name(ob)["range_plausible"].passed


def test_range_plausible_fails_for_negative_value():
    ob = make_ob(limit_value=-5.0)
    verify_obligation(ob, SEG)
    assert not checks_by_name(ob)["range_plausible"].passed


# --- domain checks: operator_consistent -----------------------------------

def test_operator_consistent_fails_when_operator_contradicts_wording():
    ob = make_ob(operator=Operator.GE)  # wording says "shall not exceed" => <=
    verify_obligation(ob, SEG)
    c = checks_by_name(ob)["operator_consistent"]
    assert not c.passed
    assert c.severity is Severity.WARNING


def test_operator_consistent_passes_for_matching_operator():
    ob = verify_obligation(make_ob(operator=Operator.LE), SEG)
    assert checks_by_name(ob)["operator_consistent"].passed


def test_operator_consistent_skipped_when_wording_is_neutral():
    ob = make_ob(description="NOx limit applies.", operator=Operator.LE,
                 source_quote="NOx is 30 ppm", )
    verify_obligation(ob, "NOx is 30 ppm")
    assert checks_by_name(ob)["operator_consistent"].passed


def test_operator_consistent_flags_inverted_minimum():
    # "shall not fall below" is a minimum (>=); LE on it must be flagged, not
    # silently treated as ambiguous (the 'below' cue collision bug).
    seg = "Dissolved oxygen shall not fall below 5.0 mg/L at any time."
    ob = make_ob(parameter="Dissolved Oxygen", limit_value=5.0, limit_unit="mg/L",
                 operator=Operator.LE, description="DO shall not fall below 5.0 mg/L",
                 source_quote=seg, citation="Condition 4.7")
    verify_obligation(ob, seg)
    assert not checks_by_name(ob)["operator_consistent"].passed


def test_operator_consistent_accepts_correct_minimum():
    seg = "Dissolved oxygen shall not fall below 5.0 mg/L at any time."
    ob = make_ob(parameter="Dissolved Oxygen", limit_value=5.0, limit_unit="mg/L",
                 operator=Operator.GE, description="DO minimum 5.0 mg/L",
                 source_quote=seg, citation="Condition 4.7")
    verify_obligation(ob, seg)
    assert checks_by_name(ob)["operator_consistent"].passed


# --- citation_present -----------------------------------------------------

def test_citation_present_fails_when_absent():
    ob = make_ob(citation=None)
    verify_obligation(ob, SEG)
    c = checks_by_name(ob)["citation_present"]
    assert not c.passed
    assert c.severity is Severity.WARNING


# --- no_duplicate ---------------------------------------------------------

def test_no_duplicate_flags_second_identical_obligation():
    seg_map = {"S0001": SEG}
    a = make_ob(obligation_id="OB1")
    b = make_ob(obligation_id="OB2")
    verify_all([a, b], seg_map)
    assert checks_by_name(a)["no_duplicate"].passed
    assert not checks_by_name(b)["no_duplicate"].passed
    assert checks_by_name(b)["no_duplicate"].severity is Severity.INFO


# --- end-to-end status routing via score ----------------------------------

def test_status_verified_for_clean_obligation():
    ob = verify_obligation(make_ob(confidence=0.9), SEG)
    score_and_route([ob], DEFAULT_THRESHOLD)
    assert ob.status is Status.VERIFIED


def test_status_flagged_for_error_failure():
    ob = verify_obligation(make_ob(limit_unit=None, confidence=0.9), SEG)
    score_and_route([ob], DEFAULT_THRESHOLD)
    assert ob.status is Status.FLAGGED


def test_status_needs_review_for_warning_only_failure():
    ob = verify_obligation(make_ob(limit_value=99999.0, confidence=0.9), SEG)
    score_and_route([ob], DEFAULT_THRESHOLD)
    assert ob.status is Status.NEEDS_REVIEW


def test_status_flagged_for_hallucination():
    ob = make_ob(source_quote="invented text not present anywhere in the source",
                 confidence=0.99)
    verify_obligation(ob, SEG)
    score_and_route([ob], DEFAULT_THRESHOLD)
    assert ob.status is Status.FLAGGED
    assert ob.match_type is MatchType.NONE
