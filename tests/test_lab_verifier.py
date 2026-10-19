"""Tests for the deterministic lab-value range checks.

This is the safety-critical component: it decides whether a value is
within, below or above its reference range, and the model is told to
repeat that verdict verbatim. A wrong answer here becomes a wrong
answer to the patient.
"""

import pytest

from services.lab_verifier import verify_lab_ranges


def statuses(context):
    """Map analyte name -> full verdict sentence."""

    return {f["name"]: f["text"] for f in verify_lab_ranges(context)}


# ------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------

def test_value_inside_range_is_within():
    result = statuses("Blood Urea 21.6 mg/dL 15 - 45")

    assert "within the reference range" in result["Blood Urea"]


def test_value_below_low_limit_is_below():
    result = statuses("Vitamin D 2 ng/mL 20 - 30")

    assert "below the reference range" in result["Vitamin D"]


def test_value_above_high_limit_is_above():
    result = statuses("HbA1c 6.2 % 4 - 5.6")

    assert "above the reference range" in result["HbA1c"]


def test_value_on_the_boundary_is_within():
    result = statuses("Sodium 135 mmol/L 135 - 145")

    assert "within the reference range" in result["Sodium"]


def test_regression_blood_urea_21_6_is_not_above_range():
    """The original defect: llama3.2 called 21.6 in 15-45 'above'."""

    verdict = statuses("Blood Urea 21.6 mg/dL 15 - 45")["Blood Urea"]

    assert "above" not in verdict
    assert "below" not in verdict


# ------------------------------------------------------------------
# Analyte names read out of messy OCR
# ------------------------------------------------------------------

def test_panel_heading_is_stripped_from_the_name():
    result = statuses("KIDNEY FUNCTION TESTS Blood Urea 21.6 mg/dL 15 - 45")

    assert "Blood Urea" in result


def test_short_abbreviation_is_kept_in_the_name():
    """'WBC Count' must not be shortened to 'Count'."""

    result = statuses("WBC Count 7900 cells/cumm 4000 - 11000")

    assert "WBC Count" in result
    assert "Count" not in result


def test_method_note_is_stripped_from_the_name():
    result = statuses(
        "(Modified Berthelot) Serum Creatinine 1.07 mg/dL 0.6 - 1.4"
    )

    assert "Serum Creatinine" in result


def test_trailing_symbol_qualifier_is_stripped():
    result = statuses("Sodium ( Na ) 139.8 mmol/L 135 - 145")

    assert "Sodium" in result


def test_value_and_range_split_across_two_lines():
    result = statuses(
        "Blood Urea : 21.6 mgs/dl\nReference Range : 15 - 45 mgs/dl"
    )

    assert "within the reference range" in result["Blood Urea"]


# ------------------------------------------------------------------
# Rows that must NOT produce a verdict
# ------------------------------------------------------------------

def test_interpretation_word_between_value_and_range_is_rejected():
    """The range belongs to another row when 'Below' sits between."""

    assert "A / G Ratio" not in statuses("A / G Ratio 1.8 Below 6.1 - 7")


def test_lipid_category_band_is_not_treated_as_a_reference_range():
    context = (
        "LDL Cholesterol 129 mg/dL Near optimal 100 - 129 "
        "Borderline 130 - 159"
    )

    assert statuses(context) == {}


def test_date_row_is_not_treated_as_a_result():
    assert statuses("Sample Collected On: 12 - 03 - 2024") == {}


def test_a_one_sided_limit_is_compared_rather_than_ignored():
    """This used to be skipped as "not a low-high range". A verdict
    IS possible - 210 is above 200 - and skipping it left the model
    to do the comparison, which produced "207 mg/dl, which is within
    the desirable range of less than 200 mg/dl"."""

    result = statuses("Cholesterol total 210 mg/dL Desirable < 200")

    assert "above that limit" in result["Cholesterol total"]


def test_a_value_under_its_limit_is_within():
    result = statuses("Cholesterol total 180 mg/dL Desirable < 200")

    assert "within that limit" in result["Cholesterol total"]


def test_the_real_lipid_row_is_judged_correctly():
    """Exactly as the row came out of the uploaded report."""

    result = statuses(
        "Total Cholesterol 207 mg/dl Desirable Cholesterol: "
        "<200mg/dl"
    )

    assert (
        "measured value 207 against the recommended limit of "
        "under 200 is above that limit"
    ) in result["Total Cholesterol"]


def test_a_minimum_is_compared_the_other_way():
    result = statuses("HDL Cholesterol 32 mg/dL Desirable > 40")

    assert "below that minimum" in result["HDL Cholesterol"]

    higher = statuses("HDL Cholesterol 48 mg/dL Desirable > 40")

    assert "at or above that minimum" in higher["HDL Cholesterol"]


def test_a_row_listing_several_limits_is_left_alone():
    """With more than one limit there is no way to know which the
    value should be judged against."""

    assert statuses(
        "Cholesterol 207 mg/dl Desirable < 200 High > 240"
    ) == {}


def test_a_limit_with_no_value_before_it_gives_nothing():
    assert statuses("High Cholesterol: > 240mg/dl") == {}


def test_heading_line_does_not_absorb_the_next_row():
    """A heading with no digits must not merge with the row below."""

    result = statuses("COMPLETE BLOOD COUNT\nWBC Count 7900 4000 - 11000")

    assert list(result) == ["WBC Count"]


def test_empty_context_returns_nothing():
    assert verify_lab_ranges("") == []
    assert verify_lab_ranges(None) == []


# ------------------------------------------------------------------
# Whole-report behaviour
# ------------------------------------------------------------------

def test_multiple_analytes_are_each_classified():
    context = (
        "Haemoglobin 15.8 g/dL 13.5 - 18\n"
        "Total WBC Count 7900 cells/cumm 4000 - 10000\n"
        "PCV 47.4 % 40 - 50\n"
    )

    result = statuses(context)

    assert set(result) == {"Haemoglobin", "Total WBC Count", "PCV"}
    assert all("within" in verdict for verdict in result.values())


def test_the_same_analyte_is_not_reported_twice():
    context = (
        "Blood Urea 21.6 mg/dL 15 - 45\n"
        "Blood Urea 21.6 mg/dL 15 - 45\n"
    )

    names = [f["name"] for f in verify_lab_ranges(context)]

    assert names == ["Blood Urea"]


# ------------------------------------------------------------------
# Multi-line sub-rows (ESR)
# ------------------------------------------------------------------

ESR_ROWS = "ESR 1/2 Hour 02 mm\n1 Hour 05 mm 0 to 20"


def test_a_row_with_its_own_value_is_not_merged_into_the_row_above():
    """Joining produced a finding named
    "ESR 1/2 Hour 02 mm 1 Hour"."""

    assert not any(
        "1/2" in name for name in statuses(ESR_ROWS)
    )


def test_a_sub_row_carries_the_label_from_the_line_above():
    """Alone, "1 Hour 05 mm 0 to 20" was reported as "Hour"."""

    result = statuses(ESR_ROWS)

    assert "ESR 1 Hour" in result
    assert "Hour" not in result
    assert "within the reference range" in result["ESR 1 Hour"]


def test_a_half_hour_row_without_a_range_gets_no_verdict():
    assert len(statuses(ESR_ROWS)) == 1


# ------------------------------------------------------------------
# Comma-grouped numbers
# ------------------------------------------------------------------

def test_a_comma_grouped_range_is_read_whole():
    """The range pattern matched "000 to 5" inside
    "1,50,000 to 5,00,000", inventing a 0 - 5 range."""

    result = statuses(
        "Platelet Count 2,89,000 Cells/cumm 1,50,000 to 5,00,000"
    )

    assert "150000 to 500000" in result["Platelet Count"]
    assert "within the reference range" in result["Platelet Count"]


def test_a_low_comma_grouped_value_is_classified_below():
    result = statuses(
        "Platelet Count 95,000 Cells/cumm 1,50,000 to 5,00,000"
    )

    assert "below the reference range" in result["Platelet Count"]


def test_western_grouping_also_works():
    result = statuses("WBC 12,500 /cumm 4,000 to 11,000")

    assert "above the reference range" in result["WBC"]


def test_a_number_split_by_extraction_is_skipped_not_guessed():
    """"2,89,000" came out of the PDF as "2,89,00 0". The value
    cannot be recovered without guessing, so no verdict is given."""

    assert statuses(
        "Platelet Count 2,89,00 0 Cells/cumm 1,50,000 to 5,00,000"
    ) == {}


# ------------------------------------------------------------------
# Sex-specific reference ranges
# ------------------------------------------------------------------

SEXED_ROWS = (
    "Haemoglobin {hb} gms% M-(13.5 to 18.0)\n"
    "F-(11.5 to 16.4)\n"
    "Total RBC Count {rbc} millions/cum m M-(4.5 to 6.5)F-(3.9 to5.6)\n"
    "PCV {pcv} % M-(40 to 50)F-(36 to 47)"
)


def female(hb="12.0", rbc="4.2", pcv="38.0"):
    return "Age/Sex : 29 /Female\n" + SEXED_ROWS.format(
        hb=hb, rbc=rbc, pcv=pcv
    )


def male(hb="15.8", rbc="5.3", pcv="47.4"):
    return "Age/Sex : 31 /Male\n" + SEXED_ROWS.format(
        hb=hb, rbc=rbc, pcv=pcv
    )


def test_a_womans_values_are_judged_against_the_female_range():
    """The first range on the row is the male one. Taking it told a
    healthy woman her haemoglobin, RBC and PCV were all below range -
    three false anaemia flags, delivered as VERIFIED CHECKS."""

    result = statuses(female())

    assert "11.5 to 16.4" in result["Haemoglobin"]
    assert all(
        "within the reference range" in text
        for text in result.values()
    )


def test_a_mans_values_still_use_the_male_range():
    result = statuses(male())

    assert "13.5 to 18" in result["Haemoglobin"]
    assert all(
        "within the reference range" in text
        for text in result.values()
    )


def test_a_genuinely_low_result_is_still_flagged_for_a_woman():
    """The fix must not simply suppress findings."""

    result = statuses(female(hb="10.2", rbc="3.1", pcv="30.0"))

    assert len(result) == 3
    assert all(
        "below the reference range" in text
        for text in result.values()
    )


def test_the_value_is_read_from_before_every_range_on_the_row():
    """Reading it before the chosen range picked 6.5 - the top of
    the male range - as the patient's RBC count."""

    result = statuses(female(rbc="4.2"))

    assert "measured value 4.2" in result["Total RBC Count"]


def test_a_sex_specific_row_is_skipped_when_the_sex_is_unknown():
    """Two ranges and no stated sex is no basis to choose."""

    assert statuses(SEXED_ROWS.format(
        hb="15.8", rbc="5.3", pcv="47.4"
    )) == {}


def test_a_caller_can_supply_the_sex():
    """A chunk retrieved on its own has no Age/Sex header, so the
    sex recorded at upload is passed in instead."""

    context = SEXED_ROWS.format(hb="12.0", rbc="4.2", pcv="38.0")

    result = {
        item["name"]: item["text"]
        for item in verify_lab_ranges(context, patient_sex="F")
    }

    assert "11.5 to 16.4" in result["Haemoglobin"]


def test_a_row_with_one_plain_range_is_unaffected():
    """Only rows that actually offer a choice engage the sex logic."""

    result = statuses(
        "Age/Sex : 29 /Female\nTotal WBC Count 7900 cells/cumm "
        "4000 to 10000"
    )

    assert "within the reference range" in result["Total WBC Count"]


@pytest.mark.parametrize(
    "header,expected",
    [
        ("Age/Sex : 31 /Male", "M"),
        ("Age/Sex: 29/F", "F"),
        ("Sex : Female", "F"),
        ("Gender: Male", "M"),
        ("Patient : Mr. Ravi Kumar H", ""),
    ],
)
def test_the_patient_sex_is_read_from_the_report(header, expected):
    from services.lab_verifier import detect_patient_sex

    assert detect_patient_sex(header) == expected

