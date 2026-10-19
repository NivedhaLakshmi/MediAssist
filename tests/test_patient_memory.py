"""Tests for patient report storage.

Uploading the same report twice used to store it twice, because the
only guard was a filename set living in the browser session. These
tests use a throwaway vector store, never the real patient_db.
"""

import pytest

from langchain_community.vectorstores import Chroma

from services import patient_memory
from services.embedding_service import embedding


REPORT = (
    "LAB REPORT\n"
    "Haemoglobin 15.8 g/dL 13.5 - 18\n"
    "Total WBC Count 7900 cells/cumm 4000 - 10000\n"
)

OTHER_REPORT = "LAB REPORT\nGlucose 92 mg/dL 70 - 110\n"


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        patient_memory,
        "patient_vectordb",
        Chroma(
            persist_directory=str(tmp_path / "test_db"),
            embedding_function=embedding,
        ),
    )


def store(text, name="report.pdf"):
    return patient_memory.add_patient_report(text, name, "Medical Report")


# ------------------------------------------------------------------
# Storing
# ------------------------------------------------------------------

def test_a_new_report_is_stored_and_returns_an_id():
    assert store(REPORT) is not None
    assert patient_memory.count_patient_reports()[0] == 1


def test_an_empty_report_is_not_stored():
    assert store("   ") is None
    assert patient_memory.count_patient_reports() == (0, 0)


def test_report_text_is_retrievable_after_storing():
    store(REPORT)

    documents = patient_memory.get_all_patient_documents()

    assert any("Haemoglobin" in d["text"] for d in documents)


# ------------------------------------------------------------------
# Duplicate detection
# ------------------------------------------------------------------

def test_the_same_report_is_not_stored_twice():
    store(REPORT)

    assert store(REPORT) is None
    assert patient_memory.count_patient_reports()[1] == 1


def test_the_same_text_under_a_new_filename_is_still_a_duplicate():
    """A filename check would have missed this."""

    store(REPORT, "original.pdf")

    assert store(REPORT, "copy of original.pdf") is None


def test_a_different_report_is_stored_alongside():
    store(REPORT, "one.pdf")
    store(OTHER_REPORT, "two.pdf")

    assert patient_memory.count_patient_reports()[0] == 2


def test_an_updated_report_with_changed_values_is_stored():
    """Different text means a different hash, so it must go in."""

    store(REPORT, "report.pdf")
    updated = REPORT.replace("15.8", "14.2")

    assert store(updated, "report.pdf") is not None


# ------------------------------------------------------------------
# Searching
# ------------------------------------------------------------------

def test_an_unrelated_question_falls_through_to_other_sources():
    """Without a distance threshold, reports matched every question."""

    store(REPORT)

    _, context, _, _ = patient_memory.search_patient_reports(
        "who won the world cup"
    )

    assert context == ""


def test_a_question_about_the_patients_own_values_retrieves_them():
    """The gate was set at 1.0, below every genuine match, so a
    question about the patient's own report retrieved nothing and was
    answered from the external sources instead."""

    store(REPORT)

    for question in (
        "what is my haemoglobin",
        "is my haemoglobin normal",
        "how is my wbc count",
    ):
        _, context, _, _ = patient_memory.search_patient_reports(
            question
        )

        assert "Haemoglobin" in context, question


def test_a_close_match_is_reported_as_strong():
    """A strong match lets the caller skip the slow network sources."""

    store(REPORT)

    _, _, _, score = patient_memory.search_patient_reports(
        "what is my haemoglobin level"
    )

    assert patient_memory.is_strong_match(score)


def test_an_unrelated_question_is_not_a_strong_match():
    store(REPORT)

    _, _, _, score = patient_memory.search_patient_reports(
        "who won the world cup"
    )

    assert not patient_memory.is_strong_match(score)


def test_no_stored_report_gives_no_score():
    _, context, _, score = patient_memory.search_patient_reports(
        "what is my haemoglobin"
    )

    assert context == ""
    assert score is None
    assert not patient_memory.is_strong_match(score)


# ------------------------------------------------------------------
# Whole-report retrieval
# ------------------------------------------------------------------

LONG_REPORT = (
    "COMPLETE BLOOD COUNT\n"
    "Haemoglobin 15.8 g/dL 13.5 - 18\n"
    "Total WBC Count 7900 cells/cumm 4000 - 10000\n"
    "Platelet Count 2.45 lakh/cumm 1.5 - 4.1\n"
    "KIDNEY FUNCTION TEST\n"
    "Blood Urea 21.6 mg/dL 15 - 45\n"
    "Serum Creatinine 0.94 mg/dL 0.7 - 1.3\n"
    "LIPID PROFILE\n"
    "Total Cholesterol 212 mg/dL 125 - 200\n"
    "Triglycerides 178 mg/dL 25 - 150\n"
    "THYROID PROFILE\n"
    "TSH 5.9 uIU/mL 0.4 - 4.0\n"
    "GLUCOSE\n"
    "Fasting Blood Sugar 112 mg/dL 70 - 100\n"
    "HbA1c 6.1 percent 4.0 - 5.6\n"
) * 3


def test_whole_report_returns_more_than_the_matching_chunks():
    """A summary needs every panel, not the chunks nearest the words
    of the question."""

    store(LONG_REPORT, "long.pdf")

    _, partial, _, _ = patient_memory.search_patient_reports(
        "summarise my report", k=2
    )

    _, whole, _, _ = patient_memory.search_patient_reports(
        "summarise my report", k=2, whole_report=True
    )

    assert len(whole) > len(partial)


def test_a_summary_question_is_not_blocked_by_the_distance_gate(
    monkeypatch,
):
    """"Summarise my report" shares no words with a lab table, and
    measured 1.69 - past the gate - against the very report it was
    asking about. The gate is closed here so the bypass is what is
    being tested, not the fixture's distance."""

    store(LONG_REPORT, "long.pdf")

    monkeypatch.setattr(patient_memory, "PATIENT_SCORE_THRESHOLD", 0.1)

    _, blocked, _, _ = patient_memory.search_patient_reports(
        "give me a summary of my report"
    )

    _, context, _, _ = patient_memory.search_patient_reports(
        "give me a summary of my report", whole_report=True
    )

    assert blocked == ""
    assert "Haemoglobin" in context


def test_a_summary_question_still_finds_nothing_when_nothing_is_stored():
    _, context, _, _ = patient_memory.search_patient_reports(
        "summarise my report", whole_report=True
    )

    assert context == ""


def test_whole_report_stays_within_the_profile():
    store_for("alice", LONG_REPORT, "alice.pdf")
    store_for("bob", OTHER_REPORT, "bob.pdf")

    docs, _, _, _ = patient_memory.search_patient_reports(
        "summarise my report",
        profile_id="alice",
        whole_report=True,
    )

    assert all(
        d.metadata.get("profile") == "alice" for d in docs
    )


# ------------------------------------------------------------------
# Deletion
# ------------------------------------------------------------------

def test_delete_removes_every_stored_chunk():
    store(REPORT, "one.pdf")
    store(OTHER_REPORT, "two.pdf")

    patient_memory.delete_all_patient_reports()

    assert patient_memory.count_patient_reports() == (0, 0)


# ------------------------------------------------------------------
# Separation between profiles
# ------------------------------------------------------------------

def store_for(profile, text, name="report.pdf"):
    return patient_memory.add_patient_report(
        text, name, "Medical Report", profile_id=profile
    )


def test_a_report_belongs_only_to_the_profile_that_added_it():
    store_for("alice", REPORT, "alice.pdf")

    assert patient_memory.count_patient_reports("alice") == (1, 1)
    assert patient_memory.count_patient_reports("bob") == (0, 0)


def test_a_new_profile_sees_no_reports():
    store_for("alice", REPORT, "alice.pdf")

    assert patient_memory.get_all_patient_documents("carol") == []


def test_search_never_returns_another_profiles_report():
    store_for("alice", REPORT, "alice.pdf")

    docs, context, _, _ = patient_memory.search_patient_reports(
        "haemoglobin", profile_id="bob"
    )

    assert docs == []
    assert context == ""


def test_the_same_report_may_exist_under_two_profiles():
    """Two people on a shared device may hold the same document."""

    assert store_for("alice", REPORT, "shared.pdf") is not None
    assert store_for("bob", REPORT, "shared.pdf") is not None


def test_duplicates_are_still_blocked_within_one_profile():
    store_for("alice", REPORT, "shared.pdf")

    assert store_for("alice", REPORT, "shared.pdf") is None


def test_deleting_one_profile_leaves_the_others_intact():
    """Deleting used to drop the whole collection, wiping everyone."""

    store_for("alice", REPORT, "alice.pdf")
    store_for("bob", OTHER_REPORT, "bob.pdf")

    patient_memory.delete_all_patient_reports("alice")

    assert patient_memory.count_patient_reports("alice") == (0, 0)
    assert patient_memory.count_patient_reports("bob") == (1, 1)
