"""Tests for query normalisation and keyword extraction.

Two defects are covered:

  * "vitilago" was not corrected to "vitiligo": it scored 87.5
    against a fuzzy threshold of 88.
  * The whole question was sent to MedlinePlus and PubMed, which are
    keyword services. "how do i find out vitiligo vs normal skin"
    returned nothing; "vitiligo skin" returned two topics.

Neither the code nor these tests rely on a hand-written stop-word
list: filler is identified by how common a word is in English.
"""

import pytest

from services.query_utils import (
    MEDICAL_VOCABULARY,
    build_search_keywords,
    english_frequency,
    find_correction,
    normalize_health_query,
)


def corrected(word):
    return find_correction(word, MEDICAL_VOCABULARY)


# ------------------------------------------------------------------
# Spelling correction
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "typo,expected",
    [
        ("vitilago", "vitiligo"),
        ("creatininee", "creatinine"),
        ("bilrubin", "bilirubin"),
        ("albumine", "albumin"),
        ("neutrofil", "neutrophil"),
    ],
)
def test_medical_misspellings_are_corrected(typo, expected):
    assert corrected(typo) == expected


def test_regression_vitilago_is_corrected():
    """Scored 87.5 against the old threshold of 88 and was missed."""

    assert corrected("vitilago") == "vitiligo"


@pytest.mark.parametrize(
    "word",
    ["vitiligo", "creatinine", "diabetes", "hypertension", "skin",
     "blood", "kidney", "doctor", "exercise"],
)
def test_correctly_spelled_words_are_left_alone(word):
    assert corrected(word) == word


@pytest.mark.parametrize(
    "word",
    ["xyzabc", "qwerty", "nivedha", "streamlit", "notarealword"],
)
def test_unknown_words_are_not_forced_onto_medical_terms(word):
    """A lower threshold must not start inventing corrections."""

    assert corrected(word) == word


def test_normalisation_reports_what_it_changed():
    result = normalize_health_query("What is my vitilago?")

    assert "vitiligo" in result["normalized"]
    assert result["corrections"]["vitilago"] == "vitiligo"


def test_normalisation_handles_empty_input():
    result = normalize_health_query("")

    assert result["normalized"] == ""
    assert result["corrections"] == {}


# ------------------------------------------------------------------
# Word frequency, used instead of a stop-word list
# ------------------------------------------------------------------

def test_a_plural_is_scored_by_its_singular():
    """'ranges' alone looks rare; 'range' shows it is filler."""

    assert english_frequency("ranges") > english_frequency("vitiligo")


def test_a_clinical_term_ending_in_s_is_not_mistaken_for_a_plural():
    """'diabetes' must not be scored as the plural of 'diabete'."""

    assert english_frequency("diabetes") < 4.5


def test_an_unknown_word_scores_zero():
    assert english_frequency("notarealwordxyz") == 0


# ------------------------------------------------------------------
# Keyword extraction for the external medical sources
# ------------------------------------------------------------------

def test_conversational_wrapper_is_removed():
    """The exact query that returned nothing from MedlinePlus."""

    assert build_search_keywords(
        "how do i find out vitiligo vs normal skin"
    ) == "vitiligo skin"


def test_clinical_terms_are_kept():
    assert build_search_keywords("what is my blood urea level") == (
        "blood urea"
    )


def test_inflected_filler_is_dropped():
    """'ranges' and 'count' cost results when sent to MedlinePlus."""

    assert build_search_keywords(
        "what are the normal ranges for my haemoglobin count"
    ) == "haemoglobin"


def test_common_word_in_mesh_is_dropped_when_too_frequent():
    """'will' is a MeSH term (a legal document) but is filler here."""

    assert build_search_keywords("will vitiligo stay forever") == "vitiligo"


def test_a_bare_term_is_left_unchanged():
    assert build_search_keywords("hypertension") == "hypertension"


def test_an_uncorrected_rare_word_is_still_kept():
    """Unknown tokens score zero frequency, so they survive."""

    assert "vitilago" in build_search_keywords("about the vitilago patches")


def test_query_with_no_clinical_terms_falls_back_to_the_whole_query():
    """A search must never be sent empty."""

    question = "how are you today"

    assert build_search_keywords(question) == question


def test_empty_query_returns_empty():
    assert build_search_keywords("") == ""


def test_keyword_count_is_capped():
    question = (
        "vitiligo psoriasis eczema dermatitis melanoma urticaria "
        "rosacea scleroderma"
    )

    assert len(build_search_keywords(question).split()) <= 6


def test_duplicate_terms_appear_once():
    keywords = build_search_keywords("vitiligo and vitiligo again")

    assert keywords.split().count("vitiligo") == 1

# ------------------------------------------------------------------
# Misspellings that appear in ordinary English corpora
# ------------------------------------------------------------------

def test_a_misspelling_common_enough_to_be_in_the_corpus_is_corrected():
    """"cholestrol" scores 1.04 in wordfreq, so any-positive-score
    treated it as a real word and it reached PubMed as typed."""

    result = normalize_health_query("how is my cholestrol")

    assert result["normalized"].endswith("cholesterol")
    assert result["corrections"]["cholestrol"] == "cholesterol"


def test_a_rare_but_real_medical_term_is_left_alone():
    """"prediabetic" scores 1.43 - below the English gate - but MeSH
    knows it, so it must not be rewritten."""

    result = normalize_health_query("am i prediabetic")

    assert "prediabetic" in result["normalized"]
    assert result["corrections"] == {}


def test_an_ordinary_english_word_below_the_gate_is_not_rewritten():
    result = normalize_health_query("my haemoglobin")

    assert "haemoglobin" in result["normalized"]
