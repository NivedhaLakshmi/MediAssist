import re

from rapidfuzz import process, fuzz, distance
from wordfreq import zipf_frequency

from services.medical_dictionary import (
    load_mesh_terms,
    load_mesh_vocabulary,
)


# ============================================================
# Load Medical Vocabulary
# ============================================================

# Broad set - terms plus the words inside multi-word terms. Used for
# spelling correction.
MEDICAL_VOCABULARY = load_mesh_vocabulary()

# Narrow set - only what MeSH publishes as a term in its own right.
# Used to pick search keywords, where "normal" and "count" would
# otherwise qualify because they appear inside descriptors.
MEDICAL_TERMS = load_mesh_terms()


# ============================================================
# Configuration
# ============================================================

# Minimum length before attempting spelling correction.
MIN_CORRECTION_LENGTH = 4

# Minimum combined fuzzy similarity.
#
# Only ever applied to tokens that are neither a known MeSH term nor
# recognised English, so it cannot alter correctly spelled words. At
# 88 a single-letter slip scored 87.5 and was missed ("vitilago" ->
# "vitiligo"); "neutrofil" scores 84.2.
#
# Validated at 84 against 1500 correctly spelled words (common
# English plus sampled MeSH terms) and a set of unknown names and
# brands: 8 of 8 known misspellings corrected, nothing else altered.
FUZZY_SCORE_THRESHOLD = 84

# Minimum edit similarity.
EDIT_SIMILARITY_THRESHOLD = 0.80

# Maximum length difference allowed between the
# misspelled word and its candidate.
MAX_LENGTH_DIFFERENCE = 4


# ============================================================
# Text Normalization
# ============================================================

def normalize_text(text):
    """
    Perform basic text normalization.

    Operations:
        - lowercase
        - remove punctuation
        - remove repeated whitespace
    """

    if not text:
        return ""

    text = text.lower().strip()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# Question Prefix Normalization
# ============================================================

def remove_question_prefix(question):
    """
    Remove common conversational prefixes.

    Example:

        "what is my hemoglobin level"

    becomes:

        "my hemoglobin level"
    """

    prefixes = [
        "what is ",
        "what are ",
        "what was ",
        "what were ",
        "how is ",
        "how are ",
        "how was ",
        "how were ",
        "tell me about ",
        "can you tell me about ",
        "could you tell me about ",
        "please tell me about ",
        "explain ",
    ]

    for prefix in prefixes:

        if question.startswith(prefix):

            return question[
                len(prefix):
            ].strip()

    return question


# ============================================================
# Tokenization
# ============================================================

def tokenize_query(question):
    """
    Split a normalized query into individual tokens.
    """

    return question.split()


# ============================================================
# English Word Recognition
# ============================================================

# A word must be at least this common in English before it is
# accepted as correctly spelled.
#
# Any score above zero used to be enough, which quietly protected
# every misspelling common enough to reach a web corpus:
# "cholestrol" scores 1.04 and "asprin" 1.68, so neither was ever
# corrected and both were sent to PubMed as typed.
#
# Words MeSH knows are accepted before this gate is reached, so a
# rare clinical term is never judged by its English frequency.
# Validated on 2,011 correctly spelled words (common English,
# sampled MeSH terms, names and brands): at 1.8 two further
# misspellings are corrected and nothing correct is altered. At 2.0
# "anthropic" starts being rewritten.
MIN_ENGLISH_ZIPF = 1.8


def is_common_english_word(word):
    """
    Determine whether a word is recognised as an English word.

    wordfreq scores a word by how often it appears in English text.
    A common misspelling still appears, just rarely, so the score
    must clear MIN_ENGLISH_ZIPF rather than merely be positive.
    """

    return zipf_frequency(word, "en") >= MIN_ENGLISH_ZIPF


# ============================================================
# Candidate Filtering
# ============================================================

def filter_candidates(word, vocabulary):
    """
    Reduce the number of medical terms considered during
    fuzzy matching.

    Candidate terms are filtered using:

        - minimum length
        - maximum length difference
        - first character similarity

    This prevents the algorithm from comparing a typo against
    every medical term in the vocabulary.
    """

    word_length = len(word)

    candidates = []

    for candidate in vocabulary:

        # Ignore extremely short terms.
        if len(candidate) < MIN_CORRECTION_LENGTH:
            continue

        # Only compare candidates with a reasonable
        # difference in length.
        if abs(
            len(candidate) - word_length
        ) > MAX_LENGTH_DIFFERENCE:

            continue

        # Candidate should begin with the same character.
        #
        # This is a lightweight candidate-generation step
        # before the more expensive similarity calculations.
        if candidate[0] != word[0]:

            continue

        candidates.append(candidate)

    return candidates


# ============================================================
# Edit Similarity
# ============================================================

def calculate_edit_similarity(
    word,
    candidate
):
    """
    Calculate normalized Levenshtein similarity.

    Returns a value between 0 and 1.

        1.0 = identical
        0.0 = completely different
    """

    distance_value = distance.Levenshtein.distance(
        word,
        candidate
    )

    max_length = max(
        len(word),
        len(candidate)
    )

    if max_length == 0:
        return 1.0

    similarity = 1 - (
        distance_value / max_length
    )

    return similarity


# ============================================================
# Find Spelling Correction
# ============================================================

def find_correction(
    word,
    medical_vocabulary,
    fuzzy_threshold=FUZZY_SCORE_THRESHOLD,
    edit_threshold=EDIT_SIMILARITY_THRESHOLD
):
    """
    Find a high-confidence medical spelling correction.

    Algorithm:

        1. Hash-table lookup.
        2. English word recognition.
        3. Candidate filtering.
        4. RapidFuzz ratio.
        5. RapidFuzz WRatio.
        6. Levenshtein edit similarity.
        7. Accept only high-confidence candidates.
    """

    # --------------------------------------------------------
    # Already recognised medical term
    # --------------------------------------------------------

    if word in medical_vocabulary:
        return word


    # --------------------------------------------------------
    # Already recognised English word
    # --------------------------------------------------------

    if is_common_english_word(word):
        return word


    # --------------------------------------------------------
    # Ignore short words
    # --------------------------------------------------------

    if len(word) < MIN_CORRECTION_LENGTH:
        return word


    # --------------------------------------------------------
    # Generate candidates
    # --------------------------------------------------------

    candidates = filter_candidates(
        word,
        medical_vocabulary
    )

    if not candidates:
        return word


    # --------------------------------------------------------
    # RapidFuzz candidate search
    # --------------------------------------------------------

    ratio_matches = process.extract(
        word,
        candidates,
        scorer=fuzz.ratio,
        limit=5
    )

    wratio_matches = process.extract(
        word,
        candidates,
        scorer=fuzz.WRatio,
        limit=5
    )


    # --------------------------------------------------------
    # Combine candidates from both methods
    # --------------------------------------------------------

    candidate_names = set()

    for match in ratio_matches:
        candidate_names.add(match[0])

    for match in wratio_matches:
        candidate_names.add(match[0])


    # --------------------------------------------------------
    # Evaluate candidates
    # --------------------------------------------------------

    best_candidate = word
    best_score = 0

    for candidate in candidate_names:

        ratio_score = fuzz.ratio(
            word,
            candidate
        )

        wratio_score = fuzz.WRatio(
            word,
            candidate
        )

        edit_score = calculate_edit_similarity(
            word,
            candidate
        )


        # ----------------------------------------------------
        # Combined fuzzy score
        # ----------------------------------------------------

        fuzzy_score = (
            ratio_score + wratio_score
        ) / 2


        # ----------------------------------------------------
        # Candidate must pass BOTH checks
        # ----------------------------------------------------

        if fuzzy_score < fuzzy_threshold:
            continue

        if edit_score < edit_threshold:
            continue


        # ----------------------------------------------------
        # Keep the strongest candidate
        # ----------------------------------------------------

        combined_score = (
            0.6 * fuzzy_score
            + 40 * edit_score
        )

        if combined_score > best_score:

            best_score = combined_score

            best_candidate = candidate


    return best_candidate


# ============================================================
# Normalize Healthcare Query
# ============================================================

def normalize_health_query(question):
    """
    Normalize a healthcare query and detect likely medical
    spelling errors.

    Returns:

        {
            "original": original question,
            "normalized": normalized query,
            "corrections": {
                "incorrect": "correct"
            }
        }
    """

    # --------------------------------------------------------
    # Empty input
    # --------------------------------------------------------

    if not question:

        return {
            "original": question,
            "normalized": "",
            "corrections": {}
        }


    # --------------------------------------------------------
    # Preserve original question
    # --------------------------------------------------------

    original_question = question


    # --------------------------------------------------------
    # Basic normalization
    # --------------------------------------------------------

    normalized = normalize_text(
        question
    )


    # --------------------------------------------------------
    # Remove question prefix
    # --------------------------------------------------------

    normalized = remove_question_prefix(
        normalized
    )


    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    tokens = tokenize_query(
        normalized
    )


    corrected_tokens = []

    corrections = {}


    # --------------------------------------------------------
    # Correct individual tokens
    # --------------------------------------------------------

    for token in tokens:

        corrected_token = find_correction(
            token,
            MEDICAL_VOCABULARY
        )

        corrected_tokens.append(
            corrected_token
        )


        # Record actual changes.
        if corrected_token != token:

            corrections[token] = (
                corrected_token
            )


    # --------------------------------------------------------
    # Reconstruct normalized query
    # --------------------------------------------------------

    normalized_question = " ".join(
        corrected_tokens
    )


    # --------------------------------------------------------
    # Return complete result
    # --------------------------------------------------------

    return {
        "original": original_question,
        "normalized": normalized_question,
        "corrections": corrections
    }

# ============================================================
# Keyword Extraction For External Medical Sources
# ============================================================

# MedlinePlus, PubMed, DailyMed and ClinicalTrials.gov are keyword
# services: a full sentence returns nothing. "how do i find out
# vitiligo vs normal skin" gave 0 MedlinePlus results, while
# "vitiligo skin" gave 2.
#
# No stop-word list is used. A token is judged by how common it is
# in ordinary English, measured by wordfreq: conversational
# scaffolding ("how", "find", "results") is frequent, clinical terms
# are not. This replaced a hand-written 56-word list that wordfreq
# already accounted for entirely.

# Below this a token is rare enough to be content-bearing on its own.
KEYWORD_MAX_ZIPF = 4.5

# Everyday words such as "blood" or "skin" are frequent but still
# meaningful when MeSH also recognises them. "will" is in MeSH as a
# legal document and sits above this, so it is dropped.
MESH_KEYWORD_MAX_ZIPF = 5.2

MAX_SEARCH_KEYWORDS = 6

_PLURAL_SUFFIXES = (("ies", "y"), ("es", ""), ("s", ""))


def english_frequency(token):
    """Frequency of a token, or of its singular form if higher.

    Plurals are rarer than their singulars, so "ranges" (4.05) looks
    technical while "range" (5.4) plainly is not. Taking the higher
    of the two stops inflected filler slipping through.
    """

    best = zipf_frequency(token, "en")

    for suffix, replacement in _PLURAL_SUFFIXES:

        if token.endswith(suffix) and len(token) - len(suffix) >= 3:

            stem = token[: -len(suffix)] + replacement

            best = max(best, zipf_frequency(stem, "en"))

    return best


def build_search_keywords(normalized_query, vocabulary=None):
    """Reduce a normalized question to terms a keyword API can use.

    Falls back to the full query when nothing survives, so a search
    is never sent empty.
    """

    if not normalized_query:
        return ""

    if vocabulary is None:
        vocabulary = MEDICAL_TERMS

    keywords = []

    for token in normalized_query.split():

        if len(token) < 3 or token in keywords:
            continue

        frequency = english_frequency(token)

        # An unknown token scores 0, so rare clinical terms and
        # uncorrected misspellings are both kept.
        is_technical = frequency < KEYWORD_MAX_ZIPF

        is_medical = (
            token in vocabulary
            and frequency < MESH_KEYWORD_MAX_ZIPF
        )

        if is_technical or is_medical:
            keywords.append(token)

    if not keywords:
        return normalized_query

    return " ".join(keywords[:MAX_SEARCH_KEYWORDS])
