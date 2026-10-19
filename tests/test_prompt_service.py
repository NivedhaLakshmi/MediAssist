"""Tests for prompt construction.

The prompt's shape matters: the question must come last so the model
answers it rather than continuing a list of instructions, and long
earlier answers must be trimmed so they are read as context rather
than copied verbatim.
"""

from services.prompt_service import (
    build_healthcare_prompt,
    format_conversation_history,
)


def turn(role, content):
    return {"role": role, "content": content}


# ------------------------------------------------------------------
# Conversation history
# ------------------------------------------------------------------

def test_no_history_is_reported_as_none():
    assert format_conversation_history([]) == "(none)"
    assert format_conversation_history(None) == "(none)"


def test_history_labels_each_speaker():
    rendered = format_conversation_history(
        [turn("user", "What is urea?"), turn("assistant", "A waste product.")]
    )

    assert "Nurse: What is urea?" in rendered
    assert "MediAssist: A waste product." in rendered


def test_only_the_most_recent_turns_are_kept():
    history = [turn("user", f"Question {i}") for i in range(20)]

    rendered = format_conversation_history(history, max_messages=4)

    assert "Question 19" in rendered
    assert "Question 0" not in rendered


def test_a_long_assistant_answer_is_truncated():
    """Full previous answers were being copied verbatim by the model."""

    rendered = format_conversation_history(
        [turn("assistant", "x" * 500)]
    )

    assert "..." in rendered
    assert len(rendered) < 400


def test_a_long_patient_question_is_not_truncated():
    rendered = format_conversation_history([turn("user", "y" * 400)])

    assert "y" * 400 in rendered


# ------------------------------------------------------------------
# Prompt assembly
# ------------------------------------------------------------------

def test_returns_a_system_and_a_user_message():
    system, user = build_healthcare_prompt("context", "question")

    assert "MediAssist" in system
    assert "question" in user


def test_the_question_is_the_last_thing_in_the_prompt():
    """Instructions after the question made the model echo it back."""

    _, user = build_healthcare_prompt(
        "some report text", "Is my urea normal?"
    )

    assert user.rstrip().endswith("Is my urea normal?")


def test_the_report_context_is_included():
    _, user = build_healthcare_prompt("Blood Urea 21.6", "Is it normal?")

    assert "Blood Urea 21.6" in user


def test_verified_findings_are_listed_when_present():
    _, user = build_healthcare_prompt(
        "Blood Urea 21.6 15 - 45",
        "Is my urea normal?",
        verified=[
            {
                "name": "Blood Urea",
                "text": "Blood Urea: measured value 21.6 ... is within "
                        "the reference range.",
            }
        ],
    )

    assert "within the reference range" in user


def test_absent_verified_findings_are_stated_explicitly():
    _, user = build_healthcare_prompt("some text", "a question", verified=[])

    assert "none" in user.lower()


def test_missing_context_is_stated_rather_than_left_blank():
    _, user = build_healthcare_prompt("", "a question")

    assert "no report content" in user.lower()


def test_system_prompt_allows_general_explanation():
    """Rule 2: refusing to explain a known term was the earlier bug."""

    system, _ = build_healthcare_prompt("ctx", "q")

    assert "general" in system.lower()
    assert "never refuse" in system.lower()


def test_system_prompt_forbids_treatment_advice():
    system, _ = build_healthcare_prompt("ctx", "q")

    assert "do not diagnose" in system.lower()
