"""Tests for reading a photographed document with a vision model.

None of these need the model installed. What matters is that
anything it returns is validated before it can reach an answer, and
that any failure falls back rather than propagating.
"""

import pytest

from services import report_extractor
from services.lab_verifier import (
    merge_findings,
    verify_extracted_rows,
)


@pytest.fixture(autouse=True)
def model_present(monkeypatch):
    """Pretend the model is installed unless a test says otherwise."""

    monkeypatch.setattr(report_extractor, "is_available", lambda: True)


def fake_reply(content, done_reason="stop"):
    return {"message": {"content": content},
            "done_reason": done_reason}


def install_reply(monkeypatch, content, done_reason="stop"):
    """Make the vision model return exactly ``content``."""

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, *args, **kwargs):
            return fake_reply(content, done_reason)

    fake_module = type("ollama", (), {"Client": FakeClient})

    monkeypatch.setitem(
        __import__("sys").modules, "ollama", fake_module
    )


def install_replies(monkeypatch, replies):
    """Answer successive calls with successive (content, reason)."""

    calls = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, *args, **kwargs):
            calls.append(kwargs.get("options", {}))
            content, reason = replies[min(len(calls) - 1,
                                          len(replies) - 1)]
            return fake_reply(content, reason)

    fake_module = type("ollama", (), {"Client": FakeClient})

    monkeypatch.setitem(
        __import__("sys").modules, "ollama", fake_module
    )

    return calls

    fake_module = type("ollama", (), {"Client": FakeClient})

    monkeypatch.setitem(
        __import__("sys").modules, "ollama", fake_module
    )


# ------------------------------------------------------------------
# Getting the JSON back out
# ------------------------------------------------------------------

GOOD = (
    '{"document_type": "lab_report", "text": "Haemoglobin 15.8", '
    '"values": [["Haemoglobin", "15.8", "g/dL", "13.5 to 18.0"]]}'
)


def test_plain_json_is_read():
    assert report_extractor._parse_payload(GOOD)["document_type"] == (
        "lab_report"
    )


def test_a_code_fence_is_stripped():
    payload = report_extractor._parse_payload(
        f"```json\n{GOOD}\n```"
    )

    assert payload["values"][0][0] == "Haemoglobin"


def test_prose_around_the_object_is_ignored():
    payload = report_extractor._parse_payload(
        f"Here is what I read:\n{GOOD}\nHope that helps."
    )

    assert payload["values"][0][1] == "15.8"


def test_unparseable_output_yields_nothing():
    assert report_extractor._parse_payload("I cannot read this") == {}
    assert report_extractor._parse_payload("") == {}
    assert report_extractor._parse_payload("[1, 2, 3]") == {}


# ------------------------------------------------------------------
# Validating what the model claims to have read
# ------------------------------------------------------------------

def row(name="Haemoglobin", value=15.8, reference="13.5 to 18.0",
        unit="g/dL", **ignored):
    """A row as the model returns it: four strings off the page."""

    return [name, value, unit, reference]


def test_a_complete_row_is_kept():
    assert len(report_extractor._clean_rows([row()])) == 1


def test_a_row_without_a_value_is_dropped():
    assert report_extractor._clean_rows([row(value=None)]) == []
    assert report_extractor._clean_rows([row(value="unreadable")]) == []


def test_a_row_with_nothing_to_compare_against_is_dropped():
    assert report_extractor._clean_rows([row(reference="")]) == []
    assert report_extractor._clean_rows(
        [row(reference="not printed")]
    ) == []


def test_a_backwards_range_is_dropped():
    assert report_extractor._clean_rows(
        [row(reference="18.0 to 13.5")]
    ) == []


def test_a_one_sided_limit_is_kept():
    kept = report_extractor._clean_rows(
        [row(name="Total Cholesterol", value=207,
             reference="<200mg/dl")]
    )

    assert kept[0]["low"] is None
    assert kept[0]["high"] == 200


def test_numbers_written_as_text_are_recovered():
    kept = report_extractor._clean_rows(
        [row(name="Platelet Count", value="2,89,000",
             reference="1,50,000 to 5,00,000")]
    )

    assert kept[0]["value"] == 289000
    assert kept[0]["low"] == 150000


def test_a_sentence_is_not_accepted_as_an_analyte_name():
    assert report_extractor._clean_rows(
        [row(name="I can see a blood report with several values " * 2)]
    ) == []


def test_junk_entries_are_ignored():
    assert report_extractor._clean_rows(
        ["not a dict", None, 42]
    ) == []
    assert report_extractor._clean_rows("not a list") == []


def test_a_plain_range_carries_no_sex():
    assert report_extractor._clean_rows([row()])[0]["sex"] == ""


def test_a_range_naming_both_sexes_becomes_two_rows():
    kept = report_extractor._clean_rows(
        [row(reference="M-(13.5 to 18.0)F-(11.5 to 16.4)")]
    )

    assert {item["sex"] for item in kept} == {"M", "F"}


# ------------------------------------------------------------------
# Reading a document end to end
# ------------------------------------------------------------------

def test_a_good_reply_gives_text_and_rows(monkeypatch):
    install_reply(monkeypatch, GOOD)

    text, rows = report_extractor.read_document(b"fake-image")

    assert text == "Haemoglobin 15.8"
    assert rows[0]["name"] == "Haemoglobin"


def test_an_empty_image_is_not_sent():
    assert report_extractor.read_document(b"") == ("", [])


def test_an_unavailable_model_returns_nothing(monkeypatch):
    monkeypatch.setattr(report_extractor, "is_available", lambda: False)

    assert report_extractor.read_document(b"image") == ("", [])


def test_a_model_error_falls_back_rather_than_raising(monkeypatch):
    class BrokenClient:
        def __init__(self, *args, **kwargs):
            pass

        def chat(self, *args, **kwargs):
            raise RuntimeError("connection refused")

    monkeypatch.setitem(
        __import__("sys").modules,
        "ollama",
        type("ollama", (), {"Client": BrokenClient}),
    )

    assert report_extractor.read_document(b"image") == ("", [])


def test_unusable_output_returns_nothing(monkeypatch):
    install_reply(monkeypatch, "I'm sorry, I can't read that.")

    assert report_extractor.read_document(b"image") == ("", [])


# ------------------------------------------------------------------
# The model never decides whether a value is normal
# ------------------------------------------------------------------

def verdicts(rows, sex=None):
    return {
        item["name"]: item["text"]
        for item in verify_extracted_rows(rows, sex)
    }


def clean(*rows):
    return report_extractor._clean_rows(list(rows))


def test_the_comparison_happens_in_python():
    result = verdicts(clean(row(value=19.2)))

    assert "above the reference range" in result["Haemoglobin"]


def test_a_value_inside_its_range_is_within():
    assert "within the reference range" in (
        verdicts(clean(row()))["Haemoglobin"]
    )


def test_a_status_invented_by_the_model_is_not_carried_through():
    """The schema has no status field, so anything it adds is
    dropped rather than repeated to the patient."""

    rows = clean(row(value=19.2, status="perfectly normal"))

    assert "above the reference range" in (
        verdicts(rows)["Haemoglobin"]
    )
    assert "perfectly normal" not in verdicts(rows)["Haemoglobin"]


def test_a_one_sided_limit_is_compared():
    rows = clean(
        row(name="Total Cholesterol", value=207,
            reference="< 200")
    )

    assert "above that limit" in verdicts(rows)["Total Cholesterol"]


def test_a_minimum_is_compared_the_other_way():
    rows = clean(
        row(name="HDL Cholesterol", value=32,
            reference="> 40")
    )

    assert "below that minimum" in verdicts(rows)["HDL Cholesterol"]


# ------------------------------------------------------------------
# Sex-specific rows
# ------------------------------------------------------------------

SEXED = None


def sexed_rows():
    return report_extractor._clean_rows([
        row(value=12.0,
            reference="M-(13.5 to 18.0)F-(11.5 to 16.4)")
    ])


def test_the_womans_range_is_used_for_a_woman():
    assert "11.5 to 16.4" in verdicts(sexed_rows(), "F")["Haemoglobin"]


def test_the_mans_range_is_used_for_a_man():
    assert "13.5 to 18" in verdicts(sexed_rows(), "M")["Haemoglobin"]


def test_a_sexed_row_is_skipped_when_the_sex_is_unknown():
    assert verdicts(sexed_rows()) == {}


def test_a_lone_sex_label_does_not_drop_the_row():
    """The model tagged a cholesterol row "M". Honouring a label
    the row never offered a choice about would lose the row
    entirely for a female patient."""

    rows = clean(
        row(name="Total Cholesterol", value=207,
            reference="M- < 200")
    )

    assert "above that limit" in (
        verdicts(rows, "F")["Total Cholesterol"]
    )
    assert "above that limit" in (
        verdicts(rows, "M")["Total Cholesterol"]
    )
    assert "above that limit" in (
        verdicts(rows)["Total Cholesterol"]
    )


# ------------------------------------------------------------------
# Two readers of the same row
# ------------------------------------------------------------------

def finding(name, text):
    return {"name": name, "text": text}


def test_readers_that_agree_produce_one_verdict():
    same = finding("Haemoglobin", "Haemoglobin: within.")

    assert merge_findings([same], [same]) == [same]


def test_readers_that_disagree_produce_none():
    """A wrong verdict is worse than no verdict."""

    merged = merge_findings(
        [finding("Haemoglobin", "Haemoglobin: 15.8 is within.")],
        [finding("Haemoglobin", "Haemoglobin: 16.8 is within.")],
    )

    assert merged == []


def test_a_row_only_one_reader_saw_is_kept():
    merged = merge_findings(
        [finding("Haemoglobin", "a")],
        [finding("Platelet Count", "b")],
    )

    assert {item["name"] for item in merged} == {
        "Haemoglobin",
        "Platelet Count",
    }


def test_merging_with_nothing_changes_nothing():
    only = [finding("Haemoglobin", "a")]

    assert merge_findings(only, []) == only
    assert merge_findings([], only) == only


# ------------------------------------------------------------------
# The compact row format
# ------------------------------------------------------------------

def test_a_row_is_four_strings_off_the_page():
    """Named low/high fields cost a third more tokens, and in a
    positional array the model put "4000 to 10000" in the low slot
    and left high empty."""

    kept = report_extractor._clean_rows(
        [["Haemoglobin", "15.8", "gms%", "M- (13.5 to 18.0)"]]
    )

    assert kept[0] == {
        "name": "Haemoglobin",
        "value": 15.8,
        "low": 13.5,
        "high": 18.0,
        "sex": "M",
    }


def test_a_row_missing_its_unit_is_tolerated():
    """The range is taken as the last item, not a fixed position."""

    kept = report_extractor._clean_rows(
        [["Haemoglobin", "15.8", "13.5 to 18.0"]]
    )

    assert kept[0]["low"] == 13.5
    assert kept[0]["high"] == 18.0


def test_a_whole_range_string_is_not_read_as_a_lower_limit():
    """The exact regression the four-string form exists to prevent."""

    kept = report_extractor._clean_rows(
        [["Total WBC Count", "7900", "4000 to 10000"]]
    )

    assert kept[0]["low"] == 4000
    assert kept[0]["high"] == 10000


def test_the_object_form_is_still_accepted():
    """A model that answers with named fields must still work."""

    assert len(report_extractor._clean_rows([{
        "name": "Haemoglobin",
        "value": 15.8,
        "unit": "g/dL",
        "reference": "13.5 to 18.0",
    }])) == 1


def test_a_one_sided_limit_is_read():
    kept = report_extractor._clean_rows(
        [["Total Cholesterol", "207", "<200mg/dl"]]
    )

    assert kept[0]["low"] is None
    assert kept[0]["high"] == 200


def test_a_short_array_is_dropped():
    assert report_extractor._clean_rows(
        [["Haemoglobin", "15.8"]]
    ) == []


def test_a_row_with_an_unreadable_range_is_dropped():
    assert report_extractor._clean_rows(
        [["Haemoglobin", "15.8", ""]]
    ) == []


def test_a_reply_cut_off_at_the_token_limit_is_discarded(monkeypatch):
    """On a photo too degraded to read, the model shifted every
    value up a row and ran out of tokens. Truncation was the only
    thing that stopped those values being used."""

    install_reply(monkeypatch, GOOD, done_reason="length")

    assert report_extractor.read_document(b"image") == ("", [])

    # The same content, finished properly, is accepted.
    install_reply(monkeypatch, GOOD, done_reason="stop")

    assert report_extractor.read_document(b"image")[1]


def test_instructions_copied_into_the_answer_are_dropped(monkeypatch):
    """Asked for "every line of text you can read", the model wrote
    that sentence into the answer as though it were printed on the
    pack, and it reached the patient's context as a line of the
    label."""

    echoed = (
        '{"document_type": "medicine_label", "text": "BONSMILE\\n'
        'every line of text you can read, in the order it appears\\n'
        'boswellic acid", "values": []}'
    )

    install_reply(monkeypatch, echoed)

    text, _ = report_extractor.read_document(b"image")

    assert "every line of text" not in text.lower()
    assert "BONSMILE" in text
    assert "boswellic acid" in text


# ------------------------------------------------------------------
# The repetition loop
# ------------------------------------------------------------------

def test_a_read_that_runs_to_the_token_limit_is_abandoned(
    monkeypatch,
):
    """A penalised retry does break the loop, and returns fluent
    text - which is the problem. On a crumpled blister pack it
    invented a different ingredient list each time, once naming a
    prescription drug the pack does not contain."""

    calls = install_replies(monkeypatch, [(GOOD, "length")])

    assert report_extractor.read_document(b"image") == ("", [])
    assert len(calls) == 1


def test_a_read_that_finishes_is_used(monkeypatch):
    calls = install_replies(monkeypatch, [(GOOD, "stop")])

    text, rows = report_extractor.read_document(b"image")

    assert rows
    assert len(calls) == 1
    assert "repeat_penalty" not in calls[0]


# ------------------------------------------------------------------
# The photograph's own orientation
# ------------------------------------------------------------------

def photograph(orientation=None, size=(60, 40)):
    """A JPEG, optionally tagged as needing rotation."""

    import io as _io

    from PIL import Image

    image = Image.new("RGB", size, "white")

    buffer = _io.BytesIO()

    if orientation is None:
        image.save(buffer, "JPEG")
    else:
        exif = image.getexif()
        exif[274] = orientation
        image.save(buffer, "JPEG", exif=exif)

    return buffer.getvalue()


def size_of(data):
    import io as _io

    from PIL import Image

    return Image.open(_io.BytesIO(data)).size


def test_a_sideways_photograph_is_turned_upright():
    """A phone records which way up the picture was taken instead of
    rotating the pixels. Read raw, a label photographed the right
    way up arrives on its side, and the model loops on it until it
    runs out of tokens."""

    turned = report_extractor.upright(photograph(orientation=6))

    assert size_of(turned) == (40, 60)


def test_a_photograph_needing_no_rotation_is_untouched():
    original = photograph(orientation=1)

    assert report_extractor.upright(original) is original


def test_a_photograph_without_the_tag_is_untouched():
    original = photograph()

    assert report_extractor.upright(original) is original


def test_something_that_is_not_an_image_is_passed_through():
    assert report_extractor.upright(b"not an image") == b"not an image"

