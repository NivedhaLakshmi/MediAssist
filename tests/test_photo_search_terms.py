"""What a photographed document may add to an outside search.

Words read from a photo are added to the search sent to the public
medical sources, so that "what is this for" can be searched by the
medicine's name. A laboratory report must add nothing: its header
carries the patient's name, identifiers and the laboratory, and the
first words of a page are the ones that were being sent.
"""

from services.image_service import is_lab_report, photo_search_terms

REPORT = (
    "CITYCARE DIAGNOSTIC LABORATORY\n"
    "Patient Name : Mr. Daniel Tan    UID : 0400103875\n"
    "Haemoglobin 14.6 g/dL 13.0 - 17.0\n"
    "Total WBC Count 11800 /uL 4000 - 10000\n"
)

LABEL = "DOLO 650 Paracetamol Tablets IP 650 mg Micro Labs Limited"

ROWS = [["Haemoglobin", "14.6", "g/dL", "13.0 - 17.0"]]


def test_a_report_read_by_the_vision_model_adds_no_terms():
    assert photo_search_terms([(REPORT, ROWS)]) == ""


def test_a_report_read_by_ocr_alone_adds_no_terms():
    """Tesseract returns no rows, so the text itself must be judged."""

    assert photo_search_terms([(REPORT, [])]) == ""


def test_no_identifier_or_name_from_a_report_is_sent():
    terms = photo_search_terms([(REPORT, [])])

    for private in ("0400103875", "tan", "daniel", "citycare"):
        assert private not in terms


def test_a_medicine_label_still_contributes_its_name():
    terms = photo_search_terms([(LABEL, [])])

    assert "dolo" in terms or "paracetamol" in terms


def test_a_label_beside_a_report_keeps_only_the_label_terms():
    terms = photo_search_terms([(REPORT, ROWS), (LABEL, [])])

    assert "dolo" in terms or "paracetamol" in terms
    assert "citycare" not in terms
    assert "0400103875" not in terms


def test_a_photo_with_no_text_adds_nothing():
    assert photo_search_terms([("", [])]) == ""
    assert photo_search_terms([]) == ""


def test_lab_report_detection():
    assert is_lab_report(REPORT, [])
    assert is_lab_report("", ROWS)
    assert not is_lab_report(LABEL, [])
