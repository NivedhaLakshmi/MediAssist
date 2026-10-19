"""Read a photo the patient took: an ingredients list, a medicine
box, a printed result slip.

Two readers, tried in order:

  * A vision model that understands the document (see
    :mod:`services.report_extractor`). It returns both the text and,
    for a lab slip, the measurements as structured rows.
  * Tesseract OCR, which transcribes characters and nothing more.

Tesseract remains the fallback, so an installation without the
vision model behaves exactly as before.
"""

import io

from services.report_extractor import read_document, upright

MIN_USEFUL_CHARACTERS = 12


def _tesseract_text(uploaded_file):
    """Transcribe an image with Tesseract, or "" if it cannot."""

    try:
        import pytesseract
        from PIL import Image

    except ImportError as error:
        print(
            "[IMAGE] OCR unavailable:",
            error,
            "- install with 'pip install pytesseract pillow'",
        )
        return ""

    try:
        uploaded_file.seek(0)

        # The same orientation tag matters here: Tesseract read a
        # sideways label as upside-down nonsense.
        image = Image.open(io.BytesIO(upright(uploaded_file.read())))

        # Grayscale helps Tesseract on photos taken at an angle
        # under uneven lighting.
        return pytesseract.image_to_string(
            image.convert("L")
        ).strip()

    except Exception as error:  # noqa: BLE001
        message = str(error)

        if "tesseract" in message.lower():
            print(
                "[IMAGE] OCR needs the Tesseract engine "
                "('brew install tesseract')."
            )
        else:
            print("[IMAGE] Could not read the image:", message)

        return ""

    finally:
        try:
            uploaded_file.seek(0)
        except Exception:  # noqa: BLE001
            pass


def read_image(uploaded_file):
    """Return ``(text, rows)`` read from an uploaded photo.

    ``rows`` are measurements the vision model could read, awaiting
    comparison by :func:`services.lab_verifier.verify_extracted_rows`.
    Tesseract produces text only, so it always returns no rows.
    """

    try:
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()

    except Exception as error:  # noqa: BLE001
        print("[IMAGE] Could not read the upload:", error)
        return "", []

    finally:
        try:
            uploaded_file.seek(0)
        except Exception:  # noqa: BLE001
            pass

    text, rows = read_document(image_bytes)

    if len(text) >= MIN_USEFUL_CHARACTERS:
        return text, rows

    # Either the vision model is not installed, or it read nothing
    # useful from this photo.
    fallback = _tesseract_text(uploaded_file)

    if len(fallback) < MIN_USEFUL_CHARACTERS:
        return "", []

    print(f"[IMAGE] Tesseract read {len(fallback)} characters.")

    return fallback, []


def extract_image_text(uploaded_file):
    """Text only, for callers that do not use the structured rows."""

    return read_image(uploaded_file)[0]


def is_lab_report(text, rows):
    """Whether a photographed document is a laboratory report.

    Decided by whether measurements can be read from it, not by the
    vision model's own label: the model has labelled a chest
    radiograph a lab report, and gives no label at all when Tesseract
    did the reading.
    """

    if rows:
        return True

    from services.lab_verifier import verify_lab_ranges

    return bool(text) and bool(verify_lab_ranges(text))


def photo_search_terms(readings):
    """Search terms to send to the public sources from attached photos.

    ``readings`` is one ``(text, rows)`` pair per photo.

    Terms are taken from a photo so that "what is this for" can be
    searched by the medicine's name, which only the photo contains.
    A laboratory report contributes nothing: its header carries the
    patient's name, identifiers and the laboratory, and a question
    about a report names its own test. Each photo is judged on its
    own, so a medicine label attached beside a report still counts.
    """

    from services.query_utils import build_search_keywords, normalize_text

    texts = [
        text
        for text, rows in readings
        if text and not is_lab_report(text, rows)
    ]

    if not texts:
        return ""

    return build_search_keywords(normalize_text(" ".join(texts)))
