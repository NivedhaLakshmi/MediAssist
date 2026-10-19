"""Read the text out of an uploaded PDF.

Three readers, tried in order of cost:

  1. The text layer, via pdfplumber. Instant, and right for a PDF
     produced by a computer.
  2. MedGemma reading a rendered page, for a scan that has no text
     layer at all, or one whose text layer came out damaged.
  3. Tesseract on the same rendered page, when the vision model is
     not installed.

Only the first ran before, so a scanned report - a photograph of
paper wrapped in a PDF, which is what most Indian labs email - gave
no text, was refused by the store, and the patient saw an upload
that appeared to work and answered nothing.
"""

import io

import pdfplumber

from services.lab_verifier import has_damaged_numbers, verify_lab_ranges
from services.report_extractor import (
    is_available as vision_is_available,
    read_document,
    rows_as_text,
)


# Below this a "text layer" is a page number and a footer, not a
# report.
MIN_TEXT_LAYER = 200

# Rendering every page of a long report would take minutes. A lab
# report that matters is a handful of pages.
MAX_VISION_PAGES = 4

# Render resolution was swept: on a photographed scan 4x read best
# (11/12 against 8/12 at 2x), but on a cleanly rendered page 4x
# collapsed to 0/12 where 2x read all twelve. The effect is not a
# resolution curve, it is variance on inputs the model finds hard,
# so the setting stays where the path that matters most - a scanned
# report that previously produced nothing at all - reads reliably.
RENDER_ZOOM = 2.0


def extract_pdf_text(uploaded_file):
    """The PDF's own text layer, or "" when it has none."""

    text = ""

    try:
        uploaded_file.seek(0)

        with pdfplumber.open(uploaded_file) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n\n"

    except Exception as error:  # noqa: BLE001
        print("[PDF] Could not read the text layer:", error)

    finally:
        try:
            uploaded_file.seek(0)
        except Exception:  # noqa: BLE001
            pass

    return text


def render_pages(uploaded_file, max_pages=MAX_VISION_PAGES):
    """Yield each page as PNG bytes."""

    try:
        import fitz  # PyMuPDF

    except ImportError:
        print(
            "[PDF] Page rendering needs PyMuPDF "
            "('pip install pymupdf')."
        )
        return

    try:
        uploaded_file.seek(0)
        data = uploaded_file.read()
        uploaded_file.seek(0)

        with fitz.open(stream=data, filetype="pdf") as document:

            for page in document[:max_pages]:

                yield page.get_pixmap(
                    matrix=fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM)
                ).tobytes("png")

    except Exception as error:  # noqa: BLE001
        print("[PDF] Could not render the pages:", error)


def _tesseract_page(image_bytes):
    try:
        import pytesseract
        from PIL import Image

    except ImportError:
        return ""

    try:
        return pytesseract.image_to_string(
            Image.open(io.BytesIO(image_bytes)).convert("L")
        ).strip()

    except Exception as error:  # noqa: BLE001
        print("[PDF] OCR failed on a page:", error)
        return ""


def _read_pages(uploaded_file, reader, on_page=None):
    """Run one reader over the rendered pages and join the result."""

    parts = []

    for number, image in enumerate(render_pages(uploaded_file), 1):

        if on_page:
            on_page(number)

        text = reader(image)

        if text:
            parts.append(text)

    return "\n\n".join(parts)


def _usable_rows(text):
    """How many results a reader's text actually yields."""

    return len(verify_lab_ranges(text)) if text else 0


def read_pdf(uploaded_file, on_page=None):
    """Return ``(text, how_it_was_read)`` for an uploaded PDF.

    The readers are compared on how many results can actually be
    read out of them, not on how much text they produced: a page of
    OCR noise is longer than a clean table and worth far less.
    """

    layer = extract_pdf_text(uploaded_file)

    layer_rows = _usable_rows(layer)

    # A text layer that is long enough, yields results, and shows no
    # sign of damage is the best reader there is - and free.
    if (
        len(layer.strip()) >= MIN_TEXT_LAYER
        and layer_rows
        and not has_damaged_numbers(layer)
    ):
        print(f"[PDF] Text layer: {len(layer)} chars, "
              f"{layer_rows} result(s).")

        return layer, "text layer"

    if layer_rows and has_damaged_numbers(layer):
        # The page could be read as an image, but the result is only
        # used where the layer gave nothing - so there is no point
        # spending thirty seconds on it here.
        print(
            f"[PDF] Text layer: {layer_rows} result(s); one or more "
            "rows have numbers split across columns and are skipped."
        )

        return layer, "text layer"

    if vision_is_available():

        def read_page(image):
            text, rows = read_document(image)

            # The measurements live in the rows, not in the text.
            # This path stores text and parses it again later, so
            # they are written back into it.
            table = rows_as_text(rows)

            return f"{text}\n\n{table}".strip() if table else text

        vision = _read_pages(uploaded_file, read_page, on_page)

        vision_rows = _usable_rows(vision)

        print(f"[PDF] MedGemma: {len(vision)} chars, "
              f"{vision_rows} result(s).")

        # Only where the text layer gave nothing usable. Letting the
        # page reading top up a layer that mostly worked was tried
        # and rejected: it recovered the split platelet count as
        # 2,890,000 instead of 2,89,000 and reported a normal count
        # as above range. A missing result is recoverable by asking
        # a doctor; a wrong one is not.
        if vision.strip() and not layer_rows:
            return vision, "MedGemma"

    ocr = _read_pages(uploaded_file, _tesseract_page, on_page)

    ocr_rows = _usable_rows(ocr)

    print(f"[PDF] Tesseract: {len(ocr)} chars, {ocr_rows} result(s).")

    if ocr.strip() and not layer_rows:
        return ocr, "OCR"

    return layer, "text layer"
