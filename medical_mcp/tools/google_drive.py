"""Read-only Google Drive report retrieval exposed through MCP."""

from io import BytesIO
import re

from pypdf import PdfReader
import requests

from mcp_instance import mcp


GOOGLE_DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
GOOGLE_DOCS_MIME_TYPE = "application/vnd.google-apps.document"
SUPPORTED_MIME_TYPES = {
    "application/pdf",
    "text/plain",
    "text/csv",
    "application/rtf",
    "text/rtf",
    GOOGLE_DOCS_MIME_TYPE,
}
MAX_FILE_BYTES = 10 * 1024 * 1024

# Optical character recognition fallback for scanned PDFs that
# carry no text layer.
OCR_DPI = 200
OCR_MAX_PAGES = 15

# Extracted text is cached for the life of the MCP server process,
# keyed by file id + last-modified time, so a scanned report is
# only OCR'd once instead of on every question.
_FILE_TEXT_CACHE = {}


def _ocr_pdf(pdf_bytes, name, log):
    """Render a scanned PDF to images and read it with Tesseract."""

    try:
        import fitz  # PyMuPDF
        import pytesseract
        from PIL import Image
    except ImportError as error:
        log(
            f"OCR is not available ({error}). Install it with "
            "'pip install pymupdf pytesseract pillow'."
        )
        return ""

    try:
        import time

        zoom = OCR_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)

        text_parts = []

        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            page_count = min(document.page_count, OCR_MAX_PAGES)

            log(
                f"OCR '{name}': {page_count} page(s) at {OCR_DPI} DPI "
                "(first time only)"
            )

            for index in range(page_count):
                started = time.monotonic()

                page = document.load_page(index)
                pixmap = page.get_pixmap(
                    matrix=matrix,
                    colorspace=fitz.csGRAY,
                )
                image = Image.frombytes(
                    "L",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )
                text_parts.append(
                    pytesseract.image_to_string(image)
                )

                log(
                    f"OCR '{name}': page {index + 1}/{page_count} "
                    f"done in {time.monotonic() - started:.1f}s"
                )

        return "\n".join(text_parts).strip()

    except Exception as error:  # noqa: BLE001
        message = str(error)

        if "tesseract" in message.lower():
            log(
                "OCR needs the Tesseract engine. Install it with "
                "'brew install tesseract'."
            )
        else:
            log(f"OCR failed: {message}")

        return ""


def _get_file_text(file_metadata, headers, log):
    """Return report text, using a per-process cache."""

    cache_key = (
        file_metadata.get("id"),
        file_metadata.get("modifiedTime"),
    )

    if cache_key in _FILE_TEXT_CACHE:
        log(f"'{file_metadata.get('name', '')}': using cached text")
        return _FILE_TEXT_CACHE[cache_key]

    text = _download_file_text(file_metadata, headers, log)

    _FILE_TEXT_CACHE[cache_key] = text

    return text


def _download_file_text(file_metadata, headers, log):
    """Download/export supported report formats as text."""

    file_id = file_metadata["id"]
    name = file_metadata.get("name", "")
    mime_type = file_metadata.get("mimeType", "")

    if mime_type == GOOGLE_DOCS_MIME_TYPE:
        response = requests.get(
            f"{GOOGLE_DRIVE_FILES_URL}/{file_id}/export",
            headers=headers,
            params={
                "mimeType": "text/plain",
                "supportsAllDrives": "true",
            },
            timeout=20,
        )
        response.raise_for_status()

        return response.text

    response = requests.get(
        f"{GOOGLE_DRIVE_FILES_URL}/{file_id}",
        headers=headers,
        params={
            "alt": "media",
            "supportsAllDrives": "true",
        },
        timeout=20,
    )
    response.raise_for_status()

    if mime_type == "application/pdf":
        reader = PdfReader(BytesIO(response.content))

        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        ).strip()

        if text:
            return text

        log(f"'{name}' has no text layer; running OCR...")

        return _ocr_pdf(response.content, name, log)

    return response.content.decode("utf-8", errors="replace")


# Words that carry no topical meaning and would otherwise inflate
# the relevance score of every document.
_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "has", "have",
    "had", "you", "your", "our", "this", "that", "these", "those",
    "what", "which", "when", "where", "why", "how", "who", "does",
    "did", "can", "could", "would", "should", "will", "about",
    "with", "from", "into", "out", "any", "all", "not", "but",
    "get", "got", "there", "here", "than", "then", "them", "they",
    "his", "her", "its", "it's", "mine", "me", "my", "i'm", "i",
    "do", "is", "in", "on", "of", "to", "a", "an", "be", "or",
    "am", "as", "at", "by", "if", "so", "up", "we",
}

# A report only contributes context when at least this many of the
# question's meaningful words appear in it.
MIN_RELEVANCE_SCORE = 1


def _score_text(text, query):
    """Deterministic word-overlap relevance score for a document."""

    terms = re.findall(r"[a-zA-Z0-9]+", query.lower())
    searchable_text = text.lower()

    return sum(
        searchable_text.count(term)
        for term in terms
        if len(term) > 2 and term not in _STOPWORDS
    )


@mcp.tool
def search_google_drive_reports(
    query: str,
    folder_id: str,
    access_token: str,
):
    """Search report and prescription files from a selected Drive folder.

    The caller provides a short-lived, read-only OAuth token for the current
    user. The tool never writes to Drive and does not persist credentials.
    """

    debug = []

    def log(message):
        print(f"[DRIVE MCP] {message}")
        debug.append(message)

    if not folder_id or not access_token:
        return {
            "source": "Google Drive",
            "context": "",
            "references": [],
            "debug": ["Missing folder id or access token."],
            "error": "A Google Drive folder and connection are required.",
        }

    log(f"Listing folder {folder_id}")

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        response = requests.get(
            GOOGLE_DRIVE_FILES_URL,
            headers=headers,
            params={
                "q": f"'{folder_id}' in parents and trashed = false",
                "pageSize": 100,
                "orderBy": "modifiedTime desc",
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
                "fields": (
                    "files(id,name,mimeType,modifiedTime,size,webViewLink)"
                ),
            },
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        status = getattr(error.response, "status_code", "n/a")
        body = ""

        if error.response is not None:
            body = error.response.text[:400]

        log(f"Listing failed: HTTP {status} {body}")

        return {
            "source": "Google Drive",
            "context": "",
            "references": [],
            "debug": debug,
            "error": (
                f"Google Drive folder could not be listed (HTTP {status}). "
                "Check that the folder ID is correct and that the connected "
                "Google account can open it."
            ),
        }

    files = response.json().get("files", [])

    log(f"Drive returned {len(files)} item(s) in the folder")

    if not files:
        log(
            "The folder is empty from this account's view. Common causes: "
            "the ID points to a different folder, the report sits in a "
            "sub-folder, or the folder was shared with a different Google "
            "account than the one that was connected."
        )

    candidates = []

    for file_metadata in files:
        name = file_metadata.get("name", "unnamed")
        mime_type = file_metadata.get("mimeType", "")
        size = int(file_metadata.get("size", 0) or 0)

        if mime_type not in SUPPORTED_MIME_TYPES:
            log(f"Skip '{name}': unsupported type {mime_type}")
            continue

        if size > MAX_FILE_BYTES:
            log(f"Skip '{name}': {size} bytes is over the 10 MB limit")
            continue

        try:
            text = _get_file_text(file_metadata, headers, log).strip()
        except requests.RequestException as error:
            status = getattr(error.response, "status_code", "n/a")
            log(f"Skip '{name}': download failed (HTTP {status})")
            continue
        except (ValueError, OSError) as error:
            log(f"Skip '{name}': could not be read ({error})")
            continue

        if not text:
            log(
                f"Skip '{name}': 0 characters extracted (even after OCR "
                "if it is a scanned PDF)."
            )
            continue

        score = _score_text(text, query)

        log(
            f"Read '{name}': {len(text)} characters, relevance score {score}"
        )

        candidates.append(
            {
                "file": file_metadata,
                "text": text,
                "score": score,
            }
        )

    ranked_documents = [
        document
        for document in sorted(
            candidates,
            key=lambda document: document["score"],
            reverse=True,
        )
        if document["score"] >= MIN_RELEVANCE_SCORE
    ][:4]

    context_parts = []
    references = []

    for document in ranked_documents:
        file_metadata = document["file"]
        text = document["text"][:6000]

        context_parts.append(
            f"Report Name: {file_metadata['name']}\n{text}"
        )
        references.append(
            {
                "title": file_metadata["name"],
                "url": file_metadata.get("webViewLink", ""),
                "modified_time": file_metadata.get("modifiedTime", ""),
            }
        )

    if ranked_documents:
        log(
            f"Using {len(ranked_documents)} relevant report(s) as context "
            f"({len(candidates)} readable of {len(files)} listed)"
        )
    else:
        log(
            f"No report scored at least {MIN_RELEVANCE_SCORE} for this "
            f"question ({len(candidates)} readable of {len(files)} listed); "
            "the folder contributes nothing."
        )

    return {
        "source": "Google Drive reports folder",
        "context": "\n\n---\n\n".join(context_parts),
        "references": references,
        "debug": debug,
    }
