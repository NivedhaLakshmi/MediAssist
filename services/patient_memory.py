from datetime import datetime

import hashlib

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import uuid
from services.embedding_service import embedding
from services.lab_verifier import detect_patient_sex
from services.profiles import DEFAULT_PROFILE_ID
from utils.formatter import format_patient_report_context


# ---------------------------------
# Persistent Patient Database
# ---------------------------------

patient_vectordb = Chroma(
    persist_directory="patient_db",
    embedding_function=embedding
)


# ---------------------------------
# Text Splitter
# ---------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Chroma returns a distance: smaller means a closer match.
#
# Two gates are used, both measured rather than guessed. Distances
# were recorded for 15 questions about the patient's own results and
# 15 general medical questions, against three report shapes (a lab
# table, prose, and OCR text with character errors):
#
#   gate   personal questions   general questions
#   1.00                   4%                  0%   <- the old value
#   1.25                  40%                  0%
#   1.65                  93%                 31%
#
# At 1.00 almost every genuine match was discarded, so questions
# about the patient's own report were answered from PubMed instead.
#
# INCLUDE is deliberately generous: a missed report means the app
# cannot answer its core question at all, whereas an unnecessary
# report in the context is filtered again downstream, where verified
# findings are matched against the words of the question.
PATIENT_SCORE_THRESHOLD = 1.65

# Below this the report clearly answers the question, so the slower
# external sources can be skipped. Measured at 0% of general
# questions, so a general question never suppresses them.
PATIENT_STRONG_MATCH_THRESHOLD = 1.25


# ---------------------------------
# Duplicate detection
# ---------------------------------

def _content_hash(pdf_text):
    """Fingerprint a report by its text, not its filename.

    This catches the same report uploaded twice under different
    names, and lets a genuinely updated report through because its
    text - and therefore its hash - differs.
    """

    return hashlib.sha256(
        pdf_text.strip().encode("utf-8")
    ).hexdigest()


def _already_stored(content_hash, profile_id):
    """A duplicate within this profile only.

    The same report may legitimately exist under two profiles on a
    shared device, so the check is scoped rather than global.
    """

    try:
        existing = patient_vectordb.get(
            where={
                "$and": [
                    {"content_hash": {"$eq": content_hash}},
                    {"profile": {"$eq": profile_id}},
                ]
            },
            limit=1,
        )

    except Exception as error:  # noqa: BLE001
        print("[PATIENT] Duplicate check failed:", error)
        return False

    return bool(existing.get("ids"))


# ---------------------------------
# Add Patient Report
# ---------------------------------

def add_patient_report(
    pdf_text, report_name, report_type, profile_id=DEFAULT_PROFILE_ID
):
    """
    Store a patient's uploaded report permanently.

    Returns the new report id, or None if nothing was stored
    because the report is empty or already present.
    """
    if not pdf_text.strip():
        print("No text found in report.")
        return None

    # Session state only remembers filenames for the current
    # browser session, so without this the same report is stored
    # again after a refresh or after the Google OAuth redirect.
    content_hash = _content_hash(pdf_text)

    if _already_stored(content_hash, profile_id):
        print(
            f"[PATIENT] '{report_name}' is already stored "
            "- skipping duplicate."
        )
        return None

    report_id = str(uuid.uuid4())

    # Read from the whole report, because the "Age/Sex" header sits
    # in the first chunk only. Rows further down print one range for
    # men and another for women, and a chunk retrieved on its own
    # would have no way to tell which applies.
    patient_sex = detect_patient_sex(pdf_text)

    chunks = text_splitter.split_text(pdf_text)

    upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    documents = []

    for index, chunk in enumerate(chunks):

        documents.append(

            Document(

                page_content=chunk,

                metadata={
                    
                    "report_id": report_id,

                    "report_name": report_name,
                    
                    "report_type": report_type,

                    "upload_date": upload_date,

                    "source": "uploaded_pdf",

                    "content_hash": content_hash,

                    "profile": profile_id,

                    "patient_sex": patient_sex,

                    "chunk": index

                }

            )

        )

    patient_vectordb.add_documents(documents)

    print(f"Stored {len(documents)} chunks from {report_name}")
    
    return report_id

def _expand_to_whole_reports(docs, profile_id):
    """Return every chunk of the reports the search matched.

    Top-k retrieval returns fragments, which is right for a question
    about one value but wrong for "summarise my report": the chunks
    holding the other panels are never fetched, so the verifier never
    sees them. Here the matched reports are re-read in full.
    """

    report_ids = {
        doc.metadata.get("report_id")
        for doc in docs
        if doc.metadata.get("report_id")
    }

    if not report_ids:
        return docs

    expanded = []

    for report_id in report_ids:

        try:
            stored = patient_vectordb.get(
                where={
                    "$and": [
                        {"report_id": {"$eq": report_id}},
                        {"profile": {"$eq": profile_id}},
                    ]
                },
                include=["documents", "metadatas"],
            )

        except Exception as error:  # noqa: BLE001
            print("[PATIENT] Could not expand report:", error)
            continue

        for text, metadata in zip(
            stored.get("documents") or [],
            stored.get("metadatas") or [],
        ):
            expanded.append(
                Document(page_content=text, metadata=metadata or {})
            )

    return expanded or docs


def search_patient_reports(
    query,
    k=6,
    profile_id=DEFAULT_PROFILE_ID,
    whole_report=False,
):
    """Search reports previously uploaded under this profile.

    Returns ``(docs, context, source, best_score)``. ``best_score`` is
    the closest distance seen, or None when the profile holds no
    reports, and lets the caller tell a strong match from a weak one
    without searching twice.

    With ``whole_report`` the matched reports are returned complete
    rather than as the retrieved fragments, for questions that ask
    about the report as a whole.
    """

    results = patient_vectordb.similarity_search_with_score(
        query,
        k=k,
        filter={"profile": profile_id},
    )

    if not results:
        return [], "", "Patient Reports", None

    best_score = min(score for _, score in results)

    docs = []

    for doc, score in results:

        keep = score < PATIENT_SCORE_THRESHOLD

        print(
            f"Retrieved: {doc.metadata.get('report_name', 'Unknown')} "
            f"| Chunk: {doc.metadata.get('chunk', 'N/A')} "
            f"| Score: {score:.2f} "
            f"| {'used' if keep else 'too far - skipped'}"
        )

        if keep:
            docs.append(doc)

    # Without this the uploaded reports match every question, and
    # the external medical sources (MedlinePlus, PubMed, DailyMed,
    # ClinicalTrials) are never reached for general questions.
    # A question about "my report" names the patient's own documents,
    # so there is nothing for a distance gate to decide. It is also
    # the question the gate handles worst: "give me a summary of my
    # report" carries none of the words a lab table contains, and
    # measured 1.69 against a report it was plainly asking about.
    if whole_report and not docs and results:

        docs = [results[0][0]]

        print(
            "[PATIENT] Summary question - using the report itself "
            f"rather than the distance gate (best {best_score:.2f})."
        )

    if not docs:

        print(
            f"No patient report chunk was close enough "
            f"(best {best_score:.2f}, need under "
            f"{PATIENT_SCORE_THRESHOLD})."
        )

        return [], "", "Patient Reports", best_score

    if whole_report:

        before = len(docs)

        docs = _expand_to_whole_reports(docs, profile_id)

        print(
            f"[PATIENT] Expanded {before} matched chunk(s) to "
            f"{len(docs)} chunk(s) covering the whole report."
        )

    docs.sort(
        key=lambda doc: (
            doc.metadata.get("report_name", ""),
            doc.metadata.get("chunk", 0),
        )
    )

    return (
        docs,
        format_patient_report_context(docs),
        "Patient Reports",
        best_score,
    )


def is_strong_match(best_score):
    """Whether the report answers the question well enough on its own."""

    return (
        best_score is not None
        and best_score < PATIENT_STRONG_MATCH_THRESHOLD
    )


# ---------------------------------
# Export / Delete (PDPA support)
# ---------------------------------

def get_all_patient_documents(profile_id=DEFAULT_PROFILE_ID):
    """Return this profile's stored chunks with their metadata."""

    try:
        stored = patient_vectordb.get(
            where={"profile": profile_id},
            include=["documents", "metadatas"],
        )

    except Exception as error:
        print("[PATIENT] Could not read stored reports:", error)
        return []

    documents = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []

    records = []

    for index, text in enumerate(documents):

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        ) or {}

        records.append(
            {
                "report_name": metadata.get(
                    "report_name",
                    "Unknown Report",
                ),
                "report_type": metadata.get(
                    "report_type",
                    "Medical Report",
                ),
                "upload_date": metadata.get(
                    "upload_date",
                    "",
                ),
                "chunk": metadata.get("chunk", 0),
                "text": text,
            }
        )

    records.sort(
        key=lambda record: (
            record["report_name"],
            record["chunk"],
        )
    )

    return records


def count_patient_reports(profile_id=DEFAULT_PROFILE_ID):
    """Return (report_count, chunk_count) for this profile."""

    records = get_all_patient_documents(profile_id)

    names = {record["report_name"] for record in records}

    return len(names), len(records)


def delete_all_patient_reports(profile_id=DEFAULT_PROFILE_ID):
    """Remove this profile's report chunks, leaving other profiles.

    Deletes by id rather than dropping the collection, which would
    take every profile's data with it.
    """

    try:
        stored = patient_vectordb.get(where={"profile": profile_id})
        ids = stored.get("ids") or []

        if ids:
            patient_vectordb.delete(ids=ids)

    except Exception as error:  # noqa: BLE001
        print("[PATIENT] Could not delete reports:", error)
        return 0

    print(
        f"[PATIENT] Deleted {len(ids)} stored chunk(s) "
        f"for profile '{profile_id}'."
    )

    return len(ids)