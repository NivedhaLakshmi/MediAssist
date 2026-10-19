"""Local storage for chat history, plus PDPA-style export and deletion.

Everything this module writes lives in ``user_data/`` inside the
project and never leaves the machine. The medical knowledge base
(``chroma_db/``, ``data/``) is system data and is deliberately kept
separate from the patient's own data.
"""

import io
import json
import os
import uuid
import zipfile
from datetime import datetime


from services.profiles import (
    DEFAULT_PROFILE_ID,
    profile_directory,
)


def conversations_file(profile_id=DEFAULT_PROFILE_ID):
    """Chat history lives inside the profile's own folder."""

    return os.path.join(
        profile_directory(profile_id),
        "conversations.json",
    )

MAX_CONVERSATIONS = 10

TITLE_MAX_LENGTH = 45


# ============================================================
# STORAGE
# ============================================================

def _ensure_storage(profile_id):
    os.makedirs(profile_directory(profile_id), exist_ok=True)


def load_conversations(profile_id=DEFAULT_PROFILE_ID):
    """Return this profile's conversations, newest first."""

    path = conversations_file(profile_id)

    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

    except (OSError, ValueError) as error:
        print("[HISTORY] Could not read conversations:", error)
        return []

    conversations = payload.get("conversations", [])

    conversations.sort(
        key=lambda item: item.get("updated_at", ""),
        reverse=True,
    )

    return conversations


def _write_conversations(conversations, profile_id):
    _ensure_storage(profile_id)

    payload = {
        "schema": 1,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "conversations": conversations[:MAX_CONVERSATIONS],
    }

    try:
        with open(
            conversations_file(profile_id), "w", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    except OSError as error:
        print("[HISTORY] Could not save conversations:", error)


def new_conversation_id():
    return str(uuid.uuid4())


def build_title(messages):
    """Use the first thing the patient asked as the title."""

    for message in messages:

        if message.get("role") == "user" and message.get("content"):

            title = " ".join(message["content"].split())

            if len(title) > TITLE_MAX_LENGTH:
                title = title[:TITLE_MAX_LENGTH].rstrip() + "..."

            return title

    return "New chat"


def save_conversation(
    conversation_id, messages, profile_id=DEFAULT_PROFILE_ID
):
    """Insert or update one conversation, keeping only the newest few."""

    if not conversation_id or not messages:
        return

    conversations = load_conversations(profile_id)

    now = datetime.now().isoformat(timespec="seconds")

    existing = None

    for conversation in conversations:
        if conversation.get("id") == conversation_id:
            existing = conversation
            break

    if existing is None:

        conversations.insert(
            0,
            {
                "id": conversation_id,
                "title": build_title(messages),
                "created_at": now,
                "updated_at": now,
                "messages": messages,
            },
        )

    else:

        existing["title"] = build_title(messages)
        existing["updated_at"] = now
        existing["messages"] = messages

    conversations.sort(
        key=lambda item: item.get("updated_at", ""),
        reverse=True,
    )

    _write_conversations(conversations, profile_id)


def delete_conversation(
    conversation_id, profile_id=DEFAULT_PROFILE_ID
):
    """Remove a single conversation. Returns True if one was removed.

    Deleting everything is a separate, deliberate action; this lets
    the patient clear one chat without losing the rest.
    """

    if not conversation_id:
        return False

    conversations = load_conversations(profile_id)

    remaining = [
        conversation
        for conversation in conversations
        if conversation.get("id") != conversation_id
    ]

    if len(remaining) == len(conversations):
        return False

    _write_conversations(remaining, profile_id)

    return True


def delete_all_conversations(profile_id=DEFAULT_PROFILE_ID):
    """Remove every conversation stored for this profile."""

    removed = len(load_conversations(profile_id))

    path = conversations_file(profile_id)

    if os.path.exists(path):

        try:
            os.remove(path)

        except OSError as error:
            print("[HISTORY] Could not delete conversations:", error)
            return 0

    return removed


# ============================================================
# EXPORT
# ============================================================

EXPORT_README = """MediAssist - export of your data
=================================

This archive contains only YOUR data. It does not contain the
application's medical knowledge base, and it does not contain any
Google credentials or access tokens.

Files
-----

chat_history.json
    Your saved conversations with MediAssist: the questions you
    asked and the answers you received.

patient_reports.json
    Text extracted from the medical reports you uploaded, together
    with the report name, type and upload date. This is the content
    that was stored as searchable embeddings.

account.json
    Non-sensitive status information, such as whether a Google Drive
    folder was connected. No access token or refresh token is ever
    included in an export.

images/
    The photographs you attached to your questions, under the names
    they are referred to by in chat_history.json.

Not included (system data, not yours)
-------------------------------------

    chroma_db/   the general medical knowledge base
    data/        source documents and the MeSH vocabulary
"""


def build_export_archive(
    conversations,
    patient_documents,
    drive_connected=False,
    drive_folder="",
    images=None,
):
    """Return a ZIP archive of the patient's own data as bytes.

    ``images`` maps an archive path to the bytes of a photograph the
    patient attached to a question.
    """

    generated_at = datetime.now().isoformat(timespec="seconds")

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:

        archive.writestr("README.txt", EXPORT_README)

        for name, data in (images or {}).items():
            archive.writestr(name, data)

        archive.writestr(
            "chat_history.json",
            json.dumps(
                {
                    "exported_at": generated_at,
                    "conversation_count": len(conversations),
                    "conversations": conversations,
                },
                indent=2,
                ensure_ascii=False,
            ),
        )

        archive.writestr(
            "patient_reports.json",
            json.dumps(
                {
                    "exported_at": generated_at,
                    "chunk_count": len(patient_documents),
                    "reports": patient_documents,
                },
                indent=2,
                ensure_ascii=False,
            ),
        )

        archive.writestr(
            "account.json",
            json.dumps(
                {
                    "exported_at": generated_at,
                    "google_drive_connected": bool(drive_connected),
                    "google_drive_folder": drive_folder or "",
                    "note": (
                        "Access and refresh tokens are never exported."
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
        )

    return buffer.getvalue()


def export_filename():
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"mediassist-my-data-{stamp}.zip"
