"""Images the patient attached to a question.

The photo is kept so the chat can show what was asked about, rather
than only the text read out of it. It lives in the profile's own
folder, never in the conversation file: a JPEG encoded into JSON
would be megabytes per message, and the ten stored conversations
would carry it all.

Because these are photographs of the patient's own documents, they
are patient data like any other: covered by "Delete my data",
included in an export, and removed with the profile that owns them.
"""

import json
import os
import re
import uuid

from services.profiles import DEFAULT_PROFILE_ID, profile_directory


# Only what a phone camera or a file picker actually produces.
ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic",
}

DEFAULT_EXTENSION = ".jpg"

MAX_BYTES = 20 * 1024 * 1024


def attachments_directory(profile_id=DEFAULT_PROFILE_ID):
    return os.path.join(profile_directory(profile_id), "attachments")


def _extension_of(filename):
    extension = os.path.splitext(filename or "")[1].lower()

    return (
        extension if extension in ALLOWED_EXTENSIONS
        else DEFAULT_EXTENSION
    )


def save_attachment(data, filename, profile_id=DEFAULT_PROFILE_ID):
    """Store one image and return the name to record in the message.

    The stored name is generated, never taken from the upload: a
    filename arriving from the browser is untrusted input, and
    "../../" in it would write outside the profile's folder.
    """

    if not data or len(data) > MAX_BYTES:
        return None

    directory = attachments_directory(profile_id)

    try:
        os.makedirs(directory, exist_ok=True)

        stored = f"{uuid.uuid4().hex}{_extension_of(filename)}"

        with open(os.path.join(directory, stored), "wb") as handle:
            handle.write(data)

    except OSError as error:
        print("[ATTACHMENT] Could not save the image:", error)
        return None

    return stored


_SAFE_NAME = re.compile(r"^[0-9a-f]{32}\.[a-z]{3,4}$")

# What the vision model read out of one image, kept beside it.
EXTRACTION_SUFFIX = ".read.json"


def attachment_path(stored_name, profile_id=DEFAULT_PROFILE_ID):
    """Absolute path of a stored image, or None if it is not there.

    The name is checked against the shape this module generates, so
    a conversation file edited by hand cannot point at an arbitrary
    file on the machine.
    """

    if not stored_name or not _SAFE_NAME.match(str(stored_name)):
        return None

    path = os.path.join(
        attachments_directory(profile_id), stored_name
    )

    return path if os.path.exists(path) else None


def _stored_names(conversations):
    """Every image name still referenced by a stored conversation."""

    names = set()

    for conversation in conversations or []:
        for message in conversation.get("messages", []):
            for name in message.get("images") or []:
                names.add(name)

    return names


def prune_unreferenced(conversations, profile_id=DEFAULT_PROFILE_ID):
    """Delete images no remaining conversation refers to.

    Only ten conversations are kept and a chat can be deleted on its
    own, so without this the images of forgotten chats would stay on
    disk for good.
    """

    directory = attachments_directory(profile_id)

    if not os.path.isdir(directory):
        return 0

    keep = _stored_names(conversations)

    removed = 0

    for name in os.listdir(directory):

        # The sidecar belongs to its image and lives or dies with it.
        owner = (
            name[: -len(EXTRACTION_SUFFIX)]
            if name.endswith(EXTRACTION_SUFFIX)
            else name
        )

        if owner in keep:
            continue

        try:
            os.remove(os.path.join(directory, name))
            removed += 1

        except OSError as error:
            print("[ATTACHMENT] Could not remove", name, error)

    if removed:
        print(f"[ATTACHMENT] Removed {removed} unreferenced image(s).")

    return removed


def delete_all_attachments(profile_id=DEFAULT_PROFILE_ID):
    """Remove every stored image for this profile."""

    directory = attachments_directory(profile_id)

    if not os.path.isdir(directory):
        return 0

    removed = 0

    for name in os.listdir(directory):

        try:
            os.remove(os.path.join(directory, name))
            removed += 1

        except OSError as error:
            print("[ATTACHMENT] Could not remove", name, error)

    try:
        os.rmdir(directory)

    except OSError:
        # Something else is in there; the images are gone regardless.
        pass

    return removed


def collect_for_export(conversations, profile_id=DEFAULT_PROFILE_ID):
    """Return {archive name: bytes} for the images still referenced."""

    files = {}

    for name in sorted(_stored_names(conversations)):

        path = attachment_path(name, profile_id)

        if not path:
            continue

        try:
            with open(path, "rb") as handle:
                files[f"images/{name}"] = handle.read()

        except OSError as error:
            print("[ATTACHMENT] Could not read", name, error)

    return files


# ------------------------------------------------------------------
# What was read out of an image
# ------------------------------------------------------------------

def _extraction_path(stored_name, profile_id):
    if not stored_name or not _SAFE_NAME.match(str(stored_name)):
        return None

    return os.path.join(
        attachments_directory(profile_id),
        stored_name + EXTRACTION_SUFFIX,
    )


def save_extraction(
    stored_name, text, rows, profile_id=DEFAULT_PROFILE_ID
):
    """Remember what was read from an image.

    Reading one costs about 25 seconds. Editing a question must not
    pay that again for a photograph that has not changed, so the
    result is kept beside the file it came from.
    """

    path = _extraction_path(stored_name, profile_id)

    if not path:
        return False

    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"text": text, "rows": rows}, handle)

    except (OSError, TypeError, ValueError) as error:
        print("[ATTACHMENT] Could not record what was read:", error)
        return False

    return True


def load_extraction(stored_name, profile_id=DEFAULT_PROFILE_ID):
    """Return ``(text, rows)`` read from an image, or ("", [])."""

    path = _extraction_path(stored_name, profile_id)

    if not path or not os.path.exists(path):
        return "", []

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

    except (OSError, ValueError) as error:
        print("[ATTACHMENT] Could not read the stored text:", error)
        return "", []

    return (
        str(payload.get("text") or ""),
        payload.get("rows") or [],
    )
