"""Named profiles for people sharing one device.

MediAssist runs locally, so there is no account server and no
password. A profile simply scopes stored data: each one has its own
chat history, its own uploaded reports and its own Google Drive
connection. Switching profile shows an empty assistant.

This is data separation on a shared machine, not authentication -
anyone at the keyboard can switch profile.
"""

import json
import os
import re
from datetime import datetime


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

USER_DATA_DIR = os.path.join(PROJECT_ROOT, "user_data")

PROFILES_FILE = os.path.join(USER_DATA_DIR, "profiles.json")

DEFAULT_PROFILE_ID = "default"

DEFAULT_PROFILE_NAME = "My profile"

NAME_MAX_LENGTH = 30


def _ensure_storage():
    os.makedirs(USER_DATA_DIR, exist_ok=True)


def slugify(name):
    """Turn a display name into something safe to use as a folder."""

    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")

    return slug or "profile"


def profile_directory(profile_id):
    """Where this profile's files live."""

    return os.path.join(USER_DATA_DIR, "profiles", profile_id)


def _read():
    if not os.path.exists(PROFILES_FILE):
        return []

    try:
        with open(PROFILES_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle).get("profiles", [])

    except (OSError, ValueError) as error:
        print("[PROFILE] Could not read profiles:", error)
        return []


def _write(profiles):
    _ensure_storage()

    try:
        with open(PROFILES_FILE, "w", encoding="utf-8") as handle:
            json.dump({"profiles": profiles}, handle, indent=2)

    except OSError as error:
        print("[PROFILE] Could not save profiles:", error)


def list_profiles():
    """Every profile, with the default always present and first."""

    profiles = _read()

    if not any(p["id"] == DEFAULT_PROFILE_ID for p in profiles):
        profiles.insert(
            0,
            {
                "id": DEFAULT_PROFILE_ID,
                "name": DEFAULT_PROFILE_NAME,
                "created_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
            },
        )

    return profiles


def profile_name(profile_id):
    for profile in list_profiles():
        if profile["id"] == profile_id:
            return profile["name"]

    return profile_id


def create_profile(name):
    """Add a profile. Returns its id, or None if the name is unusable.

    A name that collides with an existing profile returns that
    profile's id rather than creating a duplicate.
    """

    name = (name or "").strip()[:NAME_MAX_LENGTH]

    if not name:
        return None

    profile_id = slugify(name)

    profiles = list_profiles()

    for profile in profiles:
        if profile["id"] == profile_id:
            return profile_id

    profiles.append(
        {
            "id": profile_id,
            "name": name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    _write(profiles)

    return profile_id


def rename_profile(profile_id, new_name):
    """Change a profile's display name, keeping its id and its data.

    The id is what scopes stored files, so it deliberately does not
    change: renaming must not orphan a profile's chats or reports.
    """

    new_name = (new_name or "").strip()[:NAME_MAX_LENGTH]

    if not new_name:
        return False

    profiles = list_profiles()

    for profile in profiles:
        if profile["id"] == profile_id:
            profile["name"] = new_name
            _write(profiles)
            return True

    return False


def delete_profile(profile_id):
    """Remove a profile from the list. The default cannot be removed.

    Stored data is deleted separately, so the caller can clear the
    chat history and reports first.
    """

    if profile_id == DEFAULT_PROFILE_ID:
        return False

    profiles = [p for p in list_profiles() if p["id"] != profile_id]

    _write(profiles)

    return True
