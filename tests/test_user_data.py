"""Tests for local chat history, export and deletion.

Every test redirects storage into a temporary directory, so the real
user_data/ folder is never read or written.
"""

import io
import json
import zipfile

import pytest

from services import profiles, user_data


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Point profile storage at a throwaway directory.

    profile_directory() reads the module global at call time, so
    patching it redirects every profile's files.
    """

    monkeypatch.setattr(profiles, "USER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        profiles,
        "PROFILES_FILE",
        str(tmp_path / "profiles.json"),
    )


def exchange(question, answer="An answer."):
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


# ------------------------------------------------------------------
# Saving and loading
# ------------------------------------------------------------------

def test_no_conversations_before_anything_is_saved():
    assert user_data.load_conversations() == []


def test_a_saved_conversation_can_be_read_back():
    cid = user_data.new_conversation_id()
    user_data.save_conversation(cid, exchange("What is my urea?"))

    stored = user_data.load_conversations()

    assert len(stored) == 1
    assert stored[0]["id"] == cid
    assert stored[0]["messages"][0]["content"] == "What is my urea?"


def test_saving_the_same_id_updates_rather_than_duplicates():
    cid = user_data.new_conversation_id()

    user_data.save_conversation(cid, exchange("First"))
    user_data.save_conversation(cid, exchange("First", "Longer answer."))

    stored = user_data.load_conversations()

    assert len(stored) == 1
    assert stored[0]["messages"][1]["content"] == "Longer answer."


def test_title_comes_from_the_first_question():
    cid = user_data.new_conversation_id()
    user_data.save_conversation(cid, exchange("Is my haemoglobin normal?"))

    assert user_data.load_conversations()[0]["title"] == (
        "Is my haemoglobin normal?"
    )


def test_a_long_title_is_truncated():
    cid = user_data.new_conversation_id()
    user_data.save_conversation(cid, exchange("word " * 40))

    title = user_data.load_conversations()[0]["title"]

    assert len(title) <= user_data.TITLE_MAX_LENGTH + 3
    assert title.endswith("...")


def test_only_the_most_recent_conversations_are_kept():
    for index in range(user_data.MAX_CONVERSATIONS + 5):
        user_data.save_conversation(
            user_data.new_conversation_id(),
            exchange(f"Question {index}"),
        )

    assert len(user_data.load_conversations()) == (
        user_data.MAX_CONVERSATIONS
    )


def test_an_empty_conversation_is_not_saved():
    user_data.save_conversation(user_data.new_conversation_id(), [])

    assert user_data.load_conversations() == []


def test_unreadable_history_does_not_raise():
    import os

    path = user_data.conversations_file(profiles.DEFAULT_PROFILE_ID)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{ not json")

    assert user_data.load_conversations() == []


# ------------------------------------------------------------------
# Separation between profiles
# ------------------------------------------------------------------

def test_each_profile_has_its_own_history():
    user_data.save_conversation(
        user_data.new_conversation_id(), exchange("Alice question"),
        profile_id="alice",
    )
    user_data.save_conversation(
        user_data.new_conversation_id(), exchange("Bob question"),
        profile_id="bob",
    )

    alice = user_data.load_conversations("alice")
    bob = user_data.load_conversations("bob")

    assert len(alice) == 1 and len(bob) == 1
    assert alice[0]["messages"][0]["content"] == "Alice question"
    assert bob[0]["messages"][0]["content"] == "Bob question"


def test_a_new_profile_starts_with_no_history():
    user_data.save_conversation(
        user_data.new_conversation_id(), exchange("Alice question"),
        profile_id="alice",
    )

    assert user_data.load_conversations("carol") == []


def test_deleting_one_profile_leaves_the_others():
    for name in ("alice", "bob"):
        user_data.save_conversation(
            user_data.new_conversation_id(), exchange(f"{name} question"),
            profile_id=name,
        )

    user_data.delete_all_conversations("alice")

    assert user_data.load_conversations("alice") == []
    assert len(user_data.load_conversations("bob")) == 1


# ------------------------------------------------------------------
# Deletion
# ------------------------------------------------------------------

def test_delete_removes_everything_and_reports_the_count():
    for index in range(3):
        user_data.save_conversation(
            user_data.new_conversation_id(),
            exchange(f"Question {index}"),
        )

    assert user_data.delete_all_conversations() == 3
    assert user_data.load_conversations() == []


def test_delete_on_empty_storage_is_safe():
    assert user_data.delete_all_conversations() == 0


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------

def archive_of(**kwargs):
    payload = {
        "conversations": [],
        "patient_documents": [],
    }
    payload.update(kwargs)

    return zipfile.ZipFile(
        io.BytesIO(user_data.build_export_archive(**payload))
    )


def test_export_contains_the_expected_files():
    archive = archive_of()

    assert set(archive.namelist()) == {
        "README.txt",
        "chat_history.json",
        "patient_reports.json",
        "account.json",
    }


def test_export_includes_the_conversations_and_reports():
    archive = archive_of(
        conversations=[{"id": "1", "title": "T", "messages": []}],
        patient_documents=[{"report_name": "r.pdf", "text": "WBC 7900"}],
    )

    chats = json.loads(archive.read("chat_history.json"))
    reports = json.loads(archive.read("patient_reports.json"))

    assert chats["conversation_count"] == 1
    assert reports["reports"][0]["text"] == "WBC 7900"


def test_export_never_contains_an_access_token():
    """A token in an export would defeat the point of the feature."""

    archive = archive_of(drive_connected=True, drive_folder="FOLDER123")

    everything = b"".join(
        archive.read(name) for name in archive.namelist()
    ).lower()

    assert b"access_token" not in everything
    assert b"refresh_token" not in everything


def test_export_records_drive_status_without_credentials():
    account = json.loads(
        archive_of(
            drive_connected=True, drive_folder="FOLDER123"
        ).read("account.json")
    )

    assert account["google_drive_connected"] is True
    assert account["google_drive_folder"] == "FOLDER123"


# ------------------------------------------------------------------
# Deleting one conversation
# ------------------------------------------------------------------

def test_deleting_one_chat_leaves_the_others():
    user_data.save_conversation("a", exchange("First question"))
    user_data.save_conversation("b", exchange("Second question"))
    user_data.save_conversation("c", exchange("Third question"))

    assert user_data.delete_conversation("b") is True

    remaining = [c["id"] for c in user_data.load_conversations()]

    assert remaining == ["c", "a"]


def test_deleting_an_unknown_chat_changes_nothing():
    user_data.save_conversation("a", exchange("First question"))

    assert user_data.delete_conversation("nope") is False
    assert len(user_data.load_conversations()) == 1


def test_deleting_without_an_id_is_refused():
    user_data.save_conversation("a", exchange("First question"))

    assert user_data.delete_conversation(None) is False
    assert user_data.delete_conversation("") is False
    assert len(user_data.load_conversations()) == 1


def test_deleting_the_last_chat_empties_the_history():
    user_data.save_conversation("a", exchange("Only question"))

    assert user_data.delete_conversation("a") is True
    assert user_data.load_conversations() == []


def test_deleting_a_chat_touches_only_its_own_profile():
    """Two people share the device; one clearing a chat must not
    remove the other's."""

    user_data.save_conversation(
        "shared-id", exchange("Alice question"), profile_id="alice"
    )
    user_data.save_conversation(
        "shared-id", exchange("Bob question"), profile_id="bob"
    )

    user_data.delete_conversation("shared-id", profile_id="alice")

    assert user_data.load_conversations("alice") == []
    assert len(user_data.load_conversations("bob")) == 1


def test_a_deleted_chat_stays_deleted_when_another_is_saved():
    user_data.save_conversation("a", exchange("First question"))
    user_data.save_conversation("b", exchange("Second question"))

    user_data.delete_conversation("a")
    user_data.save_conversation("c", exchange("Third question"))

    assert [c["id"] for c in user_data.load_conversations()] == [
        "c",
        "b",
    ]


# ------------------------------------------------------------------
# Attached photographs in an export
# ------------------------------------------------------------------

def read_archive(data):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return set(archive.namelist())


def test_an_export_carries_the_attached_photographs():
    """They are pictures of the patient's own documents, so a
    request for "all my data" has to include them."""

    data = user_data.build_export_archive(
        conversations=[],
        patient_documents=[],
        images={"images/abc.jpg": b"bytes"},
    )

    assert "images/abc.jpg" in read_archive(data)

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.read("images/abc.jpg") == b"bytes"


def test_an_export_with_no_photographs_still_builds():
    data = user_data.build_export_archive(
        conversations=[], patient_documents=[]
    )

    assert "chat_history.json" in read_archive(data)


def test_the_readme_says_where_the_photographs_are():
    data = user_data.build_export_archive(
        conversations=[], patient_documents=[]
    )

    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert "images/" in archive.read("README.txt").decode()

