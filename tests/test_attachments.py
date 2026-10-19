"""Tests for photographs attached to a question.

These are pictures of the patient's own documents, so what matters
is that they are stored inside the right profile, cleaned up when
the chats that referred to them go, and never able to point the
application at a file outside its own folder.
"""

import os

import pytest

from services import attachments, profiles


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(profiles, "USER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        profiles, "PROFILES_FILE", str(tmp_path / "profiles.json")
    )


IMAGE = b"\xff\xd8\xff\xe0 not really a jpeg, but bytes are bytes"


def conversation(*names, conversation_id="c1"):
    return {
        "id": conversation_id,
        "messages": [
            {"role": "user", "content": "look", "images": list(names)},
            {"role": "assistant", "content": "an answer"},
        ],
    }


# ------------------------------------------------------------------
# Storing
# ------------------------------------------------------------------

def test_an_image_is_stored_and_can_be_found_again():
    name = attachments.save_attachment(IMAGE, "photo.jpg")

    assert name is not None
    assert attachments.attachment_path(name) is not None


def test_the_stored_bytes_are_unchanged():
    name = attachments.save_attachment(IMAGE, "photo.jpg")

    with open(attachments.attachment_path(name), "rb") as handle:
        assert handle.read() == IMAGE


def test_an_empty_image_is_not_stored():
    assert attachments.save_attachment(b"", "photo.jpg") is None
    assert attachments.save_attachment(None, "photo.jpg") is None


def test_an_oversized_image_is_refused():
    too_big = b"x" * (attachments.MAX_BYTES + 1)

    assert attachments.save_attachment(too_big, "photo.jpg") is None


def test_the_extension_is_kept_when_it_is_one_we_expect():
    name = attachments.save_attachment(IMAGE, "scan.png")

    assert name.endswith(".png")


def test_an_unexpected_extension_falls_back():
    name = attachments.save_attachment(IMAGE, "report.pdf.exe")

    assert name.endswith(attachments.DEFAULT_EXTENSION)


def test_two_images_of_the_same_name_do_not_collide():
    first = attachments.save_attachment(IMAGE, "photo.jpg")
    second = attachments.save_attachment(IMAGE, "photo.jpg")

    assert first != second
    assert attachments.attachment_path(first)
    assert attachments.attachment_path(second)


# ------------------------------------------------------------------
# The name in the conversation file is untrusted
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "../../../../etc/passwd",
        "../conversations.json",
        "/etc/passwd",
        "not-a-stored-name.jpg",
        "",
        None,
    ],
)
def test_a_name_this_module_did_not_generate_is_refused(name):
    """A conversation file edited by hand must not be able to point
    the application at an arbitrary file."""

    assert attachments.attachment_path(name) is None


def test_a_missing_file_reports_nothing_rather_than_a_path():
    name = attachments.save_attachment(IMAGE, "photo.jpg")

    os.remove(attachments.attachment_path(name))

    assert attachments.attachment_path(name) is None


# ------------------------------------------------------------------
# Separation between profiles
# ------------------------------------------------------------------

def test_an_image_belongs_only_to_the_profile_that_saved_it():
    name = attachments.save_attachment(IMAGE, "photo.jpg", "alice")

    assert attachments.attachment_path(name, "alice") is not None
    assert attachments.attachment_path(name, "bob") is None


def test_deleting_one_profiles_images_leaves_the_others():
    alice = attachments.save_attachment(IMAGE, "a.jpg", "alice")
    bob = attachments.save_attachment(IMAGE, "b.jpg", "bob")

    attachments.delete_all_attachments("alice")

    assert attachments.attachment_path(alice, "alice") is None
    assert attachments.attachment_path(bob, "bob") is not None


# ------------------------------------------------------------------
# Cleaning up
# ------------------------------------------------------------------

def test_delete_all_removes_every_image():
    first = attachments.save_attachment(IMAGE, "a.jpg")
    second = attachments.save_attachment(IMAGE, "b.jpg")

    assert attachments.delete_all_attachments() == 2
    assert attachments.attachment_path(first) is None
    assert attachments.attachment_path(second) is None


def test_delete_all_on_a_profile_with_no_images_is_harmless():
    assert attachments.delete_all_attachments("nobody") == 0


def test_an_image_a_conversation_still_refers_to_is_kept():
    name = attachments.save_attachment(IMAGE, "photo.jpg")

    attachments.prune_unreferenced([conversation(name)])

    assert attachments.attachment_path(name) is not None


def test_an_image_no_conversation_refers_to_is_removed():
    """Only ten conversations are kept, so an image can lose its
    last reference without anyone deleting it."""

    kept = attachments.save_attachment(IMAGE, "keep.jpg")
    orphan = attachments.save_attachment(IMAGE, "orphan.jpg")

    attachments.prune_unreferenced([conversation(kept)])

    assert attachments.attachment_path(kept) is not None
    assert attachments.attachment_path(orphan) is None


def test_pruning_with_no_conversations_left_removes_everything():
    name = attachments.save_attachment(IMAGE, "photo.jpg")

    attachments.prune_unreferenced([])

    assert attachments.attachment_path(name) is None


def test_pruning_only_touches_its_own_profile():
    alice = attachments.save_attachment(IMAGE, "a.jpg", "alice")
    bob = attachments.save_attachment(IMAGE, "b.jpg", "bob")

    attachments.prune_unreferenced([], "alice")

    assert attachments.attachment_path(alice, "alice") is None
    assert attachments.attachment_path(bob, "bob") is not None


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------

def test_a_referenced_image_is_collected_for_export():
    name = attachments.save_attachment(IMAGE, "photo.jpg")

    collected = attachments.collect_for_export([conversation(name)])

    assert collected == {f"images/{name}": IMAGE}


def test_an_image_whose_file_is_gone_is_skipped():
    name = attachments.save_attachment(IMAGE, "photo.jpg")

    os.remove(attachments.attachment_path(name))

    assert attachments.collect_for_export([conversation(name)]) == {}


def test_nothing_is_collected_when_no_message_has_an_image():
    attachments.save_attachment(IMAGE, "photo.jpg")

    assert attachments.collect_for_export([conversation()]) == {}


# ------------------------------------------------------------------
# What was read out of an image
# ------------------------------------------------------------------

ROWS = [{"name": "Haemoglobin", "value": 15.8,
         "low": 13.5, "high": 18.0, "sex": ""}]


def test_what_was_read_comes_back():
    """Reading an image costs about 25 seconds, so editing a
    question must not pay it again for an unchanged photo."""

    name = attachments.save_attachment(IMAGE, "photo.jpg")

    attachments.save_extraction(name, "Haemoglobin 15.8", ROWS)

    text, rows = attachments.load_extraction(name)

    assert text == "Haemoglobin 15.8"
    assert rows == ROWS


def test_an_image_never_read_returns_nothing():
    name = attachments.save_attachment(IMAGE, "photo.jpg")

    assert attachments.load_extraction(name) == ("", [])


def test_an_unsafe_name_cannot_write_a_sidecar():
    assert attachments.save_extraction(
        "../../escape", "text", []
    ) is False
    assert attachments.load_extraction("../../escape") == ("", [])


def test_the_sidecar_survives_pruning_with_its_image():
    name = attachments.save_attachment(IMAGE, "photo.jpg")
    attachments.save_extraction(name, "some text", ROWS)

    attachments.prune_unreferenced([conversation(name)])

    assert attachments.load_extraction(name)[0] == "some text"


def test_the_sidecar_goes_when_its_image_does():
    name = attachments.save_attachment(IMAGE, "photo.jpg")
    attachments.save_extraction(name, "some text", ROWS)

    attachments.prune_unreferenced([])

    assert attachments.attachment_path(name) is None
    assert attachments.load_extraction(name) == ("", [])


def test_a_sidecar_is_not_exported():
    """The export carries the photographs, not the machine's notes
    about them."""

    name = attachments.save_attachment(IMAGE, "photo.jpg")
    attachments.save_extraction(name, "some text", ROWS)

    collected = attachments.collect_for_export([conversation(name)])

    assert list(collected) == [f"images/{name}"]


def test_delete_all_removes_the_sidecars_too():
    name = attachments.save_attachment(IMAGE, "photo.jpg")
    attachments.save_extraction(name, "some text", ROWS)

    attachments.delete_all_attachments()

    assert attachments.load_extraction(name) == ("", [])

