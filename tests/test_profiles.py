"""Tests for named profiles.

Profiles separate data between people sharing one device. They are
not authentication: anyone at the keyboard can switch. What must
hold is that one profile never sees another's chats or reports.
"""

import pytest

from services import profiles


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(profiles, "USER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        profiles,
        "PROFILES_FILE",
        str(tmp_path / "profiles.json"),
    )


# ------------------------------------------------------------------
# Listing
# ------------------------------------------------------------------

def test_the_default_profile_always_exists():
    listed = profiles.list_profiles()

    assert listed[0]["id"] == profiles.DEFAULT_PROFILE_ID


def test_a_created_profile_is_listed():
    profiles.create_profile("Amma")

    assert "amma" in [p["id"] for p in profiles.list_profiles()]


def test_the_display_name_is_kept():
    profile_id = profiles.create_profile("Ravi Kumar")

    assert profiles.profile_name(profile_id) == "Ravi Kumar"


# ------------------------------------------------------------------
# Naming
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,expected",
    [
        ("Amma", "amma"),
        ("Ravi Kumar", "ravi-kumar"),
        ("  Test  User  ", "test-user"),
        ("A/B?C", "a-b-c"),
    ],
)
def test_names_become_safe_folder_ids(name, expected):
    assert profiles.slugify(name) == expected


def test_a_blank_name_is_rejected():
    assert profiles.create_profile("   ") is None
    assert profiles.create_profile("") is None


def test_creating_the_same_name_twice_does_not_duplicate():
    first = profiles.create_profile("Amma")
    second = profiles.create_profile("Amma")

    assert first == second
    assert [p["id"] for p in profiles.list_profiles()].count("amma") == 1


def test_a_very_long_name_is_shortened():
    profile_id = profiles.create_profile("x" * 200)

    assert len(profiles.profile_name(profile_id)) <= (
        profiles.NAME_MAX_LENGTH
    )


# ------------------------------------------------------------------
# Storage separation
# ------------------------------------------------------------------

def test_profiles_get_separate_directories():
    assert profiles.profile_directory("alice") != (
        profiles.profile_directory("bob")
    )


def test_a_profile_directory_sits_under_user_data():
    directory = profiles.profile_directory("alice")

    assert directory.startswith(profiles.USER_DATA_DIR)


# ------------------------------------------------------------------
# Deletion
# ------------------------------------------------------------------

def test_a_profile_can_be_removed():
    profiles.create_profile("Temp")

    assert profiles.delete_profile("temp") is True
    assert "temp" not in [p["id"] for p in profiles.list_profiles()]


def test_the_default_profile_cannot_be_removed():
    assert profiles.delete_profile(profiles.DEFAULT_PROFILE_ID) is False
    assert profiles.list_profiles()[0]["id"] == (
        profiles.DEFAULT_PROFILE_ID
    )


# ------------------------------------------------------------------
# Renaming
# ------------------------------------------------------------------

def test_a_profile_can_be_renamed():
    profile_id = profiles.create_profile("Amma")

    assert profiles.rename_profile(profile_id, "Grandmother") is True
    assert profiles.profile_name(profile_id) == "Grandmother"


def test_renaming_keeps_the_id_so_data_is_not_orphaned():
    profile_id = profiles.create_profile("Amma")
    before = profiles.profile_directory(profile_id)

    profiles.rename_profile(profile_id, "Completely Different")

    assert profiles.profile_directory(profile_id) == before


def test_the_default_profile_can_be_renamed():
    assert profiles.rename_profile(
        profiles.DEFAULT_PROFILE_ID, "Nivedha"
    ) is True
    assert profiles.profile_name(profiles.DEFAULT_PROFILE_ID) == "Nivedha"


def test_a_blank_new_name_is_rejected():
    profile_id = profiles.create_profile("Amma")

    assert profiles.rename_profile(profile_id, "   ") is False
    assert profiles.profile_name(profile_id) == "Amma"


def test_renaming_an_unknown_profile_reports_failure():
    assert profiles.rename_profile("nobody", "Name") is False
