"""Tests for the Google OAuth helpers.

The OAuth state is signed rather than stored, because the Streamlit
session does not survive the redirect. These tests check that a
forged or stale state is rejected.
"""

import time

import pytest

from services import google_drive_auth as auth
from services.google_drive_auth import (
    GoogleDriveAuthError,
    build_google_authorization_url,
    extract_google_drive_folder_id,
    validate_google_oauth_state,
)


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv(
        "GOOGLE_DRIVE_REDIRECT_URI", "http://localhost:8501"
    )


# ------------------------------------------------------------------
# Authorization URL
# ------------------------------------------------------------------

def test_authorization_url_carries_the_required_parameters():
    url, _ = build_google_authorization_url()

    assert url.startswith("https://accounts.google.com/")
    assert "response_type=code" in url
    assert "client_id=test-client-id" in url
    assert "access_type=offline" in url


def test_authorization_url_requests_read_only_scope():
    url, _ = build_google_authorization_url()

    assert "drive.readonly" in url
    assert "drive.file" not in url


# ------------------------------------------------------------------
# State signing
# ------------------------------------------------------------------

def test_a_freshly_issued_state_is_accepted():
    _, state = build_google_authorization_url()

    # Returns the profile carried in the state - "" when none.
    assert validate_google_oauth_state(state) == ""


def test_each_request_gets_a_different_state():
    _, first = build_google_authorization_url()
    _, second = build_google_authorization_url()

    assert first != second


def test_a_missing_state_is_rejected():
    with pytest.raises(GoogleDriveAuthError):
        validate_google_oauth_state("")


def test_a_tampered_state_is_rejected():
    _, state = build_google_authorization_url()

    forged = ("A" if state[0] != "A" else "B") + state[1:]

    with pytest.raises(GoogleDriveAuthError):
        validate_google_oauth_state(forged)


def test_a_state_signed_with_another_secret_is_rejected():
    _, state = build_google_authorization_url()

    # Same app, different client secret: the signature must not verify.
    import os

    os.environ["GOOGLE_DRIVE_CLIENT_SECRET"] = "a-different-secret"

    with pytest.raises(GoogleDriveAuthError):
        validate_google_oauth_state(state)


def test_an_expired_state_is_rejected():
    stale = auth._encode_state(
        int(time.time()) - auth.STATE_VALIDITY_SECONDS - 60,
        "nonce-value",
    )

    with pytest.raises(GoogleDriveAuthError):
        validate_google_oauth_state(stale)


def test_a_state_dated_in_the_future_is_rejected():
    future = auth._encode_state(int(time.time()) + 600, "nonce-value")

    with pytest.raises(GoogleDriveAuthError):
        validate_google_oauth_state(future)


# ------------------------------------------------------------------
# Folder reference parsing
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "reference,expected",
    [
        ("https://drive.google.com/drive/folders/ABC123", "ABC123"),
        ("https://drive.google.com/drive/folders/ABC123?usp=sharing",
         "ABC123"),
        ("https://drive.google.com/drive/folders/ABC123/", "ABC123"),
        ("ABC123", "ABC123"),
        ("  ABC123  ", "ABC123"),
        ("", ""),
        (None, ""),
    ],
)
def test_folder_id_is_extracted_from_a_url_or_used_as_is(
    reference, expected
):
    assert extract_google_drive_folder_id(reference) == expected


# ------------------------------------------------------------------
# The profile carried through the redirect
# ------------------------------------------------------------------

def test_the_profile_survives_the_round_trip():
    """Google's redirect reloads the page, so Streamlit starts a new
    session and session state is gone. Without this the connection
    was stored against the default profile no matter who asked."""

    _, state = build_google_authorization_url("pramila")

    assert validate_google_oauth_state(state) == "pramila"


def test_two_profiles_get_different_states():
    _, first = build_google_authorization_url("alice")
    _, second = build_google_authorization_url("bob")

    assert validate_google_oauth_state(first) == "alice"
    assert validate_google_oauth_state(second) == "bob"


def test_a_tampered_profile_is_rejected():
    """The profile decides where credentials are stored, so it must
    not be editable in the URL Google hands back."""

    import base64

    _, state = build_google_authorization_url("alice")

    decoded = base64.urlsafe_b64decode(state).decode()

    timestamp, nonce, _, signature = decoded.split(":", 3)

    forged = base64.urlsafe_b64encode(
        f"{timestamp}:{nonce}:bob:{signature}".encode()
    ).decode()

    with pytest.raises(GoogleDriveAuthError):
        validate_google_oauth_state(forged)

