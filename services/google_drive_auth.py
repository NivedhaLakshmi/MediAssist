"""Google Drive OAuth helpers for MediAssist.

The OAuth state is stateless and cryptographically signed so that the
Google callback does not depend on Streamlit session state surviving
the browser redirect.
"""

import base64
import hashlib
import hmac
import os
import secrets
import time
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

load_dotenv()


GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"

GOOGLE_DRIVE_SCOPE = (
    "https://www.googleapis.com/auth/drive.readonly"
)

STATE_VALIDITY_SECONDS = 600


class GoogleDriveAuthError(Exception):
    """Raised when Google OAuth cannot be completed safely."""


# ============================================================
# CONFIGURATION
# ============================================================

def get_google_drive_config():
    """Return Google OAuth settings without exposing secrets."""

    return {
        "client_id": os.getenv(
            "GOOGLE_DRIVE_CLIENT_ID",
            ""
        ).strip(),

        "client_secret": os.getenv(
            "GOOGLE_DRIVE_CLIENT_SECRET",
            ""
        ).strip(),

        "redirect_uri": os.getenv(
            "GOOGLE_DRIVE_REDIRECT_URI",
            "http://localhost:8501"
        ).strip(),
    }


def is_google_drive_configured():
    """Check whether all required OAuth settings exist."""

    config = get_google_drive_config()

    return bool(
        config["client_id"]
        and config["client_secret"]
        and config["redirect_uri"]
    )


# ============================================================
# STATE SIGNING
# ============================================================

def _get_state_secret():
    """Use the OAuth client secret as the signing secret."""

    config = get_google_drive_config()

    secret = config["client_secret"]

    if not secret:
        raise GoogleDriveAuthError(
            "Google Drive OAuth client secret is not configured."
        )

    return secret.encode("utf-8")


def _encode_state(timestamp, nonce, profile_id=""):
    """Create a signed OAuth state value.

    The profile is carried here because Google's redirect reloads the
    page, and Streamlit starts a fresh session for a fresh page load.
    Session state is therefore gone by the time the callback runs,
    and the connection would be stored against whichever profile the
    new session defaulted to.

    It travels inside the signed payload, so a tampered profile fails
    the signature check rather than redirecting someone else's
    credentials into another profile.
    """

    payload = f"{timestamp}:{nonce}:{profile_id}"

    signature = hmac.new(
        _get_state_secret(),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    state = f"{payload}:{signature}"

    return base64.urlsafe_b64encode(
        state.encode("utf-8")
    ).decode("utf-8")


def _decode_state(state):
    """Validate a signed OAuth state and return the profile in it."""

    try:

        decoded = base64.urlsafe_b64decode(
            state.encode("utf-8")
        ).decode("utf-8")

        timestamp, nonce, profile_id, signature = decoded.split(
            ":",
            3
        )

        payload = f"{timestamp}:{nonce}:{profile_id}"

        expected_signature = hmac.new(
            _get_state_secret(),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(
            signature,
            expected_signature
        ):
            raise GoogleDriveAuthError(
                "Invalid Google Drive authorization state."
            )

        timestamp = int(timestamp)

        current_time = int(time.time())

        if (
            current_time - timestamp
            > STATE_VALIDITY_SECONDS
        ):
            raise GoogleDriveAuthError(
                "Google Drive authorization expired. "
                "Please connect again."
            )

        if (
            timestamp - current_time
            > 60
        ):
            raise GoogleDriveAuthError(
                "Invalid Google Drive authorization state."
            )

        if not nonce:
            raise GoogleDriveAuthError(
                "Invalid Google Drive authorization state."
            )

        return profile_id

    except GoogleDriveAuthError:
        raise

    except Exception as error:

        raise GoogleDriveAuthError(
            "Google Drive authorization state could not "
            "be verified."
        ) from error


# ============================================================
# AUTHORIZATION URL
# ============================================================

def build_google_authorization_url(profile_id=""):
    """Build a Google OAuth authorization URL.

    ``profile_id`` is signed into the state so the callback knows
    which profile asked to connect.
    """

    config = get_google_drive_config()

    if not is_google_drive_configured():

        raise GoogleDriveAuthError(
            "Google Drive OAuth is not configured."
        )

    timestamp = int(time.time())

    nonce = secrets.token_urlsafe(32)

    state = _encode_state(
        timestamp,
        nonce,
        profile_id,
    )

    query = urlencode(
        {
            "client_id": config["client_id"],

            "redirect_uri": config["redirect_uri"],

            "response_type": "code",

            "scope": GOOGLE_DRIVE_SCOPE,

            "access_type": "offline",

            "prompt": "consent",

            "state": state,
        }
    )

    authorization_url = (
        f"{GOOGLE_AUTH_URL}?{query}"
    )

    return authorization_url, state


# ============================================================
# STATE VALIDATION
# ============================================================

def validate_google_oauth_state(state):
    """Validate the state returned by Google.

    Returns the profile that started the connection, or "" when the
    state carries none.
    """

    if not state:

        raise GoogleDriveAuthError(
            "Google did not return an authorization state."
        )

    return _decode_state(state)


# ============================================================
# TOKEN EXCHANGE
# ============================================================

def exchange_google_authorization_code(code):
    """Exchange Google's authorization code for credentials."""

    if not code:

        raise GoogleDriveAuthError(
            "Google did not return an authorization code."
        )

    config = get_google_drive_config()

    try:

        response = requests.post(
            GOOGLE_TOKEN_URL,

            data={
                "code": code,

                "client_id": config["client_id"],

                "client_secret": config["client_secret"],

                "redirect_uri": config["redirect_uri"],

                "grant_type": "authorization_code",
            },

            timeout=20,
        )

        response.raise_for_status()

    except requests.RequestException as error:

        raise GoogleDriveAuthError(
            "Google Drive authorization could not "
            "be completed."
        ) from error

    try:

        payload = response.json()

    except ValueError as error:

        raise GoogleDriveAuthError(
            "Google returned an invalid authorization response."
        ) from error

    if not payload.get("access_token"):

        raise GoogleDriveAuthError(
            "Google did not return an access token."
        )

    return {
        "access_token": payload["access_token"],

        "refresh_token": payload.get(
            "refresh_token"
        ),

        "expires_at": (
            time.time()
            + payload.get(
                "expires_in",
                3600
            )
        ),
    }


# ============================================================
# TOKEN REFRESH
# ============================================================

def get_valid_access_token(credentials):
    """Return a valid access token."""

    if not credentials:

        raise GoogleDriveAuthError(
            "Google Drive is not connected."
        )

    if (
        credentials.get("expires_at", 0)
        > time.time() + 60
    ):

        return (
            credentials["access_token"],
            credentials
        )

    refresh_token = credentials.get(
        "refresh_token"
    )

    if not refresh_token:

        raise GoogleDriveAuthError(
            "The Google Drive session expired. "
            "Please reconnect."
        )

    config = get_google_drive_config()

    try:

        response = requests.post(
            GOOGLE_TOKEN_URL,

            data={
                "client_id": config["client_id"],

                "client_secret": config["client_secret"],

                "refresh_token": refresh_token,

                "grant_type": "refresh_token",
            },

            timeout=20,
        )

        response.raise_for_status()

    except requests.RequestException as error:

        raise GoogleDriveAuthError(
            "The Google Drive session could not "
            "be refreshed."
        ) from error

    try:

        payload = response.json()

    except ValueError as error:

        raise GoogleDriveAuthError(
            "Google returned an invalid token response."
        ) from error

    if not payload.get("access_token"):

        raise GoogleDriveAuthError(
            "Google did not return a refreshed access token."
        )

    credentials["access_token"] = (
        payload["access_token"]
    )

    credentials["expires_at"] = (
        time.time()
        + payload.get(
            "expires_in",
            3600
        )
    )

    return (
        credentials["access_token"],
        credentials
    )


# ============================================================
# REVOKE ACCESS
# ============================================================

def revoke_google_access(credentials):
    """Revoke the current Google OAuth token."""

    if not credentials:
        return

    token = (
        credentials.get("refresh_token")
        or
        credentials.get("access_token")
    )

    if not token:
        return

    try:

        requests.post(
            GOOGLE_REVOKE_URL,

            params={
                "token": token
            },

            timeout=20,
        )

    except requests.RequestException:
        # The application still clears its local session.
        pass


# ============================================================
# FOLDER ID
# ============================================================

def extract_google_drive_folder_id(folder_reference):
    """Accept a Google Drive folder URL or folder ID."""

    value = (
        folder_reference or ""
    ).strip()

    if not value:
        return ""

    marker = "/folders/"

    if marker in value:

        value = value.split(
            marker,
            1
        )[1]

        value = (
            value
            .split("?", 1)[0]
            .split("/", 1)[0]
        )

    return value