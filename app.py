import base64
import html
import os
import re
import tempfile

import streamlit as st


st.set_page_config(
    page_title="MediAssist",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


from rag import search_local_database

from services.llm_service import (
    OllamaServiceError,
    stream_answer,
)

from services.prompt_service import (
    build_healthcare_prompt,
)

from services.lab_verifier import (
    merge_findings,
    verify_extracted_rows,
    verify_lab_ranges,
)

from mcp_client import (
    search_medical_sources,
)

from services.query_utils import (
    build_search_keywords,
    normalize_health_query,
)

from services.pdf_service import (
    read_pdf,
)

from services.image_service import (
    photo_search_terms,
    read_image,
)

from services.report_extractor import (
    is_available as vision_is_available,
)

from services.attachments import (
    attachment_path,
    load_extraction,
    save_extraction,
    collect_for_export,
    delete_all_attachments,
    prune_unreferenced,
    save_attachment,
)

from services.patient_memory import (
    add_patient_report,
    count_patient_reports,
    delete_all_patient_reports,
    get_all_patient_documents,
    is_strong_match,
    search_patient_reports,
)

from services.profiles import (
    DEFAULT_PROFILE_ID,
    create_profile,
    delete_profile,
    list_profiles,
    profile_name,
    rename_profile,
)

from services.user_data import (
    MAX_CONVERSATIONS,
    build_export_archive,
    delete_all_conversations,
    delete_conversation,
    export_filename,
    load_conversations,
    new_conversation_id,
    save_conversation,
)

from services.speech_service import (
    transcribe_audio,
)

from components.chat_input import (
    chat_input,
)

from services.chat_input_service import (
    convert_component_files,
)

from services.google_drive_auth import (
    GoogleDriveAuthError,
    build_google_authorization_url,
    exchange_google_authorization_code,
    extract_google_drive_folder_id,
    get_valid_access_token,
    is_google_drive_configured,
    revoke_google_access,
    validate_google_oauth_state,
)

from services.google_drive_mcp_client import (
    GoogleDriveMCPError,
    search_google_drive_reports,
)

MIN_CONTEXT_LENGTH = 200

# Kept equal to verify_lab_ranges(max_findings=...).
MAX_VERIFIED_FINDINGS = 25


def get_normalized_query(question):
    try:
        result = normalize_health_query(
            question
        )

    except Exception as error:
        print("[QUERY] Normalization error:")
        print(error)

        return (
            question,
            [],
        )

    if isinstance(result, dict):

        normalized = result.get(
            "normalized",
            question,
        )

        corrections = result.get(
            "corrections",
            [],
        )

        return (
            normalized or question,
            corrections,
        )

    return (
        result or question,
        [],
    )


def render_document_sources(label, docs, metadata_key, fallback):
    """List the distinct documents behind an answer."""

    if not docs:
        return

    with st.expander(label):

        for name in dict.fromkeys(
            doc.metadata.get(metadata_key, fallback)
            for doc in docs
        ):
            st.write(name)


def render_reference_links(label, items, fallback):
    """List external references, linked where a URL is available."""

    if not items:
        return

    with st.expander(label):

        for item in items:

            title = item.get("title", fallback)
            url = item.get("url", "")

            if url:
                st.markdown(f"- [{title}]({url})")
            else:
                st.write(f"• {title}")


def render_google_connect_link(authorization_url):
    """Render the connect link as a same-tab anchor.

    st.link_button opens a new tab, which would leave the app
    running in two tabs after the callback. A plain anchor with
    target="_self" navigates this tab to Google and brings the
    user straight back here.

    The markup is written on one line on purpose: an indented
    HTML block inside st.markdown is parsed as a code block, which
    would render the tag as visible text.
    """

    safe_url = html.escape(authorization_url, quote=True)

    st.markdown(
        f'<a class="gdrive-connect" href="{safe_url}" target="_self">'
        f"Connect Google Drive</a>",
        unsafe_allow_html=True,
    )


# Session state does not survive the OAuth redirect - the callback
# loads a fresh Streamlit session - so the token lives in one
# process-wide store. Memory only; cleared on disconnect or exit.


@st.cache_resource
def _google_drive_credential_store():
    """Keyed by profile, so a connection is not shared between them."""

    return {}


def get_google_drive_credentials():
    store = _google_drive_credential_store()

    profile = st.session_state.get("profile_id", DEFAULT_PROFILE_ID)

    stored = store.get(profile)

    if stored and not st.session_state.get(
        "google_drive_credentials"
    ):
        st.session_state.google_drive_credentials = stored

    return (
        st.session_state.get("google_drive_credentials")
        or stored
    )


def set_google_drive_credentials(credentials):
    profile = st.session_state.get("profile_id", DEFAULT_PROFILE_ID)

    _google_drive_credential_store()[profile] = credentials

    st.session_state.google_drive_credentials = credentials


def clear_google_drive_credentials():
    profile = st.session_state.get("profile_id", DEFAULT_PROFILE_ID)

    _google_drive_credential_store().pop(profile, None)

    st.session_state.google_drive_credentials = None


for key, default in {
    "messages": [],
    "dictation_text": "",
    "uploaded_pdf_names": set(),
    "last_component_id": None,
    "voice_completion_id": 0,
    "last_context": "",
    "last_source": "",
    "profile_id": DEFAULT_PROFILE_ID,
    "current_conversation_id": None,
    "export_bytes": None,
    "export_name": "",
    "data_notice": "",
    "clear_drive_folder": False,
    "show_delete_panel": False,
    "show_profile_settings": False,
    "pending_chat_delete": None,
    "image_context": "",
    "image_rows": [],
    "image_terms": "",
    "editing_index": None,
    "pending_question": "",
    "pending_images": [],
    "editing_images": [],
    "editing_added": [],
    "google_drive_credentials": None,
    "google_drive_folder_reference": "",
}.items():

    if key not in st.session_state:
        st.session_state[key] = default


if st.session_state.current_conversation_id is None:
    st.session_state.current_conversation_id = new_conversation_id()


# The Drive folder box is a widget, so its value has to be reset
# before the widget is created rather than during the run that
# requested the reset.
if st.session_state.clear_drive_folder:
    st.session_state.google_drive_folder_reference = ""
    st.session_state.clear_drive_folder = False


active_profile = st.session_state.profile_id


def switch_profile(profile_id):
    """Move to another profile, leaving nothing of the last one on screen."""

    st.session_state.profile_id = profile_id
    st.session_state.messages = []
    st.session_state.current_conversation_id = new_conversation_id()
    st.session_state.last_context = ""
    st.session_state.last_source = ""
    st.session_state.uploaded_pdf_names = set()
    st.session_state.export_bytes = None
    st.session_state.data_notice = ""
    st.session_state.show_delete_panel = False

    st.session_state.image_context = ""
    st.session_state.image_rows = []
    st.session_state.image_terms = ""
    st.session_state.editing_index = None
    st.session_state.pending_question = ""
    st.session_state.editing_images = []
    st.session_state.editing_added = []

    # Each profile has its own Drive connection and folder.
    st.session_state.google_drive_credentials = None
    st.session_state.clear_drive_folder = True

    print(f"[PROFILE] Switched to '{profile_id}'.")


def start_new_conversation():
    """Begin an empty chat without touching what is already saved."""

    st.session_state.current_conversation_id = new_conversation_id()
    st.session_state.messages = []
    st.session_state.last_context = ""
    st.session_state.last_source = ""
    st.session_state.pending_chat_delete = None
    st.session_state.image_context = ""
    st.session_state.image_rows = []
    st.session_state.image_terms = ""
    st.session_state.editing_index = None
    st.session_state.pending_question = ""
    st.session_state.editing_images = []
    st.session_state.editing_added = []


def open_conversation(conversation):
    """Load a saved chat back into the session."""

    st.session_state.current_conversation_id = conversation.get("id")
    st.session_state.messages = list(
        conversation.get("messages", [])
    )
    st.session_state.last_context = ""
    st.session_state.last_source = ""
    st.session_state.pending_chat_delete = None

    # A photo belongs to the chat it was attached to, not to the
    # next one opened.
    st.session_state.image_context = ""
    st.session_state.image_rows = []
    st.session_state.image_terms = ""
    st.session_state.editing_index = None
    st.session_state.pending_question = ""
    st.session_state.editing_images = []
    st.session_state.editing_added = []


print("========== MEDIASSIST START ==========")


# Rendered first so it keeps element index 0. Streamlit re-mounts a
# custom component when its position shifts, orphaning the old iframe
# and silently dropping everything it sends. CSS moves it to the
# bottom with `order: 999`.
chat_result = chat_input(
    value=st.session_state.dictation_text,
    voice_completion_id=(
        st.session_state.voice_completion_id
    ),
)


oauth_error = st.query_params.get(
    "error"
)

oauth_code = st.query_params.get(
    "code"
)

oauth_state = st.query_params.get(
    "state"
)

print(
    "[OAUTH] Callback:",
    f"error={bool(oauth_error)}",
    f"code={bool(oauth_code)}",
    f"state={bool(oauth_state)}",
)


if oauth_error:

    print(
        "[OAUTH] Google returned error:",
        oauth_error,
    )

    st.error(
        "Google Drive authorization was cancelled "
        "or denied."
    )

    st.query_params.clear()

elif oauth_code:

    print(
        "[OAUTH] Authorization code received."
    )

    try:

        print(
            "[OAUTH] Validating OAuth state."
        )

        state_profile = validate_google_oauth_state(
            oauth_state
        )

        print(
            "[OAUTH] OAuth state validated."
        )

        # Google's redirect is a fresh page load, so this is a new
        # Streamlit session and profile_id has fallen back to the
        # default. Restore the profile that started the connection,
        # or the token is stored against the wrong one.
        if state_profile and state_profile != active_profile:

            known = {
                profile["id"] for profile in list_profiles()
            }

            if state_profile in known:

                st.session_state.profile_id = state_profile

                active_profile = state_profile

                print(
                    "[OAUTH] Restored profile from state:",
                    state_profile,
                )

            else:

                print(
                    "[OAUTH] Profile in state no longer exists:",
                    state_profile,
                )

        print(
            "[OAUTH] Exchanging authorization code."
        )

        credentials = (
            exchange_google_authorization_code(
                oauth_code
            )
        )

        print(
            "[OAUTH] Token exchange successful."
        )

        print(
            "[OAUTH] Access token:",
            bool(
                credentials.get(
                    "access_token"
                )
            ),
        )

        print(
            "[OAUTH] Refresh token:",
            bool(
                credentials.get(
                    "refresh_token"
                )
            ),
        )

        set_google_drive_credentials(
            credentials
        )

        st.query_params.clear()

        # Coming back from Google is a fresh page load, so the
        # in-memory chat is gone. Re-open the most recent saved
        # conversation so the user lands where they left off.
        if not st.session_state.messages:

            recent = load_conversations(active_profile)

            if recent:
                open_conversation(recent[0])

                print(
                    "[OAUTH] Restored the most recent conversation."
                )

        print(
            "[OAUTH] Credentials stored."
        )

        print(
            "[OAUTH] OAuth callback completed."
        )

        st.rerun()

    except GoogleDriveAuthError as error:

        print(
            "[OAUTH] Authentication error:",
            error,
        )

        st.error(
            f"Google Drive connection failed: {error}"
        )

        st.query_params.clear()

    except Exception as error:

        print(
            "[OAUTH] Unexpected callback error:",
            type(error).__name__,
        )

        print(
            error
        )

        st.error(
            "An unexpected error occurred while "
            "connecting Google Drive."
        )

        st.query_params.clear()


st.markdown(
    """
    <style>

    [data-testid="stDeployButton"] { display: none !important; }

    /* Hide only the toolbar's action items. Hiding the whole
       toolbar also hides stExpandSidebarButton, which is the only
       way to re-open a collapsed sidebar. */
    [data-testid="stToolbarActions"] { display: none !important; }

    [data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    /* Sidebar toggle pinned to the very top-left of the window. */
    [data-testid="stExpandSidebarButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 10px !important;
        left: 10px !important;
        z-index: 1000000 !important;
    }

    [data-testid="stExpandSidebarButton"] button {
        border-radius: 8px !important;
    }

    [data-testid="stExpandSidebarButton"] button:hover {
        background: rgba(255, 255, 255, 0.10) !important;
    }

    /* Sidebar: readable width, still user-resizable by dragging. */
    [data-testid="stSidebar"] {
        min-width: 260px !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.4rem !important;
    }

    /* Only the saved-chat list reads as a plain list. Every other
       sidebar button keeps a normal, consistent button look. */
    [data-testid="stSidebar"] [class*="st-key-open_chat_"] button {
        justify-content: flex-start !important;
        text-align: left !important;
        font-weight: 400 !important;
        border: none !important;
        background: transparent !important;
        padding: 0.35rem 0.6rem !important;
    }

    [data-testid="stSidebar"] [class*="st-key-open_chat_"] button:hover {
        background: rgba(255, 255, 255, 0.07) !important;
    }

    [data-testid="stSidebar"] [class*="st-key-open_chat_"] button:disabled {
        background: rgba(255, 255, 255, 0.10) !important;
        color: inherit !important;
        opacity: 1 !important;
    }

    /* The profile gear sits beside the picker, not under it. */
    [class*="st-key-action_profile_settings"] button {
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        background: rgba(255, 255, 255, 0.04) !important;
        border-radius: 10px !important;
        padding: 0.42rem 0.5rem !important;
        width: 100% !important;
    }

    [class*="st-key-action_profile_settings"] button:hover {
        background: rgba(255, 255, 255, 0.12) !important;
    }

    /* Sidebar sections read as a list of headings, not as boxes. */
    [data-testid="stSidebar"] [data-testid="stExpander"] details {
        border: none !important;
        background: transparent !important;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        padding-left: 0 !important;
        padding-right: 0 !important;
        font-weight: 600 !important;
    }

    /* The pencil sits just below the question bubble, at its right
       edge. Inside the bubble it stretched the box and left a band
       of empty space under the text. */
    [class*="st-key-edit_open_"] {
        margin-top: -0.55rem;
        margin-bottom: 0.1rem;
    }

    [class*="st-key-edit_open_"] button {
        border: none !important;
        background: transparent !important;
        padding: 0.15rem 0.3rem !important;
        min-height: 0 !important;
        opacity: 0.3;
        transition: opacity 0.12s ease, background 0.12s ease;
    }

    /* The icon carries the interface's own colour rather than the
       emoji's yellow. */
    [class*="st-key-edit_open_"] button span,
    [class*="st-key-edit_open_"] button [data-testid="stIconMaterial"] {
        color: inherit !important;
        font-size: 1rem !important;
    }

    [class*="st-key-edit_open_"] button:hover {
        opacity: 1 !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }

    /* Editing turns the bubble itself into the input. */
    [class*="st-key-edit_text_"] textarea {
        border-radius: 16px !important;
        border: 1px solid rgba(122, 162, 255, 0.85) !important;
        background: rgba(255, 255, 255, 0.02) !important;
        padding: 0.95rem 1.1rem !important;
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        box-shadow: none !important;
    }

    [class*="st-key-edit_text_"] textarea:focus {
        border-color: rgba(122, 162, 255, 1) !important;
        box-shadow: 0 0 0 1px rgba(122, 162, 255, 0.45) !important;
    }

    /* Cancel is a way out, so it reads as plain text. Save is the
       committed action and carries the only fill. */
    [class*="st-key-edit_cancel_"] button {
        border: none !important;
        background: transparent !important;
        color: inherit !important;
        opacity: 0.75;
        padding: 0.35rem 0.6rem !important;
    }

    [class*="st-key-edit_cancel_"] button:hover {
        opacity: 1;
        background: rgba(255, 255, 255, 0.07) !important;
    }

    [class*="st-key-edit_save_"] button {
        border: none !important;
        background: rgba(255, 255, 255, 0.16) !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        border-radius: 9px !important;
        padding: 0.35rem 0.6rem !important;
    }

    [class*="st-key-edit_save_"] button:hover {
        background: rgba(255, 255, 255, 0.26) !important;
    }

    [class*="st-key-edit_save_"] button:disabled {
        opacity: 0.4 !important;
    }

    /* The bin sits inside the chat row, so it stays quiet until
       the row is hovered. */
    [data-testid="stSidebar"] [class*="st-key-delete_chat_"] button {
        border: none !important;
        background: transparent !important;
        padding: 0.35rem 0.2rem !important;
        opacity: 0.45;
    }

    [data-testid="stSidebar"] [class*="st-key-delete_chat_"] button:hover {
        opacity: 1;
        background: rgba(255, 255, 255, 0.07) !important;
    }

    /* Confirming a deletion is the one place a red button helps. */
    [data-testid="stSidebar"]
    [class*="st-key-confirm_delete_chat_"] button {
        border: 1px solid rgba(255, 107, 107, 0.45) !important;
        color: #ff8a8a !important;
        background: rgba(255, 107, 107, 0.08) !important;
        padding: 0.3rem 0.5rem !important;
    }

    [data-testid="stSidebar"]
    [class*="st-key-cancel_delete_chat_"] button {
        padding: 0.3rem 0.5rem !important;
    }

    /* Shared look for the action buttons (New chat, Export, Delete). */
    [data-testid="stSidebar"] [class*="st-key-action_"] button {
        justify-content: flex-start !important;
        text-align: left !important;
        border: 1px solid rgba(255, 255, 255, 0.16) !important;
        background: rgba(255, 255, 255, 0.04) !important;
        border-radius: 10px !important;
        padding: 0.45rem 0.7rem !important;
    }

    [data-testid="stSidebar"] [class*="st-key-action_"] button:hover {
        background: rgba(255, 255, 255, 0.10) !important;
    }

    /* Google Drive connect link, styled as a primary button. */
    a.gdrive-connect {
        display: block;
        width: 100%;
        box-sizing: border-box;
        padding: 0.5rem 0.75rem;
        border-radius: 10px;
        background: #4285F4;
        color: #ffffff !important;
        text-align: center;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 0.92rem;
    }

    a.gdrive-connect:hover {
        background: #3b78e7;
    }

    /* Content column. The chat input lives inside it, so the two
       line up with no sidebar-width maths. */
    .stMain .block-container {
        max-width: 820px !important;
        padding-top: 3rem !important;
        padding-bottom: 1.5rem !important;
    }

    /* Fill the window so the input sits at the bottom, not
       floating half-way up a short page. */
    .stMain .block-container > [data-testid="stVerticalBlock"] {
        min-height: calc(100vh - 6rem) !important;
    }

    /* The input is rendered first (for component stability) and
       moved to the bottom here, where it sticks while scrolling. */
    [data-testid="stElementContainer"]:has(
        iframe[title*="mediassist_chat_input"]
    ) {
        order: 999 !important;
        margin-top: auto !important;
        position: sticky !important;
        bottom: 12px !important;
        z-index: 500 !important;
        width: 100% !important;
    }

    iframe[title*="mediassist_chat_input"] {
        width: 100% !important;
        min-height: 112px !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
        color-scheme: dark !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:

    st.markdown("### 🏥 MediAssist")

    # ------------------------------------------------
    # Profile
    # ------------------------------------------------

    all_profiles = list_profiles()

    profile_ids = [p["id"] for p in all_profiles]

    if active_profile not in profile_ids:
        active_profile = DEFAULT_PROFILE_ID
        st.session_state.profile_id = active_profile

    st.caption("Profile")

    # Switching is the common action, so it stays in reach. Renaming,
    # adding and deleting are occasional, and three open panels were
    # most of what filled the sidebar - they now live behind the gear.
    picker_column, gear_column = st.columns([5, 1])

    with picker_column:

        chosen = st.selectbox(
            "Profile",
            options=profile_ids,
            index=profile_ids.index(active_profile),
            format_func=profile_name,
            key="profile_selector",
            label_visibility="collapsed",
        )

    with gear_column:

        if st.button(
            "",
            icon=":material/settings:",
            key="action_profile_settings",
            help="Rename, add or delete a profile",
        ):
            st.session_state.show_profile_settings = (
                not st.session_state.show_profile_settings
            )
            st.rerun()

    if chosen != active_profile:
        switch_profile(chosen)
        st.rerun()

    if st.session_state.show_profile_settings:

        with st.expander("✏️  Rename this profile"):

            # Keyed per profile. With one shared key Streamlit keeps
            # whatever was typed for the previous profile and ignores
            # ``value``, so the box showed another profile's name.
            renamed = st.text_input(
                "New name",
                value=profile_name(active_profile),
                key=f"rename_profile_name_{active_profile}",
                label_visibility="collapsed",
            )

            if st.button(
                "Save name",
                key="action_rename_profile",
                use_container_width=True,
                disabled=(
                    not renamed.strip()
                    or renamed.strip() == profile_name(active_profile)
                ),
            ):
                if rename_profile(active_profile, renamed):
                    st.rerun()

        with st.expander("➕  Add a profile"):

            new_name = st.text_input(
                "Name",
                key="new_profile_name",
                placeholder="Please input name",
                label_visibility="collapsed",
            )

            if st.button(
                "Create and switch",
                key="action_create_profile",
                use_container_width=True,
                disabled=not new_name.strip(),
            ):
                created = create_profile(new_name)

                if created:
                    switch_profile(created)
                    st.rerun()

        with st.expander("🗑️  Delete this profile"):

            if len(all_profiles) < 2:

                st.info(
                    "This is your only profile, so it cannot be "
                    "deleted. To clear what is stored, use "
                    "**Delete my data** below."
                )

            elif active_profile == DEFAULT_PROFILE_ID:

                st.info(
                    "The first profile cannot be deleted. Switch to "
                    "another profile to remove it."
                )

            else:

                st.warning(
                    f"This removes **{profile_name(active_profile)}** "
                    "along with its chats and uploaded reports. It "
                    "cannot be undone."
                )

                if st.checkbox(
                    "I understand this cannot be undone",
                    key="confirm_delete_profile",
                ) and st.button(
                    "Delete this profile",
                    key="action_delete_profile",
                    type="primary",
                    use_container_width=True,
                ):
                    removed_chunks = delete_all_patient_reports(
                        active_profile
                    )
                    removed_chats = delete_all_conversations(
                        active_profile
                    )

                    credentials = get_google_drive_credentials()

                    if credentials:
                        try:
                            revoke_google_access(credentials)
                        except Exception as error:
                            print("[PROFILE] Revoke error:", error)

                    clear_google_drive_credentials()

                    gone = profile_name(active_profile)

                    delete_profile(active_profile)

                    switch_profile(DEFAULT_PROFILE_ID)

                    st.session_state.data_notice = (
                        f"Deleted the profile '{gone}' with "
                        f"{removed_chunks} report chunk(s) and "
                        f"{removed_chats} saved chat(s)."
                    )

                    print(f"[PROFILE] Deleted '{gone}'.")

                    st.rerun()

    st.caption(
        "Each profile keeps its own chats, reports and Drive "
        "connection on this device."
    )

    st.divider()

    # ------------------------------------------------
    # Chats
    # ------------------------------------------------

    with st.expander("💬  Chats", expanded=True):

        if st.button(
            "➕  New chat",
            key="action_new_chat",
            use_container_width=True,
        ):
            start_new_conversation()
            st.rerun()

        saved_conversations = load_conversations(active_profile)

        if not saved_conversations:

            st.caption(
                "Your chats are saved on this device as you use "
                "MediAssist."
            )

        else:

            for conversation in saved_conversations:

                conversation_id = str(conversation.get("id"))

                is_active = (
                    conversation.get("id")
                    == st.session_state.current_conversation_id
                )

                # A chat cannot be brought back once removed, so the bin
                # asks first rather than deleting on the click itself.
                if st.session_state.pending_chat_delete == conversation_id:

                    st.caption(
                        "Delete “"
                        + conversation.get("title", "Chat")
                        + "”?"
                    )

                    confirm_column, cancel_column = st.columns(2)

                    with confirm_column:

                        if st.button(
                            "Delete",
                            key="confirm_delete_chat_" + conversation_id,
                            use_container_width=True,
                        ):
                            delete_conversation(
                                conversation.get("id"), active_profile
                            )

                            prune_unreferenced(
                                load_conversations(active_profile),
                                active_profile,
                            )

                            # The open chat is written back to storage
                            # after every answer, so deleting it while
                            # it is still loaded would simply save it
                            # again on the next question.
                            if is_active:
                                start_new_conversation()

                            st.session_state.pending_chat_delete = None

                            st.rerun()

                    with cancel_column:

                        if st.button(
                            "Cancel",
                            key="cancel_delete_chat_" + conversation_id,
                            use_container_width=True,
                        ):
                            st.session_state.pending_chat_delete = None
                            st.rerun()

                    continue

                open_column, delete_column = st.columns(
                    [5, 1], gap="small"
                )

                with open_column:

                    if st.button(
                        ("• " if is_active else "")
                        + conversation.get("title", "Chat"),
                        key="open_chat_" + conversation_id,
                        use_container_width=True,
                        disabled=is_active,
                    ):
                        open_conversation(conversation)
                        st.rerun()

                with delete_column:

                    if st.button(
                        "🗑",
                        key="delete_chat_" + conversation_id,
                        use_container_width=True,
                        help="Delete this chat",
                    ):
                        st.session_state.pending_chat_delete = (
                            conversation_id
                        )
                        st.rerun()

            st.caption(
                f"The most recent {MAX_CONVERSATIONS} chats are kept "
                "on this device."
            )


    # ------------------------------------------------
    # Google Drive
    # ------------------------------------------------

    with st.expander("☁️  Google Drive", expanded=False):

        st.caption(
            "Optionally connect one Google Drive folder "
            "for read-only access to personal medical records."
        )

        folder_reference = st.text_input(
            "Reports folder URL or ID",
            key="google_drive_folder_reference",
            placeholder=(
                "https://drive.google.com/drive/folders/..."
            ),
        )

        folder_id = extract_google_drive_folder_id(
            folder_reference
        )

        credentials = get_google_drive_credentials()

        if credentials:

            st.success(
                "🟢 Connected — read-only access"
            )

            if not folder_id:

                st.warning(
                    "Add the Google Drive folder URL or ID "
                    "to search your reports."
                )

            else:

                st.caption(
                    "MediAssist can search this folder only "
                    "for personal medical-record questions."
                )

            if st.button(
                "Disconnect Google Drive",
                type="secondary",
                use_container_width=True,
            ):

                print(
                    "[OAUTH] Disconnect clicked."
                )

                try:

                    revoke_google_access(
                        credentials
                    )

                    print(
                        "[OAUTH] Google access revoked."
                    )

                except Exception as error:

                    print(
                        "[OAUTH] Revoke error:",
                        error,
                    )

                finally:

                    clear_google_drive_credentials()

                    # The folder box is a widget, so it can only be
                    # reset before it is instantiated on the next run.
                    st.session_state.clear_drive_folder = True

                    print(
                        "[OAUTH] Local credentials cleared."
                    )

                    st.rerun()

        elif not is_google_drive_configured():

            print(
                "[OAUTH] Google Drive is not configured."
            )

            st.info(
                "Google Drive is not configured yet. "
                "Add the OAuth settings described in "
                "GOOGLE_DRIVE_SETUP.md."
            )

        else:

            st.caption(
                "⚪ Not connected"
            )

            # One link, one click. The authorization URL is rebuilt on
            # every run (the signed state carries its own timestamp),
            # so there is no need for a separate "generate" step.
            try:

                authorization_url, _ = build_google_authorization_url(
                    active_profile
                )

                render_google_connect_link(authorization_url)

                st.caption(
                    "Opens Google's read-only consent screen in this "
                    "tab and returns you here."
                )

            except GoogleDriveAuthError as error:

                print(
                    "[OAUTH] Failed to create URL:",
                    error,
                )

                st.error(
                    f"Google Drive connection failed: {error}"
                )

        st.caption(
            "Disconnect removes the active session token "
            "and requests Google to revoke access."
        )


    # ------------------------------------------------
    # Data
    # ------------------------------------------------

    with st.expander("🔒  Data", expanded=False):

        if st.session_state.get("data_notice"):

            st.success(st.session_state.data_notice)

            st.session_state.data_notice = ""

        report_count, chunk_count = count_patient_reports(active_profile)

        st.caption(
            f"{report_count} uploaded report(s) · "
            f"{chunk_count} stored chunk(s) · "
            f"{len(saved_conversations)} saved chat(s)"
        )

        # --- Export -------------------------------------

        if st.button(
            "⬇️  Export my data",
            key="action_export",
            use_container_width=True,
        ):

            st.session_state.export_bytes = build_export_archive(
                conversations=saved_conversations,
                patient_documents=get_all_patient_documents(active_profile),
                drive_connected=bool(credentials),
                drive_folder=folder_id,
                images=collect_for_export(
                    saved_conversations, active_profile
                ),
            )

            st.session_state.export_name = export_filename()

        if st.session_state.get("export_bytes"):

            st.download_button(
                "Download (.zip)",
                data=st.session_state.export_bytes,
                file_name=st.session_state.export_name,
                mime="application/zip",
                use_container_width=True,
            )

            st.caption(
                "Your chats and the text of your uploaded reports. "
                "Google tokens are never exported."
            )

        # --- Delete -------------------------------------

        if st.button(
            "🗑️  Delete my data",
            key="action_delete",
            use_container_width=True,
        ):
            st.session_state.show_delete_panel = (
                not st.session_state.get("show_delete_panel", False)
            )

        if st.session_state.get("show_delete_panel"):

            st.warning(
                "This permanently deletes your uploaded reports and "
                "their embeddings, your saved chats, and disconnects "
                "Google Drive. It cannot be undone."
            )

            st.caption(
                "The general medical knowledge base is system data "
                "and is not affected."
            )

            confirmed = st.checkbox(
                "I understand this cannot be undone",
                key="confirm_delete_my_data",
            )

            if st.button(
                "Delete everything",
                type="primary",
                disabled=not confirmed,
                use_container_width=True,
            ):

                deleted_chunks = delete_all_patient_reports(active_profile)

                deleted_chats = delete_all_conversations(active_profile)

                # Photographs of the patient's documents are their data
                # too, so "delete everything" has to mean it.
                deleted_images = delete_all_attachments(active_profile)

                if credentials:

                    try:
                        revoke_google_access(credentials)

                    except Exception as error:
                        print("[DELETE] Revoke error:", error)

                clear_google_drive_credentials()

                # Cleared on the next run, before the widget exists.
                st.session_state.clear_drive_folder = True

                st.session_state.uploaded_pdf_names = set()

                st.session_state.messages = []

                st.session_state.last_context = ""

                st.session_state.last_source = ""

                st.session_state.current_conversation_id = (
                    new_conversation_id()
                )

                st.session_state.export_bytes = None

                st.session_state.show_delete_panel = False

                st.session_state.data_notice = (
                    f"Deleted {deleted_chunks} report chunk(s) and "
                    f"{deleted_chats} saved chat(s), and disconnected "
                    "Google Drive."
                )

                print(
                    "[DELETE] Patient data removed:",
                    f"{deleted_chunks} chunk(s),",
                    f"{deleted_chats} chat(s).",
                )

                st.rerun()


greeting_slot = st.empty()

if not st.session_state.messages:

    greeting_slot.markdown(
        """
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 46vh;
            text-align: center;
            opacity: 0.92;
        ">
            <div style="font-size: 2.1rem; font-weight: 600;">
                🏥 MediAssist
            </div>
            <div style="
                margin-top: 0.6rem;
                font-size: 1rem;
                opacity: 0.62;
                max-width: 30rem;
            ">
                Your assistant for clinical questions and patient records.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def question_was_answered(index):
    """Whether an assistant reply follows this message.

    Only an answered question can be edited: editing the one still
    being answered would fork the conversation mid-flight.
    """

    messages = st.session_state.messages

    return (
        index + 1 < len(messages)
        and messages[index + 1]["role"] == "assistant"
    )


ATTACHMENT_WIDTH = 240


def rebuild_image_context(names, profile_id):
    """Set the image context from a list of stored attachments.

    Used when a question is edited: the photographs may have been
    added to or removed from, and what each one says was recorded
    when it was first read.
    """

    parts = []
    rows = []
    readings = []

    for name in names:

        text, image_rows = load_extraction(name, profile_id)

        if text:
            parts.append(f"Text read from an attached image:\n{text}")

        readings.append((text, image_rows))
        rows.extend(image_rows)

    st.session_state.image_context = "\n\n".join(parts)
    st.session_state.image_rows = rows
    st.session_state.image_terms = photo_search_terms(readings)


def render_attachments(message, profile_id):
    """Show the photos that were attached to a question.

    An image whose file has gone - deleted data, a profile removed -
    is simply not shown, so an old conversation still opens.
    """

    names = message.get("images") or []

    if not names:
        return

    paths = [
        path
        for path in (
            attachment_path(name, profile_id) for name in names
        )
        if path
    ]

    if not paths:
        return

    for column, path in zip(st.columns(len(paths)), paths):

        with column:
            st.image(path, width=ATTACHMENT_WIDTH)


def render_edit_button(index):
    """The pencil under an answered question.

    Used both by the history loop and, once an answer is stored, by
    the live block - otherwise the question just asked is the one
    message on screen that cannot be edited, because nothing reruns
    to hand it back to the history loop.
    """

    if st.session_state.editing_index is not None:
        return

    # A Material outline icon rather than the pencil emoji, which
    # renders as a yellow glyph that does not belong beside the
    # monochrome interface.
    if st.button(
        "",
        icon=":material/edit:",
        key=f"edit_open_{index}",
        help="Edit this question and ask again",
    ):
        # A keyed text_area ignores value= once it has a stored
        # value, so reopening the editor would show whatever was
        # typed and abandoned last time rather than the question.
        st.session_state.pop(f"edit_text_{index}", None)

        st.session_state.editing_index = index

        st.session_state.editing_images = list(
            st.session_state.messages[index].get("images") or []
        )

        st.session_state.editing_added = []

        st.rerun()


for index, message in enumerate(st.session_state.messages):

    with st.chat_message(
        message["role"]
    ):

        editing = (
            message["role"] == "user"
            and st.session_state.editing_index == index
        )

        if editing:

            content = message["content"]

            editing_images = st.session_state.editing_images

            # The photographs are part of the question, so they are
            # shown while it is being edited and can be changed.
            # Previously they vanished from the editor and were then
            # dropped from the message the edit produced.
            if editing_images:

                thumbnails = st.columns(
                    max(len(editing_images), 3)
                )

                for column, name in zip(thumbnails, editing_images):

                    path = attachment_path(name, active_profile)

                    if not path:
                        continue

                    with column:

                        st.image(path, width=ATTACHMENT_WIDTH)

                        if st.button(
                            "Remove",
                            key=f"edit_drop_{index}_{name}",
                            use_container_width=True,
                        ):
                            st.session_state.editing_images = [
                                other
                                for other in editing_images
                                if other != name
                            ]
                            st.rerun()

            added = st.file_uploader(
                "Add a photo",
                type=["jpg", "jpeg", "png", "webp", "bmp", "gif"],
                accept_multiple_files=True,
                key=f"edit_add_{index}",
                label_visibility="collapsed",
            )

            new_images = [
                upload for upload in (added or [])
                if upload.name not in st.session_state.editing_added
            ]

            if new_images:

                with st.spinner(
                    "Reading the image with MedGemma..."
                    if vision_is_available()
                    else "Reading the image..."
                ):

                    for upload in new_images:

                        text, rows = read_image(upload)

                        upload.seek(0)

                        stored = save_attachment(
                            upload.read(), upload.name, active_profile
                        )

                        if not stored:
                            continue

                        save_extraction(
                            stored, text, rows, active_profile
                        )

                        st.session_state.editing_images.append(stored)

                        st.session_state.editing_added.append(
                            upload.name
                        )

                st.rerun()

            # Grow with the question rather than making the patient
            # scroll a three-line box to re-read what they asked.
            lines = content.count("\n") + len(content) // 60 + 2

            edited = st.text_area(
                "Edit this question",
                value=content,
                key=f"edit_text_{index}",
                label_visibility="collapsed",
                height=max(80, min(420, 26 * lines)),
            )

            _, cancel_column, save_column = st.columns([5, 1, 1])

            with save_column:

                if st.button(
                    "Save",
                    key=f"edit_save_{index}",
                    use_container_width=True,
                    disabled=not edited.strip(),
                ):
                    # Everything after this point answered the old
                    # question, so it is dropped rather than left
                    # sitting under a question that no longer
                    # produced it.
                    st.session_state.messages = (
                        st.session_state.messages[:index]
                    )

                    st.session_state.pending_question = edited.strip()

                    # The edited question keeps its photographs, and
                    # the context is rebuilt from whatever the set is
                    # now - one may have been added or removed.
                    st.session_state.pending_images = list(
                        st.session_state.editing_images
                    )

                    rebuild_image_context(
                        st.session_state.editing_images,
                        active_profile,
                    )

                    st.session_state.editing_index = None
                    st.session_state.editing_images = []
                    st.session_state.editing_added = []

                    st.session_state.pop(f"edit_text_{index}", None)

                    st.rerun()

            with cancel_column:

                if st.button(
                    "Cancel",
                    key=f"edit_cancel_{index}",
                    use_container_width=True,
                ):
                    st.session_state.editing_index = None
                    st.session_state.editing_images = []
                    st.session_state.editing_added = []

                    st.session_state.pop(f"edit_text_{index}", None)

                    st.rerun()

        else:

            render_attachments(message, active_profile)

            st.markdown(
                message["content"]
            )

    # Outside the bubble, as Claude places it. Inside, the button
    # stretched the bubble and left a band of empty space under the
    # question.
    if (
        message["role"] == "user"
        and not editing
        and question_was_answered(index)
    ):
        _, pencil_column = st.columns([13, 1])

        with pencil_column:
            render_edit_button(index)


component_id = (
    chat_result.get("id")
    if chat_result
    else None
)


print(
    "[COMPONENT] received:",
    "type=" + str(
        (chat_result or {}).get("type")
    ),
    "id=" + str(component_id),
    "last_id=" + str(
        st.session_state.last_component_id
    ),
)


# An edited question is answered by the same path as a typed one.
# It arrives here instead of through the chat component, which has
# no message to re-send.
pending_question = st.session_state.pending_question

st.session_state.pending_question = ""

is_new_message = (
    component_id is not None
    and
    component_id
    != st.session_state.last_component_id
)


if is_new_message or pending_question:

    if is_new_message:

        st.session_state.last_component_id = (
            component_id
        )

    result_type = (
        (chat_result or {}).get("type", "message")
        if is_new_message
        else "message"
    )

    if result_type == "audio":

        audio_base64 = chat_result.get(
            "audio"
        )

        if audio_base64:

            temp_audio_path = None

            try:

                audio_bytes = base64.b64decode(
                    audio_base64
                )

                with tempfile.NamedTemporaryFile(
                    suffix=".webm",
                    delete=False,
                ) as temp_audio:

                    temp_audio.write(
                        audio_bytes
                    )

                    temp_audio_path = (
                        temp_audio.name
                    )

                print(
                    "[VOICE] Audio received."
                )

                print(
                    "[VOICE] Audio file:",
                    temp_audio_path,
                )

                print(
                    "[VOICE] Audio size:",
                    len(audio_bytes),
                    "bytes",
                )

                with st.spinner(
                    "Transcribing..."
                ):

                    transcription = (
                        transcribe_audio(
                            temp_audio_path
                        )
                    )

                print(
                    "[VOICE] Transcription:",
                    transcription,
                )

                if transcription:

                    st.session_state.dictation_text = (
                        transcription.strip()
                    )

                else:

                    st.warning(
                        "No speech was detected. "
                        "Please try recording again."
                    )

            except Exception as error:

                print(
                    "[VOICE] Transcription error:",
                    type(error).__name__,
                )

                print(
                    error
                )

                st.error(
                    "Unable to transcribe the audio."
                )

            finally:

                if (
                    temp_audio_path
                    and
                    os.path.exists(
                        temp_audio_path
                    )
                ):

                    try:

                        os.remove(
                            temp_audio_path
                        )

                    except Exception as error:

                        print(
                            "[VOICE] Temporary file cleanup error:",
                            error,
                        )

        st.session_state.voice_completion_id += 1

        st.rerun()

    elif result_type == "message":

        question = pending_question or (
            (chat_result or {}).get("text", "").strip()
        )

        st.session_state.dictation_text = ""

        # An edited question carries no new attachments. Anything
        # already read from an image stays in session state.
        component_files = (
            (chat_result or {}).get("files", [])
            if is_new_message
            else []
        )

        uploaded_files = []

        if component_files:

            try:

                uploaded_files = (
                    convert_component_files(
                        component_files
                    )
                )

            except Exception as error:

                print(
                    "[UPLOAD] File conversion error:"
                )

                print(
                    error
                )

                st.error(
                    "Unable to process the attached file(s)."
                )

        uploaded_pdfs = [
            uploaded_file
            for uploaded_file in uploaded_files
            if (
                getattr(
                    uploaded_file,
                    "type",
                    "",
                )
                == "application/pdf"
                or
                uploaded_file.name.lower().endswith(
                    ".pdf"
                )
            )
        ]

        uploaded_images = [
            uploaded_file
            for uploaded_file in uploaded_files
            if (
                getattr(
                    uploaded_file,
                    "type",
                    "",
                )
                and
                uploaded_file.type.startswith(
                    "image/"
                )
            )
        ]

        if uploaded_pdfs:

            new_upload_count = 0

            duplicate_count = 0

            readers_used = set()

            for uploaded_pdf in uploaded_pdfs:

                filename = uploaded_pdf.name

                if (
                    filename
                    not in
                    st.session_state.uploaded_pdf_names
                ):

                    try:

                        status = st.empty()

                        def note_page(number, name=filename):
                            status.caption(
                                f"Reading page {number} of {name}..."
                            )

                        pdf_text, how = read_pdf(
                            uploaded_pdf,
                            on_page=note_page,
                        )

                        status.empty()

                        readers_used.add(how)

                        print(
                            f"[UPLOAD] {filename} read by {how}: "
                            f"{len(pdf_text)} characters."
                        )

                        report_id = add_patient_report(
                            pdf_text,
                            filename,
                            "Medical Report",
                            profile_id=active_profile,
                        )

                        st.session_state.uploaded_pdf_names.add(
                            filename
                        )

                        if report_id:

                            new_upload_count += 1

                            print(
                                "[UPLOAD] Processed:",
                                filename,
                            )

                        else:

                            duplicate_count += 1

                            print(
                                "[UPLOAD] Already stored:",
                                filename,
                            )

                    except Exception as error:

                        print(
                            "[UPLOAD] PDF processing error:",
                            error,
                        )

                        st.error(
                            f"Unable to process {filename}."
                        )

            if new_upload_count:

                # Naming the reader matters here: a scan takes half a
                # minute and reads differently from a digital PDF, so
                # the patient can see which happened.
                how_read = (
                    f" (read with {', '.join(sorted(readers_used))})"
                    if readers_used
                    else ""
                )

                st.success(
                    f"{new_upload_count} medical "
                    f"report(s) uploaded successfully{how_read}."
                )

            if duplicate_count:

                st.info(
                    f"{duplicate_count} report(s) were already "
                    "stored and were not added again."
                )

        if uploaded_images:

            image_parts = []
            image_rows = []
            image_readings = []
            saved_images = []

            with st.spinner(
                "Reading the image..."
                if not vision_is_available()
                else "Reading the image with MedGemma..."
            ):

                for uploaded_image in uploaded_images:

                    text, rows = read_image(uploaded_image)

                    # Kept so the chat can show what was asked
                    # about, not only the text read out of it.
                    try:
                        uploaded_image.seek(0)
                        stored = save_attachment(
                            uploaded_image.read(),
                            uploaded_image.name,
                            active_profile,
                        )
                        uploaded_image.seek(0)

                    except Exception as error:  # noqa: BLE001
                        print("[ATTACHMENT] Skipped:", error)
                        stored = None

                    if stored:
                        saved_images.append(stored)

                        # Kept so editing the question later does not
                        # spend another 25 seconds re-reading a photo
                        # that has not changed.
                        save_extraction(
                            stored, text, rows, active_profile
                        )

                    print(
                        "[IMAGE]",
                        uploaded_image.name,
                        f"-> {len(text)} characters,"
                        f" {len(rows)} measurement(s)",
                    )

                    if text:
                        image_parts.append(
                            f"Text read from the uploaded image "
                            f"'{uploaded_image.name}':\n{text}"
                        )

                    image_readings.append((text, rows))
                    image_rows.extend(rows)

            st.session_state.pending_images = saved_images

            new_image_context = "\n\n".join(image_parts)

            if new_image_context:

                st.session_state.image_context = new_image_context
                st.session_state.image_rows = image_rows

                # Taken from the text itself, not the wrapper around
                # it, or "uploaded" and "image" become search terms.
                # A photographed lab report contributes none: its
                # header holds the patient's name and identifiers.
                st.session_state.image_terms = photo_search_terms(
                    image_readings
                )

                st.success(
                    f"Read text from {len(image_parts)} of "
                    f"{len(uploaded_images)} image(s)."
                )

            else:

                st.info(
                    f"{len(uploaded_images)} image(s) received, but "
                    "the text could not be read clearly enough to "
                    "rely on.\n\n"
                    "A photo taken flat, filled with the label and "
                    "in even light usually reads. Glare, a curved "
                    "or crumpled surface and very small print are "
                    "what defeat it.\n\n"
                    "MediAssist reads printed text - a report, an "
                    "ingredients list, a medicine box. It cannot "
                    "interpret photographs of the body or scan "
                    "images."
                )

        # A photo stays part of the conversation after the turn it
        # was attached to. Without this the text was discarded at
        # the end of that run, so "when is it generally prescribed?"
        # reached the model with no idea which ointment was meant,
        # and searched the medical sources for "prescribed".
        image_context = st.session_state.image_context
        image_rows = st.session_state.image_rows
        image_terms = st.session_state.image_terms

        if question:

            (
                search_query,
                corrections,
            ) = get_normalized_query(
                question
            )

            print(
                "================================"
            )

            print(
                "[QUERY] Original:",
                question,
            )

            print(
                "[QUERY] Normalized:",
                search_query,
            )

            print(
                "[QUERY] Corrections:",
                corrections,
            )

            print(
                "================================"
            )

            # The welcome panel was rendered before this question
            # arrived, so clear it now that a conversation exists.
            greeting_slot.empty()

            attached_images = st.session_state.pending_images

            st.session_state.pending_images = []

            user_message = {
                "role": "user",
                "content": question,
            }

            if attached_images:
                user_message["images"] = attached_images

            st.session_state.messages.append(user_message)

            # Held in a slot so the pencil can be added once the
            # answer exists - a question with no answer yet must not
            # be editable.
            user_slot = st.empty()

            with user_slot.container():

                with st.chat_message("user"):

                    render_attachments(user_message, active_profile)

                    st.markdown(question)

            # Retrieval can take several seconds - Drive files are
            # OCR'd, the external sources are network calls - and
            # nothing was shown while it ran, so the app looked
            # frozen. Each stage reports itself as it starts.
            retrieval_status = st.status(
                "Searching your reports…",
                expanded=False,
            )

            # --------------------------------------------------
            # Retrieval
            #
            # Most questions need two different things: the value
            # from the patient's own report, and the meaning of that
            # value from reference material. "Is my cholesterol
            # high?" needs both, so every local source is gathered
            # and combined.
            #
            # This was previously an if/else cascade, which meant a
            # hit in the knowledge base stopped the patient's own
            # report being searched at all - "am I prediabetic"
            # returned a general leaflet while the patient's own
            # HbA1c sat unread.
            # --------------------------------------------------

            # Asked before retrieval, because a question about the
            # report as a whole needs the whole report, not just the
            # few chunks nearest the question.
            wants_all_values = any(
                term in question.lower()
                for term in (
                    "summary",
                    "summarise",
                    "summarize",
                    "overall",
                    "all my",
                    "full report",
                    "entire report",
                    "whole report",
                    "everything",
                )
            )

            kb_docs = []
            patient_docs = []
            references = []
            drive_references = []

            # The patient's own material and general reference
            # material are kept apart all the way to the prompt.
            # Merged, the model read a PubMed abstract as though
            # it were the patient's report.
            patient_parts = []
            reference_parts = []
            sources_used = []

            # 1. The patient's own reports. Local, about 8 ms.
            patient_context = ""
            patient_score = None

            try:

                (
                    patient_docs,
                    patient_context,
                    _,
                    patient_score,
                ) = search_patient_reports(
                    search_query,
                    profile_id=active_profile,
                    whole_report=wants_all_values,
                )

                if patient_context:
                    patient_parts.append(patient_context)
                    sources_used.append("Patient Reports")

            except Exception as error:

                print("[PATIENT] Search error:", error)

            # 2. The offline knowledge base. Also local and fast.
            retrieval_status.update(
                label="Searching the medical knowledge base…"
            )

            try:

                (
                    kb_docs,
                    local_context,
                    local_source,
                ) = search_local_database(search_query)

                if len(local_context.strip()) > MIN_CONTEXT_LENGTH:

                    reference_parts.append(local_context)

                    sources_used.append(
                        local_source or "Local Knowledge Base"
                    )

                else:
                    kb_docs = []

            except Exception as error:

                print("[RAG] Local database error:", error)

            # 3. The external sources are the only ones needing the
            #    network, so they are asked last and only when the
            #    local material has not already answered the
            #    question.
            answered_locally = (
                # The report clearly answers the question.
                is_strong_match(patient_score)
                # A summary of the patient's own report is answered
                # by the report. Searching PubMed for "summary"
                # spends a network call on nothing.
                or (wants_all_values and bool(patient_context))
                # The offline knowledge base already covers it.
                or any(
                    source != "Patient Reports"
                    for source in sources_used
                )
            )

            if answered_locally:

                print(
                    "[SEARCH] Answered from local sources - "
                    "external sources not called."
                )

            else:

                # MedlinePlus, PubMed and DailyMed are keyword
                # services and return nothing for a full sentence,
                # so the question is reduced to its medical terms.
                # Vector search above keeps the whole sentence,
                # which its embeddings need.
                retrieval_status.update(
                    label="Searching MedlinePlus, PubMed and DailyMed…"
                )

                keyword_query = build_search_keywords(search_query)

                # "When is it generally prescribed?" names nothing to
                # search for - the medicine's name is in the photo,
                # not the question. Terms read from the image are
                # added so the sources are asked about the right
                # thing, with the question's own terms kept first.
                if image_terms:

                    from_image = [
                        term
                        for term in image_terms.split()
                        if term not in keyword_query
                    ][:2]

                    if from_image:

                        keyword_query = " ".join(
                            [keyword_query] + from_image
                        ).strip()

                        print(
                            "[QUERY] Added from the image:",
                            " ".join(from_image),
                        )

                print(
                    "[QUERY] Keywords for external sources:",
                    keyword_query,
                )

                try:

                    (
                        external_context,
                        external_source,
                        references,
                    ) = search_medical_sources(
                        keyword_query,
                        question=search_query,
                    )

                    if external_context:

                        reference_parts.append(external_context)

                        sources_used.append(external_source)

                except Exception as error:

                    print("[MEDICAL SOURCES] Search error:", error)

                    references = []

            drive_is_connected = bool(
                get_google_drive_credentials()
                and folder_id
            )

            # When a folder is connected it is always searched.
            # The MCP tool decides, by relevance score, whether
            # anything in the folder is actually used, so there is
            # no question-intent guessing here.
            if drive_is_connected:

                retrieval_status.update(
                    label=(
                        "Reading your Google Drive folder… "
                        "scanned reports are read with OCR, which "
                        "takes a moment the first time."
                    )
                )

                print(
                    "[DRIVE] Searching connected folder "
                    f"(other_sources={sources_used or None})."
                )

                print(
                    "[DRIVE] Folder ID:",
                    folder_id,
                )

                try:

                    (
                        access_token,
                        refreshed_credentials,
                    ) = get_valid_access_token(
                        get_google_drive_credentials()
                    )

                    set_google_drive_credentials(
                        refreshed_credentials
                    )

                    print(
                        "[DRIVE] Access token valid."
                    )

                    drive_result = (
                        search_google_drive_reports(
                            search_query,
                            folder_id,
                            access_token,
                        )
                    )

                    for note in drive_result.get(
                        "debug",
                        [],
                    ):
                        print("[DRIVE]", note)

                    drive_context = (
                        drive_result.get(
                            "context",
                            "",
                        )
                    )

                    drive_references = (
                        drive_result.get(
                            "references",
                            [],
                        )
                    )

                    if drive_context:

                        # A document in the patient's own Drive
                        # folder is their own material, not general
                        # reference.
                        patient_parts.append(
                            "--- Google Drive reports ---\n\n"
                            f"{drive_context}"
                        )

                        sources_used.append("Google Drive")

                        print(
                            "[DRIVE] Relevant context found."
                        )

                    else:

                        print(
                            "[DRIVE] No relevant report context found."
                        )

                except GoogleDriveAuthError as error:

                    print(
                        "[DRIVE] Authentication error:",
                        error,
                    )

                    clear_google_drive_credentials()

                    st.warning(
                        "Google Drive connection expired. "
                        "Please reconnect if you want to search "
                        "your personal reports."
                    )

                except GoogleDriveMCPError as error:

                    print(
                        "[DRIVE] MCP error:",
                        error,
                    )

                    for note in getattr(error, "debug", []):
                        print("[DRIVE]", note)

                    st.warning(
                        "Google Drive could not be searched. "
                        "The other medical sources will still be used."
                    )

                except Exception as error:

                    print(
                        "[DRIVE] Unexpected error:",
                        type(error).__name__,
                    )

                    print(
                        error
                    )

                    st.warning(
                        "Google Drive search was unavailable. "
                        "The other medical sources will still be used."
                    )

            # Text read from an image attached to this question is
            # the most direct answer to it, so it leads the context.
            # It is the patient's own material too.
            if image_context:

                patient_parts.insert(0, image_context)

                sources_used.insert(0, "Uploaded image")

                print(
                    "[IMAGE] Added",
                    len(image_context),
                    "characters of image text to the context.",
                )

            patient_context = "\n\n".join(patient_parts)

            reference_context = "\n\n".join(reference_parts)

            # Kept for the checks that only ask whether anything at
            # all was found, and for the follow-up carry-over.
            context = "\n\n".join(
                part for part in (patient_context, reference_context)
                if part
            )

            source_used = (
                " + ".join(sources_used)
                if sources_used
                else "No reliable source found"
            )

            retrieval_status.update(
                label=f"Searched: {source_used}",
                state="complete",
            )

            print(
                "========== SEARCH RESULT =========="
            )

            print(
                "[SEARCH] Source:",
                source_used,
            )

            print(
                "[SEARCH] Query:",
                search_query,
            )

            print(
                "[SEARCH] Context length:",
                len(context),
            )

            print(
                "===================================="
            )

            recent_history = (
                st.session_state.messages[:-1]
            )

            if (
                not context
                and st.session_state.get("last_context")
                and recent_history
            ):

                # A follow-up such as "and the second one?" retrieves
                # nothing of its own, so the previous turn's material
                # is carried over - each half back to its own side.
                patient_context = st.session_state.last_context

                reference_context = st.session_state.get(
                    "last_reference_context", ""
                )

                context = patient_context or reference_context

                source_used = (
                    f"{st.session_state.get('last_source') or 'Previous answer'}"
                    " (follow-up)"
                )

                print(
                    "[CONTEXT] Reusing previous context for "
                    "follow-up question."
                )

            # Only the patient's own documents are checked. Run over
            # the general reference material as well, this parsed
            # numbers out of PubMed abstracts and presented them as
            # verified findings about the patient.
            # The sex was read from the whole report at upload, so
            # a middle chunk retrieved on its own still resolves the
            # sex-specific haematology ranges. Reports disagreeing
            # means no basis to choose, so nothing is passed.
            stored_sexes = {
                doc.metadata.get("patient_sex")
                for doc in patient_docs
                if doc.metadata.get("patient_sex")
            }

            known_sex = (
                stored_sexes.pop() if len(stored_sexes) == 1 else None
            )

            verified_all = merge_findings(
                verify_lab_ranges(
                    patient_context,
                    patient_sex=known_sex,
                ),
                # Rows the vision model read from a photo. It reports
                # what is printed; the comparison is still done in
                # Python.
                verify_extracted_rows(image_rows, known_sex),
            )

            for finding in verified_all:
                print("[VERIFY]", finding["text"])

            question_words = set(
                re.findall(r"[a-z]{3,}", question.lower())
            )

            def names_in_question(finding):
                return bool(
                    question_words
                    & set(
                        re.findall(
                            r"[a-z]{3,}",
                            finding["name"].lower(),
                        )
                    )
                )

            if verified_all and not wants_all_values:

                verified_findings = [
                    finding
                    for finding in verified_all
                    if names_in_question(finding)
                ]

            else:
                # A summary keeps every finding, but the ones the
                # question names are put first so the cap can only
                # ever drop the least relevant. sorted() is stable,
                # so report order survives within each group.
                verified_findings = sorted(
                    verified_all,
                    key=lambda finding: not names_in_question(finding),
                )

            # Matches the verifier's own limit, so the two cannot
            # silently disagree. At 15 a summary of several reports
            # lost the later ones - their values still reached the
            # model, with no check to judge them by, and it invented
            # verdicts: a cholesterol of 207 "in the desirable
            # range". All 25 cost about 250 tokens of 8192.
            verified_findings = verified_findings[
                :MAX_VERIFIED_FINDINGS
            ]

            if verified_findings:
                print(
                    "[VERIFY] Using",
                    len(verified_findings),
                    "finding(s) for this question.",
                )

            (
                system_prompt,
                prompt,
            ) = build_healthcare_prompt(
                patient_context,
                question,
                history=recent_history,
                verified=verified_findings,
                reference_context=reference_context,
            )

            answer = ""

            with st.chat_message(
                "assistant"
            ):

                # Created before the answer streams. Streamlit leaves
                # the previous run's elements on screen until the
                # script reaches their position again, so the last
                # answer and its source expanders stayed visible under
                # the new one while it was still being written.
                #
                # All three are st.empty(), which clears its slot the
                # moment it is created. An empty st.container() does
                # not: it opens the block but leaves whatever was
                # rendered inside it last time, which is why the
                # caption disappeared here and the answer did not.
                answer_slot = st.empty()
                caption_slot = st.empty()
                sources_slot = st.empty()

                with answer_slot.container():
                    try:

                        if not context:

                            answer = (
                                "I could not find enough "
                                "reliable information in the "
                                "available medical sources "
                                "for this question."
                            )

                            source_used = (
                                "No reliable source found"
                            )

                            references = []

                            st.markdown(answer)

                        else:

                            # Streamed so the answer appears as it is
                            # written rather than after a long pause.
                            with st.spinner("Thinking..."):

                                answer = st.write_stream(
                                    stream_answer(
                                        prompt,
                                        system_prompt=system_prompt,
                                    )
                                )

                            if isinstance(answer, list):
                                answer = "".join(
                                    str(part) for part in answer
                                )

                            answer = (answer or "").strip()

                            st.session_state.last_context = (
                                patient_context
                            )

                            st.session_state.last_reference_context = (
                                reference_context
                            )

                            st.session_state.last_source = (
                                source_used
                            )

                    except OllamaServiceError as error:

                        answer = str(
                            error
                        )

                        st.warning(
                            answer
                        )

                        print(
                            "[LLM] Ollama error:",
                            error,
                        )

                    except Exception as error:

                        answer = (
                            "Error generating answer. "
                            "Please check the terminal "
                            "logs for details."
                        )

                        print(
                            "[LLM] Unexpected error:",
                            type(error).__name__,
                        )

                        print(
                            error
                        )

                        st.error(
                            answer
                        )

                caption_slot.caption(
                    f"Source Used: {source_used}"
                )

                with sources_slot.container():
                    source_label = str(source_used)

                    render_document_sources(
                        "Knowledge Base Sources",
                        kb_docs,
                        "source",
                        "Unknown",
                    )

                    render_document_sources(
                        "Patient Report Sources",
                        patient_docs,
                        "report_name",
                        "Unknown Report",
                    )

                    render_reference_links(
                        "Google Drive Reports",
                        drive_references,
                        "Google Drive report",
                    )

                    # (source name as it appears in source_used, expander
                    # label, fallback title)
                    for source_name, label, fallback in (
                        ("MedlinePlus", "MedlinePlus References",
                         "MedlinePlus topic"),
                        ("PubMed", "PubMed References",
                         "PubMed article"),
                        ("ClinicalTrials.gov", "Clinical Trials",
                         "Clinical trial"),
                    ):
                        render_reference_links(
                            label,
                            references if source_name in source_label else [],
                            fallback,
                        )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            save_conversation(
                st.session_state.current_conversation_id,
                st.session_state.messages,
                profile_id=active_profile,
            )

            # Only the newest few conversations are kept, so this is
            # where an image can lose its last reference.
            prune_unreferenced(
                load_conversations(active_profile), active_profile
            )

            # The answer exists now, so the question can be edited.
            # Redrawing the slot adds the pencil without disturbing
            # the answer or the sources already on screen.
            with user_slot.container():

                with st.chat_message("user"):

                    render_attachments(user_message, active_profile)

                    st.markdown(question)

                # Outside the bubble, matching the history loop.
                _, pencil_column = st.columns([13, 1])

                with pencil_column:
                    render_edit_button(
                        len(st.session_state.messages) - 2
                    )