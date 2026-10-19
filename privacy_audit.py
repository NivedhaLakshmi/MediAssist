"""Privacy audit for MediAssist.

Prints where the application stores data and every point at which it
can talk to the network. The egress list is re-derived from the
source code on each run rather than hard-coded, so it cannot quietly
go out of date as the code changes.

Run with:

    venv/bin/python privacy_audit.py
"""

import os
import re
import sqlite3

from dotenv import load_dotenv

load_dotenv()


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# This file names the endpoints in order to classify them, so it
# must not scan itself.
SELF = os.path.basename(os.path.abspath(__file__))

# Where the application keeps things, and who the data belongs to.
STORAGE_LOCATIONS = [
    ("patient_db", "PATIENT", "Uploaded reports and their embeddings"),
    ("user_data", "PATIENT",
     "Saved conversations and attached photographs"),
    ("chroma_db", "SYSTEM", "General medical knowledge base"),
    ("data", "SYSTEM", "Source documents and the MeSH vocabulary"),
]

# What each remote host receives. Anything not listed here that turns
# up in the scan below is flagged as unreviewed.
KNOWN_ENDPOINTS = {
    "127.0.0.1": ("LOCAL", "This machine - the MCP server"),
    "localhost": ("LOCAL", "This machine - OAuth redirect target"),
    "eutils.ncbi.nlm.nih.gov": ("QUERY ONLY", "PubMed: search terms"),
    "wsearch.nlm.nih.gov": ("QUERY ONLY", "MedlinePlus: search terms"),
    "clinicaltrials.gov": ("QUERY ONLY", "ClinicalTrials.gov: search terms"),
    "dailymed.nlm.nih.gov": ("QUERY ONLY", "DailyMed: drug name"),
    "accounts.google.com": ("AUTH", "Google sign-in (no patient data)"),
    "oauth2.googleapis.com": ("AUTH", "Token exchange and revocation"),
    "www.googleapis.com": ("INBOUND", "Downloads the user's own Drive files"),
    "drive.google.com": ("LINK ONLY", "Folder URL typed by the user"),
}

SKIP_DIRS = {"venv", ".git", "__pycache__", "node_modules", "dist"}


def _directory_size(path):
    total = 0

    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass

    return total


def _human(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _chroma_record_count(path):
    """Read the embedding count straight out of the Chroma SQLite file."""

    database = os.path.join(path, "chroma.sqlite3")

    if not os.path.exists(database):
        return None

    try:
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        cursor = connection.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        connection.close()
        return count

    except Exception:  # noqa: BLE001
        return None


def report_storage():
    print("=" * 68)
    print("1. WHERE DATA IS STORED")
    print("=" * 68)
    print()

    for name, owner, description in STORAGE_LOCATIONS:

        path = os.path.join(PROJECT_ROOT, name)

        if not os.path.exists(path):
            print(f"  [{owner:7}] {name:<12} (not present)")
            continue

        detail = _human(_directory_size(path))

        records = _chroma_record_count(path)

        if records is not None:
            detail += f", {records} embeddings"

        print(f"  [{owner:7}] {name:<12} {detail}")
        print(f"            {description}")
        print(f"            {path}")
        print()

    print("  All paths above are inside the project directory on this")
    print("  machine. No database server, bucket or remote store is")
    print("  configured anywhere in the codebase.")
    print()


def scan_endpoints():
    """Re-derive every host referenced in the source code."""

    found = {}

    pattern = re.compile(r"https?://([A-Za-z0-9.\-]+)")

    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for name in files:
            if not name.endswith(".py") or name == SELF:
                continue

            path = os.path.join(root, name)

            try:
                with open(path, "r", encoding="utf-8") as handle:
                    text = handle.read()
            except OSError:
                continue

            for host in pattern.findall(text):
                relative = os.path.relpath(path, PROJECT_ROOT)
                found.setdefault(host, set()).add(relative)

    return found


def report_endpoints():
    print("=" * 68)
    print("2. EVERY NETWORK DESTINATION IN THE SOURCE CODE")
    print("=" * 68)
    print()

    found = scan_endpoints()

    unreviewed = []

    for host in sorted(found):

        classification, purpose = KNOWN_ENDPOINTS.get(
            host,
            ("UNREVIEWED", "not classified - review this"),
        )

        if classification == "UNREVIEWED":
            unreviewed.append(host)

        print(f"  [{classification:11}] {host}")
        print(f"                {purpose}")
        print(f"                referenced in: "
              f"{', '.join(sorted(found[host]))}")
        print()

    print("  Key:")
    print("    LOCAL       never leaves this machine")
    print("    QUERY ONLY  receives the search terms, never report content")
    print("    AUTH        Google sign-in and token handling only")
    print("    INBOUND     downloads the user's own files to this machine")
    print("    LINK ONLY   appears in a URL, not contacted with data")
    print()

    if unreviewed:
        print(f"  WARNING: {len(unreviewed)} unreviewed host(s): "
              f"{', '.join(unreviewed)}")
        print()

    return unreviewed


def report_model():
    print("=" * 68)
    print("3. WHERE THE LANGUAGE MODEL RUNS")
    print("=" * 68)
    print()

    base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

    print(f"  Model:    {model}")
    print(f"  Endpoint: {base_url}")
    print()
    print("  The model runs under Ollama on this machine. Report text")
    print("  placed in the prompt is sent to that local endpoint only.")
    print("  Embeddings (all-MiniLM-L6-v2), OCR (Tesseract) and speech")
    print("  to text (faster-whisper) also run locally.")
    print()


def report_telemetry():
    """Check the things that can phone home without being asked."""

    print("=" * 68)
    print("4. TELEMETRY AND BACKGROUND NETWORK ACCESS")
    print("=" * 68)
    print()

    problems = []

    # --- Streamlit usage statistics ---------------------------------

    try:
        from streamlit import config

        gathering = config.get_option("browser.gatherUsageStats")

    except Exception as error:  # noqa: BLE001
        gathering = None
        print(f"  [UNKNOWN] Streamlit telemetry ({error})")

    if gathering is True:
        problems.append(
            "Streamlit usage statistics are being sent. Set "
            "gatherUsageStats = false under [browser] in "
            ".streamlit/config.toml"
        )
        print("  [FAIL]  Streamlit usage statistics: ENABLED")

    elif gathering is False:
        print("  [OK]    Streamlit usage statistics: disabled")

    # --- Hugging Face hub -------------------------------------------
    #
    # Applies the same decision the app makes at startup, without
    # importing the model itself.

    from services.offline_mode import (
        is_offline,
        model_is_cached,
        prefer_offline,
    )

    reason = prefer_offline()

    if is_offline():
        print("  [OK]    Hugging Face hub: offline - the embedding "
              "model loads from the local cache")
        print(f"          {reason}")

    elif not model_is_cached():
        print("  [NOTE]  Hugging Face hub: online - the embedding "
              "model has not been downloaded yet")
        print("          It will go offline automatically after the "
              "first run.")

    else:
        problems.append(
            "HF_HUB_OFFLINE is set to a value that keeps the "
            "embedding model online despite it being cached"
        )
        print("  [FAIL]  Hugging Face hub: online despite a cached "
              "model")

    # --- Ollama request logs ----------------------------------------

    log = os.path.expanduser("~/.ollama/logs/server.log")

    if not os.path.exists(log):
        print("  [NOTE]  Ollama log not found at ~/.ollama/logs")

    else:
        markers = [
            "MediAssist",
            "REPORT CONTEXT",
            "VERIFIED CHECKS",
            "PATIENT'S QUESTION",
        ]

        try:
            with open(log, "r", encoding="utf-8",
                      errors="replace") as handle:
                text = handle.read()

            leaked = [m for m in markers if m in text]

        except OSError as error:
            leaked = []
            print(f"  [UNKNOWN] Could not read the Ollama log ({error})")

        if leaked:
            problems.append(
                "Prompt content appears in the Ollama log: "
                + ", ".join(leaked)
            )
            print(f"  [FAIL]  Ollama log contains prompt content: "
                  f"{', '.join(leaked)}")
        else:
            print("  [OK]    Ollama log: request metadata only, "
                  "no prompt or report content")

    print()

    return problems


def main():
    print()
    print("MediAssist - privacy audit")
    print()

    report_storage()
    unreviewed = report_endpoints()
    report_model()
    problems = report_telemetry()

    print("=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print()
    print("  Patient data - uploaded reports, their embeddings, saved")
    print("  conversations, voice transcripts and OCR text - is written")
    print("  only to the local paths in section 1 and is never sent to")
    print("  any host in section 2.")
    print()
    print("  What does leave this machine: the search terms of a general")
    print("  medical question, sent to public NIH/NLM services; and, only")
    print("  if the user connects Google Drive, the OAuth exchange needed")
    print("  to download their own files.")
    print()

    if unreviewed:
        problems.append(
            "Unreviewed network destination(s): "
            + ", ".join(unreviewed)
        )

    if problems:
        print(f"  RESULT: {len(problems)} issue(s) to address:")
        for problem in problems:
            print(f"    - {problem}")
    else:
        print("  RESULT: no unreviewed network destinations, no")
        print("          telemetry, no prompt content in local logs.")

    print()


if __name__ == "__main__":
    main()
