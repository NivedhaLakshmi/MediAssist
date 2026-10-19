"""Keep the embedding model from contacting Hugging Face.

Sentence-transformers checks the Hub for updates every time it loads
a model. That leaks the fact the app is running and makes startup
depend on a network round trip, neither of which suits a local-first
application.

This lives apart from ``embedding_service`` so that tools such as
``privacy_audit.py`` can report the setting without importing - and
therefore loading - the model itself.
"""

import os


CACHE_DIRECTORY_NAME = (
    "models--sentence-transformers--all-MiniLM-L6-v2"
)


def model_is_cached():
    """True when the embedding weights are already on this machine."""

    hub = os.path.join(
        os.environ.get(
            "HF_HOME",
            os.path.expanduser("~/.cache/huggingface"),
        ),
        "hub",
    )

    return os.path.isdir(
        os.path.join(hub, CACHE_DIRECTORY_NAME)
    )


def is_offline():
    return os.environ.get("HF_HUB_OFFLINE") in ("1", "true", "True")


def prefer_offline():
    """Switch on offline mode, but only once the weights are cached.

    A first run still needs the network to download the model. An
    explicit HF_HUB_OFFLINE in .env always wins.

    Returns the reason for the decision, for reporting.
    """

    if os.environ.get("HF_HUB_OFFLINE") is not None:
        return "set explicitly in the environment"

    if not model_is_cached():
        return "model not cached yet - first download needs the network"

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    return "enabled automatically (model already cached)"
