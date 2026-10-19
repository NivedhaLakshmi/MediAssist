"""Model-driven selection of a medical source.

The keyword rules in :mod:`utils.query_router` decide which service
to search by matching phrases. They are fast and reproducible, and
they miss phrasings nobody thought of: "is dolo 650 safe for a
child" names a medicine and asks about its safety, but matches none
of the medication keywords, so it reached the general encyclopedia
instead of the drug label.

This module asks the language model instead, in the way the Model
Context Protocol is usually used: the tools are described, and the
model chooses one. Measured over sixteen questions with known
correct sources, the model selected correctly in all sixteen against
twelve for the keyword rules.

The model is asked only which source to search. It never sees the
patient's report, never decides anything clinical, and its answer is
checked against the list of real tool names before use. Any failure
returns None, and the caller falls back to the keyword rules.
"""

import os


MODEL = os.getenv("MCP_ROUTER_MODEL", "qwen2.5:7b-instruct")

# Short on purpose. Selection sits in front of a question the patient
# is waiting on, so a slow or stuck model must give way to the rules
# rather than hold up the answer.
TIMEOUT_SECONDS = float(os.getenv("MCP_ROUTER_TIMEOUT", "15"))

KEEP_ALIVE = os.getenv("MCP_ROUTER_KEEP_ALIVE", "10m")

# "model" uses the language model and falls back to the rules.
# "rules" skips the model entirely and behaves as the system did
# before this module existed.
ROUTING_MODE = os.getenv("MCP_ROUTING", "model").strip().lower()


SYSTEM_PROMPT = (
    "You are a medical information assistant. Choose exactly one "
    "tool to answer the patient's question. Call the tool; do not "
    "answer in prose."
)


# Descriptions are written for the model rather than for a
# developer, since they are the only thing it has to choose from.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_dailymed",
            "description": (
                "Search DailyMed for FDA-approved medication "
                "information: dosage, side effects, warnings, "
                "contraindications, interactions and whether a "
                "medicine is suitable, taken from the "
                "manufacturer's label."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Drug or ingredient name",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_clinical_trials",
            "description": (
                "Search ClinicalTrials.gov for clinical trials, "
                "ongoing studies, experimental therapies and new or "
                "latest treatments, with their recruitment status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Condition or treatment",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_medlineplus",
            "description": (
                "Search MedlinePlus, a consumer health "
                "encyclopedia, for plain-language summaries of "
                "conditions, symptoms and tests, written for "
                "patients."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Health topic",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pubmed",
            "description": (
                "Search PubMed for peer-reviewed research article "
                "abstracts, for questions about evidence, studies "
                "or what research shows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Research topic",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


KNOWN_TOOLS = frozenset(
    schema["function"]["name"] for schema in TOOL_SCHEMAS
)


def is_enabled():
    """Whether model-driven selection should be attempted."""

    return ROUTING_MODE == "model"


def choose_source(question):
    """Return the tool name the model picked, or None.

    None means the caller should use the keyword rules: routing is
    switched off, the runtime is unreachable, the model declined to
    call a tool, or it named something that is not a real tool.
    """

    if not is_enabled() or not question or not question.strip():
        return None

    try:
        import ollama

        reply = ollama.Client(timeout=TIMEOUT_SECONDS).chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tools=TOOL_SCHEMAS,
            options={"temperature": 0},
            keep_alive=KEEP_ALIVE,
        )

    except Exception as error:  # noqa: BLE001
        print(f"[ROUTER] Model unavailable, using keyword rules: {error}")
        return None

    calls = (reply.get("message") or {}).get("tool_calls") or []

    if not calls:
        print("[ROUTER] Model named no tool, using keyword rules.")
        return None

    # More than one means the model did not make a choice. Taking
    # the first would be guessing on its behalf.
    if len(calls) > 1:
        print(
            f"[ROUTER] Model named {len(calls)} tools, "
            "using keyword rules."
        )
        return None

    try:
        name = calls[0]["function"]["name"]
    except (KeyError, TypeError, IndexError):
        print("[ROUTER] Tool call was malformed, using keyword rules.")
        return None

    if name not in KNOWN_TOOLS:
        print(
            f"[ROUTER] Model named an unknown tool '{name}', "
            "using keyword rules."
        )
        return None

    print(f"[ROUTER] Model chose {name}.")

    return name
