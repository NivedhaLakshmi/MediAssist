"""Compare model-driven source selection against the rule-based router.

MediAssist selects a medical source in Python. The usual way to use
MCP is to give the model the tool list and let it choose. This script
measures both on the same questions so the choice can be reported as
a result rather than a preference.

Nothing here is imported by MediAssist. The rule-based router is
imported read-only to obtain its decision; no product code changes.

    python experiments/probe_tool_selection.py
    python experiments/probe_tool_selection.py --repeat 3
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "medical_mcp"))

from utils.query_router import (          # noqa: E402
    should_search_clinical_trials,
    should_search_dailymed,
)

MODEL = "qwen2.5:7b-instruct"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The four tools, described as an agentic setup would describe them.
# Wording is taken from the real tool docstrings so the model is not
# handicapped by a worse description than the server actually gives.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_dailymed",
            "description": (
                "Search DailyMed for FDA-approved medication "
                "information: dosage, side effects, warnings, "
                "contraindications and drug interactions, taken from "
                "the manufacturer's label."
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
                "Search MedlinePlus, a consumer health encyclopedia, "
                "for plain-language summaries of conditions, symptoms "
                "and tests, written for patients."
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
                "abstracts, for questions about evidence, studies or "
                "what research shows."
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

# (question, expected tool). Expected source follows the system's own
# definition of each service's role.
CASES = [
    ("what are the side effects of metformin",            "search_dailymed"),
    ("what is the usual dosage for amoxicillin",           "search_dailymed"),
    ("can I take ibuprofen with warfarin",                 "search_dailymed"),
    ("what are the warnings for atorvastatin",             "search_dailymed"),
    ("is dolo 650 safe for a child",                       "search_dailymed"),

    ("are there any clinical trials for type 2 diabetes",  "search_clinical_trials"),
    ("what new treatments are being studied for psoriasis","search_clinical_trials"),
    ("is there an ongoing study for alzheimer's disease",  "search_clinical_trials"),
    ("any experimental therapy for sickle cell",           "search_clinical_trials"),

    ("what is hypertension",                               "search_medlineplus"),
    ("what causes anaemia",                                "search_medlineplus"),
    ("how is asthma usually managed",                      "search_medlineplus"),
    ("what does a thyroid test measure",                   "search_medlineplus"),

    ("what does recent research say about intermittent fasting", "search_pubmed"),
    ("is there evidence linking sleep and immune function",      "search_pubmed"),
    ("what do studies show about vitamin d supplementation",     "search_pubmed"),
]

SYSTEM = (
    "You are a medical information assistant. Choose exactly one "
    "tool to answer the patient's question. Call the tool; do not "
    "answer in prose."
)


def rule_based(question):
    """The deterministic router's decision, in cascade order."""

    if should_search_dailymed(question):
        return "search_dailymed"
    if should_search_clinical_trials(question):
        return "search_clinical_trials"
    return "search_medlineplus"   # general fallback, PubMed only if empty


def model_based(question):
    """Ask the model to pick a tool. Returns (tool_name, seconds)."""

    import ollama

    started = time.monotonic()

    try:
        reply = ollama.Client(timeout=120).chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
            ],
            tools=TOOLS,
            options={"temperature": 0},
            keep_alive="10m",
        )
    except Exception as error:  # noqa: BLE001
        return f"ERROR: {error}", round(time.monotonic() - started, 1)

    seconds = round(time.monotonic() - started, 1)
    calls = (reply.get("message") or {}).get("tool_calls") or []

    if not calls:
        return "NO_TOOL_CALLED", seconds

    names = [c["function"]["name"] for c in calls]

    if len(names) > 1:
        return "MULTIPLE:" + "+".join(names), seconds

    return names[0], seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    model_right = rule_right = 0
    total = 0
    model_seconds = 0.0
    inconsistent = 0

    for question, expected in CASES:
        rule = rule_based(question)

        picks = []
        for _ in range(args.repeat):
            pick, secs = model_based(question)
            picks.append(pick)
            model_seconds += secs

        pick = picks[0]
        stable = len(set(picks)) == 1
        if not stable:
            inconsistent += 1

        total += 1
        m_ok = pick == expected
        r_ok = rule == expected
        model_right += m_ok
        rule_right += r_ok

        print(
            f"{'OK ' if m_ok else 'BAD'} model={pick:<24} "
            f"{'OK ' if r_ok else 'BAD'} rule={rule:<24} "
            f"| {question[:46]}"
        )

        rows.append({
            "question": question,
            "expected": expected,
            "model": pick,
            "model_all": picks,
            "model_correct": m_ok,
            "rule": rule,
            "rule_correct": r_ok,
            "stable": stable,
        })

    print()
    print(f"model-driven : {model_right}/{total} "
          f"({100*model_right/total:.0f}%)  "
          f"avg {model_seconds/(total*args.repeat):.1f}s per decision")
    print(f"rule-based   : {rule_right}/{total} "
          f"({100*rule_right/total:.0f}%)  "
          f"effectively instant")
    if args.repeat > 1:
        print(f"unstable across {args.repeat} runs: {inconsistent}/{total}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"tool-selection-{stamp}.json"
    out.write_text(json.dumps({
        "model": MODEL,
        "run": datetime.now().isoformat(timespec="seconds"),
        "repeats": args.repeat,
        "model_correct": model_right,
        "rule_correct": rule_right,
        "total": total,
        "cases": rows,
    }, indent=2))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
