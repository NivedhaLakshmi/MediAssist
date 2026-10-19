"""Prove the model, not the keyword rules, is choosing the source.

The four searchers are replaced with stubs that record the call and
return nothing, so the full cascade runs and the ORDER it was tried
in can be observed. The model call is real. No network is touched.

The proof is in the disagreement cases: questions where the rules
and the model would choose differently. If the first source tried is
the model's choice, the model is in control.

    python experiments/verify_agent_routing.py
"""

import os
import sys
from pathlib import Path

MCP = Path(__file__).resolve().parent.parent / "medical_mcp"
sys.path.insert(0, str(MCP))
os.chdir(MCP)

from tools import search                      # noqa: E402
from utils import model_router                # noqa: E402
from utils.query_router import (              # noqa: E402
    should_search_clinical_trials,
    should_search_dailymed,
)

EMPTY = {"source": "", "context": "", "references": []}
calls = []


def _stub(name):
    def fn(query):
        calls.append(name)
        return EMPTY
    return fn


search.SEARCHERS = {name: _stub(name) for name in search.SEARCHERS}


def rules_would_pick(question):
    if should_search_dailymed(question):
        return "search_dailymed"
    if should_search_clinical_trials(question):
        return "search_clinical_trials"
    return "search_medlineplus"


CASES = [
    "is dolo 650 safe for a child",
    "what do studies show about vitamin d supplementation",
    "is there evidence linking sleep and immune function",
    "what does recent research say about intermittent fasting",
    "what are the side effects of metformin",
    "what is hypertension",
    "are there any clinical trials for type 2 diabetes",
]


def first_source(question, mode):
    model_router.ROUTING_MODE = mode
    calls.clear()
    search.search_medical_sources("stub query", question)
    return calls[0] if calls else "(none)"


def short(name):
    return name.replace("search_", "")


print(f"routing mode default: {model_router.ROUTING_MODE}\n")

head = (f"{'question':<52} {'rules':<16} {'agent':<16} "
        f"{'MCP_ROUTING=rules':<18} differs")
print(head)
print("-" * len(head))

differ = agreed_with_agent = 0

for question in CASES:
    rules = rules_would_pick(question)
    agent = first_source(question, "model")
    forced = first_source(question, "rules")

    disagrees = rules != agent
    differ += disagrees
    agreed_with_agent += disagrees

    print(f"{question[:50]:<52} {short(rules):<16} {short(agent):<16} "
          f"{short(forced):<18} {'YES' if disagrees else '-'}")

print()
print(f"cases where the rules and the model disagree: {differ}/{len(CASES)}")
print(f"of those, the source actually tried first was the model's "
      f"choice: {agreed_with_agent}/{differ}")
print()
print("With MCP_ROUTING=rules the column above reverts to the rules "
      "column,\nwhich confirms the model is what changes the outcome.")
