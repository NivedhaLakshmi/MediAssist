from mcp_instance import mcp

from tools.medlineplus import search_medlineplus
from tools.pubmed import search_pubmed
from tools.clinicaltrials import search_clinical_trials
from tools.dailymed import search_dailymed

from utils.model_router import choose_source

from utils.query_router import (
    should_search_clinical_trials,
    should_search_dailymed,
)


# Name to function, so a name chosen by the model can be turned into
# a call without a chain of comparisons.
SEARCHERS = {
    "search_dailymed": search_dailymed,
    "search_clinical_trials": search_clinical_trials,
    "search_medlineplus": search_medlineplus,
    "search_pubmed": search_pubmed,
}


# For the log line only.
DISPLAY_NAMES = {
    "search_dailymed": "DailyMed",
    "search_clinical_trials": "ClinicalTrials.gov",
    "search_medlineplus": "MedlinePlus",
    "search_pubmed": "PubMed",
}


EMPTY_RESULT = {
    "source": "",
    "context": "",
    "references": [],
}


def _keyword_order(intent):
    """The sources the keyword rules would try, most specific first.

    MedlinePlus used to be tried before everything else and returned
    on any hit. It is a general consumer-health encyclopedia, so it
    answers almost any medical word with something - which meant
    DailyMed, ClinicalTrials.gov and PubMed were never reached. A
    question about a tablet came back with "Medicines", "Drug
    Reactions" and "Anabolic Steroids" instead of the drug's label.

    So a question that names its own kind of source goes there
    first, and MedlinePlus is the general fallback it was always
    better suited to be.
    """

    order = []

    if should_search_dailymed(intent):
        order.append("search_dailymed")

    if should_search_clinical_trials(intent):
        order.append("search_clinical_trials")

    order.append("search_medlineplus")
    order.append("search_pubmed")

    return order


@mcp.tool
def search_medical_sources(query: str, question: str = ""):
    """Search the medical sources, most promising first.

    Selection is attempted twice. The language model is asked which
    source fits the question, in the way the Model Context Protocol
    is normally used. Where it declines, is unreachable, or names
    something that is not a real tool, the keyword rules decide
    instead, and the system behaves exactly as it did before model
    routing was added.

    Whichever picks first, the remaining sources are still tried in
    order when the chosen one returns nothing, so a bad choice costs
    a wasted request rather than an unanswered question.

    ``query`` is the keyword form the search APIs need. ``question``
    is the full question, used only to choose a source: the keyword
    step strips exactly the words that identify intent, so
    "what are the side effects of dolo 650" arrives as "dolo 650"
    and no longer looks like a medication question at all.
    """

    intent = question or query

    order = _keyword_order(intent)

    chosen = choose_source(intent)

    if chosen:
        # The model's pick goes first. The keyword order is kept
        # behind it rather than discarded, so a source the model
        # passed over is still reachable if its choice comes back
        # empty.
        order = [chosen] + [name for name in order if name != chosen]

    else:
        print("[SEARCH] Using keyword rules to choose a source.")

    for name in order:

        print(f"Searching {DISPLAY_NAMES[name]}...")

        result = SEARCHERS[name](query)

        if result["context"]:
            return result

    return EMPTY_RESULT
