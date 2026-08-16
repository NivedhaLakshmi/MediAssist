from mcp_instance import mcp

from clients.clinicaltrials_client import (
    search_clinical_trials as clinicaltrials_client
)

from utils.formatter import (
    format_clinical_trials_context
)


@mcp.tool
def search_clinical_trials(query: str):
    """
    Search ClinicalTrials.gov for clinical studies.
    """

    trials = clinicaltrials_client(query)

    return {
        "source": "ClinicalTrials.gov",
        "context": format_clinical_trials_context(trials),
        "references": trials
    }