from mcp_instance import mcp

from tools.medlineplus import search_medlineplus
from tools.pubmed import search_pubmed
from tools.clinicaltrials import search_clinical_trials
from tools.dailymed import search_dailymed

from utils.query_router import (
    should_search_clinical_trials,
    should_search_dailymed,
)

@mcp.tool
def search_medical_sources(query: str):

    # -----------------------------
    # MedlinePlus
    # -----------------------------
    
    result = search_medlineplus(query)

    if result["context"]:
        return result
    
    # ---------------------------------
    # DailyMed (Medication Questions)
    # ---------------------------------

    if should_search_dailymed(query):

        result = search_dailymed(query)

        if result["context"]:
            return result

    
    # ---------------------------------
    # ClinicalTrials.gov
    # ---------------------------------

    if should_search_clinical_trials(query):
        
        result = search_clinical_trials(query)

        if result["context"]:
            return result


    # -----------------------------
    # PubMed
    # -----------------------------
    
    print("Searching PubMed...")

    result = search_pubmed(query)

    if result["context"]:
        return result

    # ---------------------------------
    # No Result
    # ---------------------------------

    return {
        "source": "",
        "context": "",
        "references": []
    }