import requests

from utils.constants import (
    CLINICAL_TRIALS_BASE_URL,
    DEFAULT_PUBMED_RESULTS,
)


def search_clinical_trials(
    query: str,
    max_results: int = DEFAULT_PUBMED_RESULTS,
):
    """
    Search ClinicalTrials.gov and return structured trial information.
    """

    try:

        # ---------------------------------------------------
        # Step 1: Search ClinicalTrials.gov
        # ---------------------------------------------------

        response = requests.get(
            CLINICAL_TRIALS_BASE_URL,
            params={
                "query.term": query,
                "pageSize": max_results,
            },
            timeout=20,
        )

        response.raise_for_status()

        studies = response.json().get("studies", [])

        if not studies:
            return []

        trials = []

        # ---------------------------------------------------
        # Step 2: Parse JSON
        # ---------------------------------------------------

        for study in studies:

            protocol = study.get(
                "protocolSection",
                {}
            )

            # -----------------------------
            # Study Title
            # -----------------------------

            title = (
                protocol
                .get("identificationModule", {})
                .get("briefTitle", "No title")
            )

            # -----------------------------
            # Condition
            # -----------------------------

            conditions = (
                protocol
                .get("conditionsModule", {})
                .get("conditions", [])
            )

            condition = ", ".join(conditions)

            # -----------------------------
            # Intervention
            # -----------------------------

            interventions = (
                protocol
                .get("armsInterventionsModule", {})
                .get("interventions", [])
            )

            intervention = ", ".join(
                item.get("name", "")
                for item in interventions
            )

            # -----------------------------
            # Overall Status
            # -----------------------------

            status = (
                protocol
                .get("statusModule", {})
                .get("overallStatus", "Unknown")
            )

            # -----------------------------
            # Phase
            # -----------------------------

            phases = (
                protocol
                .get("designModule", {})
                .get("phases", [])
            )

            phase = ", ".join(phases)

            # -----------------------------
            # Brief Summary
            # -----------------------------

            summary = (
                protocol
                .get("descriptionModule", {})
                .get("briefSummary", "")
            )

            # -----------------------------
            # NCT ID
            # -----------------------------

            nct_id = (
                protocol
                .get("identificationModule", {})
                .get("nctId", "")
            )

            # -----------------------------
            # ClinicalTrials.gov URL
            # -----------------------------

            url = (
                f"https://clinicaltrials.gov/study/{nct_id}"
                if nct_id
                else ""
            )

            trials.append(
                {
                    "title": title,
                    "condition": condition,
                    "intervention": intervention,
                    "status": status,
                    "phase": phase,
                    "summary": summary,
                    "nct_id": nct_id,
                    "url": url,
                }
            )

        return trials

    except Exception as e:

        print("ClinicalTrials Client Error:")
        print(e)

        return []