def should_search_clinical_trials(query: str) -> bool:
    """
    Return True if the user is asking about
    clinical trials or new treatments.
    """

    query = query.lower()

    keywords = [
        "clinical trial",
        "clinical trials",
        "new treatment",
        "latest treatment",
        "experimental therapy",
        "experimental",
        "ongoing study",
        "ongoing studies",
        "research study",
        "research studies",
        "trial",
    ]

    return any(keyword in query for keyword in keywords)

def should_search_dailymed(query: str) -> bool:
    """
    Return True if the user is asking about
    medication information.
    """

    query = query.lower()

    keywords = [
        "side effect",
        "side effects",
        "dosage",
        "dose",
        "drug",
        "medicine",
        "medication",
        "tablet",
        "capsule",
        "injection",
        "warning",
        "warnings",
        "contraindication",
        "contraindications",
        "drug interaction",
        "interactions",
        "can i take",
        "can i use",
    ]

    return any(keyword in query for keyword in keywords)