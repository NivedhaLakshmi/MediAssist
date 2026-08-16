def normalize_health_query(question):
    normalized = question.lower().strip()
    normalized = normalized.rstrip("?!.")

    prefixes = [
        "what is ",
        "what are ",
        "how to ",
        "how do i ",
        "tell me about ",
        "explain ",
    ]

    for prefix in prefixes:
        if normalized.startswith(prefix):
            return normalized[len(prefix):].strip()

    return normalized