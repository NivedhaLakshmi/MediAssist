def format_pubmed_context(papers):

    context = ""

    for paper in papers[:2]:

        abstract = paper["abstract"][:600]

        context += f"""
Title:
{paper['title']}

Journal:
{paper['journal']}

Year:
{paper['year']}

Abstract:
{abstract}

----------------------------------------

"""

    return context


def format_medlineplus_context(topics):

    context = ""

    for topic in topics[:2]:

        summary = topic["summary"][:600]

        context += f"""
Title:
{topic['title']}

Summary:
{summary}

----------------------------------------

"""

    return context

def format_clinical_trials_context(trials):

    context = ""

    for trial in trials[:2]:

        summary = trial["summary"][:600]

        context += f"""
Study Title:
{trial['title']}

Condition:
{trial['condition']}

Intervention:
{trial['intervention']}

Status:
{trial['status']}

Phase:
{trial['phase']}

Summary:
{summary}

----------------------------------------

"""

    return context

def format_dailymed_context(drug):

    if not drug:
        return ""

    context = f"""
Drug:
{drug.get("title", "")}

Indications:
{drug.get("indications", "")[:800]}

Dosage:
{drug.get("dosage", "")[:800]}

Warnings:
{drug.get("warnings", "")[:800]}

Contraindications:
{drug.get("contraindications", "")[:800]}

Adverse Reactions:
{drug.get("adverse_reactions", "")[:800]}

----------------------------------------

"""

    return context

