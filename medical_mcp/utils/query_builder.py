def build_pubmed_query(question: str):

    return f"""
({question})

AND

(
review[Publication Type]

OR

systematic review

OR

meta-analysis

OR

guideline

OR

practice guideline
)
"""