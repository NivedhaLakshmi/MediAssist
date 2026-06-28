from services.pubmed_service import search_pubmed

papers = search_pubmed("hypertension")

for paper in papers:

    print("\n=================")

    print("TITLE:")
    print(paper["title"])

    print("\nABSTRACT:")
    print(paper["abstract"][:500])

    print("\n=================")