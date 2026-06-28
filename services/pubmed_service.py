import requests
import xml.etree.ElementTree as ET


def search_pubmed(query):

    # Search PubMed
    search_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    )

    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": 10,
        "retmode": "json"
    }

    search_response = requests.get(
        search_url,
        params=search_params
    )

    ids = (
        search_response.json()
        ["esearchresult"]
        ["idlist"]
    )

    if not ids:
        return []

    # Fetch abstracts
    fetch_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    )

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml"
    }

    fetch_response = requests.get(
        fetch_url,
        params=fetch_params
    )

    root = ET.fromstring(
        fetch_response.text
    )

    papers = []

    for article in root.findall(".//PubmedArticle"):

        title = ""

        title_element = article.find(
            ".//ArticleTitle"
        )

        if title_element is not None:
            title = "".join(
                title_element.itertext()
            )

        abstract = ""

        abstract_elements = article.findall(
            ".//AbstractText"
        )
        
        if not abstract.strip():
            continue

        for item in abstract_elements:
            abstract += (
                "".join(item.itertext()) + "\n"
            )

        papers.append(
            {
                "title": title,
                "abstract": abstract
            }
        )

    return papers