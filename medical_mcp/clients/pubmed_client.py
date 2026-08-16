import requests
import xml.etree.ElementTree as ET

from utils.constants import (
    PUBMED_BASE_URL,
    DEFAULT_PUBMED_RESULTS,
)

from utils.query_builder import build_pubmed_query


def search_pubmed(query: str, max_results: int = DEFAULT_PUBMED_RESULTS):
    """
    Search PubMed and return structured paper information.
    """

    try:

        query = build_pubmed_query(query)

        # ---------------------------------------------------
        # Step 1: Search PubMed IDs
        # ---------------------------------------------------

        search_response = requests.get(
            f"{PUBMED_BASE_URL}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max_results,
                "sort": "relevance",
            },
            timeout=15,
        )

        search_response.raise_for_status()

        ids = search_response.json()["esearchresult"]["idlist"]

        if not ids:
            return []

        # ---------------------------------------------------
        # Step 2: Download complete article XML
        # ---------------------------------------------------

        fetch_response = requests.get(
            f"{PUBMED_BASE_URL}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "xml",
            },
            timeout=20,
        )

        fetch_response.raise_for_status()

        root = ET.fromstring(fetch_response.content)

        papers = []

        # ---------------------------------------------------
        # Step 3: Parse XML
        # ---------------------------------------------------

        for article in root.findall(".//PubmedArticle"):

            # -----------------------------
            # Title
            # -----------------------------

            title = article.findtext(".//ArticleTitle", default="No title")

            # -----------------------------
            # Journal
            # -----------------------------

            journal = article.findtext(
                ".//Journal/Title",
                default="Unknown Journal",
            )

            # -----------------------------
            # Publication Year
            # -----------------------------

            year = article.findtext(
                ".//PubDate/Year",
                default="Unknown",
            )

            # -----------------------------
            # Abstract
            # -----------------------------

            abstract_sections = []

            for section in article.findall(".//Abstract/AbstractText"):

                label = section.attrib.get("Label")

                text = "".join(section.itertext()).strip()

                if not text:
                    continue

                if label:
                    abstract_sections.append(
                        f"{label}: {text}"
                    )
                else:
                    abstract_sections.append(text)

            abstract = "\n\n".join(abstract_sections)

            # -----------------------------
            # PMID
            # -----------------------------

            pmid = article.findtext(
                ".//PMID",
                default=""
            )

            # -----------------------------
            # DOI
            # -----------------------------

            doi = ""

            for article_id in article.findall(".//ArticleId"):

                if article_id.attrib.get("IdType") == "doi":

                    doi = article_id.text or ""

                    break

            papers.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "year": year,
                    "doi": doi,
                    "abstract": abstract,
                }
            )

        # ---------------------------------------------------
        # Step 4: Sort by year (Newest first)
        # ---------------------------------------------------

        papers.sort(
            key=lambda x: x["year"],
            reverse=True,
        )

        return papers

    except Exception as e:

        print("PubMed Client Error:")
        print(e)

        return []