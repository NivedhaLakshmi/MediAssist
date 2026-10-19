import html
import re
import requests
import xml.etree.ElementTree as ET

MEDLINEPLUS_API = "https://wsearch.nlm.nih.gov/ws/query"


# MedlinePlus marks the words that matched the search, so a title
# arrives as 'Prescription <span class="qt0">Drug</span> Misuse'.
# Shown to the patient, the markup was visible in the reference list.
_TAG = re.compile(r"<[^>]+>")


def _plain_text(value):
    """Strip MedlinePlus highlight markup and decode entities."""

    if not value:
        return ""

    return html.unescape(_TAG.sub("", value)).strip()


def search_medlineplus(query: str):
    """
    Search MedlinePlus Health Topics.
    Returns a list of patient-friendly health topics.
    """

    try:

        response = requests.get(
            MEDLINEPLUS_API,
            params={
                "db": "healthTopics",
                "term": query,
            },
            timeout=10,
        )

        response.raise_for_status()

        root = ET.fromstring(response.content)

        topics = []

        for document in root.findall(".//document"):

            title = ""
            summary = ""
            url = ""

            for content in document.findall("content"):

                name = content.attrib.get("name")

                if name == "title":
                    title = _plain_text(content.text)

                elif name == "FullSummary":
                    summary = _plain_text(content.text)

                elif name == "url":
                    url = content.text or ""

            topics.append(
                {
                    "title": title,
                    "summary": summary,
                    "url": url,
                }
            )

        return topics

    except Exception as e:

        print("MedlinePlus Client Error:", e)

        return []