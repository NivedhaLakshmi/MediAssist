import requests
import xml.etree.ElementTree as ET

MEDLINEPLUS_API = "https://wsearch.nlm.nih.gov/ws/query"


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
                    title = content.text or ""

                elif name == "FullSummary":
                    summary = content.text or ""

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