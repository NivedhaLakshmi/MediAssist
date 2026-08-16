import requests
import re
import html
import xml.etree.ElementTree as ET

from utils.constants import DAILYMED_BASE_URL

def _search_label(query: str):

    response = requests.get(
        DAILYMED_BASE_URL,
        params={
            "drug_name": query
        },
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    drugs = data.get("data", [])

    if not drugs:
        return None

    return drugs[0]

def _download_xml(setid: str):

    url = (
        f"https://dailymed.nlm.nih.gov/"
        f"dailymed/services/v2/spls/{setid}.xml"
    )

    response = requests.get(
        url,
        timeout=20,
    )

    response.raise_for_status()

    return response.text

def _extract_text(xml: str, heading: str):

    """
    Extract the text that follows a section heading.
    """

    start = xml.find(heading)

    if start == -1:
        return ""

    # Extract approximately the next 4000 characters
    snippet = xml[start:start + 4000]

    # Remove XML tags
    snippet = re.sub("<.*?>", " ", snippet)

    # Decode HTML entities
    snippet = html.unescape(snippet)

    # Remove extra whitespace
    snippet = re.sub(r"\s+", " ", snippet)

    return snippet.strip()

def _extract_sections(xml: str):

    root = ET.fromstring(xml)

    sections = {}

    namespace = {
        "hl7": "urn:hl7-org:v3"
    }

    for section in root.findall(".//hl7:section", namespace):

        title_element = section.find("hl7:title", namespace)

        text_element = section.find("hl7:text", namespace)

        if title_element is None or text_element is None:
            continue

        title = "".join(title_element.itertext()).strip().upper()
        
        text = " ".join(text_element.itertext()).strip()

        sections[title] = text

    return sections

def search_dailymed(query: str):

    drug = _search_label(query)

    if not drug:
        return {}

    xml = _download_xml(
        drug["setid"]
    )

    sections = _extract_sections(xml)

    return {

        "title": drug["title"],

        "published_date": drug["published_date"],

        "indications":
            sections.get(
                "INDICATIONS & USAGE",
                ""
            ),

        "dosage":
            sections.get(
                "DOSAGE & ADMINISTRATION",
                ""
            ),

        "warnings":
            sections.get(
                "WARNINGS",
                ""
            ),

        "contraindications":
            sections.get(
                "CONTRAINDICATIONS",
                ""
            ),

        "adverse_reactions":
            sections.get(
                "ADVERSE REACTIONS",
                ""
            ),

        "url":
            f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={drug['setid']}"
    }
    
"""
def search_dailymed(query: str):
    
    Search DailyMed by drug name.
    

    try:

        response = requests.get(
            DAILYMED_BASE_URL,
            params={
                "drug_name": query,
            },
            timeout=20,
        )

        response.raise_for_status()

        data = response.json()

        print(data)

        return data

    except Exception as e:

        print("DailyMed Client Error:")
        print(e)

        return {}
"""