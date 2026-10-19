import requests
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
