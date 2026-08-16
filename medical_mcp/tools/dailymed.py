from mcp_instance import mcp

from clients.dailymed_client import search_dailymed as dailymed_client
from utils.formatter import format_dailymed_context


@mcp.tool
def search_dailymed(query: str):
    """
    Search DailyMed for FDA-approved medication information.
    """

    drug = dailymed_client(query)

    if not drug:
        return {
            "source": "DailyMed",
            "context": "",
            "references": []
        }

    context = format_dailymed_context(drug)

    return {
        "source": "DailyMed",
        "context": context,
        "references": [drug]
    }