from mcp_instance import mcp

from clients.medlineplus_client import (
    search_medlineplus as medlineplus_client
)

from utils.formatter import format_medlineplus_context


@mcp.tool
def search_medlineplus(query: str):

    topics = medlineplus_client(query)

    return {
        "source": "MedlinePlus",
        "context": format_medlineplus_context(topics),
        "references": topics
    }