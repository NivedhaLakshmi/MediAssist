from mcp_instance import mcp

from clients.pubmed_client import search_pubmed as pubmed_client

from utils.formatter import format_pubmed_context


@mcp.tool
def search_pubmed(query: str):
    """
    Search PubMed for peer-reviewed medical research.
    """

    papers = pubmed_client(query)

    return {
        "source": "PubMed",
        "context": format_pubmed_context(papers),
        "references": papers
    }