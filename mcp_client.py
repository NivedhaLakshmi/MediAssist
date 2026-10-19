import asyncio
import ast

from fastmcp import Client


async def _call_medical_sources(query, question):

    async with Client("http://127.0.0.1:8000/mcp") as client:

        result = await client.call_tool(
            "search_medical_sources",
            {
                "query": query,
                "question": question,
            }
        )

        return result


def search_medical_sources(query: str, question: str = ""):
    """``query`` is the keyword form; ``question`` is the full text,
    used only to choose which source to search."""

    result = asyncio.run(
        _call_medical_sources(query, question)
    )

    response_text = result.content[0].text

    response = ast.literal_eval(response_text)

    return (
        response["context"],
        response["source"],
        response["references"]
    )