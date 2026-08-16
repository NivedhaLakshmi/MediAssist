import asyncio
import ast

from fastmcp import Client


async def _call_medical_sources(query):

    async with Client("http://127.0.0.1:8000/mcp") as client:

        result = await client.call_tool(
            "search_medical_sources",
            {
                "query": query
            }
        )

        return result


def search_medical_sources(query: str):

    result = asyncio.run(
        _call_medical_sources(query)
    )

    response_text = result.content[0].text

    response = ast.literal_eval(response_text)

    return (
        response["context"],
        response["source"],
        response["references"]
    )