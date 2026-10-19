"""Client for the local Google Drive MCP retrieval tool."""

import asyncio
import ast
import json

from fastmcp import Client


class GoogleDriveMCPError(Exception):
    """Raised when the local Google Drive MCP tool is unavailable."""

    def __init__(self, message, debug=None):
        super().__init__(message)
        self.debug = debug or []


async def _search_google_drive_reports(query, folder_id, access_token):
    async with Client("http://127.0.0.1:8000/mcp") as client:
        return await client.call_tool(
            "search_google_drive_reports",
            {
                "query": query,
                "folder_id": folder_id,
                "access_token": access_token,
            },
        )


def search_google_drive_reports(query, folder_id, access_token):
    """Search the selected Drive folder through the MCP server."""

    try:
        result = asyncio.run(
            _search_google_drive_reports(
                query,
                folder_id,
                access_token,
            )
        )

        response_text = result.content[0].text

        try:
            response = json.loads(response_text)
        except json.JSONDecodeError:
            response = ast.literal_eval(response_text)
    except Exception as error:
        raise GoogleDriveMCPError(
            "Google Drive search is currently unavailable."
        ) from error

    if response.get("error"):
        raise GoogleDriveMCPError(
            response["error"],
            debug=response.get("debug"),
        )

    return response
