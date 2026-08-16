from mcp_instance import mcp


@mcp.tool
def echo(message: str) -> str:
    """
    Echo back the user's message.
    """
    return f"You said: {message}"