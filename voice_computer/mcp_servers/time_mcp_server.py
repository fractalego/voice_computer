"""
Time MCP server with date and time tools for the voice computer system.
"""

import asyncio
import argparse
from datetime import datetime

from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server

mcp = FastMCP("time-tools")


@mcp.tool()
def current_time() -> str:
    """
    Get the current time in HH:MM format
    """
    now = datetime.now()
    return now.strftime("%H:%M")


@mcp.tool()
def current_date() -> str:
    """
    Get the current date in YYYY-MM-DD format
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d")


@mcp.tool()
def current_day_of_week() -> str:
    """
    Get the current day of the week
    """
    now = datetime.now()
    return now.strftime("%A")


@mcp.tool()
def current_datetime() -> str:
    """
    Get the current date and time in a readable format
    """
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %H:%M:%S")


async def run_stdio_server(mcp_server: Server) -> None:
    """Run an MCP server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server
    parser = argparse.ArgumentParser(description="Run MCP stdio-based time server")
    args = parser.parse_args()

    asyncio.run(run_stdio_server(mcp_server))