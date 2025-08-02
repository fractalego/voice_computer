"""
Math MCP server with basic math tools for the voice computer system.
"""

import asyncio
import argparse
import math
from typing import Union

from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server

mcp = FastMCP("math-tools")


@mcp.tool()
def add_two_numbers(lhs: Union[int, float], rhs: Union[int, float]) -> Union[int, float]:
    """
    Calculate the sum of two numbers
    """
    return lhs + rhs


@mcp.tool()
def subtract_two_numbers(lhs: Union[int, float], rhs: Union[int, float]) -> Union[int, float]:
    """
    Calculate the difference of two numbers (lhs - rhs)
    """
    return lhs - rhs


@mcp.tool()
def multiply_two_numbers(lhs: Union[int, float], rhs: Union[int, float]) -> Union[int, float]:
    """
    Calculate the product of two numbers
    """
    return lhs * rhs


@mcp.tool()
def divide_two_numbers(lhs: Union[int, float], rhs: Union[int, float]) -> Union[int, float]:
    """
    Calculate the division of two numbers (lhs / rhs)
    """
    if rhs == 0:
        raise ValueError("Cannot divide by zero")
    return lhs / rhs


@mcp.tool()
def square_root(number: Union[int, float]) -> float:
    """
    Calculate the square root of a number
    """
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(number)


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
    parser = argparse.ArgumentParser(description="Run MCP stdio-based math server")
    args = parser.parse_args()

    asyncio.run(run_stdio_server(mcp_server))