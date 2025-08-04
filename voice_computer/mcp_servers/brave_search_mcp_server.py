"""
Brave Search MCP server with web search tools using Brave Search API for the voice computer system.
"""

import asyncio
import argparse
import os
import aiohttp
import json
from typing import Optional, Dict, Any, List

from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server

mcp = FastMCP("brave-search-tools")


def get_api_key() -> str:
    """Get the Brave Search API key from environment variable."""
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise ValueError("BRAVE_SEARCH_API_KEY environment variable not set")
    return api_key


async def make_brave_search_request(query: str, count: int = 10) -> List[Dict[str, Any]]:
    """Make a request to the Brave Search API."""
    api_key = get_api_key()
    url = "https://api.search.brave.com/res/v1/web/search"
    
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }
    
    params = {
        "q": query,
        "count": min(count, 20)  # Limit to maximum of 20 results
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                # Extract web results
                if "web" in data and "results" in data["web"]:
                    return data["web"]["results"]
                else:
                    return []
            else:
                error_text = await response.text()
                raise Exception(f"Brave Search API error ({response.status}): {error_text}")


@mcp.tool()
async def web_search(query: str, count: int = 5) -> str:
    """
    Search the web using Brave Search API
    
    Args:
        query: Search query string
        count: Number of results to return (1-10, default 5)
    """
    try:
        # Limit count to reasonable range
        count = max(1, min(10, count))
        
        results = await make_brave_search_request(query, count)
        
        if not results:
            return f"No search results found for: {query}"
        
        result_text = f"Web search results for '{query}':\n\n"
        
        for i, result in enumerate(results[:count], 1):
            title = result.get("title", "No title")
            description = result.get("description", "No description")
            url = result.get("url", "No URL")
            
            result_text += f"{i}. {title}\n"
            result_text += f"   {description}\n"
            result_text += f"   URL: {url}\n\n"
        
        return result_text
        
    except Exception as e:
        return f"Error performing web search: {str(e)}"


@mcp.tool()
async def search_news(query: str, count: int = 5) -> str:
    """
    Search for news articles using Brave Search API
    
    Args:
        query: News search query string
        count: Number of results to return (1-10, default 5)
    """
    try:
        # Limit count to reasonable range
        count = max(1, min(10, count))
        
        # Add "news" to the query to get more news-focused results
        news_query = f"{query} news"
        results = await make_brave_search_request(news_query, count)
        
        if not results:
            return f"No news results found for: {query}"
        
        result_text = f"News search results for '{query}':\n\n"
        
        for i, result in enumerate(results[:count], 1):
            title = result.get("title", "No title")
            description = result.get("description", "No description")
            url = result.get("url", "No URL")
            
            result_text += f"{i}. {title}\n"
            result_text += f"   {description}\n"
            result_text += f"   URL: {url}\n\n"
        
        return result_text
        
    except Exception as e:
        return f"Error performing news search: {str(e)}"


@mcp.tool()
async def search_recent(query: str, count: int = 5) -> str:
    """
    Search for recent web content using Brave Search API
    
    Args:
        query: Search query string
        count: Number of results to return (1-10, default 5)
    """
    try:
        # Limit count to reasonable range
        count = max(1, min(10, count))
        
        # Add time-based keywords to get more recent results
        recent_query = f"{query} recent OR latest OR 2024 OR 2025"
        results = await make_brave_search_request(recent_query, count)
        
        if not results:
            return f"No recent results found for: {query}"
        
        result_text = f"Recent search results for '{query}':\n\n"
        
        for i, result in enumerate(results[:count], 1):
            title = result.get("title", "No title")
            description = result.get("description", "No description")
            url = result.get("url", "No URL")
            
            result_text += f"{i}. {title}\n"
            result_text += f"   {description}\n"
            result_text += f"   URL: {url}\n\n"
        
        return result_text
        
    except Exception as e:
        return f"Error performing recent search: {str(e)}"


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
    parser = argparse.ArgumentParser(description="Run MCP stdio-based Brave Search server")
    args = parser.parse_args()

    asyncio.run(run_stdio_server(mcp_server))