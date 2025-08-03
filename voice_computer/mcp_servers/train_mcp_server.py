"""
Train MCP server with UK train timetable information using transportapi.com for the voice computer system.
"""

import asyncio
import argparse
import os
import aiohttp
import json
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server

mcp = FastMCP("train-tools")


def get_api_credentials() -> tuple[str, str]:
    """Get the Transport API credentials from environment variables."""
    app_id = os.getenv("TRANSPORT_API_ID")
    app_key = os.getenv("TRANSPORT_API_KEY")
    
    if not app_id or not app_key:
        raise ValueError("TRANSPORT_API_ID and TRANSPORT_API_KEY environment variables must be set")
    
    return app_id, app_key


async def make_transport_request(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Make a request to the transportapi.com API."""
    app_id, app_key = get_api_credentials()
    base_url = "https://transportapi.com/v3"
    
    # Add API credentials to parameters
    params.update({
        "app_id": app_id,
        "app_key": app_key
    })
    
    url = f"{base_url}/{endpoint}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Transport API error ({response.status}): {error_text}")


@mcp.tool()
async def find_stations_near_postcode(postcode: str, distance: int = 1000) -> str:
    """
    Find train stations near a postcode
    
    Args:
        postcode: UK postcode (e.g., "SW1A 1AA")
        distance: Search radius in meters (default 1000)
    """
    try:
        # Clean postcode (remove spaces and make uppercase)
        clean_postcode = postcode.replace(" ", "").upper()
        
        data = await make_transport_request(
            f"uk/places.json",
            {
                "query": clean_postcode,
                "type": "train_station"
            }
        )
        
        if not data.get("member"):
            return f"No train stations found near postcode {postcode}"
        
        result = f"Train stations near {postcode}:\n\n"
        
        for i, station in enumerate(data["member"][:10], 1):  # Limit to top 10
            result += f"{i}. {station['name']}\n"
            result += f"   Code: {station.get('station_code', 'N/A')}\n"
            result += f"   Distance: {station.get('distance', 'N/A')}m\n"
            if station.get('accuracy'):
                result += f"   Accuracy: {station['accuracy']}\n"
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error finding stations near postcode: {str(e)}"


@mcp.tool()
async def get_station_departures(station_code: str, limit: int = 10) -> str:
    """
    Get live departure times from a train station
    
    Args:
        station_code: Three-letter station code (e.g., "PAD" for London Paddington)
        limit: Maximum number of departures to show (default 10)
    """
    try:
        data = await make_transport_request(
            f"uk/train/station/{station_code}/live.json",
            {}
        )
        
        departures = data.get("departures", {}).get("all", [])
        
        if not departures:
            return f"No departures found for station {station_code}"
        
        station_name = data.get("station_name", station_code)
        result = f"Live departures from {station_name} ({station_code}):\n\n"
        
        for i, dep in enumerate(departures[:limit], 1):
            scheduled_time = dep.get("aimed_departure_time", "N/A")
            expected_time = dep.get("expected_departure_time", scheduled_time)
            
            result += f"{i}. {dep.get('destination_name', 'Unknown destination')}\n"
            result += f"   Scheduled: {scheduled_time}\n"
            
            if expected_time != scheduled_time:
                result += f"   Expected: {expected_time}"
                # Calculate delay
                try:
                    scheduled = datetime.strptime(scheduled_time, "%H:%M")
                    expected = datetime.strptime(expected_time, "%H:%M")
                    delay = (expected - scheduled).total_seconds() / 60
                    if delay > 0:
                        result += f" (DELAYED by {int(delay)} min)"
                    elif delay < 0:
                        result += f" (EARLY by {int(abs(delay))} min)"
                except:
                    pass
                result += "\n"
            else:
                result += f"   Status: On time\n"
            
            result += f"   Platform: {dep.get('platform', 'TBC')}\n"
            result += f"   Operator: {dep.get('operator_name', 'N/A')}\n"
            
            if dep.get("status"):
                result += f"   Status: {dep['status']}\n"
            
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error getting departures for station {station_code}: {str(e)}"


@mcp.tool()
async def get_station_arrivals(station_code: str, limit: int = 10) -> str:
    """
    Get live arrival times at a train station
    
    Args:
        station_code: Three-letter station code (e.g., "PAD" for London Paddington)
        limit: Maximum number of arrivals to show (default 10)
    """
    try:
        data = await make_transport_request(
            f"uk/train/station/{station_code}/live.json",
            {}
        )
        
        arrivals = data.get("arrivals", {}).get("all", [])
        
        if not arrivals:
            return f"No arrivals found for station {station_code}"
        
        station_name = data.get("station_name", station_code)
        result = f"Live arrivals at {station_name} ({station_code}):\n\n"
        
        for i, arr in enumerate(arrivals[:limit], 1):
            scheduled_time = arr.get("aimed_arrival_time", "N/A")
            expected_time = arr.get("expected_arrival_time", scheduled_time)
            
            result += f"{i}. From {arr.get('origin_name', 'Unknown origin')}\n"
            result += f"   Scheduled: {scheduled_time}\n"
            
            if expected_time != scheduled_time:
                result += f"   Expected: {expected_time}"
                # Calculate delay
                try:
                    scheduled = datetime.strptime(scheduled_time, "%H:%M")
                    expected = datetime.strptime(expected_time, "%H:%M")
                    delay = (expected - scheduled).total_seconds() / 60
                    if delay > 0:
                        result += f" (DELAYED by {int(delay)} min)"
                    elif delay < 0:
                        result += f" (EARLY by {int(abs(delay))} min)"
                except:
                    pass
                result += "\n"
            else:
                result += f"   Status: On time\n"
            
            result += f"   Platform: {arr.get('platform', 'TBC')}\n"
            result += f"   Operator: {arr.get('operator_name', 'N/A')}\n"
            
            if arr.get("status"):
                result += f"   Status: {arr['status']}\n"
            
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error getting arrivals for station {station_code}: {str(e)}"


@mcp.tool()
async def get_train_timetable_from_postcode(postcode: str, direction: str = "departures", limit: int = 5) -> str:
    """
    Get train timetable from the nearest station to a postcode
    
    Args:
        postcode: UK postcode (e.g., "SW1A 1AA")
        direction: "departures" or "arrivals" (default "departures")
        limit: Maximum number of trains to show (default 5)
    """
    try:
        # First find stations near the postcode
        clean_postcode = postcode.replace(" ", "").upper()
        
        stations_data = await make_transport_request(
            f"uk/places.json",
            {
                "query": clean_postcode,
                "type": "train_station"
            }
        )
        
        if not stations_data.get("member"):
            return f"No train stations found near postcode {postcode}"
        
        # Get the nearest station
        nearest_station = stations_data["member"][0]
        station_code = nearest_station.get("station_code")
        station_name = nearest_station.get("name")
        distance = nearest_station.get("distance", "unknown")
        
        if not station_code:
            return f"No valid station code found for nearest station to {postcode}"
        
        # Get timetable for the nearest station
        if direction.lower() == "arrivals":
            timetable_result = await get_station_arrivals(station_code, limit)
        else:
            timetable_result = await get_station_departures(station_code, limit)
        
        result = f"Train information for postcode {postcode}\n"
        result += f"Nearest station: {station_name} ({station_code}) - {distance}m away\n\n"
        result += timetable_result
        
        return result
        
    except Exception as e:
        return f"Error getting train timetable from postcode {postcode}: {str(e)}"


@mcp.tool()
async def search_station_codes(query: str) -> str:
    """
    Search for train station codes by name
    
    Args:
        query: Station name or partial name to search for
    """
    try:
        data = await make_transport_request(
            f"uk/places.json",
            {
                "query": query,
                "type": "train_station"
            }
        )
        
        if not data.get("member"):
            return f"No train stations found matching '{query}'"
        
        result = f"Train stations matching '{query}':\n\n"
        
        for i, station in enumerate(data["member"][:15], 1):  # Limit to top 15
            result += f"{i}. {station['name']}\n"
            result += f"   Code: {station.get('station_code', 'N/A')}\n"
            if station.get('accuracy'):
                result += f"   Accuracy: {station['accuracy']}\n"
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error searching for station codes: {str(e)}"


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
    parser = argparse.ArgumentParser(description="Run MCP stdio-based train server")
    args = parser.parse_args()

    asyncio.run(run_stdio_server(mcp_server))