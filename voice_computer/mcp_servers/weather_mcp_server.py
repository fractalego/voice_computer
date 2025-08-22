"""
Weather MCP server with weather prediction tools using weatherapi.com for the voice computer system.
"""

import asyncio
import argparse
import os
import aiohttp
import json
from typing import Optional, Dict, Any

from mcp.server import Server
from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server

mcp = FastMCP("weather-tools")


def get_api_key() -> str:
    """Get the weather API key from environment variable."""
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        raise ValueError("WEATHER_API_KEY environment variable not set")
    return api_key


async def make_weather_request(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Make a request to the weatherapi.com API."""
    api_key = get_api_key()
    base_url = "http://api.weatherapi.com/v1"
    
    # Add API key to parameters
    params["key"] = api_key
    
    url = f"{base_url}/{endpoint}"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Weather API error ({response.status}): {error_text}")


@mcp.tool()
async def get_current_weather(location: str) -> str:
    """
    Get current weather conditions for a location
    
    Args:
        location: City name, coordinates (lat,lon), IP address, or airport code
    """
    try:
        data = await make_weather_request("current.json", {"q": location})
        
        current = data["current"]
        location_info = data["location"]
        
        result = f"Current weather in {location_info['name']}, {location_info['region']}, {location_info['country']}:\n"
        result += f"Temperature: {current['temp_c']}°C ({current['temp_f']}°F)\n"
        result += f"Condition: {current['condition']['text']}\n"
        result += f"Feels like: {current['feelslike_c']}°C ({current['feelslike_f']}°F)\n"
        result += f"Humidity: {current['humidity']}%\n"
        result += f"Wind: {current['wind_kph']} km/h ({current['wind_mph']} mph) {current['wind_dir']}\n"
        result += f"Pressure: {current['pressure_mb']} mb\n"
        result += f"UV Index: {current['uv']}\n"
        result += f"Last updated: {current['last_updated']}"
        
        return result
        
    except Exception as e:
        return f"Error getting current weather: {str(e)}"


@mcp.tool()
async def get_weather_forecast(location: str, days: int = 3) -> str:
    """
    Get weather forecast for a location
    
    Args:
        location: City name, coordinates (lat,lon), IP address, or airport code
        days: Number of days to forecast (1-10, default 3)
    """
    try:
        # Limit days to valid range
        if days is None:
            days = 3
        days = max(1, min(10, days))
        
        data = await make_weather_request("forecast.json", {"q": location, "days": days})
        
        location_info = data["location"]
        forecast = data["forecast"]["forecastday"]
        
        result = f"Weather forecast for {location_info['name']}, {location_info['region']}, {location_info['country']}:\n\n"
        
        for day_data in forecast:
            day = day_data["day"]
            date = day_data["date"]
            
            result += f"Date: {date}\n"
            result += f"Condition: {day['condition']['text']}\n"
            result += f"High: {day['maxtemp_c']}°C ({day['maxtemp_f']}°F)\n"
            result += f"Low: {day['mintemp_c']}°C ({day['mintemp_f']}°F)\n"
            result += f"Chance of rain: {day['daily_chance_of_rain']}%\n"
            result += f"Max wind: {day['maxwind_kph']} km/h ({day['maxwind_mph']} mph)\n"
            result += f"Average humidity: {day['avghumidity']}%\n"
            result += f"UV Index: {day['uv']}\n\n"
        
        return result
        
    except Exception as e:
        return f"Error getting weather forecast: {str(e)}"


@mcp.tool()
async def search_locations(query: str) -> str:
    """
    Search for locations to get weather data
    
    Args:
        query: Location search query (city, airport, etc.)
    """
    try:
        data = await make_weather_request("search.json", {"q": query})
        
        if not data:
            return f"No locations found for '{query}'"
        
        result = f"Location search results for '{query}':\n\n"
        
        for i, location in enumerate(data[:10], 1):  # Limit to top 10 results
            result += f"{i}. {location['name']}"
            if location.get('region'):
                result += f", {location['region']}"
            result += f", {location['country']}"
            if location.get('lat') and location.get('lon'):
                result += f" (Lat: {location['lat']}, Lon: {location['lon']})"
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error searching locations: {str(e)}"


@mcp.tool()
async def get_weather_alerts(location: str) -> str:
    """
    Get weather alerts for a location
    
    Args:
        location: City name, coordinates (lat,lon), IP address, or airport code
    """
    try:
        data = await make_weather_request("current.json", {"q": location, "alerts": "yes"})
        
        location_info = data["location"]
        alerts = data.get("alerts", {}).get("alert", [])
        
        if not alerts:
            return f"No weather alerts for {location_info['name']}, {location_info['region']}, {location_info['country']}"
        
        result = f"Weather alerts for {location_info['name']}, {location_info['region']}, {location_info['country']}:\n\n"
        
        for i, alert in enumerate(alerts, 1):
            result += f"Alert {i}:\n"
            result += f"Headline: {alert.get('headline', 'N/A')}\n"
            result += f"Severity: {alert.get('severity', 'N/A')}\n"
            result += f"Areas: {alert.get('areas', 'N/A')}\n"
            result += f"Event: {alert.get('event', 'N/A')}\n"
            if alert.get('effective'):
                result += f"Effective: {alert['effective']}\n"
            if alert.get('expires'):
                result += f"Expires: {alert['expires']}\n"
            if alert.get('desc'):
                result += f"Description: {alert['desc']}\n"
            result += "\n"
        
        return result
        
    except Exception as e:
        return f"Error getting weather alerts: {str(e)}"


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
    parser = argparse.ArgumentParser(description="Run MCP stdio-based weather server")
    args = parser.parse_args()

    asyncio.run(run_stdio_server(mcp_server))