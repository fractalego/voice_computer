#!/usr/bin/env python3
"""
TFL (Transport for London) MCP Server

Provides tools to check the status of London's transport network including:
- Tube lines
- DLR (Docklands Light Railway)
- Bus routes and real-time countdown information
- Other transport modes

Uses the official TFL API to get real-time status and countdown information.
"""

import asyncio
import json
import logging
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from fuzzywuzzy import fuzz

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tfl-mcp-server")

# Initialize the MCP server
mcp = FastMCP("TFL Transport Status")

def _get_transport_status(
    modes: str = "dlr,tube",
    line_ids: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get the current status of London transport lines.
    
    Args:
        modes: Comma-separated transport modes (e.g., "dlr,tube", "bus", "overground")
        line_ids: Optional comma-separated line IDs to filter specific lines
        
    Returns:
        Dictionary containing status information for requested transport modes
    """
    try:
        # Build the API URL
        if line_ids:
            # Query specific lines
            url = f"https://api.tfl.gov.uk/Line/{line_ids}/Status"
        else:
            # Query by transport modes
            url = f"https://api.tfl.gov.uk/Line/Mode/{modes}/Status"
        
        logger.info(f"Querying TFL API: {url}")
        
        # Set up headers
        headers = {
            'Cache-Control': 'no-cache',
            'User-Agent': 'TFL-MCP-Server/1.0'
        }
        
        # Make the request
        req = urllib.request.Request(url, headers=headers)
        req.get_method = lambda: 'GET'
        
        with urllib.request.urlopen(req, timeout=10) as response:
            status_code = response.getcode()
            data = response.read().decode('utf-8')
            
            if status_code == 200:
                # Parse JSON response
                transport_data = json.loads(data)
                
                # Process and format the response
                result = {
                    "status": "success",
                    "query": {
                        "modes": modes,
                        "line_ids": line_ids,
                        "url": url
                    },
                    "lines": []
                }
                
                for line in transport_data:
                    line_info = {
                        "id": line.get("id", ""),
                        "name": line.get("name", ""),
                        "mode": line.get("modeName", ""),
                        "disruptions": [],
                        "status": "Unknown"
                    }
                    
                    # Extract line status
                    line_statuses = line.get("lineStatuses", [])
                    if line_statuses:
                        primary_status = line_statuses[0]
                        line_info["status"] = primary_status.get("statusSeverityDescription", "Unknown")
                        
                        # Check for disruptions
                        if primary_status.get("statusSeverity", 10) < 10:
                            line_info["disruptions"].append({
                                "category": primary_status.get("statusSeverityDescription", "Disruption"),
                                "description": primary_status.get("reason", "No details available")
                            })
                        
                        # Add additional disruption details if available
                        disruption = primary_status.get("disruption")
                        if disruption:
                            line_info["disruptions"].append({
                                "category": disruption.get("categoryDescription", "Service Update"),
                                "description": disruption.get("description", "No details available"),
                                "additional_info": disruption.get("additionalInfo", "")
                            })
                    
                    result["lines"].append(line_info)
                
                # Sort lines by name for consistent output
                result["lines"].sort(key=lambda x: x["name"])
                
                logger.info(f"Successfully retrieved status for {len(result['lines'])} lines")
                return result
                
            else:
                error_msg = f"TFL API returned status code {status_code}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "status_code": status_code
                }
                
    except urllib.error.HTTPError as e:
        error_msg = f"HTTP error accessing TFL API: {e.code} - {e.reason}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "status_code": e.code
        }
    except urllib.error.URLError as e:
        error_msg = f"URL error accessing TFL API: {e.reason}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg
        }
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse TFL API response as JSON: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg
        }
    except Exception as e:
        error_msg = f"Unexpected error querying TFL API: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg
        }

def _get_bus_countdown(
    station_name: str,
    route: Optional[str] = None,
    max_results: int = 10,
    fuzzy_threshold: int = 70
) -> Dict[str, Any]:
    """
    Get bus countdown information for a specific station and optionally filter by route.
    
    Args:
        station_name: Name of the bus station to search for
        route: Optional bus route number/identifier to filter by
        max_results: Maximum number of results to return
        fuzzy_threshold: Minimum fuzzy match score (0-100) for station name matching
        
    Returns:
        Dictionary containing countdown information for buses at the station
    """
    try:
        # Query the TFL countdown API
        url = "https://countdown.api.tfl.gov.uk/interfaces/ura/instant_V1"
        logger.info(f"Querying TFL countdown API: {url}")
        
        # Set up headers
        headers = {
            'Cache-Control': 'no-cache',
            'User-Agent': 'TFL-MCP-Server/1.0'
        }
        
        # Make the request
        req = urllib.request.Request(url, headers=headers)
        req.get_method = lambda: 'GET'
        
        with urllib.request.urlopen(req, timeout=15) as response:
            status_code = response.getcode()
            data = response.read().decode('utf-8')
            
            if status_code == 200:
                # Parse the response - it's in a specific format, not JSON
                lines = data.strip().split('\n')
                
                # Process each line and extract bus information
                all_buses = []
                for line in lines:
                    if line.strip():
                        try:
                            # Parse the line - format is typically: [1, "STOP NAME", "ROUTE", TIMESTAMP, ...]
                            # Remove brackets and split by comma
                            cleaned_line = line.strip()
                            if cleaned_line.startswith('[') and cleaned_line.endswith(']'):
                                cleaned_line = cleaned_line[1:-1]
                            
                            # Split by comma and clean up quotes
                            parts = [part.strip().strip('"') for part in cleaned_line.split(',')]
                            
                            if len(parts) >= 4:
                                stop_name = parts[1]
                                bus_route = parts[2]
                                timestamp_str = parts[3]
                                
                                # Convert timestamp to readable format
                                try:
                                    # The timestamp appears to be in milliseconds
                                    timestamp_ms = int(timestamp_str)
                                    # Convert to seconds for datetime
                                    timestamp_s = timestamp_ms / 1000
                                    arrival_time = datetime.fromtimestamp(timestamp_s)
                                    
                                    # Calculate minutes until arrival
                                    now = datetime.now()
                                    time_diff = arrival_time - now
                                    minutes_until = int(time_diff.total_seconds() / 60)
                                    
                                except (ValueError, OSError):
                                    # If timestamp parsing fails, use the raw value
                                    arrival_time = None
                                    minutes_until = None
                                
                                all_buses.append({
                                    'stop_name': stop_name,
                                    'route': bus_route,
                                    'timestamp': timestamp_str,
                                    'arrival_time': arrival_time.isoformat() if arrival_time else None,
                                    'minutes_until': minutes_until
                                })
                                
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Failed to parse line: {line.strip()[:50]}... Error: {e}")
                            continue
                
                # Filter by station name using fuzzy matching
                matching_buses = []
                for bus in all_buses:
                    # Use fuzzy string matching to find similar station names
                    similarity = fuzz.ratio(station_name.lower(), bus['stop_name'].lower())
                    if similarity >= fuzzy_threshold:
                        bus['name_similarity'] = similarity
                        matching_buses.append(bus)
                
                # Sort by similarity score (highest first)
                matching_buses.sort(key=lambda x: x['name_similarity'], reverse=True)
                
                # Filter by route if specified
                if route:
                    route_filtered = []
                    for bus in matching_buses:
                        if fuzz.ratio(route.lower(), bus['route'].lower()) >= 80:
                            route_filtered.append(bus)
                    matching_buses = route_filtered
                
                # Limit results
                matching_buses = matching_buses[:max_results]
                
                # Remove similarity score from final results (it was just for sorting)
                for bus in matching_buses:
                    bus.pop('name_similarity', None)
                
                result = {
                    "status": "success",
                    "query": {
                        "station_name": station_name,
                        "route": route,
                        "fuzzy_threshold": fuzzy_threshold,
                        "url": url
                    },
                    "total_buses_found": len(all_buses),
                    "matching_buses": len(matching_buses),
                    "buses": matching_buses
                }
                
                logger.info(f"Found {len(matching_buses)} matching buses for station '{station_name}'" + 
                           (f" on route '{route}'" if route else ""))
                return result
                
            else:
                error_msg = f"TFL countdown API returned status code {status_code}"
                logger.error(error_msg)
                return {
                    "status": "error",
                    "error": error_msg,
                    "status_code": status_code
                }
                
    except urllib.error.HTTPError as e:
        error_msg = f"HTTP error accessing TFL countdown API: {e.code} - {e.reason}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "status_code": e.code
        }
    except urllib.error.URLError as e:
        error_msg = f"URL error accessing TFL countdown API: {e.reason}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg
        }
    except Exception as e:
        error_msg = f"Unexpected error querying TFL countdown API: {e}"
        logger.error(error_msg)
        return {
            "status": "error",
            "error": error_msg
        }

# Create MCP tool wrappers that call the internal functions
@mcp.tool()
def get_transport_status(
    modes: str = "dlr,tube,bus,overground",
    line_ids: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get the current status of London transport lines.
    
    Args:
        modes: Comma-separated transport modes (e.g., "dlr,tube", "bus", "overground")
        line_ids: Optional comma-separated line IDs to filter specific lines
        
    Returns:
        Dictionary containing status information for requested transport modes
    """
    return _get_transport_status(modes, line_ids)

@mcp.tool()
def get_tube_status() -> Dict[str, Any]:
    """
    Get the current status of all London Underground (Tube) lines.
    This is a convenience function that queries only tube lines.
    
    Returns:
        Dictionary containing status information for all tube lines
    """
    return _get_transport_status(modes="tube")

@mcp.tool()
def get_dlr_status() -> Dict[str, Any]:
    """
    Get the current status of the DLR (Docklands Light Railway).
    
    Returns:
        Dictionary containing status information for DLR
    """
    return _get_transport_status(modes="dlr")

@mcp.tool()
def get_overground_status() -> Dict[str, Any]:
    """
    Get the current status of London Overground lines.
    
    Returns:
        Dictionary containing status information for Overground lines
    """
    return _get_transport_status(modes="overground")

@mcp.tool()
def get_bus_status() -> Dict[str, Any]:
    """
    Get the current status of London bus services.
    Note: This may return a large amount of data as there are many bus routes.
    
    Returns:
        Dictionary containing status information for bus services
    """
    return _get_transport_status(modes="bus")

@mcp.tool()
def get_specific_line_status(line_id: str) -> Dict[str, Any]:
    """
    Get the status of a specific transport line by its ID.
    
    Args:
        line_id: The TFL line ID (e.g., "central", "piccadilly", "dlr")
        
    Returns:
        Dictionary containing status information for the specified line
    """
    return _get_transport_status(modes="", line_ids=line_id)

@mcp.tool()
def get_bus_countdown(
    station_name: str,
    route: Optional[str] = None,
    max_results: int = 10,
    fuzzy_threshold: int = 70
) -> Dict[str, Any]:
    """
    Get real-time bus countdown information for a specific station.
    
    This tool uses fuzzy string matching to find bus stops that match the provided
    station name and optionally filters by bus route number.
    
    Args:
        station_name: Name of the bus station to search for (e.g., "Oxford Circus", "King's Cross")
        route: Optional bus route number/identifier to filter by (e.g., "73", "N73")
        max_results: Maximum number of results to return (default: 10)
        fuzzy_threshold: Minimum fuzzy match score 0-100 for station name matching (default: 70)
        
    Returns:
        Dictionary containing:
        - status: "success" or "error"
        - query: Details about the search parameters
        - total_buses_found: Total number of buses in the system
        - matching_buses: Number of buses matching the criteria
        - buses: List of matching buses with arrival times and countdown information
        
    Example:
        get_bus_countdown("Oxford Street", "73") - Get route 73 buses at Oxford Street
        get_bus_countdown("Kings Cross") - Get all buses at King's Cross station
    """
    return _get_bus_countdown(station_name, route, max_results, fuzzy_threshold)

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()