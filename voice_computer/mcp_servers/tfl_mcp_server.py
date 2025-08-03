#!/usr/bin/env python3
"""
TFL (Transport for London) MCP Server

Provides tools to check the status of London's transport network including:
- Tube lines
- DLR (Docklands Light Railway)
- Bus routes
- Other transport modes

Uses the official TFL API to get real-time status information.
"""

import asyncio
import json
import logging
import urllib.request
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

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

# Create MCP tool wrappers that call the internal functions
@mcp.tool()
def get_transport_status(
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

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()