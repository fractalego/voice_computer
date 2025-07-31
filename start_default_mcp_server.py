#!/usr/bin/env python3
"""
Test the default MCP server for the voice computer system.

Note: With stdio transport, the MCP server is automatically started
by the voice computer client. This script is for testing the server directly.
"""

import subprocess
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mcp_server():
    """Test the default MCP server."""
    try:
        # Get the path to the default MCP server
        server_path = Path(__file__).parent / "voice_computer" / "default_mcp_server.py"
        
        if not server_path.exists():
            logger.error(f"Default MCP server not found at: {server_path}")
            return False
        
        logger.info("Testing default MCP server with stdio transport")
        logger.info("This server provides tools like 'add_two_numbers'")
        logger.info("Note: In normal operation, the server is started automatically by the voice client")
        
        # Test the server directly
        cmd = [sys.executable, str(server_path)]
        
        logger.info("Starting server test...")
        logger.info("You can send MCP commands via stdin (Ctrl+C to exit)")
        
        try:
            process = subprocess.run(cmd, check=False)
            return process.returncode == 0
                
        except FileNotFoundError:
            logger.error("Python interpreter not found")
            return False
        except Exception as e:
            logger.error(f"Error testing MCP server: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to test MCP server: {e}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("MCP Server Test Utility")
    logger.info("=" * 60)
    logger.info("The default MCP server uses stdio transport and is automatically")
    logger.info("started by the voice computer client. No separate server process needed!")
    logger.info("")
    logger.info("To run the voice computer:")
    logger.info("  python run_voice_computer.py --test")
    logger.info("=" * 60)
    
    success = test_mcp_server()
    sys.exit(0 if success else 1)