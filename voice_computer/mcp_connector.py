"""
MCP (Model Context Protocol) connector for integrating with MCP servers.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
import subprocess
import json
import asyncio
import logging

_logger = logging.getLogger(__name__)


class ToolDescription(BaseModel):
    """Description of an MCP tool."""
    name: str
    description: str
    inputSchema: Dict[str, Any]


class MCPTools(BaseModel):
    """Container for MCP tools and server information."""
    server_description: str
    tools: List[ToolDescription]
    server_name: str
    server_path: str
    server_args: List[str]

    class Config:
        arbitrary_types_allowed = True

    def __getitem__(self, index: int) -> ToolDescription:
        """Allow indexing to get tools by index"""
        return self.tools[index]

    def __len__(self) -> int:
        """Return number of tools"""
        return len(self.tools)

    def get_tools_descriptions(self) -> str:
        """Get a string representation of available tool descriptions"""
        return "\n".join(
            f"{i}) Name: {tool.name}, Description: {tool.description.strip()}, Input schema: {str(tool.inputSchema).strip()}"
            for i, tool in enumerate(self.tools)
        )

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool by name with given arguments.
        
        This implementation uses subprocess to call MCP servers directly.
        """
        try:
            # Prepare the initialization request first
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "clientInfo": {
                        "name": "voice-computer",
                        "version": "1.0.0"
                    }
                }
            }
            
            # Prepare the initialized notification
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }
            
            # Prepare the tool call request
            tool_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            # Start the MCP server process
            process = await asyncio.create_subprocess_exec(
                self.server_path,
                *self.server_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Send initialization, notification, and tool call requests
            init_data = json.dumps(init_request) + "\n"
            notification_data = json.dumps(initialized_notification) + "\n"
            tool_data = json.dumps(tool_request) + "\n"
            request_data = init_data + notification_data + tool_data
            
            stdout, stderr = await process.communicate(request_data.encode())
            
            if process.returncode != 0:
                _logger.error(f"MCP server exited with code {process.returncode}: {stderr.decode()}")
                return f"Error: MCP server failed with exit code {process.returncode}"
            
            # Parse the responses (expecting two JSON-RPC responses)
            response_text = stdout.decode().strip()
            if response_text:
                lines = response_text.split('\n')
                # Look for the tool call response (id=2)
                for line in lines:
                    if line.strip():
                        try:
                            response = json.loads(line)
                            if response.get("id") == 2:  # Tool call response
                                if "result" in response:
                                    result = response["result"]
                                    if isinstance(result, dict):
                                        # Check for structured content first
                                        if "structuredContent" in result:
                                            return str(result["structuredContent"])
                                        # Then check for content array
                                        elif "content" in result and isinstance(result["content"], list) and len(result["content"]) > 0:
                                            content_item = result["content"][0]
                                            if isinstance(content_item, dict) and "text" in content_item:
                                                return content_item["text"]
                                            else:
                                                return str(content_item)
                                        else:
                                            return str(result)
                                    else:
                                        return str(result)
                                elif "error" in response:
                                    return f"Error: {response['error']['message']}"
                        except json.JSONDecodeError:
                            continue
            
            return "No valid response from MCP server"
            
        except Exception as e:
            _logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return f"Error calling tool: {str(e)}"

    async def call_tool_by_index(self, index: int, arguments: Dict[str, Any]) -> Any:
        """Call a tool by index with given arguments"""
        if index < 0 or index >= len(self.tools):
            raise IndexError(f"Tool index {index} out of range")

        tool_name = self.tools[index].name
        return await self.call_tool(tool_name, arguments)


class MCPConnector:
    """Base connector for MCP servers."""

    def __init__(self, server_name: str, server_path: str, server_args: Optional[List[str]] = None):
        self.server_name = server_name
        self.server_path = server_path  
        self.server_args = server_args or []

    async def get_tools(self) -> MCPTools:
        """
        Connect to MCP server and get available tools.
        
        This is a simplified implementation that returns mock tools.
        In a real implementation, this would query the MCP server for available tools.
        """
        try:
            # For now, return some common tools based on server type
            tools = []
            
            if "default" in self.server_name.lower():
                tools = [
                    ToolDescription(
                        name="add_two_numbers",
                        description="Calculate the sum of two integers",
                        inputSchema={
                            "type": "object", 
                            "properties": {
                                "lhs": {"type": "integer"}, 
                                "rhs": {"type": "integer"}
                            }, 
                            "required": ["lhs", "rhs"]
                        }
                    )
                ]
            elif "filesystem" in self.server_name.lower():
                tools = [
                    ToolDescription(
                        name="read_file",
                        description="Read the contents of a file",
                        inputSchema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                    ),
                    ToolDescription(
                        name="write_file", 
                        description="Write content to a file",
                        inputSchema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}
                    ),
                    ToolDescription(
                        name="list_directory",
                        description="List files and directories in a path",
                        inputSchema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
                    )
                ]
            elif "search" in self.server_name.lower() or "brave" in self.server_name.lower():
                tools = [
                    ToolDescription(
                        name="web_search",
                        description="Search the web using Brave Search",
                        inputSchema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
                    )
                ]
            else:
                # Generic tools for unknown servers
                tools = [
                    ToolDescription(
                        name="generic_tool",
                        description="A generic tool provided by the MCP server",
                        inputSchema={"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}
                    )
                ]

            return MCPTools(
                server_description=f"MCP server: {self.server_name}",
                tools=tools,
                server_name=self.server_name,
                server_path=self.server_path,
                server_args=self.server_args
            )

        except Exception as e:
            _logger.error(f"Error connecting to MCP server {self.server_name}: {e}")
            # Return empty tools list on error
            return MCPTools(
                server_description=f"MCP server: {self.server_name} (error connecting)",
                tools=[],
                server_name=self.server_name,
                server_path=self.server_path,
                server_args=self.server_args
            )


class MCPStdioConnector(MCPConnector):
    """MCP Connector for stdio-based servers"""

    def __init__(self, server_name: str, server_path: str, server_args: Optional[List[str]] = None):
        super().__init__(server_name, server_path, server_args)