"""
Main voice computer client that integrates all components.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any

from .voice_interface import VoiceInterface
from .ollama_client import OllamaClient
from .data_types import Messages, Utterance
from .tool_handler import ToolHandler
from .mcp_connector import MCPStdioConnector
from .config import Config

_logger = logging.getLogger(__name__)


class SimpleConfig:
    """Simple configuration class."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or {
            "listener_model": {
                "listener_silence_timeout": 2,
                "listener_volume_threshold": 1,
                "listener_hotword_logp": -8
            },
            "waking_up_sound": True,
            "deactivate_sound": True,
            "ollama_host": "http://localhost:11434",
            "ollama_model": "qwen2.5:32b",
            "mcp_servers": []
        }
    
    def get_value(self, key: str) -> Any:
        """Get a configuration value."""
        return self._config.get(key)
    
    def set_value(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._config[key] = value


class VoiceComputerClient:
    """Main voice computer client that coordinates all components."""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        
        # Initialize Ollama client
        self.ollama_client = OllamaClient(
            model=self.config.get_value("ollama_model"),
            host=self.config.get_value("ollama_host")
        )
        
        # Initialize voice interface
        self.voice_interface = VoiceInterface(self.config)
        
        # Initialize MCP tools and handler
        self.mcp_tools = []
        self.tool_handler = None
        
        # Tool results queue (last 5 results)
        self.tool_results_queue = []
        self._initialize_mcp_tools()
        
        _logger.info("VoiceComputerClient initialized")
    
    def _initialize_mcp_tools(self) -> None:
        """Initialize MCP tools from configuration."""
        mcp_servers = self.config.get_value("mcp_servers") or []
        
        for server_config in mcp_servers:
            try:
                connector = MCPStdioConnector(
                    command=server_config["path"],
                    description=f"MCP server: {server_config['name']}",
                    args=server_config.get("args", [])
                )
                
                # Get tools asynchronously (we'll do this in the run method)
                self.mcp_tools.append(connector)
                _logger.info(f"Configured MCP connector: {server_config['name']}")
                
            except Exception as e:
                _logger.error(f"Failed to configure MCP connector {server_config['name']}: {e}")
        
        _logger.info(f"Configured {len(self.mcp_tools)} MCP connectors")
    
    async def _setup_mcp_tools(self) -> None:
        """Setup MCP tools asynchronously."""
        tools = []
        for connector in self.mcp_tools:
            try:
                mcp_tools = await connector.get_tools()
                if mcp_tools.tools:  # Only add if tools are available
                    tools.append(mcp_tools)
                    _logger.info(f"Loaded {len(mcp_tools.tools)} tools from {mcp_tools.server_description}")
                else:
                    _logger.warning(f"No tools available from {mcp_tools.server_description}")
            except Exception as e:
                _logger.error(f"Failed to get tools from connector: {e}")
        
        if tools:
            self.tool_handler = ToolHandler(self.ollama_client, tools, self.config)
            _logger.info(f"ToolHandler initialized with {len(tools)} MCP tool groups")
        else:
            _logger.warning("No MCP tools available - running in basic mode")
    
    async def process_query(self, query: str) -> str:
        """
        Process a user query using MCP tools if available.
        
        Args:
            query: The user's query
            
        Returns:
            The response to the query
        """
        if not self.tool_handler:
            # Fallback to direct Ollama if no MCP tools
            messages = Messages(utterances=[
                Utterance(role="user", content=query)
            ])
            messages = self._add_tool_results_to_system_prompt(messages)
            response = await self.ollama_client.predict(messages)
            return response.message
        
        try:
            # Execute relevant tools using the handler
            tool_results = await self.tool_handler.handle_query(query)
            
            # Update tool results queue (keep only last 5)
            self.tool_results_queue.extend(tool_results)
            if len(self.tool_results_queue) > 5:
                self.tool_results_queue = self.tool_results_queue[-5:]
            
            # Generate response using LLM with tool results context
            messages = Messages(utterances=[
                Utterance(role="user", content=query)
            ])
            messages = self._add_tool_results_to_system_prompt(messages)
            
            response = await self.ollama_client.predict(messages)
            return response.message
            
        except Exception as e:
            _logger.error(f"Error processing query with MCP: {e}")
            return f"I encountered an error: {str(e)}"
    
    def _add_tool_results_to_system_prompt(self, messages: Messages) -> Messages:
        """Add recent tool results to the system prompt."""
        if not self.tool_results_queue:
            return messages
        
        # Build tool results context
        tool_context = "Recent tool execution results:\n\n"
        for i, result in enumerate(self.tool_results_queue):
            tool_context += f"Tool Result {i+1}:\n"
            tool_context += f"Query: {result.original_query}\n"
            tool_context += f"Tool: {result.tool_description}\n"
            tool_context += f"Result: {result.tool_result}\n\n"
        
        # Create system prompt with tool context
        system_prompt = f"""You are a helpful voice assistant. Use the recent tool results below to provide informed responses to user queries.

{tool_context}

Instructions:
1. Use the tool results to answer questions when relevant
2. Reference specific tool results when helpful
3. Be conversational and helpful
4. If tool results don't contain relevant information, use your general knowledge"""
        
        return messages.add_system_prompt(system_prompt)
    
    async def run_voice_loop(self) -> None:
        """Main voice interaction loop."""
        _logger.info("Starting voice interaction loop...")
        
        # Setup MCP tools
        await self._setup_mcp_tools()
        
        # Activate voice interface
        self.voice_interface.activate()
        
        try:
            await self.voice_interface.output("Voice assistant ready. How can I help you?")
            
            while True:
                try:
                    # Get voice input
                    user_input = await self.voice_interface.input()
                    
                    if not user_input or not user_input.strip():
                        continue
                    
                    # Handle exit commands
                    exit_commands = ['exit', 'quit', 'stop', 'goodbye', 'bye']
                    if user_input.lower().strip() in exit_commands:
                        await self.voice_interface.output("Goodbye!")
                        break
                    
                    # Process with agent
                    response = await self.process_query(user_input)
                    
                    # Speak response
                    if response:
                        await self.voice_interface.output(response)
                    else:
                        await self.voice_interface.output("I'm sorry, I couldn't process that.")
                        
                except KeyboardInterrupt:
                    _logger.info("Voice loop interrupted by user")
                    break
                except Exception as e:
                    _logger.error(f"Error in voice loop: {e}")
                    await self.voice_interface.output("Sorry, I encountered an error.")
                    
        finally:
            self.voice_interface.deactivate()
            _logger.info("Voice interaction loop ended")
    
    async def run_text_loop(self) -> None:
        """Run in text-only mode for testing."""
        _logger.info("Starting text interaction loop...")
        
        # Setup MCP tools
        await self._setup_mcp_tools()
        
        try:
            print("Voice assistant ready in text mode. Type 'exit' to quit.")
            
            while True:
                try:
                    # Get text input
                    user_input = input("\nuser> ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle exit commands
                    exit_commands = ['exit', 'quit', 'stop', 'goodbye', 'bye']
                    if user_input.lower() in exit_commands:
                        print("bot> Goodbye!")
                        break
                    
                    # Process query
                    response = await self.process_query(user_input)
                    
                    # Print response
                    if response:
                        print(f"bot> {response}")
                    else:
                        print("bot> I'm sorry, I couldn't process that.")
                        
                except KeyboardInterrupt:
                    _logger.info("Text loop interrupted by user")
                    break
                except EOFError:
                    break
                except Exception as e:
                    _logger.error(f"Error in text loop: {e}")
                    print("bot> Sorry, I encountered an error.")
                    
        finally:
            _logger.info("Text interaction loop ended")
    
    def add_mcp_server(self, name: str, path: str, args: Optional[List[str]] = None) -> None:
        """
        Add an MCP server configuration.
        
        Args:
            name: Name of the MCP server
            path: Path to the MCP server executable
            args: Optional command-line arguments
        """
        server_config = {
            "name": name,
            "path": path,
            "args": args or []
        }
        
        current_servers = self.config.get_value("mcp_servers") or []
        current_servers.append(server_config)
        self.config.set_value("mcp_servers", current_servers)
        
        # Reinitialize MCP tools
        self._initialize_mcp_tools()
        
        _logger.info(f"Added MCP server: {name}")


async def main():
    """Example usage."""
    # Create client
    client = VoiceComputerClient()
    
    # Example: Add MCP servers (configure based on your setup)
    # client.add_mcp_server("filesystem", "mcp-server-filesystem", ["--root", "/tmp"])
    # client.add_mcp_server("web-search", "mcp-server-brave-search", ["--api-key", "your-key"])
    
    # Run the voice loop
    await client.run_voice_loop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())