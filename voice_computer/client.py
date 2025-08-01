"""
Main voice computer client that integrates all components.
"""

import asyncio
import logging
import sys
from io import StringIO
from typing import Optional, List, Dict, Any

from .voice_interface import VoiceInterface
from .ollama_client import OllamaClient
from .data_types import Messages, Utterance
from .tool_handler import ToolHandler
from .mcp_connector import MCPStdioConnector
from .config import Config
from .streaming_display import stream_colored_to_console

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
            "mcp_servers": [],
            "streaming": {
                "enabled": True,
                "token_batch_size": 4,
                "flush_delay": 0.1
            }
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
                    description=f"{server_config['name']}",
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
    
    async def process_query(self, query: str, use_streaming: bool = True, use_colored_output: bool = True) -> str:
        """
        Process a user query using MCP tools if available.
        
        Args:
            query: The user's query
            use_streaming: Whether to use streaming output (default True)
            use_colored_output: Whether to use colored output in streaming mode (default True)
            
        Returns:
            The response to the query
        """
        if not self.tool_handler:
            # Fallback to direct Ollama if no MCP tools
            messages = Messages(utterances=[
                Utterance(role="user", content=query)
            ])
            messages = self._add_tool_results_to_system_prompt(messages)
            
            if use_streaming and self._is_streaming_enabled():
                return await self._process_streaming_query(messages, use_colored_output)
            else:
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
            
            if use_streaming and self._is_streaming_enabled():
                return await self._process_streaming_query(messages, use_colored_output)
            else:
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
    
    async def _handle_voice_activation_cycle(self) -> bool:
        """
        Handle a single voice activation cycle (hotword detection + conversation).
        
        Returns:
            True to continue the main loop, False to exit
        """
        try:
            # Wait for hotword to activate conversation
            _logger.info("Waiting for activation hotword...")
            detected_hotword, immediate_instruction = await self.voice_interface._wait_for_hotword()
            
            # Check if this is a single-command activation or persistent conversation
            if immediate_instruction:
                # Single-command mode: process the instruction immediately and return to hotword listening
                _logger.info(f"Single-command mode activated with instruction: '{immediate_instruction}'")
                
                # Process the immediate instruction
                response = await self.process_query(immediate_instruction, use_streaming=True, use_colored_output=True)
                
                # Speak response
                if response:
                    await self.voice_interface.output(response)
                else:
                    await self.voice_interface.output("I'm sorry, I couldn't process that.")
                
                # Continue to next hotword detection cycle (no persistent conversation)
                return True
            else:
                # Persistent conversation mode
                conversation_is_on = True
                await self.voice_interface.output("Voice assistant ready. How can I help you?")
            
            # Inner loop: Handle conversation until exit command
            while conversation_is_on:
                try:
                    # Get voice input (skipping hotword detection since we're already activated)
                    from .voice_interface import COLOR_START, COLOR_END
                    print(f"{COLOR_START}✨ Listening for your command...{COLOR_END}")
                    user_input = ""
                    while not user_input:
                        user_input = await self.voice_interface._listener.input()
                        if not user_input or not user_input.strip() or user_input.strip() == "[unclear]":
                            continue
                        
                        user_input = self.voice_interface._remove_activation_word_and_normalize(user_input)
                        break

                    # Simple quality check - if text seems too short or unclear, ask for repeat
                    while self.voice_interface._is_listening and self.voice_interface._not_good_enough(user_input):
                        print(f"{COLOR_START}user> {user_input}{COLOR_END}")
                        await self.voice_interface.output("Sorry? Can you repeat?")
                        user_input = await self.voice_interface._listener.input()
                        user_input = self.voice_interface._remove_activation_word_and_normalize(user_input)

                    user_input = user_input.lower().capitalize()
                    user_input = self.voice_interface._remove_unclear(user_input)
                    print(f"{COLOR_START}user> {user_input}{COLOR_END}")
                    
                    if not user_input or not user_input.strip():
                        continue
                    
                    # Handle exit commands
                    exit_commands = ['exit', 'quit', 'stop', 'goodbye', 'bye']
                    if user_input.lower().strip() in exit_commands:
                        await self.voice_interface.output("Goodbye!")
                        conversation_is_on = False  # End conversation, return to hotword listening
                        break
                    
                    # Process with agent (enable streaming with colors for console output in voice mode)
                    response = await self.process_query(user_input, use_streaming=True, use_colored_output=True)
                    
                    # Speak response
                    if response:
                        await self.voice_interface.output(response)
                    else:
                        await self.voice_interface.output("I'm sorry, I couldn't process that.")
                        
                except KeyboardInterrupt:
                    _logger.info("Voice loop interrupted by user")
                    return False
                except Exception as e:
                    _logger.error(f"Error in voice loop: {e}")
                    await self.voice_interface.output("Sorry, I encountered an error.")
            
            return True
            
        except KeyboardInterrupt:
            _logger.info("Voice loop interrupted by user")
            return False
        except Exception as e:
            _logger.error(f"Error in outer voice loop: {e}")
            # Continue to next hotword detection cycle
            await asyncio.sleep(1)
            return True
    
    async def run_voice_loop(self) -> None:
        """Main voice interaction loop with hotword activation."""
        _logger.info("Starting voice interaction loop...")
        
        # Setup MCP tools
        await self._setup_mcp_tools()
        
        # Activate voice interface
        self.voice_interface.activate()
        
        try:
            # Main loop: Handle voice activation cycles
            while True:
                should_continue = await self._handle_voice_activation_cycle()
                if not should_continue:
                    break
                    
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
                    
                    # Process query with streaming (streaming display will handle "bot> " prefix)
                    response = await self.process_query(user_input, use_streaming=True)
                    
                    # Response is already printed via streaming, just handle empty response
                    if not response:
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
    
    def _is_streaming_enabled(self) -> bool:
        """Check if streaming is enabled in configuration."""
        streaming_config = self.config.get_value("streaming") or {}
        return streaming_config.get("enabled", True)
    
    def _get_token_batch_size(self) -> int:
        """Get the configured token batch size."""
        streaming_config = self.config.get_value("streaming") or {}
        return streaming_config.get("token_batch_size", 4)
    
    def _get_flush_delay(self) -> float:
        """Get the configured flush delay."""
        streaming_config = self.config.get_value("streaming") or {}
        return streaming_config.get("flush_delay", 0.1)
    
    async def _process_streaming_query(self, messages: Messages, use_colored_output: bool = True) -> str:
        """Process query with streaming output to console."""
        token_queue = asyncio.Queue()
        batch_size = self._get_token_batch_size()
        flush_delay = self._get_flush_delay()
        
        # Create streaming display task with colored output for text mode
        if use_colored_output:
            display_task = await stream_colored_to_console(
                token_queue=token_queue,
                prefix="bot> ",
                batch_size=batch_size,
                flush_delay=flush_delay
            )
        else:
            # For voice mode or when colored output is disabled
            from .streaming_display import stream_to_console
            display_task = await stream_to_console(
                token_queue=token_queue,
                batch_size=batch_size,
                flush_delay=flush_delay
            )
        
        try:
            # Start the streaming prediction (this will populate the queue)
            prediction_task = asyncio.create_task(
                self.ollama_client.predict(
                    messages,
                    stream=True,
                    token_queue=token_queue
                )
            )
            
            # Wait for both tasks to complete
            response, _ = await asyncio.gather(prediction_task, display_task)
            
            return response.message
            
        except Exception as e:
            display_task.cancel()
            raise e


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