"""
Main voice computer client that integrates all components.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any

from .voice_interface import VoiceInterface
from .data_types import Messages, Utterance
from .tool_handler import ToolHandler
from .mcp_connector import MCPStdioConnector
from .config import Config
from .streaming_display import stream_colored_to_console_with_tts, stream_colored_to_console
from .speaker import TTSSpeaker
from .entailer import Entailer
from .model_factory import get_model_factory
from .prompt import get_voice_assistant_system_prompt, format_tool_context

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
            },
            "exit_sentences": [
                "please stop this conversation",
                "good bye",
                "thank you",
                "shut up",
                "stop talking",
                "end conversation",
                "I'm done",
                "that's all"
            ],
            "exit_entailment_threshold": 0.7,
            "facts": [
                "The name of this chatbot is 'Computer'"
            ]
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
        
        # Initialize Ollama client using model factory
        model_factory = get_model_factory()
        self.ollama_client = model_factory.get_ollama_client(
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
        
        # Conversation history (list of utterances)
        self.conversation_history = []
        
        self._initialize_mcp_tools()
        
        # Initialize entailer for exit detection
        self.entailer = Entailer(self.config)
        
        # Initialize TTS speaker for streaming speech
        try:
            self.tts_speaker = TTSSpeaker()
            _logger.info("TTSSpeaker initialized successfully")
        except Exception as e:
            _logger.warning(f"Failed to initialize TTSSpeaker: {e}")
            self.tts_speaker = None

        self._initialize_models()

        _logger.info("VoiceComputerClient initialized")
    
    def _reset_conversation_state(self):
        """Reset conversation history and tool results when conversation ends."""
        _logger.info("Resetting conversation state - clearing history and tool results")
        self.conversation_history.clear()
        self.tool_results_queue.clear()
    
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
    
    async def process_query(self, query: str, use_streaming: bool = True, use_colored_output: bool = True, use_tts: bool = False) -> str:
        """
        Process a user query using MCP tools if available.
        
        Args:
            query: The user's query
            use_streaming: Whether to use streaming output (default True)
            use_colored_output: Whether to use colored output in streaming mode (default True)
            
        Returns:
            The response to the query
        """
        # Add user query to conversation history
        self.conversation_history.append(Utterance(role="user", content=query))
        
        if not self.tool_handler:
            # Fallback to direct Ollama if no MCP tools
            messages = self._build_messages_with_history()
            messages = self._add_tool_results_to_system_prompt(messages)
            
            if use_streaming and self._is_streaming_enabled():
                response = await self._process_streaming_query(messages, use_colored_output, use_tts)
            else:
                response_obj = await self.ollama_client.predict(messages)
                response = response_obj.message
            
            # Add assistant response to conversation history
            self.conversation_history.append(Utterance(role="assistant", content=response))
            return response
        
        try:
            # Execute relevant tools using the handler
            tool_results = await self.tool_handler.handle_query(query)
            
            # Update tool results queue (keep only last 5)
            self.tool_results_queue.extend(tool_results)
            if len(self.tool_results_queue) > 5:
                self.tool_results_queue = self.tool_results_queue[-5:]
            
            # Generate response using LLM with conversation history and tool results context
            messages = self._build_messages_with_history()
            messages = self._add_tool_results_to_system_prompt(messages)
            
            if use_streaming and self._is_streaming_enabled():
                response = await self._process_streaming_query(messages, use_colored_output, use_tts)
            else:
                response_obj = await self.ollama_client.predict(messages)
                response = response_obj.message
            
            # Add assistant response to conversation history
            self.conversation_history.append(Utterance(role="assistant", content=response))
            return response
            
        except Exception as e:
            _logger.error(f"Error processing query with MCP: {e}")
            error_response = f"I encountered an error: {str(e)}"
            # Add error response to conversation history
            self.conversation_history.append(Utterance(role="assistant", content=error_response))
            return error_response
    
    def _build_messages_with_history(self) -> Messages:
        """Build Messages object with conversation history."""
        # Keep only recent conversation history (last 10 exchanges to prevent context overflow)
        max_history = 20  # 10 user + 10 assistant messages
        recent_history = self.conversation_history[-max_history:] if len(self.conversation_history) > max_history else self.conversation_history
        
        return Messages(utterances=recent_history)
    
    def _add_tool_results_to_system_prompt(self, messages: Messages) -> Messages:
        """Add recent tool results to the system prompt."""
        # Format tool context and get facts from config
        tool_context = format_tool_context(self.tool_results_queue)
        facts = self.config.get_value("facts") or []
        system_prompt = get_voice_assistant_system_prompt(tool_context, facts)
        
        return messages.add_system_prompt(system_prompt)
    
    def _is_exit_command(self, user_input: str) -> bool:
        """
        Check if user input is an exit command using entailment.
        
        Args:
            user_input: The user's input text
            
        Returns:
            True if the input entails an exit command, False otherwise
        """
        if not user_input or not user_input.strip():
            return False
        
        # Get configured exit sentences and threshold
        exit_sentences = self.config.get_value("exit_sentences") or [
            "please stop this conversation",
            "good bye", 
            "thank you",
            "shut up"
        ]
        threshold = self.config.get_value("exit_entailment_threshold") or 0.7
        
        user_input_clean = user_input.strip().lower()
        
        try:
            # Check entailment against each exit sentence
            for exit_sentence in exit_sentences:
                score = self.entailer.judge(user_input_clean, exit_sentence.lower())
                _logger.debug(f"Exit entailment score for '{user_input_clean}' -> '{exit_sentence}': {score}")
                
                if score >= threshold:
                    _logger.info(f"Exit command detected: '{user_input_clean}' entails '{exit_sentence}' (score: {score})")
                    return True
            
            return False
            
        except Exception as e:
            _logger.error(f"Error checking exit command with entailer: {e}")
            # Fallback to simple keyword matching
            exit_keywords = ["goodbye", "bye", "stop", "quit", "exit", "shut up", "thank you"]
            user_lower = user_input_clean
            for keyword in exit_keywords:
                if keyword in user_lower:
                    _logger.info(f"Exit command detected via fallback keyword matching: '{keyword}' in '{user_input_clean}'")
                    return True
            return False
    
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
                
                # Play activation sound when conversation starts
                await self.voice_interface._play_activation_sound()
                
                # Process the immediate instruction
                response = await self.process_query(immediate_instruction, use_streaming=True, use_colored_output=True, use_tts=True)
                
                # TTS is handled by streaming, no need for additional speech output
                if not response:
                    # Only speak error messages that weren't generated by streaming
                    await self.voice_interface.output("I'm sorry, I couldn't process that.", skip_print=True)
                
                # Continue to next hotword detection cycle (no persistent conversation)
                return True
            else:
                # Persistent conversation mode
                conversation_is_on = True
                
                # Play activation sound when conversation starts
                await self.voice_interface._play_activation_sound()
                await self.voice_interface.output("How can I help you?")
            
            # Inner loop: Handle conversation until exit command
            while conversation_is_on:
                try:
                    # Get voice input (skipping hotword detection since we're already activated)
                    from .voice_interface import COLOR_START, COLOR_END, COLOR_GREEN
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
                        print(f"{COLOR_GREEN}user> {user_input}{COLOR_END}")
                        await self.voice_interface.output("Sorry? Can you repeat?")
                        user_input = await self.voice_interface._listener.input()
                        user_input = self.voice_interface._remove_activation_word_and_normalize(user_input)

                    user_input = user_input.lower().capitalize()
                    user_input = self.voice_interface._remove_unclear(user_input)
                    print(f"{COLOR_GREEN}user> {user_input}{COLOR_END}")
                    
                    if not user_input or not user_input.strip():
                        continue
                    
                    # Handle exit commands using entailer
                    if self._is_exit_command(user_input):
                        await self.voice_interface.output("Goodbye!")
                        # Reset conversation state when conversation ends
                        self._reset_conversation_state()
                        # Play deactivation sound when conversation ends
                        await self.voice_interface._play_deactivation_sound()
                        conversation_is_on = False  # End conversation, return to hotword listening
                        break
                    
                    # Process with agent (enable streaming with TTS for voice mode)
                    response = await self.process_query(user_input, use_streaming=True, use_colored_output=True, use_tts=True)
                    
                    # TTS is handled by streaming, no need for additional speech output
                    if not response:
                        # Only speak error messages that weren't generated by streaming
                        await self.voice_interface.output("I'm sorry, I couldn't process that.", skip_print=True)
                        
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
        
        await self._setup_mcp_tools()
        self.voice_interface.activate()

        if self.tts_speaker:
            activation_text = "Please say the activation word to start the conversation."
            self.tts_speaker.speak(activation_text)
            _logger.info(activation_text)

        try:
            # Main loop: Handle voice activation cycles
            while True:
                should_continue = await self._handle_voice_activation_cycle()
                if not should_continue:
                    break
                    
        finally:
            self.voice_interface.deactivate()
            # Clean up TTS speaker
            if hasattr(self, 'tts_speaker') and self.tts_speaker:
                try:
                    self.tts_speaker.cleanup()
                except Exception as e:
                    _logger.debug(f"Error cleaning up TTS speaker: {e}")
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
                    
                    # Handle exit commands using entailer
                    if self._is_exit_command(user_input):
                        print("bot> Goodbye!")
                        # Reset conversation state when conversation ends
                        self._reset_conversation_state()
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
    
    async def _process_streaming_query(self, messages: Messages, use_colored_output: bool = True, use_tts: bool = False) -> str:
        """Process query with streaming output to console and optional TTS."""
        token_queue = asyncio.Queue()
        batch_size = self._get_token_batch_size()
        flush_delay = self._get_flush_delay()
        
        # Create streaming display task
        if use_tts and self.tts_speaker:
            # Use TTS streaming for voice mode
            display_task = await stream_colored_to_console_with_tts(
                token_queue=token_queue,
                tts_speaker=self.tts_speaker,
                prefix="bot> ",
                batch_size=batch_size,
                flush_delay=flush_delay
            )
        elif use_colored_output:
            # Use colored output for text mode
            display_task = await stream_colored_to_console(
                token_queue=token_queue,
                prefix="bot> ",
                batch_size=batch_size,
                flush_delay=flush_delay
            )
        else:
            # Basic streaming without color or TTS
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

    def _initialize_models(self):
        # Initialize all the models and components
        self.voice_interface.initialize()
        self.entailer.initialize()
        if self.tts_speaker:
            self.tts_speaker.initialize()
        else:
            _logger.warning("TTSSpeaker is not initialized, TTS functionality will be disabled.")





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