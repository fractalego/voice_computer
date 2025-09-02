"""
Main voice computer client that integrates all components.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Tuple

from voice_computer.listeners.base_listener import VoiceInterruptionException
from voice_computer.voice_interface import VoiceInterface
from voice_computer.data_types import Messages, Utterance
from voice_computer.tool_handler import ToolHandler, ToolResult, FailedTool
from voice_computer.mcp_connector import MCPStdioConnector
from voice_computer.config import Config
from voice_computer.streaming_display import (
    stream_colored_to_console_with_tts,
    stream_colored_to_console, StreamingCompletionException
)
from voice_computer.speaker import TTSSpeaker, ServerSoundFileSpeaker
from voice_computer.entailer import Entailer
from voice_computer.model_factory import get_model_factory
from voice_computer.prompt import get_voice_assistant_system_prompt, format_tool_context

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
                "flush_delay": 1.0
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


class Conversation:
    """Main voice computer handler that coordinates all components."""
    
    @classmethod
    def create_with_shared_tools(cls, config: Optional[Config] = None, voice_listener=None, tts_speaker=None, sound_speaker=None,
                                shared_mcp_tools=None, shared_tool_handler=None, shared_entailer=None):
        """
        Factory method to create a Conversation with pre-shared MCP tools and entailer.
        
        This prevents MCP tools and entailer models from being reloaded for each client in server mode.
        
        Args:
            config: Configuration object
            voice_listener: Voice listener instance (for server mode)
            tts_speaker: TTS speaker instance (for server mode)
            sound_speaker: Sound file speaker instance (for server mode)
            shared_mcp_tools: Pre-initialized MCP tools to share
            shared_tool_handler: Pre-initialized tool handler to share
            shared_entailer: Pre-initialized entailer to share
            
        Returns:
            Conversation with shared MCP tools and entailer assigned
        """
        # Create instance without triggering MCP/entailer setup
        instance = cls.__new__(cls)
        instance._init_without_mcp_setup(config, voice_listener, tts_speaker, sound_speaker)
        
        # Assign shared MCP tools
        if shared_mcp_tools is not None:
            instance.mcp_tools = shared_mcp_tools
        if shared_tool_handler is not None:
            instance.tool_handler = shared_tool_handler
        if shared_entailer is not None:
            instance.entailer = shared_entailer
        
        _logger.debug("Created Conversation with shared MCP tools and entailer")
        return instance
    
    def _init_without_mcp_setup(self, config: Optional[Config] = None, voice_listener=None, tts_speaker=None, sound_speaker=None):
        """Initialize Conversation without setting up MCP tools."""
        self.config = config or Config()
        
        # Initialize Ollama client using model factory
        model_factory = get_model_factory()
        self.ollama_client = model_factory.get_ollama_client(
            model=self.config.get_value("ollama_model"),
            host=self.config.get_value("ollama_host")
        )
        
        # Initialize voice components - use provided ones or default to local
        if voice_listener is not None:
            # Server mode - use provided listener and speakers
            self.voice_listener = voice_listener
            self.tts_speaker = tts_speaker
            self.sound_speaker = sound_speaker
            self.voice_interface = VoiceInterface(self.config, voice_listener=voice_listener, tts_speaker=tts_speaker, sound_speaker=sound_speaker)
            self.server_mode = True
        else:
            # Local mode - use traditional voice interface
            self.voice_interface = VoiceInterface(self.config)
            self.voice_listener = None
            self.tts_speaker = None
            self.sound_speaker = None
            self.server_mode = False
        
        # Initialize empty MCP tools, handler, and entailer (will be assigned later)
        self.mcp_tools = []
        self.tool_handler = None
        self.entailer = None
        
        # Tool results queue (last 5 results)
        self.tool_results_queue = []
        
        # Failed tools queue (last 5 failed tools)
        self.failed_tools_queue = []
        
        # Conversation history (list of utterances)
        self.conversation_history = []
    
    def __init__(self, config: Optional[Config] = None, voice_listener=None, tts_speaker=None, sound_speaker=None):
        self.config = config or Config()
        
        # Initialize Ollama client using model factory
        model_factory = get_model_factory()
        self.ollama_client = model_factory.get_ollama_client(
            model=self.config.get_value("ollama_model"),
            host=self.config.get_value("ollama_host")
        )
        
        # Initialize voice components - use provided ones or default to local
        if voice_listener is not None:
            # Server mode - use provided listener and speakers
            self.voice_listener = voice_listener
            self.tts_speaker = tts_speaker
            self.sound_speaker = sound_speaker
            self.voice_interface = VoiceInterface(self.config, voice_listener=voice_listener, tts_speaker=tts_speaker, sound_speaker=sound_speaker)
            self.server_mode = True
        else:
            # Local mode - use traditional voice interface
            self.voice_interface = VoiceInterface(self.config)
            self.voice_listener = None
            self.tts_speaker = None
            self.sound_speaker = None
            self.server_mode = False
        
        # Initialize MCP tools and handler
        self.mcp_tools = []
        self.tool_handler = None
        
        # Tool results queue (last 5 results)
        self.tool_results_queue = []
        
        # Failed tools queue (last 5 failed tools)
        self.failed_tools_queue = []
        
        # Conversation history (list of utterances)
        self.conversation_history = []
        
        # Constant listening state
        self._constant_listening_mode = False
        self._command_queue = asyncio.Queue()
        self._processing_command = False
        
        self._initialize_mcp_tools()
        
        # Initialize entailer for exit detection
        self.entailer = Entailer(self.config)
        
        # Initialize local TTS speaker only in local mode
        if not self.server_mode:
            try:
                if self.tts_speaker is None:  # Only if not provided
                    self.tts_speaker = TTSSpeaker(config=self.config)
                    _logger.info("TTSSpeaker initialized successfully")
            except Exception as e:
                _logger.warning(f"Failed to initialize TTSSpeaker: {e}")
                self.tts_speaker = None

        self._initialize_models()

        _logger.info("Handler initialized")
    
    def _reset_conversation_state(self):
        """Reset conversation history and tool results when conversation ends."""
        _logger.info("Resetting conversation state - clearing history and tool results")
        self.conversation_history.clear()
        self.tool_results_queue.clear()
        self.failed_tools_queue.clear()
    
    def _initialize_mcp_tools(self) -> None:
        """Initialize MCP tools from configuration."""
        mcp_servers = self.config.get_value("mcp_servers") or []
        
        for server_config in mcp_servers:
            try:
                connector = MCPStdioConnector(
                    command=server_config["path"],
                    description=f"{server_config['name']}",
                    args=server_config.get("args", []),
                    env_vars=server_config.get("env_vars", {})
                )
                
                # Get tools asynchronously (we'll do this in the run method)
                self.mcp_tools.append(connector)
                _logger.info(f"Configured MCP connector: {server_config['name']}")
                
            except Exception as e:
                _logger.error(f"Failed to configure MCP connector {server_config['name']}: {e}")
        
        _logger.info(f"Configured {len(self.mcp_tools)} MCP connectors")
    
    async def _setup_mcp_tools(self) -> None:
        """Setup MCP tools asynchronously."""
        # Skip setup if tools are already configured (e.g., from shared tools in server mode)
        if self.tool_handler is not None:
            _logger.debug("MCP tools already configured, skipping setup")
            return
            
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
            # Pass voice_interface only in local mode
            voice_interface_for_tools = self.voice_interface if not self.server_mode else None
            self.tool_handler = ToolHandler(self.ollama_client, tools, self.config, self.conversation_history, voice_interface_for_tools)
            _logger.info(f"ToolHandler initialized with {len(tools)} MCP tool groups")
        else:
            _logger.warning("No MCP tools available - running in basic mode")
        
        # Validate special sentences after MCP tools are loaded
        await self._validate_special_sentences()
    
    async def _validate_special_sentences(self) -> None:
        """
        Validate that all special sentences reference existing MCP server tools.
        Log warnings for any invalid tool references.
        """
        special_sentences = self.config.get_value("special_sentences") or {}
        if not special_sentences:
            return
            
        if not self.tool_handler:
            _logger.warning("No tool handler available to validate special sentences")
            return
            
        # Get all available tools
        available_tools = {}
        for mcp_tools in self.tool_handler.tools:
            # Parse server name from server_description (e.g., "math" from "math")
            server_name = mcp_tools.server_description
            for tool in mcp_tools.tools:
                tool_key = f"{server_name}.{tool.name}"
                available_tools[tool_key] = True
                # Also register just tool name as fallback
                available_tools[tool.name] = True
        
        _logger.info(f"Validating {len(special_sentences)} special sentences against {len(available_tools)} available tools")
        
        # Validate each special sentence
        invalid_sentences = []
        for sentence, sentence_config in special_sentences.items():
            # Handle both simple and extended formats
            if isinstance(sentence_config, str):
                # Simple format: "server.tool"
                mcp_tool = sentence_config
            elif isinstance(sentence_config, dict):
                # Extended format: {"tool": "server.tool", "default_args": {...}}
                mcp_tool = sentence_config.get("tool")
                if not mcp_tool:
                    _logger.warning(f"Special sentence '{sentence}' missing 'tool' key")
                    invalid_sentences.append(sentence)
                    continue
            else:
                _logger.warning(f"Special sentence '{sentence}' has invalid format: {type(sentence_config)}")
                invalid_sentences.append(sentence)
                continue
            
            if "." not in mcp_tool:
                _logger.warning(f"Special sentence '{sentence}' has invalid tool format '{mcp_tool}', expected SERVER.TOOL")
                invalid_sentences.append(sentence)
                continue
                
            server_name, tool_name = mcp_tool.split(".", 1)
            tool_key = f"{server_name}.{tool_name}"
            
            # Check if exact match exists
            if tool_key not in available_tools:
                # Check if tool name exists in any server (fallback)
                if tool_name in available_tools:
                    _logger.warning(f"Special sentence '{sentence}' references '{mcp_tool}' but tool '{tool_name}' exists in a different server")
                else:
                    _logger.error(f"Special sentence '{sentence}' references non-existent tool '{mcp_tool}'")
                    invalid_sentences.append(sentence)
            else:
                _logger.debug(f"Special sentence '{sentence}' -> {mcp_tool} [valid]")
        
        if invalid_sentences:
            _logger.warning(f"Found {len(invalid_sentences)} invalid special sentence mappings. These will be ignored during processing.")
        else:
            _logger.info("All special sentences validated successfully")
    
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
        _logger.info(f"Processing query in {'server' if self.server_mode else 'local'} mode: '{query}' (streaming={use_streaming}, tts={use_tts})")
        
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
            # Check for special sentences first (bypasses normal tool selection)
            special_tool = await self._check_special_sentences(query)
            if special_tool:
                server_name, tool_name, default_args = special_tool
                # Handle special sentence with specific tool
                tool_results, failed_tools = await self._handle_special_sentence_tool(server_name, tool_name, query, default_args)
            else:
                # Update tool handler with current conversation history
                if self.tool_handler:
                    self.tool_handler.update_conversation_history(self.conversation_history)
                
                # Execute relevant tools using the handler (normal flow)
                tool_results, failed_tools = await self.tool_handler.handle_query(query)
            
            # Update tool results queue (keep only last 5)
            self.tool_results_queue.extend(tool_results)
            if len(self.tool_results_queue) > 5:
                self.tool_results_queue = self.tool_results_queue[-5:]
            
            # Update failed tools queue (keep only last 5)
            self.failed_tools_queue.extend(failed_tools)
            if len(self.failed_tools_queue) > 5:
                self.failed_tools_queue = self.failed_tools_queue[-5:]
            
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
        # Format tool context including failed tools and get facts from config
        tool_context = format_tool_context(self.tool_results_queue, self.failed_tools_queue)
        facts = self.config.get_value("facts") or []
        
        # Get available tool names from tool handler
        available_tools = []
        if self.tool_handler:
            available_tools = self.tool_handler.get_available_tool_names()
        
        system_prompt = get_voice_assistant_system_prompt(tool_context, facts, available_tools)
        
        return messages.add_system_prompt(system_prompt)
    
    async def _is_exit_command(self, user_input: str) -> bool:
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
        user_input_clean = user_input.strip().lower()
        
        try:
            # Check entailment against all exit sentences at once
            matching_indices = await self.entailer.judge_list(user_input_clean, exit_sentences)
            
            if matching_indices:
                matched_sentences = [exit_sentences[i] for i in matching_indices]
                _logger.info(f"Exit command detected: '{user_input_clean}' matches {matched_sentences}")
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
    
    async def _check_special_sentences(self, user_input: str) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """
        Check if user input matches any special sentences.
        
        Args:
            user_input: The user's input text
            
        Returns:
            Tuple of (server_name, tool_name, default_args) if match found, None otherwise.
            Priority is based on order in special_sentences config (top to bottom).
        """
        if not user_input or not user_input.strip():
            return None
            
        # Get configured special sentences
        special_sentences = self.config.get_value("special_sentences") or {}
        if not special_sentences:
            return None
            
        # Convert to list preserving order for priority
        sentences_list = list(special_sentences.keys())
        user_input_clean = user_input.strip().lower()
        
        try:
            # Check entailment against all special sentences at once
            matching_indices = await self.entailer.judge_list(user_input_clean, sentences_list)
            
            if matching_indices:
                # Use first match (highest priority) based on order in config
                first_match_idx = min(matching_indices)  # Get the earliest index for priority
                matched_sentence = sentences_list[first_match_idx]
                sentence_config = special_sentences[matched_sentence]
                
                # Handle both simple and extended formats
                if isinstance(sentence_config, str):
                    # Simple format: "server.tool"
                    mcp_tool = sentence_config
                    default_args = {}
                elif isinstance(sentence_config, dict):
                    # Extended format: {"tool": "server.tool", "default_args": {...}}
                    mcp_tool = sentence_config.get("tool")
                    default_args = sentence_config.get("default_args", {})
                    if not mcp_tool:
                        _logger.warning(f"Special sentence '{matched_sentence}' missing 'tool' key")
                        return None
                else:
                    _logger.warning(f"Invalid special sentence format for '{matched_sentence}': {type(sentence_config)}")
                    return None
                
                # Parse server_name.tool_name
                if "." not in mcp_tool:
                    _logger.warning(f"Invalid MCP tool format '{mcp_tool}', expected SERVER.TOOL")
                    return None
                    
                server_name, tool_name = mcp_tool.split(".", 1)
                _logger.info(f"Special sentence detected: '{user_input_clean}' matches '{matched_sentence}' -> {server_name}.{tool_name} with default args: {default_args}")
                return (server_name, tool_name, default_args)
            
            return None
            
        except Exception as e:
            _logger.error(f"Error checking special sentences with entailer: {e}")
            return None
    
    async def _handle_special_sentence_tool(self, server_name: str, tool_name: str, query: str, default_args: Optional[Dict[str, Any]] = None) -> Tuple[List[ToolResult], List[FailedTool]]:
        """
        Handle a special sentence by directly calling the specified MCP tool.
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            query: Original user query for argument extraction
            default_args: Default arguments to use for the tool call
            
        Returns:
            Tuple of (tool_results, failed_tools) matching ToolHandler format
        """
        tool_results = []
        failed_tools = []
        default_args = default_args or {}
        
        if not self.tool_handler:
            _logger.error("No tool handler available for special sentence")
            failed_tools.append(FailedTool(
                tool_name=f"{server_name}.{tool_name}",
                tool_description=f"Special sentence tool: {server_name}.{tool_name}",
                missing_parameters=[],
                score=0.0
            ))
            return tool_results, failed_tools
        
        try:
            # Find the specific tool in the tool handler
            target_tool = None
            target_mcp_tools = None
            
            for mcp_tools in self.tool_handler.tools:
                for tool in mcp_tools.tools:
                    # Check if this tool matches server and tool name
                    if (mcp_tools.server_description == server_name and tool.name == tool_name):
                        target_tool = tool
                        target_mcp_tools = mcp_tools
                        break
                    # Fallback: check if tool name matches (for any server)
                    elif tool.name == tool_name:
                        target_tool = tool
                        target_mcp_tools = mcp_tools
                        break
                if target_tool:
                    break
            
            if not target_tool:
                _logger.warning(f"Special sentence tool not found: {server_name}.{tool_name}")
                failed_tools.append(FailedTool(
                    tool_name=f"{server_name}.{tool_name}",
                    tool_description=f"Special sentence tool not found: {server_name}.{tool_name}",
                    missing_parameters=[],
                    score=0.0
                ))
                return tool_results, failed_tools
            
            # Extract arguments using the argument extractor and merge with defaults
            arguments = default_args.copy()  # Start with default arguments
            
            if hasattr(self.tool_handler, 'extractor') and self.tool_handler.extractor:
                conversation_history = self.conversation_history[-3:] if len(self.conversation_history) > 3 else self.conversation_history
                facts = self.config.get_value("facts") or []
                
                extracted_args = await self.tool_handler.extractor.extract_arguments(
                    query, 
                    target_tool.name, 
                    target_tool.description,
                    target_tool.inputSchema,
                    conversation_history,
                    facts
                )
                
                # Merge extracted args with defaults (extracted args take precedence)
                arguments.update(extracted_args)
            else:
                _logger.warning(f"No argument extractor available for special sentence tool {server_name}.{tool_name}, cannot extract arguments from query")
                if not default_args:
                    _logger.error(f"Special sentence tool {server_name}.{tool_name} has no default_args and no argument extractor - tool call will likely fail")
            
            _logger.info(f"Executing special sentence tool {server_name}.{tool_name} with arguments: {arguments}")
            
            # Execute the tool directly
            result = await target_mcp_tools.call_tool(target_tool.name, arguments)
            
            # Create ToolResult object matching ToolHandler format
            tool_result = ToolResult(
                original_query=query,
                tool_description=f"{server_name}.{tool_name}: {target_tool.description}",
                tool_result=result
            )
            tool_results.append(tool_result)
            
            _logger.info(f"Special sentence tool {server_name}.{tool_name} executed successfully")
            
        except Exception as e:
            _logger.error(f"Error executing special sentence tool {server_name}.{tool_name}: {e}")
            failed_tools.append(FailedTool(
                tool_name=f"{server_name}.{tool_name}",
                tool_description=f"Special sentence tool execution failed: {server_name}.{tool_name}",
                missing_parameters=[],
                score=0.0
            ))
        
        return tool_results, failed_tools
    
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

                    # Play sound to indicate processing has started
                    await self.voice_interface.play_computer_starting_to_work()

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
                    if await self._is_exit_command(user_input):
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

        while True:
            try:
                # Wait for hotword activation and handle conversation cycles
                should_continue = await self._handle_voice_activation_cycle()
                if not should_continue:
                    break

            except KeyboardInterrupt:
                _logger.info("Voice loop interrupted by user")
                break

            except Exception as e:
                _logger.error(f"Error in voice loop: {e}")
                await self.voice_interface.output("Sorry, I encountered an error.")

            finally:
                self.voice_interface.deactivate()

        if hasattr(self, 'tts_speaker') and self.tts_speaker:
            try:
                self.tts_speaker.cleanup()
            except Exception as e:
                _logger.debug(f"Error cleaning up TTS speaker: {e}")
                
        await self._cleanup_mcp_connections()
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
                    if await self._is_exit_command(user_input):
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
            # Clean up MCP connections
            await self._cleanup_mcp_connections()
            _logger.info("Text interaction loop ended")
    
    async def _cleanup_mcp_connections(self) -> None:
        """Clean up all MCP server connections."""
        # Instead of trying to properly close connections (which can cause cancel scope issues),
        # we'll just mark them as disconnected to prevent further use
        if hasattr(self, 'tool_handler') and self.tool_handler:
            try:
                # Mark all MCP server references as None to prevent further use
                for mcp_tools in self.tool_handler.tools:
                    if hasattr(mcp_tools, 'server') and mcp_tools.server:
                        mcp_tools.server = None
                        _logger.debug(f"Marked MCP server as disconnected: {mcp_tools.server_description}")
                        
            except Exception as e:
                _logger.debug(f"Error during MCP cleanup: {e}")
        
        # Mark all connectors as disconnected
        for connector in self.mcp_tools:
            try:
                if hasattr(connector, '_server'):
                    connector._server = None
                _logger.debug(f"Marked MCP connector as disconnected: {connector._description}")
            except Exception as e:
                _logger.debug(f"Error marking connector as disconnected: {e}")
        
        # Clear tool handler reference
        if hasattr(self, 'tool_handler'):
            self.tool_handler = None
            _logger.debug("Cleared tool handler reference")
    
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
        return streaming_config.get("flush_delay", 1.0)
    
    async def _process_streaming_query(self, messages: Messages, use_colored_output: bool = True, use_tts: bool = False) -> str:
        """Process query with streaming output to console and optional TTS."""
        token_queue = asyncio.Queue()
        batch_size = self._get_token_batch_size()
        flush_delay = self._get_flush_delay()
        
        # Create streaming display task with voice interruption capability
        if use_tts and self.tts_speaker:
            # Use TTS streaming with voice interruption for voice mode
            display_task, display_instance = await stream_colored_to_console_with_tts(
                token_queue=token_queue,
                tts_speaker=self.tts_speaker,
                prefix="bot> ",
                batch_size=batch_size,
                flush_delay=flush_delay
            )
        elif use_colored_output:
            # Use colored output with voice interruption for text mode
            display_task, display_instance = await stream_colored_to_console(
                token_queue=token_queue,
                prefix="bot> ",
                batch_size=batch_size,
                flush_delay=flush_delay
            )
        else:
            # Basic streaming without color or TTS
            from .streaming_display import stream_to_console
            display_task, display_instance = await stream_to_console(
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

            # Create a wrapper for the listening task that cancels other tasks immediately
            async def listening_with_cancellation():
                await self.voice_interface.throw_exception_on_voice_activity()
            
            listening_task = asyncio.create_task(listening_with_cancellation())
            task_list = [prediction_task, display_task, listening_task]
            _logger.debug("Created voice activity monitoring task for local mode")
            
            # Wait for the first exception (voice interruption) or all tasks to complete
            done, pending = await asyncio.wait(
                task_list,
                return_when=asyncio.FIRST_EXCEPTION
            )
            ## logs the status of the tasks
            _logger.debug(f"Tasks done: {[task.get_name() for task in done]}")
            _logger.debug(f"Tasks pending: {[task.get_name() for task in pending]}")

            # Check if listening task completed with voice interruption exception
            if listening_task in done:
                try:
                    await listening_task  # This will re-raise the VoiceInterruptionException if it occurred
                except VoiceInterruptionException as e:
                    _logger.info(f"Voice activity detected in {'server' if self.server_mode else 'local'} mode, cancelling prediction and display tasks")
                    _logger.debug(f"Voice activity details: {e}")
                    if self.tts_speaker:
                        self.tts_speaker.cancel_playback()
                    if self.voice_interface:
                        await self.voice_interface.play_deny_sound()
                        await asyncio.sleep(0.05)
                        self.tts_speaker.speak("Stopping.")

                except Exception as e:
                    _logger.debug(f"Listening task completed with unexpected exception: {e}")

            # Cancel all pending tasks
            for task in pending:
                _logger.debug(f"Cancelling pending task: {task.get_name()}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    _logger.debug(f"Successfully cancelled task: {task.get_name()}")
                    pass
                except Exception as e:
                    _logger.debug(f"Task {task.get_name()} raised exception during cancellation: {e}")

            # Get the result from the prediction task if it completed
            if prediction_task in done:
                try:
                    _logger.debug("Prediction task completed successfully, getting result")
                    response = await prediction_task
                    return response.message

                except Exception as e:
                    _logger.debug(f"Error getting result from prediction task: {e}")
                    # Treat as interrupted
                    accumulated_text = getattr(display_instance, 'accumulated_text', '')
                    return accumulated_text + " [interrupted]"
            else:
                # If prediction task didn't complete, we need to handle this case
                _logger.debug("Prediction task did not complete - was interrupted")
                # Return accumulated text with interrupted marker
                accumulated_text = getattr(display_instance, 'accumulated_text', '')
                return accumulated_text + " [interrupted]"
            
        except Exception as e:
            # Cancel any remaining tasks in case of exception
            for task in [prediction_task, display_task, listening_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass  # Ignore exceptions during cleanup
            raise e

    def _initialize_models(self):
        # Initialize all the models and components
        if self.server_mode:
            # Server mode - initialize server components
            if self.voice_listener:
                self.voice_listener.initialize()
            if self.tts_speaker:
                self.tts_speaker.initialize()
        else:
            # Local mode - initialize local components
            self.voice_interface.initialize()
            if self.tts_speaker:
                self.tts_speaker.initialize()
            else:
                _logger.warning("TTSSpeaker is not initialized, TTS functionality will be disabled.")
        
        self.entailer.initialize()
    
    # Server mode specific methods
    async def add_audio_chunk(self, audio_data: bytes):
        """Add audio chunk from WebSocket to the voice listener (server mode only)."""
        if self.server_mode and self.voice_listener:
            await self.voice_listener.add_audio_chunk(audio_data)
    
    async def process_accumulated_audio(self) -> Optional[Dict[str, Any]]:
        """Process accumulated audio and return result (server mode only)."""
        if not self.server_mode or not self.voice_listener:
            return None
            
        try:
            # Transcribe accumulated audio
            transcribed_text = await self.voice_listener.transcribe_accumulated_audio()
            
            if not transcribed_text or not transcribed_text.strip():
                return None
            
            # Check for activation words
            activation_word = await self.voice_listener.detect_activation_words(transcribed_text)
            
            if activation_word:
                # Extract command after activation word
                command = await self.voice_listener.extract_command_after_activation(
                    transcribed_text, activation_word
                )
                
                result = {
                    "type": "activation_detected",
                    "activation_word": activation_word,
                    "full_text": transcribed_text,
                    "command": command.strip() if command else None
                }
                
                # If we have a command, process it
                if command and command.strip():
                    response = await self.process_query(command.strip())
                    result["response"] = response
                
                return result
            else:
                # No activation word, just return transcription
                return {
                    "type": "transcription_only",
                    "text": transcribed_text
                }
                
        except Exception as e:
            _logger.error(f"Error processing accumulated audio: {e}")
            return {
                "type": "error",
                "error": str(e)
            }
    
    def get_conversation_length(self) -> int:
        """Get current conversation length."""
        return len(self.conversation_history)
    
    def get_available_tool_names(self) -> List[str]:
        """Get list of available tool names."""
        if self.tool_handler:
            return self.tool_handler.get_available_tool_names()
        return []
    
    def get_buffer_duration(self) -> float:
        """Get current audio buffer duration (server mode only)."""
        if self.server_mode and self.voice_listener:
            return self.voice_listener.get_buffer_duration()
        return 0.0
    
    def clear_audio_buffer(self):
        """Clear the audio buffer (server mode only)."""
        if self.server_mode and self.voice_listener:
            self.voice_listener.clear_audio_buffer()
    
    async def is_exit_command(self, user_input: str) -> bool:
        """Check if user input is an exit command using entailment."""
        return await self._is_exit_command(user_input)

    async def run_constant_listening_mode(self):
        """
        Run in constant listening mode where:
        1. Background task continuously listens for hotwords
        2. Commands are queued if they arrive during processing
        3. Queued commands are joined together and processed as one
        4. Processing uses existing streaming architecture
        """
        _logger.info("Starting constant listening mode...")
        self._constant_listening_mode = True
        
        # Initialize voice interface
        self.voice_interface.activate()
        self._initialize_models()
        
        try:
            # Start background listening task
            background_listening_task = asyncio.create_task(
                self._background_hotword_listener(),
                name="background_listening"
            )
            
            # Start command processing task
            command_processing_task = asyncio.create_task(
                self._process_command_queue(),
                name="command_processing"
            )
            
            # Run both tasks concurrently
            await asyncio.gather(
                background_listening_task,
                command_processing_task,
                return_exceptions=True
            )
            
        except KeyboardInterrupt:
            _logger.info("Constant listening mode interrupted by user")
        except Exception as e:
            _logger.error(f"Error in constant listening mode: {e}")
        finally:
            self._constant_listening_mode = False
            self.voice_interface.deactivate()
            _logger.info("Constant listening mode stopped")

    async def _background_hotword_listener(self):
        """
        Continuously listen for hotwords and commands in the background.
        When a command is detected, add it to the queue.
        """
        _logger.info("Background hotword listener started")
        
        while self._constant_listening_mode:
            try:
                # Wait for hotword activation (reuse existing voice interface logic)
                _logger.debug("Listening for hotword activation...")
                hotword, instruction = await self.voice_interface._wait_for_hotword()
                
                if instruction:
                    # Hotword came with instruction - add directly to queue
                    _logger.info(f"Hotword '{hotword}' detected with instruction: '{instruction}'")
                    await self._command_queue.put(instruction)
                else:
                    # Just hotword detected - listen for command
                    _logger.info(f"Hotword '{hotword}' detected, listening for command...")
                    try:
                        # Listen for the actual command (with timeout from config)
                        constant_listening_config = self.config.get_value("constant_listening") or {}
                        command_timeout = constant_listening_config.get("command_timeout", 10.0)
                        
                        command = await asyncio.wait_for(
                            self.voice_interface._listener.input(),
                            timeout=command_timeout
                        )
                        
                        if command and command.strip():
                            # Clean up the command
                            cleaned_command = self.voice_interface._remove_activation_word_and_normalize(command)
                            cleaned_command = self.voice_interface._remove_unclear(cleaned_command)
                            
                            if cleaned_command and len(cleaned_command.strip()) > 2:
                                _logger.info(f"Command received: '{cleaned_command}'")
                                await self._command_queue.put(cleaned_command)
                            else:
                                _logger.debug("Command too short or unclear, ignoring")
                        
                    except asyncio.TimeoutError:
                        _logger.debug("Timeout waiting for command after hotword")
                    except Exception as e:
                        _logger.debug(f"Error getting command after hotword: {e}")
            
            except Exception as e:
                _logger.error(f"Error in background hotword listener: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retrying

    async def _process_command_queue(self):
        """
        Process commands from the queue. If multiple commands are queued,
        join them together and process as a single command.
        """
        _logger.info("Command queue processor started")
        
        while self._constant_listening_mode:
            try:
                # Wait for at least one command
                first_command = await self._command_queue.get()
                _logger.info(f"Processing command: '{first_command}'")
                
                # Set processing flag
                self._processing_command = True
                
                # Collect any additional commands that arrived while we were waiting
                additional_commands = []
                
                # Non-blocking check for more commands (with configurable delay to collect rapid commands)
                constant_listening_config = self.config.get_value("constant_listening") or {}
                join_delay = constant_listening_config.get("command_join_delay", 0.1)
                await asyncio.sleep(join_delay)
                
                while not self._command_queue.empty():
                    try:
                        additional_command = self._command_queue.get_nowait()
                        additional_commands.append(additional_command)
                        _logger.info(f"Additional command queued: '{additional_command}'")
                    except asyncio.QueueEmpty:
                        break
                
                # Join all commands together
                if additional_commands:
                    all_commands = [first_command] + additional_commands
                    joined_command = " ".join(all_commands)
                    _logger.info(f"Joined {len(all_commands)} commands: '{joined_command}'")
                else:
                    joined_command = first_command
                
                # Process the joined command using existing infrastructure
                await self._process_single_command(joined_command)
                
                # Clear processing flag
                self._processing_command = False
                
            except Exception as e:
                _logger.error(f"Error processing command queue: {e}")
                self._processing_command = False
                await asyncio.sleep(1.0)  # Brief pause before retrying

    async def _process_single_command(self, command: str):
        """
        Process a single command using the existing voice computer infrastructure.
        This reuses all existing logic including streaming, tools, etc.
        """
        try:
            # Check for exit command
            if await self.entailer.should_exit(command, self.conversation_history):
                _logger.info("Exit command detected in constant listening mode")
                self._constant_listening_mode = False
                return
            
            # Add user utterance to conversation history
            user_utterance = Utterance(role="user", content=command)
            self.conversation_history.append(user_utterance)
            
            # Process tools if available
            tool_results = []
            failed_tools = []
            
            if self.tool_handler:
                try:
                    # Play starting to work sound
                    await self.voice_interface.play_computer_starting_to_work()
                    
                    # Check for special sentences first (bypasses normal tool selection)
                    special_tool = await self._check_special_sentences(command)
                    if special_tool:
                        server_name, tool_name, default_args = special_tool
                        # Handle special sentence with specific tool
                        tool_results, failed_tools = await self._handle_special_sentence_tool(server_name, tool_name, command, default_args)
                    else:
                        # Update tool handler with current conversation history
                        self.tool_handler.update_conversation_history(self.conversation_history)
                        
                        # Handle the query using tools (normal flow)
                        tool_results, failed_tools = await self.tool_handler.handle_query(command)
                    
                    # Update tool results queues
                    self.tool_results_queue.extend(tool_results)
                    self.tool_results_queue = self.tool_results_queue[-5:]  # Keep last 5
                    
                    self.failed_tools_queue.extend(failed_tools)
                    self.failed_tools_queue = self.failed_tools_queue[-5:]  # Keep last 5
                    
                except Exception as e:
                    _logger.error(f"Error in tool handling: {e}")
            
            # Build messages for LLM
            messages = self._build_messages_with_context(
                command, tool_results, failed_tools, self.conversation_history
            )
            
            # Process with streaming (reuse existing method)
            response = await self._process_streaming_query(
                messages, 
                use_colored_output=True, 
                use_tts=True
            )
            
            # Add assistant response to conversation history
            if response:
                assistant_utterance = Utterance(role="assistant", content=response)
                self.conversation_history.append(assistant_utterance)
                
                # Trim conversation history if too long (configurable)
                constant_listening_config = self.config.get_value("constant_listening") or {}
                max_history = constant_listening_config.get("max_conversation_history", 20)
                if len(self.conversation_history) > max_history:
                    self.conversation_history = self.conversation_history[-max_history:]
        
        except Exception as e:
            _logger.error(f"Error processing single command: {e}")
            # Try to provide error feedback
            try:
                await self.voice_interface.output("Sorry, I encountered an error processing that command.")
            except Exception:
                pass  # If even error output fails, just continue


async def main():
    """Example usage."""
    # Create handler
    handler = Conversation()
    
    # Example: Add MCP servers (configure based on your setup)
    # handler.add_mcp_server("filesystem", "mcp-server-filesystem", ["--root", "/tmp"])
    # handler.add_mcp_server("web-search", "mcp-server-brave-search", ["--api-key", "your-key"])
    
    # Run the voice loop
    await handler.run_voice_loop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())