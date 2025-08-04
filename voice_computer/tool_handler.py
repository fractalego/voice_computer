"""
Tool handler for selecting and executing MCP tools based on entailment scoring.
"""

import logging
from collections import deque
from typing import List, Dict, Any, Optional, NamedTuple, TYPE_CHECKING
from .config import Config
from .entailer import Entailer
from .extractor import ArgumentExtractor
from .ollama_client import OllamaClient
from .mcp_connector import MCPTools

if TYPE_CHECKING:
    from .voice_interface import VoiceInterface

_logger = logging.getLogger(__name__)


class ToolResult(NamedTuple):
    """Result from tool execution."""
    original_query: str
    tool_description: str
    tool_result: Any


class FailedTool(NamedTuple):
    """Information about a tool that couldn't be executed."""
    tool_name: str
    tool_description: str
    missing_parameters: List[str]
    score: float


class ToolHandler:
    """Handler that uses entailment scoring to select and execute relevant tools."""
    
    def __init__(self, ollama_client: OllamaClient, tools: List[MCPTools], config: Optional[Config] = None, conversation_history: Optional[List] = None, voice_interface: Optional['VoiceInterface'] = None):
        """
        Initialize the tool handler.
        
        Args:
            ollama_client: Ollama client for LLM communication
            tools: List of MCP tool groups
            config: Configuration object
            conversation_history: Recent conversation history for context
            voice_interface: Voice interface for playing sounds (voice mode only)
        """
        self.ollama_client = ollama_client
        self.tools = tools
        self.config = config
        self.conversation_history = conversation_history or []
        self.voice_interface = voice_interface
        
        # Initialize tool results queue
        queue_length = config.get_value("tool_results_queue_length") if config else 2
        self.tool_results_queue = deque(maxlen=queue_length)
        
        # Initialize entailer and extractor
        self.entailer = Entailer(config)
        self.extractor = ArgumentExtractor(config=config)
        
        # Update entailer with tools once they're available
        self._update_entailer_tools()
        
        _logger.info(f"ToolHandler initialized with {len(tools)} tool groups and queue length {queue_length}")
    
    def update_conversation_history(self, conversation_history: List) -> None:
        """Update the conversation history for context in tool extraction."""
        self.conversation_history = conversation_history
    
    def _update_entailer_tools(self) -> None:
        """Update the entailer with the current tools list."""
        # Build a flattened list of all tools from all groups
        all_tools = []
        for group in self.tools:
            for tool in group.tools:
                all_tools.append({
                    'name': tool.name,
                    'description': tool.description
                })
        
        self.entailer.update_tools(all_tools)
    
    async def handle_query(self, query: str) -> tuple[List[ToolResult], List[FailedTool]]:
        """
        Handle a query by selecting and executing relevant tools.
        
        Args:
            query: The user's natural language query
            
        Returns:
            Tuple of (successful_results, failed_tools)
        """
        if not self.tools:
            _logger.warning("No tools available")
            return [], []
        
        # Build table of all available tools with their descriptions
        tool_table = self._build_tool_table()
        
        if not tool_table:
            _logger.warning("No tools found in tool groups")
            return [], []
        
        # Build conversation context for entailment
        conversation_context = self._build_entailment_context(query)
        
        # Get relevant tool indices from entailer
        relevant_indices = await self.entailer.select_relevant_tools(conversation_context)
        
        if not relevant_indices:
            _logger.info("No tools were determined to be relevant for this query")
            return [], []
        
        _logger.info(f"Found {len(relevant_indices)} relevant tools: {relevant_indices}")
        
        # Execute each relevant tool
        results = []
        failed_tools = []
        
        for tool_index in relevant_indices:
            if tool_index < len(tool_table):
                tool_info = tool_table[tool_index]
                try:
                    result, failed_tool = await self._execute_tool(query, tool_info, True)
                    if result:
                        results.append(result)
                    if failed_tool:
                        failed_tools.append(failed_tool)
                except Exception as e:
                    _logger.error(f"Error executing tool {tool_info['name']}: {e}")
            else:
                _logger.warning(f"Invalid tool index returned by entailer: {tool_index}")
        
        return results, failed_tools
    
    def _build_tool_table(self) -> List[Dict[str, Any]]:
        """Build a table of all available tools with their metadata."""
        tool_table = []
        
        for group_index, tool_group in enumerate(self.tools):
            group_description = tool_group.server_description
            
            for tool_index, tool in enumerate(tool_group.tools):
                # Combine group description and tool description for better context
                combined_description = f"{group_description}: {tool.description}"
                
                tool_info = {
                    'group_index': group_index,
                    'tool_index': tool_index,
                    'group': tool_group,
                    'name': tool.name,
                    'description': tool.description,
                    'combined_description': combined_description,
                    'input_schema': tool.inputSchema
                }
                tool_table.append(tool_info)
        
        return tool_table
    
    
    async def _execute_tool(self, query: str, tool_info: Dict[str, Any], score: bool) -> tuple[Optional[ToolResult], Optional[FailedTool]]:
        """Execute a single tool with extracted arguments."""
        try:
            _logger.info(f"Executing tool {tool_info['name']} with entailment: {score}")
            
            # Get recent conversation history for context
            history_length = self.config.get_value("extractor_conversation_history_length") if self.config else 2
            recent_history = self._get_recent_conversation_history(history_length)
            
            # Get facts from config
            facts = self.config.get_value("facts") if self.config else None
            
            # Extract arguments for this tool with context
            arguments = await self.extractor.extract_arguments(
                query=query,
                tool_name=tool_info['name'],
                tool_description=tool_info['description'],
                input_schema=tool_info['input_schema'],
                conversation_history=recent_history,
                facts=facts
            )
            
            _logger.debug(f"Extracted arguments for {tool_info['name']}: {arguments}")

            # Check if tool execution should proceed based on argument requirements
            should_execute, missing_params = self._should_execute_tool_with_details(tool_info, arguments)
            if not should_execute:
                # Create failed tool information
                failed_tool = FailedTool(
                    tool_name=tool_info['name'],
                    tool_description=tool_info['description'],
                    missing_parameters=missing_params,
                    score=score
                )
                return None, failed_tool

            # Play computer work beep sound if in voice mode
            if self.voice_interface:
                try:
                    await self.voice_interface.play_computer_work_beep()
                except Exception as e:
                    _logger.debug(f"Failed to play computer work beep: {e}")

            # Execute the tool
            tool_group = tool_info['group']
            result = await tool_group.call_tool(tool_info['name'], arguments)
            
            # Create result object
            tool_result = ToolResult(
                original_query=query,
                tool_description=tool_info['description'],
                tool_result=result
            )
            
            # Add to results queue
            self.tool_results_queue.append(tool_result)
            _logger.debug(f"Added tool result to queue. Queue size: {len(self.tool_results_queue)}")
            
            return tool_result, None
            
        except Exception as e:
            _logger.error(f"Error executing tool {tool_info['name']}: {e}")
            raise
    
    def _should_execute_tool(self, tool_info: Dict[str, Any], arguments: Dict[str, Any]) -> bool:
        """
        Determine if a tool should be executed based on its argument requirements.
        
        Args:
            tool_info: Tool information including input schema
            arguments: Extracted arguments from the query
            
        Returns:
            True if the tool should be executed, False otherwise
        """
        if arguments is None:
            _logger.warning(f"No valid arguments extracted for tool {tool_info['name']}")
            return False
        
        input_schema = tool_info.get('input_schema', {})
        required_params = input_schema.get('required', [])
        properties = input_schema.get('properties', {})
        
        # If tool has no parameters defined, empty arguments are fine
        if not properties:
            _logger.debug(f"Tool {tool_info['name']} requires no parameters, proceeding with execution")
            return True
        
        # If tool has required parameters, check if they were extracted
        if required_params:
            missing_required = [param for param in required_params if param not in arguments]
            if missing_required:
                _logger.warning(f"Tool {tool_info['name']} missing required parameters: {missing_required}")
                return False
        
        # If tool has optional parameters only, empty arguments are acceptable
        _logger.debug(f"Tool {tool_info['name']} argument validation passed")
        return True
    
    def _should_execute_tool_with_details(self, tool_info: Dict[str, Any], arguments: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Determine if a tool should be executed and return missing parameters.
        
        Args:
            tool_info: Tool information including input schema
            arguments: Extracted arguments from the query
            
        Returns:
            Tuple of (should_execute, missing_parameters)
        """
        if arguments is None:
            _logger.warning(f"No valid arguments extracted for tool {tool_info['name']}")
            return False, ["Failed to extract any arguments"]
        
        input_schema = tool_info.get('input_schema', {})
        required_params = input_schema.get('required', [])
        properties = input_schema.get('properties', {})
        
        # If tool has no parameters defined, empty arguments are fine
        if not properties:
            _logger.debug(f"Tool {tool_info['name']} requires no parameters, proceeding with execution")
            return True, []
        
        # If tool has required parameters, check if they were extracted
        if required_params:
            missing_required = [param for param in required_params if param not in arguments]
            if missing_required:
                _logger.warning(f"Tool {tool_info['name']} missing required parameters: {missing_required}")
                return False, missing_required
        
        # If tool has optional parameters only, empty arguments are acceptable
        _logger.debug(f"Tool {tool_info['name']} argument validation passed")
        return True, []
    
    def _get_recent_conversation_history(self, history_length: int) -> List:
        """Get the most recent conversation exchanges for context."""
        if not self.conversation_history or history_length <= 0:
            return []
        
        # Get the last N exchanges (each exchange is user + assistant pair)
        # We want history_length * 2 utterances (user + assistant pairs)
        max_utterances = history_length * 2
        return self.conversation_history[-max_utterances:] if len(self.conversation_history) > max_utterances else self.conversation_history
    
    def _build_entailment_context(self, current_query: str) -> str:
        """Build conversation context for entailment scoring."""
        # Get configuration for how many utterances to include
        entailer_history_length = self.config.get_value("entailer_conversation_history_length") if self.config else 3
        
        if not self.conversation_history or entailer_history_length <= 0:
            return current_query
        
        # Get the last N utterances (not exchanges, just utterances)
        # We want to end with the current user query, so we get N-1 previous utterances
        previous_utterances = self.conversation_history[-(entailer_history_length-1):-1] if len(self.conversation_history) >= entailer_history_length-1 else self.conversation_history
        
        # Build conversation context
        context_parts = []
        if previous_utterances:
            context_parts.append("This is the prior conversation for context:")
            for utterance in previous_utterances[:-1]:
                if utterance.role == "user":
                    context_parts.append(f"User: {utterance.content}")
                elif utterance.role == "assistant":
                    context_parts.append(f"Assistant: {utterance.content}")
        
        # Add current query
        context_parts.append("")
        context_parts.append(f"The current query from the user is: {current_query}")
        
        return "\n".join(context_parts)
    
    def get_available_tool_names(self) -> List[str]:
        """Get a list of all available tool names."""
        if not self.tools:
            return []
        
        tool_names = []
        for tool_group in self.tools:
            for tool in tool_group.tools:
                tool_names.append(tool.name)
        
        return tool_names
    
    def get_tool_summary(self) -> str:
        """Get a summary of available tools."""
        if not self.tools:
            return "No tools available"
        
        summary_lines = []
        total_tools = 0
        
        for group_index, tool_group in enumerate(self.tools):
            group_tools = len(tool_group.tools)
            total_tools += group_tools
            
            summary_lines.append(
                f"Group {group_index} ({tool_group.server_description}): {group_tools} tools"
            )
        
        summary = f"Tool Handler Summary:\n"
        summary += f"- Total tool groups: {len(self.tools)}\n"
        summary += f"- Total tools: {total_tools}\n"
        summary += f"- Entailment model: Ollama-based Y/N judgment\n\n"
        summary += "Groups:\n" + "\n".join(f"  {line}" for line in summary_lines)
        
        return summary
    
    def get_tool_results_queue(self) -> List[ToolResult]:
        """Get all tool results currently in the queue."""
        return list(self.tool_results_queue)
    
    def get_latest_tool_result(self) -> Optional[ToolResult]:
        """Get the most recent tool result from the queue."""
        return self.tool_results_queue[-1] if self.tool_results_queue else None
    
    def clear_tool_results_queue(self) -> None:
        """Clear all tool results from the queue."""
        self.tool_results_queue.clear()
        _logger.debug("Tool results queue cleared")
    
    def get_tool_results_queue_size(self) -> int:
        """Get the current size of the tool results queue."""
        return len(self.tool_results_queue)