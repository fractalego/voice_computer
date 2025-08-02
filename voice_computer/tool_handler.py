"""
Tool handler for selecting and executing MCP tools based on entailment scoring.
"""

import logging
from typing import List, Dict, Any, Optional, NamedTuple
from .config import Config
from .entailer import Entailer
from .extractor import ArgumentExtractor
from .ollama_client import OllamaClient
from .mcp_connector import MCPTools

_logger = logging.getLogger(__name__)


class ToolResult(NamedTuple):
    """Result from tool execution."""
    original_query: str
    tool_description: str
    tool_result: Any


class ToolHandler:
    """Handler that uses entailment scoring to select and execute relevant tools."""
    
    def __init__(self, ollama_client: OllamaClient, tools: List[MCPTools], config: Optional[Config] = None):
        """
        Initialize the tool handler.
        
        Args:
            ollama_client: Ollama client for LLM communication
            tools: List of MCP tool groups
            config: Configuration object
        """
        self.ollama_client = ollama_client
        self.tools = tools
        self.config = config
        
        # Initialize entailer and extractor
        self.entailer = Entailer(config)
        self.extractor = ArgumentExtractor(config=config)
        
        # Get threshold from config
        self.threshold = config.get_value("entailment_threshold") if config else 0.5
        
        _logger.info(f"ToolHandler initialized with {len(tools)} tool groups and threshold {self.threshold}")
    
    async def handle_query(self, query: str) -> List[ToolResult]:
        """
        Handle a query by selecting and executing relevant tools.
        
        Args:
            query: The user's natural language query
            
        Returns:
            List of ToolResult objects for each executed tool
        """
        if not self.tools:
            _logger.warning("No tools available")
            return []
        
        # Build table of all available tools with their descriptions
        tool_table = self._build_tool_table()
        
        if not tool_table:
            _logger.warning("No tools found in tool groups")
            return []
        
        # Score each tool against the query using entailment
        scored_tools = await self._score_tools(query, tool_table)
        
        # Filter tools above threshold
        relevant_tools = [
            (tool_info, score) for tool_info, score in scored_tools 
            if score >= self.threshold
        ]
        
        if not relevant_tools:
            _logger.info(f"No tools scored above threshold {self.threshold}")
            return []
        
        _logger.info(f"Found {len(relevant_tools)} relevant tools above threshold")
        
        # Execute each relevant tool
        results = []
        for tool_info, score in relevant_tools:
            try:
                result = await self._execute_tool(query, tool_info, score)
                if result:
                    results.append(result)
            except Exception as e:
                _logger.error(f"Error executing tool {tool_info['name']}: {e}")
                # Create error result
                results.append(ToolResult(
                    original_query=query,
                    tool_description=tool_info['description'],
                    tool_result=f"Error: {str(e)}"
                ))
        
        return results
    
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
    
    async def _score_tools(self, query: str, tool_table: List[Dict[str, Any]]) -> List[tuple[Dict[str, Any], float]]:
        """Score each tool against the query using entailment."""
        scored_tools = []
        
        for tool_info in tool_table:
            try:
                # Use combined description for better entailment scoring
                description = tool_info['combined_description']
                
                # Judge entailment: does the query entail that this tool should be used?
                # Create modified versions without changing the original query
                entailment_query = "The user asked: " + query
                entailment_description = "The user wants to use: " + description
                score = self.entailer.judge(entailment_query, entailment_description)
                
                scored_tools.append((tool_info, score))
                
                _logger.debug(f"Tool {tool_info['name']} scored {score:.3f} for query: {query}")
                
            except Exception as e:
                _logger.error(f"Error scoring tool {tool_info['name']}: {e}")
                scored_tools.append((tool_info, 0.0))
        
        # Sort by score (highest first)
        scored_tools.sort(key=lambda x: x[1], reverse=True)
        
        return scored_tools
    
    async def _execute_tool(self, query: str, tool_info: Dict[str, Any], score: float) -> Optional[ToolResult]:
        """Execute a single tool with extracted arguments."""
        try:
            _logger.info(f"Executing tool {tool_info['name']} with score {score:.3f}")
            
            # Extract arguments for this tool
            arguments = await self.extractor.extract_arguments(
                query=query,
                tool_name=tool_info['name'],
                tool_description=tool_info['description'],
                input_schema=tool_info['input_schema']
            )
            
            _logger.debug(f"Extracted arguments for {tool_info['name']}: {arguments}")

            if not arguments:
                _logger.warning(f"No valid arguments extracted for tool {tool_info['name']}")
                return None

            # Execute the tool
            tool_group = tool_info['group']
            result = await tool_group.call_tool(tool_info['name'], arguments)
            
            # Create result object
            return ToolResult(
                original_query=query,
                tool_description=tool_info['description'],
                tool_result=result
            )
            
        except Exception as e:
            _logger.error(f"Error executing tool {tool_info['name']}: {e}")
            raise
    
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
        summary += f"- Entailment threshold: {self.threshold}\n\n"
        summary += "Groups:\n" + "\n".join(f"  {line}" for line in summary_lines)
        
        return summary