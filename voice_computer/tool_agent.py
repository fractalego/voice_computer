"""
Tool agent for handling MCP tool execution and coordination.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional

from .data_types import Messages, Utterance, ClientResponse
from .ollama_client import OllamaClient
from .mcp_connector import MCPTools

_logger = logging.getLogger(__name__)

# Constants
TASK_COMPLETED_TAG = "[TASK_COMPLETED]"
TOOL_OUTPUT_TAG = "```tools"


class ToolAgent:
    """Agent that coordinates between the LLM and MCP tools."""
    
    def __init__(self, client: OllamaClient, tools: List[MCPTools]):
        self.client = client
        self.tools = tools
        self.max_steps = 5
        
        # Build tools description for the system prompt
        self.tools_description = "\n".join(
            f"Tool group index {group_index}:\n{tool_group.get_tools_descriptions()}\n\n"
            for group_index, tool_group in enumerate(tools)
        )

    async def query(self, messages: Messages) -> str:
        """
        Process a query using MCP tools.
        
        Args:
            messages: The conversation messages
            
        Returns:
            The final response after tool execution
        """
        if not self.tools:
            # No tools available, fallback to direct LLM
            response = await self.client.predict(messages)
            return response.message

        # Add system prompt with tool descriptions
        system_prompt = self._build_system_prompt()
        messages = messages.add_system_prompt(system_prompt)
        
        all_tool_results = []
        all_tool_calls = []

        for step_idx in range(self.max_steps):
            response = await self.client.predict(
                messages=messages, 
                stop_sequences=[TASK_COMPLETED_TAG]
            )
            answer = response.message

            # Check if task is completed
            if TASK_COMPLETED_TAG in answer or answer.strip() == "":
                break

            # Look for tool calls in the response
            tool_calls_found = self._extract_tool_calls(answer)
            
            if tool_calls_found:
                # Execute the tool calls
                step_results = []
                for tool_call in tool_calls_found:
                    try:
                        group_index = int(tool_call['group_index'])
                        tool_index = int(tool_call['tool_index'])
                        arguments = tool_call['arguments']
                        
                        if group_index < len(self.tools):
                            tool_group = self.tools[group_index]
                            if tool_index < len(tool_group.tools):
                                tool = tool_group.tools[tool_index]
                                result = await tool_group.call_tool(tool.name, arguments)
                                step_results.append(result)
                                all_tool_results.append(result)
                                all_tool_calls.append(f"Tool {tool_index} ({tool.name}): {arguments}")
                            else:
                                step_results.append(f"Error: Tool index {tool_index} out of range")
                        else:
                            step_results.append(f"Error: Tool group index {group_index} out of range")
                    except Exception as e:
                        _logger.error(f"Error executing tool call: {e}")
                        step_results.append(f"Error executing tool: {str(e)}")

                # Add results to conversation
                results_text = "\n".join(f"Tool result: {result}" for result in step_results)
                messages = messages.add_user_utterance(
                    f"The tool results are:\n{results_text}\n\n"
                    f"If the user's question is answered, write {TASK_COMPLETED_TAG} at the beginning of your answer. "
                    f"Otherwise, analyze the results and make additional tool calls if needed."
                )
            else:
                # No tool calls found in response
                messages = messages.add_user_utterance(
                    f"The response was: {answer}\n"
                    f"If the user's question is answered, write {TASK_COMPLETED_TAG} at the beginning of your answer. "
                    f"Otherwise, provide the appropriate tool calls in the specified format."
                )

        # Return the final result
        if all_tool_results:
            return f"Based on the tool results, here's what I found:\n\n" + "\n".join(str(result) for result in all_tool_results)
        else:
            return answer

    def _build_system_prompt(self) -> str:
        """Build the system prompt with tool descriptions."""
        return f"""You are a helpful assistant that can use various tools to answer questions and perform tasks.

Available tools:
{self.tools_description}

When you need to use tools, format your response like this:
{TOOL_OUTPUT_TAG}
| group_index | tool_index | arguments |
|-------------|------------|-----------|
| 0           | 1          | {{"key": "value"}} |
```

Important instructions:
1. Always use the exact table format shown above for tool calls
2. The arguments column should contain valid JSON
3. When the task is complete, start your response with {TASK_COMPLETED_TAG}
4. You can make multiple tool calls in a single response
5. Analyze tool results before deciding if more calls are needed
"""

    def _extract_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from the LLM response.
        
        Args:
            text: The response text from the LLM
            
        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        
        # Look for tool table in the response
        matches = re.findall(
            rf"{TOOL_OUTPUT_TAG}(.+?)```",
            text,
            re.DOTALL | re.MULTILINE,
        )
        
        if matches:
            table_content = matches[0].strip()
            lines = table_content.split('\n')
            
            # Skip header and separator lines
            data_lines = [line for line in lines if '|' in line and not line.strip().startswith('|--')]
            
            for line in data_lines[1:]:  # Skip header row
                parts = [part.strip() for part in line.split('|') if part.strip()]
                if len(parts) >= 3:
                    try:
                        group_index = parts[0]
                        tool_index = parts[1] 
                        arguments_str = parts[2]
                        
                        # Parse JSON arguments
                        arguments = json.loads(arguments_str) if arguments_str else {}
                        
                        tool_calls.append({
                            'group_index': group_index,
                            'tool_index': tool_index,
                            'arguments': arguments
                        })
                    except (json.JSONDecodeError, IndexError, ValueError) as e:
                        _logger.warning(f"Failed to parse tool call from line '{line}': {e}")
                        continue
        
        return tool_calls