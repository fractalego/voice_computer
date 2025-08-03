"""
System prompts for the voice computer system.
"""

def get_voice_assistant_system_prompt(tool_context: str = "", facts: list = None) -> str:
    """
    Get the system prompt for the voice assistant.
    
    Args:
        tool_context: Optional context from recent tool execution results
        facts: Optional list of facts to include in the system prompt
        
    Returns:
        The formatted system prompt
    """
    base_prompt = "You are a helpful voice assistant."
    
    # Add facts section if facts are provided
    facts_section = ""
    if facts:
        facts_section = "\n\nKey Facts:\n" + "\n".join(f"- {fact}" for fact in facts)
    
    if tool_context:
        return f"""{base_prompt}{facts_section}

Use the recent tool results below to provide informed responses to user queries.

{tool_context}

Instructions:
1. Use the tool results to answer questions when relevant
2. Reference specific tool results when helpful
3. Be conversational and helpful
4. If tool results don't contain relevant information, use your general knowledge"""
    
    return f"{base_prompt}{facts_section}"


def get_argument_extraction_system_prompt(tool_name: str, tool_description: str, params_text: str, facts: list = None) -> str:
    """
    Get the system prompt for argument extraction.
    
    Args:
        tool_name: Name of the tool to extract arguments for
        tool_description: Description of what the tool does
        params_text: Formatted parameter descriptions
        facts: Optional list of facts for additional context
        
    Returns:
        The formatted system prompt for argument extraction
    """
    # Add facts section if facts are provided
    facts_section = ""
    if facts:
        facts_section = f"\n\nKey Facts:\n" + "\n".join(f"- {fact}" for fact in facts)
    
    return f"""You are an argument extraction assistant. Your job is to extract tool arguments from natural language queries.{facts_section}

Tool Information:
- Name: {tool_name}
- Description: {tool_description}

Parameters:
{params_text}

Instructions:
1. Analyze the conversation history and the user's latest query to extract the relevant arguments for this tool
2. Use context from previous messages to understand references and implied values
3. Return ONLY a valid JSON object with the extracted arguments
4. Use the exact parameter names from the schema
5. If a required parameter cannot be determined from the conversation, use a reasonable default or null
6. If an optional parameter is not mentioned, omit it from the response
7. Do not include any explanation, just the JSON object

Example response format:
{{"param1": "value1", "param2": 123, "param3": true}}"""


def format_tool_context(tool_results_queue: list) -> str:
    """
    Format tool results into context string for the system prompt.
    
    Args:
        tool_results_queue: List of recent tool execution results
        
    Returns:
        Formatted tool context string
    """
    if not tool_results_queue:
        return ""
    
    tool_context = "Recent tool execution results:\n\n"
    for i, result in enumerate(tool_results_queue):
        tool_context += f"Tool Result {i+1}:\n"
        tool_context += f"Query: {result.original_query}\n"
        tool_context += f"Tool: {result.tool_description}\n"
        tool_context += f"Result: {result.tool_result}\n\n"
    
    return tool_context


def format_parameter_descriptions(properties: dict, required_params: list) -> str:
    """
    Format parameter descriptions for the argument extraction prompt.
    
    Args:
        properties: Dictionary of parameter properties from JSON schema
        required_params: List of required parameter names
        
    Returns:
        Formatted parameter descriptions string
    """
    param_descriptions = []
    for param_name, param_info in properties.items():
        param_type = param_info.get("type", "string")
        param_desc = param_info.get("description", "No description available")
        is_required = param_name in required_params
        
        required_str = " (REQUIRED)" if is_required else " (optional)"
        param_descriptions.append(f"- {param_name} ({param_type}){required_str}: {param_desc}")
    
    return "\n".join(param_descriptions) if param_descriptions else "No parameters required"