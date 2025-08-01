"""
Argument extractor for extracting tool arguments from natural language queries using LLM.
"""

import json
import logging
from typing import Dict, Any, Optional
from .config import Config
from .ollama_client import OllamaClient
from .data_types import Messages

_logger = logging.getLogger(__name__)


class ArgumentExtractor:
    """Extracts tool arguments from natural language using LLM."""
    
    def __init__(self, ollama_client: Optional[OllamaClient] = None, config: Optional[Config] = None):
        """
        Initialize the argument extractor.
        
        Args:
            ollama_client: Ollama client for LLM communication (optional, will create from config if None)
            config: Configuration object (optional)
        """
        self.config = config
        
        # Determine which client to use
        if ollama_client is not None:
            # Use provided client
            self.client = ollama_client
        elif config is not None:
            # Create extractor-specific client from config
            extractor_host = config.get_value("extractor_host") or config.get_value("ollama_host")
            extractor_model = config.get_value("extractor_model") or config.get_value("ollama_model")
            
            self.client = OllamaClient(
                model=extractor_model,
                host=extractor_host
            )
            _logger.info(f"Created extractor-specific OllamaClient with model {extractor_model} at {extractor_host}")
        else:
            raise ValueError("Either ollama_client or config must be provided")
    
    async def extract_arguments(self, query: str, tool_name: str, tool_description: str, input_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract tool arguments from a natural language query.
        
        Args:
            query: The user's natural language query
            tool_name: Name of the tool to extract arguments for
            tool_description: Description of what the tool does
            input_schema: JSON schema describing the tool's input parameters
            
        Returns:
            Dictionary of extracted arguments matching the input schema
        """
        try:
            # Build the extraction prompt
            system_prompt = self._build_extraction_prompt(tool_name, tool_description, input_schema)
            
            # Create messages for the extraction
            messages = Messages()
            messages = messages.add_system_prompt(system_prompt)
            messages = messages.add_user_utterance(query)
            
            # Get LLM response
            response = await self.client.predict(messages)
            
            # Parse the JSON response
            arguments = self._parse_arguments_response(response.message)
            
            # Validate against schema (basic validation)
            validated_args = self._validate_arguments(arguments, input_schema)
            
            return validated_args
            
        except Exception as e:
            _logger.error(f"Error extracting arguments for tool {tool_name}: {e}")
            return {}
    
    def _build_extraction_prompt(self, tool_name: str, tool_description: str, input_schema: Dict[str, Any]) -> str:
        """Build the system prompt for argument extraction."""
        
        # Extract required and optional parameters from schema
        properties = input_schema.get("properties", {})
        required_params = input_schema.get("required", [])
        
        param_descriptions = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "string")
            param_desc = param_info.get("description", "No description available")
            is_required = param_name in required_params
            
            required_str = " (REQUIRED)" if is_required else " (optional)"
            param_descriptions.append(f"- {param_name} ({param_type}){required_str}: {param_desc}")
        
        params_text = "\n".join(param_descriptions) if param_descriptions else "No parameters required"
        
        return f"""You are an argument extraction assistant. Your job is to extract tool arguments from natural language queries.

Tool Information:
- Name: {tool_name}
- Description: {tool_description}

Parameters:
{params_text}

Instructions:
1. Analyze the user's query and extract the relevant arguments for this tool
2. Return ONLY a valid JSON object with the extracted arguments
3. Use the exact parameter names from the schema
4. If a required parameter cannot be determined from the query, use a reasonable default or null
5. If an optional parameter is not mentioned in the query, omit it from the response
6. Do not include any explanation, just the JSON object

Example response format:
{{"param1": "value1", "param2": 123, "param3": true}}"""

    def _parse_arguments_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response to extract JSON arguments."""
        try:
            # Try to find JSON in the response
            response = response.strip()
            
            # If response starts and ends with braces, it's likely pure JSON
            if response.startswith('{') and response.endswith('}'):
                return json.loads(response)
            
            # Look for JSON block in the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # If no JSON found, return empty dict
            _logger.warning(f"No JSON found in LLM response: {response}")
            return {}
            
        except json.JSONDecodeError as e:
            _logger.error(f"Failed to parse JSON from LLM response: {e}")
            _logger.error(f"Response was: {response}")
            return {}
    
    def _validate_arguments(self, arguments: Dict[str, Any], input_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Basic validation of extracted arguments against schema."""
        if not input_schema or not arguments:
            return arguments
        
        properties = input_schema.get("properties", {})
        required_params = input_schema.get("required", [])
        validated_args = {}
        
        # Check each extracted argument
        for param_name, param_value in arguments.items():
            if param_name in properties:
                # Parameter exists in schema, include it
                validated_args[param_name] = param_value
            else:
                _logger.warning(f"Parameter '{param_name}' not found in schema, skipping")
        
        # Check for missing required parameters
        for required_param in required_params:
            if required_param not in validated_args:
                _logger.warning(f"Required parameter '{required_param}' missing from extracted arguments")
                # Set to None or reasonable default
                param_info = properties.get(required_param, {})
                param_type = param_info.get("type", "string")
                
                if param_type == "string":
                    validated_args[required_param] = ""
                elif param_type == "number" or param_type == "integer":
                    validated_args[required_param] = 0
                elif param_type == "boolean":
                    validated_args[required_param] = False
                elif param_type == "array":
                    validated_args[required_param] = []
                elif param_type == "object":
                    validated_args[required_param] = {}
                else:
                    validated_args[required_param] = None
        
        return validated_args