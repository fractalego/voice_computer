"""
Argument extractor for extracting tool arguments from natural language queries using configurable LLM.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Union
from .config import Config
from .client import OllamaClient
from .client.hf_client import HFClient
from .data_types import Messages, Utterance
from .model_factory import get_model_factory
from .prompt import get_argument_extraction_system_prompt, format_parameter_descriptions

_logger = logging.getLogger(__name__)


class ArgumentExtractor:
    """Extracts tool arguments from natural language using configurable LLM."""
    
    def __init__(self, client: Optional[Union[OllamaClient, HFClient]] = None, config: Optional[Config] = None):
        """
        Initialize the argument extractor.
        
        Args:
            client: LLM client for communication (optional, will create from config if None)
            config: Configuration object (optional)
        """
        self.config = config
        
        # Determine which client to use
        if client is not None:
            # Use provided client
            self.client = client
        elif config is not None:
            # Get client from factory based on config
            self.client = self._create_client_from_config(config)
        else:
            raise ValueError("Either client or config must be provided")

    def _create_client_from_config(self, config: Config) -> Union[OllamaClient, HFClient]:
        """Create appropriate client based on configuration."""
        client_type = config.get_value("extractor_client_type")
        if client_type is None:
            # Default to main LLM client type if extractor type not specified
            client_type = config.get_value("llm_client_type") or "ollama"

        model_factory = get_model_factory()
        
        if client_type == "ollama":
            extractor_host = config.get_value("extractor_host") or config.get_value("ollama_host")
            extractor_model = config.get_value("extractor_model") or config.get_value("ollama_model")
            
            client = model_factory.get_ollama_client(
                model=extractor_model,
                host=extractor_host
            )
            _logger.info(f"Using Ollama client for extractor with model {extractor_model} at {extractor_host}")
            
        elif client_type == "huggingface":
            extractor_model = config.get_value("extractor_model") or config.get_value("huggingface_model")
            
            # Pass quantization settings from main config for extractor
            device = config.get_value("huggingface_device")
            torch_dtype = config.get_value("huggingface_torch_dtype")
            quantization = config.get_value("huggingface_quantization")
            
            client = model_factory.get_hf_client(
                model=extractor_model,
                device=device,
                torch_dtype=torch_dtype,
                quantization=quantization
            )
            _logger.info(f"Using HuggingFace client for extractor with model {extractor_model}")
            
        else:
            raise ValueError(f"Unknown extractor client type: {client_type}")
            
        return client
    
    async def extract_arguments(self, query: str, tool_name: str, tool_description: str, input_schema: Dict[str, Any], 
                              conversation_history: Optional[List[Utterance]] = None, facts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract tool arguments from a natural language query.
        
        Args:
            query: The user's natural language query
            tool_name: Name of the tool to extract arguments for
            tool_description: Description of what the tool does
            input_schema: JSON schema describing the tool's input parameters
            conversation_history: Recent conversation history for context
            facts: System facts for additional context
            
        Returns:
            Dictionary of extracted arguments matching the input schema
        """
        try:
            # Build the extraction prompt with context
            system_prompt = self._build_extraction_prompt(tool_name, tool_description, input_schema, facts)
            
            # Create messages for the extraction
            messages = Messages()
            messages = messages.add_system_prompt(system_prompt)
            
            # Add conversation history for context if provided
            if conversation_history:
                for utterance in conversation_history:
                    if utterance.role == "user":
                        messages = messages.add_user_utterance(utterance.content)
                    elif utterance.role == "assistant":
                        messages = messages.add_assistant_utterance(utterance.content)
            
            # Add the current query as the final user message
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
    
    def _build_extraction_prompt(self, tool_name: str, tool_description: str, input_schema: Dict[str, Any], facts: Optional[List[str]] = None) -> str:
        """Build the system prompt for argument extraction."""
        # Extract required and optional parameters from schema
        properties = input_schema.get("properties", {})
        required_params = input_schema.get("required", [])
        
        # Format parameter descriptions
        params_text = format_parameter_descriptions(properties, required_params)
        
        return get_argument_extraction_system_prompt(tool_name, tool_description, params_text, facts)

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