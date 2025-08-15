"""
Entailment model for selecting relevant tools from a conversation context.

Uses Ollama language models to determine which tools are relevant for a given query.
"""

import logging
from typing import Optional, List, Dict, Any
from .config import Config
from .client import OllamaClient
from .model_factory import get_model_factory
from .data_types import Messages, Utterance

_logger = logging.getLogger(__name__)


class Entailer:
    """Entailment model for selecting relevant tools using Ollama."""
    
    def __init__(self, config: Config, tools: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize the entailer with Ollama client and tools list.
        
        Args:
            config: Configuration object containing model settings
            tools: List of tool dictionaries with name and description
        """
        self.config = config
        self.tools = tools or []

        # Use entailer-specific settings or fall back to main Ollama settings
        self.host = config.get_value("entailer_host") or config.get_value("ollama_host")
        self.model = config.get_value("entailer_model") or config.get_value("ollama_model")

        
        self.ollama_client = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the Ollama client."""
        if self.initialized:
            return
            
        _logger.info(f"Initializing entailer with Ollama model {self.model} at {self.host}")
        
        try:
            # Get cached Ollama client from factory if available
            model_factory = get_model_factory()
            self.ollama_client = model_factory.get_ollama_client(self.model, self.host)
            
            if self.ollama_client is None:
                # Create new client if not cached
                self.ollama_client = OllamaClient(model=self.model, host=self.host)
            
            self.initialized = True
            _logger.info(f"Entailer initialized successfully with model {self.model}")
            
        except Exception as e:
            _logger.error(f"Failed to initialize entailer with model {self.model}: {e}")
            raise
    
    async def select_relevant_tools(self, conversation_context: str) -> List[int]:
        """
        Select relevant tools for the given conversation context.
        
        Args:
            conversation_context: The conversation context to analyze
            
        Returns:
            List of tool indices that are relevant for the context
        """
        if not self.initialized:
            self.initialize()
        
        if not self.tools:
            return []
        
        try:
            # Create tool selection prompt
            prompt = self._create_tool_selection_prompt(conversation_context)
            
            # Create Messages object for Ollama
            messages = Messages()
            messages = messages.add_user_utterance(prompt)
            
            # Get response from Ollama
            response = await self.ollama_client.predict(messages)
            
            # Parse the response to get tool indices
            return self._parse_tool_indices(response.message)
                
        except Exception as e:
            _logger.error(f"Error during tool selection: {e}")
            # Default to empty list on error to be conservative
            return []
    
    def _create_tool_selection_prompt(self, conversation_context: str) -> str:
        """
        Create a prompt for tool selection.
        
        Args:
            conversation_context: The conversation context to analyze
            
        Returns:
            Formatted prompt string
        """
        # Format tools list
        tools_list = ""
        for i, tool in enumerate(self.tools):
            tools_list += f"{i}: {tool['name']} - {tool['description']}\n"
        
        prompt = f"""Task: Select the relevant tools for the user's request. Be strict and only select tools that are directly needed.

Available Tools:
{tools_list}

User Request: "{conversation_context}"

Which tools are relevant for this request? Be very selective - only choose tools that are directly needed to fulfill the user's request.

Respond ONLY with the indices of relevant tools separated by commas (e.g., "0,2,5"). If no tools are relevant, respond with "NONE".

Answer:"""
        
        return prompt
    
    def _create_sentence_matching_prompt(self, user_input: str, target_sentences: List[str]) -> str:
        """
        Create a prompt for sentence matching.
        
        Args:
            user_input: The user's input text
            target_sentences: List of target sentences to check against
            
        Returns:
            Formatted prompt string
        """
        # Format sentences list
        sentences_list = ""
        for i, sentence in enumerate(target_sentences):
            sentences_list += f"{i}: {sentence}\n"
        
        prompt = f"""Task: Determine which of the target sentences the user's input entails or matches in meaning.

User Input: "{user_input}"

Target Sentences:
{sentences_list}

Which target sentences does the user input entail or match in meaning? Be strict - only select sentences that clearly match the user's intent.

Respond ONLY with the indices of matching sentences separated by commas (e.g., "0,2,5"). If no sentences match, respond with "NONE".

Answer:"""
        
        return prompt
    
    def _parse_tool_indices(self, response: str) -> List[int]:
        """
        Parse the tool indices from the model response.
        
        Args:
            response: Raw response from the model
            
        Returns:
            List of tool indices
        """
        if not response:
            return []
        
        # Clean and normalize response
        response = response.strip().upper()
        
        # Check for NONE response
        if "NONE" in response:
            return []
        
        # Extract numbers from the response
        import re
        numbers = re.findall(r'\d+', response)
        
        try:
            indices = [int(num) for num in numbers]
            # Filter out invalid indices
            valid_indices = [i for i in indices if 0 <= i < len(self.tools)]
            
            if len(valid_indices) != len(indices):
                _logger.warning(f"Some tool indices were invalid in response: '{response}'")
            
            return valid_indices
            
        except ValueError:
            _logger.warning(f"Could not parse tool indices from response: '{response}'")
            return []
    
    def _parse_sentence_indices(self, response: str, max_sentences: int) -> List[int]:
        """
        Parse the sentence indices from the model response.
        
        Args:
            response: Raw response from the model
            max_sentences: Maximum number of sentences available
            
        Returns:
            List of sentence indices
        """
        if not response:
            return []
        
        # Clean and normalize response
        response = response.strip().upper()
        
        # Check for NONE response
        if "NONE" in response:
            return []
        
        # Extract numbers from the response
        import re
        numbers = re.findall(r'\d+', response)
        
        try:
            indices = [int(num) for num in numbers]
            # Filter out invalid indices
            valid_indices = [i for i in indices if 0 <= i < max_sentences]
            
            if len(valid_indices) != len(indices):
                _logger.warning(f"Some sentence indices were invalid in response: '{response}'")
            
            return valid_indices
            
        except ValueError:
            _logger.warning(f"Could not parse sentence indices from response: '{response}'")
            return []
    
    def update_tools(self, tools: List[Dict[str, Any]]) -> None:
        """
        Update the tools list.
        
        Args:
            tools: New list of tool dictionaries with name and description
        """
        self.tools = tools
        _logger.debug(f"Updated entailer with {len(tools)} tools")
    
    async def judge_list(self, user_input: str, target_sentences: List[str]) -> List[int]:
        """
        Judge which target sentences the user input entails.
        
        Args:
            user_input: The user's input text
            target_sentences: List of target sentences to check against
            
        Returns:
            List of indices of target sentences that the user input entails
        """
        if not self.initialized:
            self.initialize()
        
        if not target_sentences:
            return []
        
        try:
            # Create entailment prompt with all target sentences
            prompt = self._create_sentence_matching_prompt(user_input, target_sentences)
            
            # Create Messages object for Ollama
            messages = Messages()
            messages = messages.add_user_utterance(prompt)
            
            # Get response from Ollama
            response = await self.ollama_client.predict(messages)
            
            # Parse the response to get sentence indices
            return self._parse_sentence_indices(response.message, len(target_sentences))
                
        except Exception as e:
            _logger.error(f"Error during sentence matching: {e}")
            return []
    
    async def judge(self, lhs: str, rhs: str) -> bool:
        """
        Fallback method for simple pairwise entailment.
        
        Args:
            lhs: The premise text (left-hand side)
            rhs: The hypothesis text (right-hand side)
            
        Returns:
            Boolean: True if lhs entails rhs, False otherwise
        """
        # Use judge_list for consistency
        indices = await self.judge_list(lhs, [rhs])
        return len(indices) > 0