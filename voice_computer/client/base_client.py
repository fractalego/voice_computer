"""
Abstract base class for LLM clients.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import asyncio

from ..data_types import Messages, Tool, ClientResponse


class BaseClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def predict(
        self,
        messages: Messages,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        token_queue: Optional[asyncio.Queue] = None,
    ) -> ClientResponse:
        """
        Make a prediction request to the LLM provider.
        
        Args:
            messages: The conversation messages
            stop_sequences: Optional list of stop sequences
            tools: Optional list of tools available to the model
            stream: Whether to stream the response
            token_queue: Asyncio queue for streaming tokens in real-time
            
        Returns:
            ClientResponse containing the response message and any tool calls
        """
        pass