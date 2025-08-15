"""
HuggingFace client for communicating with HuggingFace Inference API.
"""

import requests
import json
import logging
import asyncio
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

from .base_client import BaseClient
from ..data_types import Messages, Tool, ClientResponse, ToolCall

_logger = logging.getLogger(__name__)


class HFConnectionError(Exception):
    """Exception raised when there's a connection error to HuggingFace API."""

    def __init__(self, model: str, original_error: Exception):
        self.model = model
        self.original_error = original_error
        super().__init__(
            f"Failed to connect to HuggingFace API for model '{model}': {original_error}"
        )


class HFResponseError(Exception):
    """Exception raised when HuggingFace returns an error response."""

    def __init__(self, model: str, status_code: int, response_text: str):
        self.model = model
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(
            f"HuggingFace error for model '{model}': HTTP {status_code} - {response_text}"
        )


class HFClient(BaseClient):
    """Client for HuggingFace Inference API."""

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        base_url: str = "https://api-inference.huggingface.co/models",
    ):
        super().__init__(model, temperature, max_tokens)
        self.api_key = api_key
        self.base_url = base_url
        self.api_url = f"{base_url}/{model}"

        # Log HF connection details
        _logger.info(f"Initializing HFClient for model '{model}' at '{self.api_url}'")

    def _messages_to_prompt(self, messages: Messages) -> str:
        """
        Convert Messages to a single prompt string for HF models that don't support chat format.
        """
        prompt_parts = []
        for msg in messages.utterances:
            role = msg.role
            content = msg.content
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    async def predict(
        self,
        messages: Messages,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        token_queue: Optional[asyncio.Queue] = None,
    ) -> ClientResponse:
        """
        Make a prediction request to HuggingFace Inference API.
        
        Args:
            messages: The conversation messages
            stop_sequences: Optional list of stop sequences
            tools: Optional list of tools available to the model
            stream: Whether to stream the response
            token_queue: Asyncio queue for streaming tokens in real-time
            
        Returns:
            ClientResponse containing the response message and any tool calls
        """
        _logger.debug(f"Making request to HuggingFace API for model '{self.model}'")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages)
        
        # Build parameters
        parameters = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_tokens,
            "return_full_text": False,
        }
        
        if stop_sequences:
            parameters["stop"] = stop_sequences

        data = {
            "inputs": prompt,
            "parameters": parameters,
            "stream": stream,
        }

        # Note: Tools are not directly supported by most HF models via API
        # This would require model-specific formatting or fine-tuned models
        if tools:
            _logger.warning("Tool calls are not currently supported by HFClient")

        try:
            if stream and token_queue:
                return await self._handle_async_streaming_response(
                    self.api_url, headers, data, token_queue
                )
            else:
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=600,
                )
        except asyncio.exceptions.CancelledError:
            _logger.debug("HuggingFace streaming request was cancelled")
            return ClientResponse(message="", tool_calls=None)
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error to HuggingFace API for model {self.model}"
            _logger.error(error_msg)
            raise HFConnectionError(self.model, e)
        except requests.exceptions.Timeout as e:
            error_msg = f"Timeout connecting to HuggingFace API for model {self.model}"
            _logger.error(error_msg)
            raise HFConnectionError(self.model, e)
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error connecting to HuggingFace API for model {self.model}: {e}"
            _logger.error(error_msg)
            raise HFConnectionError(self.model, e)

        if response.status_code == 200:
            _logger.debug(f"Successfully received response from HuggingFace API")
            try:
                response_data = response.json()
                
                # Handle different response formats
                if isinstance(response_data, list) and len(response_data) > 0:
                    # Standard text generation response
                    message_content = response_data[0].get("generated_text", "")
                elif isinstance(response_data, dict):
                    # Alternative response format
                    message_content = response_data.get("generated_text", "")
                else:
                    message_content = str(response_data)

                # HF API typically doesn't return tool calls for most models
                tool_calls = None

                return ClientResponse(message=message_content, tool_calls=tool_calls)
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                error_msg = f"Invalid response format from HuggingFace API: {e}"
                _logger.error(error_msg)
                raise HFResponseError(self.model, response.status_code, str(e))
        else:
            _logger.error(f"Error response from HuggingFace API: {response.status_code}, {response.text}")
            raise HFResponseError(self.model, response.status_code, response.text)

    async def _handle_async_streaming_response(
        self,
        url: str,
        headers: dict,
        data: dict,
        token_queue: asyncio.Queue,
    ) -> ClientResponse:
        """
        Handle streaming response from HuggingFace API using async queue.
        
        Args:
            url: The API endpoint URL
            headers: Request headers
            data: Request data
            token_queue: Asyncio queue to put tokens into as they arrive
            
        Returns:
            ClientResponse with complete message
        """
        def _stream_request(loop):
            """Run the streaming request in a thread."""
            try:
                _logger.debug("Starting streaming request to HuggingFace API...")
                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=600,
                    stream=True,
                )
                _logger.debug(f"Got response from HuggingFace API, status: {response.status_code}")
                
                if response.status_code != 200:
                    raise HFResponseError(self.model, response.status_code, response.text)
                
                complete_message = ""
                token_count = 0
                
                _logger.debug("Starting to process streaming response lines...")
                for line in response.iter_lines(decode_unicode=True):
                    if line.strip():
                        # HuggingFace streaming format: data: {json}
                        if line.startswith("data: "):
                            json_str = line[6:]  # Remove "data: " prefix
                            if json_str.strip() == "[DONE]":
                                break
                            
                            try:
                                chunk_data = json.loads(json_str)
                                
                                # Extract token from different possible formats
                                token = None
                                if isinstance(chunk_data, dict):
                                    # Check various possible fields for the token
                                    token = (chunk_data.get("token") or 
                                            chunk_data.get("generated_text") or
                                            chunk_data.get("text"))
                                    
                                    # Handle choices format (similar to OpenAI)
                                    if "choices" in chunk_data and chunk_data["choices"]:
                                        delta = chunk_data["choices"][0].get("delta", {})
                                        token = delta.get("content")
                                
                                if token:
                                    complete_message += token
                                    token_count += 1

                                    # Put token in queue for immediate display
                                    asyncio.run_coroutine_threadsafe(
                                        token_queue.put(token), 
                                        loop
                                    )
                            except json.JSONDecodeError:
                                # Skip invalid JSON lines
                                continue
                
                # Signal completion
                asyncio.run_coroutine_threadsafe(
                    token_queue.put(None),  # None signals end of stream
                    loop
                )
                
                return ClientResponse(message=complete_message, tool_calls=None)
                
            except requests.exceptions.ConnectionError as e:
                raise HFConnectionError(self.model, e)
            except requests.exceptions.Timeout as e:
                raise HFConnectionError(self.model, e)
            except requests.exceptions.RequestException as e:
                raise HFConnectionError(self.model, e)
            except HFResponseError:
                # Re-raise HFResponseError without modification
                raise
            except Exception as e:
                error_msg = f"Error processing streaming response from HuggingFace API: {e}"
                _logger.error(error_msg)
                raise HFResponseError(self.model, 200, str(e))
        
        # Run the streaming request in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(executor, _stream_request, loop)
            return await future