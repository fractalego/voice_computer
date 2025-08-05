"""
Ollama client for communicating with Ollama API.
"""

import requests
import json
import logging
import asyncio
from io import StringIO
from typing import Optional, List, AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor

from .data_types import Messages, Tool, ClientResponse, ToolCall

_logger = logging.getLogger(__name__)


class OllamaConnectionError(Exception):
    """Exception raised when there's a connection error to Ollama."""

    def __init__(self, host: str, model: str, original_error: Exception):
        self.host = host
        self.model = model
        self.original_error = original_error
        super().__init__(
            f"Failed to connect to Ollama at {host} for model '{model}': {original_error}"
        )


class OllamaResponseError(Exception):
    """Exception raised when Ollama returns an error response."""

    def __init__(self, host: str, model: str, status_code: int, response_text: str):
        self.host = host
        self.model = model
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(
            f"Ollama error at {host} for model '{model}': HTTP {status_code} - {response_text}"
        )


class OllamaClient:
    """Client for Ollama API."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        host: str = "http://localhost:11434",
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.host = host

        # Log Ollama connection details
        _logger.info(f"Initializing OllamaClient for model '{model}' on host '{host}'")

    async def predict(
        self,
        messages: Messages,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        token_queue: Optional[asyncio.Queue] = None,
    ) -> ClientResponse:
        """
        Make a prediction request to Ollama.
        
        Args:
            messages: The conversation messages
            stop_sequences: Optional list of stop sequences
            tools: Optional list of tools available to the model
            stream: Whether to stream the response
            token_queue: Asyncio queue for streaming tokens in real-time
            
        Returns:
            ClientResponse containing the response message and any tool calls
        """
        _logger.debug(
            f"Making request to Ollama instance at {self.host} with model '{self.model}'"
        )

        headers = {"Content-Type": "application/json"}

        # Convert tools to dict format for API if provided
        tools_dict = None
        if tools:
            tools_dict = [tool.model_dump() for tool in tools]

        data = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages.model_dump()["utterances"],
            "options": {
                "stop": stop_sequences,
            },
            "stream": stream,
            "tools": tools_dict,
            "think": False,
        }

        try:
            if stream and token_queue:
                return await self._handle_async_streaming_response(
                    f"{self.host}/api/chat", headers, data, token_queue
                )
            else:
                response = requests.post(
                    f"{self.host}/api/chat",
                    headers=headers,
                    data=json.dumps(data),
                    timeout=600,
                )
        except asyncio.exceptions.CancelledError:
            _logger.debug("Ollama streaming request was cancelled")
            return ClientResponse(message="", tool_calls=None)
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection refused to Ollama at {self.host}. Please ensure Ollama is running and accessible."
            _logger.error(error_msg)
            raise OllamaConnectionError(self.host, self.model, e)
        except requests.exceptions.Timeout as e:
            error_msg = f"Timeout connecting to Ollama at {self.host}. The server may be overloaded or unreachable."
            _logger.error(error_msg)
            raise OllamaConnectionError(self.host, self.model, e)
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error connecting to Ollama at {self.host}: {e}"
            _logger.error(error_msg)
            raise OllamaConnectionError(self.host, self.model, e)

        if response.status_code == 200:
            _logger.debug(f"Successfully received response from {self.host}")
            try:
                response_data = json.loads(response.text)

                message_content = response_data["message"]["content"]

                # Extract tool calls if present
                tool_calls = None
                if (
                    "message" in response_data
                    and "tool_calls" in response_data["message"]
                ):
                    tool_calls = []
                    for tool_call_data in response_data["message"]["tool_calls"]:
                        tool_call = ToolCall(
                            id=tool_call_data.get("id", ""),
                            type=tool_call_data.get("type", "function"),
                            function=tool_call_data.get("function", {}),
                        )
                        tool_calls.append(tool_call)

                return ClientResponse(message=message_content, tool_calls=tool_calls)
            except (json.JSONDecodeError, KeyError) as e:
                error_msg = f"Invalid response format from Ollama at {self.host}: {e}"
                _logger.error(error_msg)
                raise OllamaResponseError(
                    self.host, self.model, response.status_code, str(e)
                )
        else:
            _logger.error(
                f"Error response from {self.host}: {response.status_code}, {response.text}"
            )
            raise OllamaResponseError(
                self.host, self.model, response.status_code, response.text
            )
    
    async def _handle_async_streaming_response(
        self,
        url: str,
        headers: dict,
        data: dict,
        token_queue: asyncio.Queue,
    ) -> ClientResponse:
        """
        Handle streaming response from Ollama using async queue.
        
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
                _logger.debug("Starting streaming request to Ollama...")
                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=600,
                    stream=True,
                )
                _logger.debug(f"Got response from Ollama, status: {response.status_code}")
                
                if response.status_code != 200:
                    raise OllamaResponseError(
                        self.host, self.model, response.status_code, response.text
                    )
                
                complete_message = ""
                tool_calls = None
                token_count = 0
                
                _logger.debug("Starting to process streaming response lines...")
                for line in response.iter_lines(decode_unicode=True):
                    if line.strip():
                        chunk_data = json.loads(line)
                        
                        if "message" in chunk_data and "content" in chunk_data["message"]:
                            token = chunk_data["message"]["content"]
                            if not token:
                                continue
                            complete_message += token
                            token_count += 1

                            # Put token in queue for immediate display
                            asyncio.run_coroutine_threadsafe(
                                token_queue.put(token), 
                                loop
                            )
                        
                        # Check for tool calls in the final chunk
                        if (
                            chunk_data.get("done", False) and
                            "message" in chunk_data and
                            "tool_calls" in chunk_data["message"]
                        ):
                            tool_calls = []
                            for tool_call_data in chunk_data["message"]["tool_calls"]:
                                tool_call = ToolCall(
                                    id=tool_call_data.get("id", ""),
                                    type=tool_call_data.get("type", "function"),
                                    function=tool_call_data.get("function", {}),
                                )
                                tool_calls.append(tool_call)
                        
                        # Break if done
                        if chunk_data.get("done", False):
                            break
                
                # Signal completion
                asyncio.run_coroutine_threadsafe(
                    token_queue.put(None),  # None signals end of stream
                    loop
                )
                
                return ClientResponse(message=complete_message, tool_calls=tool_calls)
                
            except requests.exceptions.ConnectionError as e:
                raise OllamaConnectionError(self.host, self.model, e)
            except requests.exceptions.Timeout as e:
                raise OllamaConnectionError(self.host, self.model, e)
            except requests.exceptions.RequestException as e:
                raise OllamaConnectionError(self.host, self.model, e)
            except (json.JSONDecodeError, KeyError) as e:
                error_msg = f"Invalid streaming response format from Ollama at {self.host}: {e}"
                _logger.error(error_msg)
                raise OllamaResponseError(
                    self.host, self.model, 200, str(e)
                )
        
        # Run the streaming request in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(executor, _stream_request, loop)
            return await future