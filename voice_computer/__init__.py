"""
Voice Computer - A voice-driven assistant with MCP integration.

This module provides a complete voice-driven interface that integrates
speech recognition, text-to-speech, and MCP (Model Context Protocol) tools
with Ollama for intelligent responses.
"""

from .client import VoiceComputerClient
from .voice_interface import VoiceInterface
from .whisper_listener import WhisperListener
from .ollama_client import OllamaClient
from .mcp_connector import MCPTools, MCPStdioConnector
from .data_types import Messages, Utterance, ClientResponse, ToolCall
from .tool_handler import ToolHandler
from .entailer import Entailer
from .extractor import ArgumentExtractor

__version__ = "1.0.0"
__all__ = [
    "VoiceComputerClient",
    "VoiceInterface", 
    "WhisperListener",
    "OllamaClient",
    "MCPTools",
    "MCPStdioConnector",
    "Messages",
    "Utterance",
    "ClientResponse", 
    "ToolCall",
    "ToolHandler",
    "Entailer",
    "ArgumentExtractor",
]