"""
Voice Computer - A voice-driven assistant with MCP integration.

This module provides a complete voice-driven interface that integrates
speech recognition, text-to-speech, and MCP (Model Context Protocol) tools
with Ollama for intelligent responses.
"""

from .handler import Handler
from .voice_interface import VoiceInterface
from .whisper_listener import WhisperListener
from .client import OllamaClient, HFClient
from .mcp_connector import MCPTools, MCPStdioConnector
from .data_types import Messages, Utterance, ClientResponse, ToolCall
from .tool_handler import ToolHandler
from .entailer import Entailer
from .extractor import ArgumentExtractor
from .model_factory import get_model_factory, ModelFactory

__version__ = "1.0.0"
__all__ = [
    "Handler",
    "VoiceInterface", 
    "WhisperListener",
    "OllamaClient",
    "HFClient",
    "MCPTools",
    "MCPStdioConnector",
    "Messages",
    "Utterance",
    "ClientResponse", 
    "ToolCall",
    "ToolHandler",
    "Entailer",
    "ArgumentExtractor",
    "ModelFactory",
    "get_model_factory",
]