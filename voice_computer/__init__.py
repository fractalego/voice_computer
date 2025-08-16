"""
Voice Computer - A voice-driven assistant with MCP integration.

This module provides a complete voice-driven interface that integrates
speech recognition, text-to-speech, and MCP (Model Context Protocol) tools
with Ollama for intelligent responses.
"""

from voice_computer.conversation_handler import ConversationHandler
from voice_computer.voice_interface import VoiceInterface
from voice_computer.whisper_listener import WhisperListener
from voice_computer.client import OllamaClient, HFClient
from voice_computer.mcp_connector import MCPTools, MCPStdioConnector
from voice_computer.data_types import Messages, Utterance, ClientResponse, ToolCall
from voice_computer.tool_handler import ToolHandler
from voice_computer.entailer import Entailer
from voice_computer.extractor import ArgumentExtractor
from voice_computer.model_factory import get_model_factory, ModelFactory

__version__ = "1.0.0"
__all__ = [
    "ConversationHandler",
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