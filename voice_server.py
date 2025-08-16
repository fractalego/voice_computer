#!/usr/bin/env python3
"""
Voice Computer Server - exposes voice computer input/output streams over WebSocket.

This server allows remote clients to connect and interact with the voice computer
system over the network. The client sends audio data and receives text responses
and audio responses.
"""

import asyncio
import websockets
import json
import logging
import signal
import time
from typing import Optional, Dict, Any
import base64
from pathlib import Path

from voice_computer import Handler
from voice_computer.config import load_config
from voice_computer.data_types import Utterance
from voice_computer.whisper_listener import WhisperListener

_logger = logging.getLogger(__name__)


class VoiceComputerServer:
    """WebSocket server that exposes voice computer functionality."""
    
    def __init__(self, config_path: Optional[str] = None, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.config = load_config(config_path)
        self.handler = Handler(self.config)
        self.connected_clients = set()
        self.server = None
        
        # Initialize Whisper listener for transcription
        self.whisper_listener = WhisperListener(self.config)
        
    async def start(self):
        """Start the WebSocket server."""
        _logger.info(f"Starting Voice Computer Server on {self.host}:{self.port}")
        
        # Initialize MCP tools
        await self.handler._setup_mcp_tools()
        self.handler._initialize_models()
        
        # Initialize Whisper listener
        _logger.info("Initializing Whisper listener...")
        self.whisper_listener.initialize()
        
        # Start WebSocket server
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10
        )
        
        _logger.info(f"Voice Computer Server started on ws://{self.host}:{self.port}")
        
    async def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            
        # Cleanup handler
        await self.handler._cleanup_mcp_connections()
        _logger.info("Voice Computer Server stopped")
        
    async def handle_client(self, websocket):
        """Handle a new client connection."""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        _logger.info(f"Client connected: {client_id}")
        
        self.connected_clients.add(websocket)
        
        try:
            # Send welcome message
            await self.send_message(websocket, {
                "type": "welcome",
                "message": "Connected to Voice Computer Server",
                "server_config": {
                    "streaming_enabled": self.handler._is_streaming_enabled(),
                    "available_tools": self.handler.tool_handler.get_available_tool_names() if self.handler.tool_handler else []
                }
            })
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.process_message(websocket, data)
                except json.JSONDecodeError:
                    await self.send_error(websocket, "Invalid JSON format")
                except Exception as e:
                    _logger.error(f"Error processing message from {client_id}: {e}")
                    await self.send_error(websocket, f"Server error: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            _logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            _logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self.connected_clients.discard(websocket)
            
    async def process_message(self, websocket, data: Dict[str, Any]):
        """Process a message from a client."""
        message_type = data.get("type")
        
        if message_type == "text_query":
            await self.handle_text_query(websocket, data)
        elif message_type == "audio_chunk":
            await self.handle_audio_chunk(websocket, data)
        elif message_type == "start_listening":
            await self.handle_start_listening(websocket)
        elif message_type == "stop_listening":
            await self.handle_stop_listening(websocket)
        elif message_type == "reset_conversation":
            await self.handle_reset_conversation(websocket)
        elif message_type == "get_status":
            await self.handle_get_status(websocket)
        else:
            await self.send_error(websocket, f"Unknown message type: {message_type}")
            
    async def handle_text_query(self, websocket, data: Dict[str, Any]):
        """Handle a text query from the client."""
        query = data.get("query", "").strip()
        if not query:
            await self.send_error(websocket, "Empty query")
            return
            
        _logger.info(f"Processing text query: {query}")
        
        try:
            # Send query started notification
            await self.send_message(websocket, {
                "type": "query_started",
                "query": query
            })
            
            # Check for exit command
            if await self.handler._is_exit_command(query):
                await self.send_message(websocket, {
                    "type": "exit_command",
                    "message": "Goodbye!"
                })
                self.handler._reset_conversation_state()
                return
            
            # Process query with streaming
            response = await self.handler.process_query(
                query, 
                use_streaming=True, 
                use_colored_output=False, 
                use_tts=False
            )
            
            # Send final response
            await self.send_message(websocket, {
                "type": "query_response",
                "query": query,
                "response": response,
                "conversation_length": len(self.handler.conversation_history)
            })
            
        except Exception as e:
            _logger.error(f"Error processing text query: {e}")
            await self.send_error(websocket, f"Error processing query: {str(e)}")
            
    async def handle_audio_chunk(self, websocket, data: Dict[str, Any]):
        """Handle real-time audio chunk from client."""
        if not hasattr(websocket, 'audio_buffer'):
            websocket.audio_buffer = []
            websocket.is_listening = True  # Always listening by default
            websocket.last_audio_time = time.time()
            
        # Store audio chunk
        audio_data = data.get("audio_data", "")
        if audio_data:
            try:
                decoded_audio = base64.b64decode(audio_data)
                websocket.audio_buffer.append(decoded_audio)
                websocket.last_audio_time = time.time()
                
                # Process audio buffer periodically
                if len(websocket.audio_buffer) >= 50:  # About 3 seconds of audio at 16kHz
                    asyncio.create_task(self._process_audio_buffer(websocket))
                    
            except Exception as e:
                _logger.error(f"Error decoding audio data: {e}")
                
    async def handle_start_listening(self, websocket):
        """Start listening for voice commands from audio stream."""
        if not hasattr(websocket, 'audio_buffer'):
            websocket.audio_buffer = []
            
        websocket.is_listening = True
        websocket.audio_buffer.clear()  # Clear previous audio
        
        await self.send_message(websocket, {
            "type": "start_listening",
            "message": "Server started listening"
        })
        
        # Start processing audio after a short delay to collect some audio
        asyncio.create_task(self._process_audio_after_delay(websocket))
        
    async def handle_stop_listening(self, websocket):
        """Stop listening and process collected audio."""
        if hasattr(websocket, 'is_listening'):
            websocket.is_listening = False
            
        await self.send_message(websocket, {
            "type": "stop_listening", 
            "message": "Server stopped listening"
        })
        
    async def _process_audio_buffer(self, websocket):
        """Process accumulated audio buffer for activation words and commands."""
        if not hasattr(websocket, 'audio_buffer') or len(websocket.audio_buffer) == 0:
            return
            
        # Get current audio buffer
        audio_data = b''.join(websocket.audio_buffer)
        websocket.audio_buffer.clear()
        
        # Transcribe the audio
        try:
            transcribed_text = await self._transcribe_audio(audio_data)
            
            if transcribed_text and transcribed_text.strip():
                text_lower = transcribed_text.lower().strip()
                _logger.info(f"Transcribed: {text_lower}")
                
                # Check for activation words
                activation_words = ['computer', 'hey computer']
                for activation_word in activation_words:
                    if activation_word in text_lower:
                        _logger.info(f"Activation word '{activation_word}' detected")
                        
                        # Send notification to client
                        await self.send_message(websocket, {
                            "type": "activation_detected",
                            "activation_word": activation_word,
                            "full_text": transcribed_text
                        })
                        
                        # Extract command after activation word
                        parts = text_lower.split(activation_word, 1)
                        if len(parts) > 1 and parts[1].strip():
                            command = parts[1].strip()
                            _logger.info(f"Processing command: {command}")
                            
                            # Process as text query
                            await self.handle_text_query(websocket, {
                                "query": command
                            })
                        else:
                            # Just activation word, ask what to do
                            await self.send_message(websocket, {
                                "type": "query_response",
                                "query": "activation",
                                "response": "How can I help you?"
                            })
                        return
                
                # If no activation word found, just log the transcription
                await self.send_message(websocket, {
                    "type": "audio_processed",
                    "transcribed_text": transcribed_text
                })
                
        except Exception as e:
            _logger.error(f"Error processing audio buffer: {e}")
    
    async def _process_audio_after_delay(self, websocket, delay: float = 3.0):
        """Legacy method - now using continuous processing."""
        pass
                
    async def _transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio data using the voice computer's WhisperListener."""
        try:
            # Convert bytes to numpy array (same format as WhisperListener expects)
            import numpy as np
            
            # Convert bytes to int16 array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert to float32 and normalize to [-1, 1] range
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Use the whisper listener to transcribe
            text = await self.whisper_listener._transcribe_audio(audio_float)
            
            if text:
                _logger.debug(f"Transcribed: '{text}'")
                return text.strip()
            else:
                _logger.debug("No transcription result")
                return ""
                
        except Exception as e:
            _logger.error(f"Error transcribing audio with WhisperListener: {e}")
            return ""
        
    async def handle_reset_conversation(self, websocket):
        """Reset the conversation state."""
        self.handler._reset_conversation_state()
        await self.send_message(websocket, {
            "type": "conversation_reset",
            "message": "Conversation history cleared"
        })
        
    async def handle_get_status(self, websocket):
        """Get server status information."""
        status = {
            "type": "status",
            "conversation_length": len(self.handler.conversation_history),
            "tool_results_count": len(self.handler.tool_results_queue),
            "failed_tools_count": len(self.handler.failed_tools_queue),
            "available_tools": self.handler.tool_handler.get_available_tool_names() if self.handler.tool_handler else [],
            "streaming_enabled": self.handler._is_streaming_enabled(),
            "connected_clients": len(self.connected_clients)
        }
        await self.send_message(websocket, status)
        
    async def send_message(self, websocket, data: Dict[str, Any]):
        """Send a message to a client."""
        try:
            message = json.dumps(data)
            await websocket.send(message)
        except Exception as e:
            _logger.error(f"Error sending message: {e}")
            
    async def send_error(self, websocket, error_message: str):
        """Send an error message to a client."""
        await self.send_message(websocket, {
            "type": "error",
            "error": error_message
        })


async def main():
    """Main entry point for the voice server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Computer Server")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create and start server
    server = VoiceComputerServer(args.config, args.host, args.port)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler():
        _logger.info("Received shutdown signal")
        asyncio.create_task(server.stop())
        
    try:
        # Register signal handlers
        loop = asyncio.get_event_loop()
        for sig in [signal.SIGTERM, signal.SIGINT]:
            loop.add_signal_handler(sig, signal_handler)
            
        await server.start()
        
        # Keep server running
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        _logger.info("Shutdown requested by user")
    except Exception as e:
        _logger.error(f"Server error: {e}")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())