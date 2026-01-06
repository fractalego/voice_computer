#!/usr/bin/env python3
"""
Voice Computer Client - connects to remote voice computer server.

This client streams real-time audio to a remote voice computer server
and receives text responses that are spoken locally. All processing
(speech-to-text, MCP tools, LLM) happens on the server.
"""

import asyncio
import websockets
import json
import logging
import pyaudio
import threading
import queue
import base64
import numpy as np
from typing import Optional, Dict, Any
import time

_logger = logging.getLogger(__name__)


class AudioStreamer:
    """Handles real-time audio streaming to server."""
    
    def __init__(self, chunk_size: int = 1024, format=pyaudio.paInt16, channels: int = 1, rate: int = 16000):
        self.chunk_size = chunk_size
        self.format = format
        self.channels = channels
        self.rate = rate
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.streaming = False
        self.audio_queue = queue.Queue()
        
    def start_streaming(self):
        """Start streaming audio."""
        if self.streaming:
            return
            
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        self.streaming = True
        self.streaming_thread = threading.Thread(target=self._stream_loop)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        
    def stop_streaming(self):
        """Stop streaming audio."""
        if not self.streaming:
            return
            
        self.streaming = False
        if self.streaming_thread:
            self.streaming_thread.join()
            
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
    def _stream_loop(self):
        """Internal streaming loop."""
        while self.streaming:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                self.audio_queue.put(data)
            except Exception as e:
                _logger.error(f"Error streaming audio: {e}")
                break
                
    def get_audio_chunk(self, timeout: float = 0.1) -> Optional[bytes]:
        """Get audio chunk for streaming."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def cleanup(self):
        """Cleanup audio resources."""
        self.stop_streaming()
        self.audio.terminate()




class VoiceComputerClient:
    """Client that connects to a remote voice computer server."""

    def __init__(self, server_uri: str = "ws://localhost:8765", auto_reconnect: bool = True, reconnect_interval: float = 60.0):
        self.server_uri = server_uri
        self.websocket = None
        self.connected = False
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval = reconnect_interval

        # Audio components
        self.audio_streamer = AudioStreamer()

        # State
        self.streaming_audio = False
        self.processing = False
        self.conversation_active = False
        self.shutdown_requested = False

        # Audio streaming task
        self.streaming_task = None

        # Reconnection task
        self.reconnect_task = None

        # Message handler task (track to clean up on reconnect)
        self.message_handler_task = None
        
    async def connect(self):
        """Connect to the voice computer server."""
        try:
            # Clean up any existing connection first
            await self._cleanup_old_connection()

            _logger.info(f"Connecting to {self.server_uri}")
            # Use ping settings that match server (ping_interval=30, ping_timeout=60)
            # This prevents premature disconnects during heavy processing
            self.websocket = await websockets.connect(
                self.server_uri,
                ping_interval=30,
                ping_timeout=60,
                close_timeout=10
            )
            self.connected = True
            _logger.info("Connected to voice computer server")

            # Start message handler
            self.message_handler_task = asyncio.create_task(self._message_handler())

        except Exception as e:
            _logger.error(f"Failed to connect to server: {e}")
            raise

    async def _cleanup_old_connection(self):
        """Clean up any existing connection state before reconnecting."""
        # Cancel old message handler task if it exists
        if self.message_handler_task and not self.message_handler_task.done():
            self.message_handler_task.cancel()
            try:
                await self.message_handler_task
            except asyncio.CancelledError:
                pass
            self.message_handler_task = None

        # Close old websocket if it exists
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                _logger.debug(f"Error closing old websocket: {e}")
            self.websocket = None

        self.connected = False
            
    async def _attempt_reconnect(self):
        """Attempt to reconnect to the server with retry logic."""
        if not self.auto_reconnect or self.shutdown_requested:
            return

        # Stop audio streaming before reconnection attempts
        was_streaming = self.streaming_audio
        if self.streaming_audio:
            self.streaming_audio = False
            self.audio_streamer.stop_streaming()
            if self.streaming_task:
                self.streaming_task.cancel()
                try:
                    await self.streaming_task
                except asyncio.CancelledError:
                    pass
                self.streaming_task = None

        reconnect_attempt = 1
        max_interval = 60.0  # Cap the interval at 60 seconds
        current_interval = min(5.0, self.reconnect_interval)  # Start with 5s or less

        _logger.info(f"🔄 Starting auto-reconnection process")
        print(f"🔄 Connection lost - attempting to reconnect...")

        while not self.connected and not self.shutdown_requested:
            try:
                _logger.info(f"🔄 Reconnection attempt #{reconnect_attempt} to {self.server_uri}")
                print(f"🔄 Reconnection attempt #{reconnect_attempt}...")

                await self.connect()

                _logger.info("✅ Successfully reconnected to server")
                print(f"✅ Reconnected successfully after {reconnect_attempt} attempt(s)")

                # If we were streaming audio before disconnection, resume it
                if was_streaming or self.conversation_active:
                    await self.start_audio_streaming()
                    _logger.info("🎤 Resumed audio streaming after reconnection")
                    print("🎤 Audio streaming resumed")

                break

            except Exception as e:
                _logger.warning(f"❌ Reconnection attempt #{reconnect_attempt} failed: {e}")
                print(f"❌ Reconnection attempt #{reconnect_attempt} failed: {e}")

                if not self.shutdown_requested:
                    _logger.info(f"⏳ Waiting {current_interval:.1f} seconds before next attempt")
                    print(f"⏳ Waiting {current_interval:.1f} seconds before next attempt...")
                    await asyncio.sleep(current_interval)
                    reconnect_attempt += 1
                    # Exponential backoff with cap
                    current_interval = min(current_interval * 1.5, max_interval)
            
    async def disconnect(self):
        """Disconnect from the server."""
        self.shutdown_requested = True
        await self.stop_audio_streaming()

        # Cancel reconnection task if running
        if self.reconnect_task and not self.reconnect_task.done():
            self.reconnect_task.cancel()
            try:
                await self.reconnect_task
            except asyncio.CancelledError:
                pass

        # Cancel message handler task if running
        if self.message_handler_task and not self.message_handler_task.done():
            self.message_handler_task.cancel()
            try:
                await self.message_handler_task
            except asyncio.CancelledError:
                pass

        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                _logger.debug(f"Error closing websocket: {e}")
        self.connected = False
        self.audio_streamer.cleanup()
        _logger.info("Disconnected from server")
        
    async def _message_handler(self):
        """Handle incoming messages from the server."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_server_message(data)
                except json.JSONDecodeError as e:
                    _logger.error(f"JSON decode error: {e}")
                    _logger.error(f"Raw message received: {repr(message)}")
                    # Continue to next message instead of crashing
                    continue
        except websockets.exceptions.ConnectionClosed:
            _logger.info("🔌 Server connection closed")
            print("🔌 Connection to server lost")
            self.connected = False
            # Trigger reconnection if not shutting down
            if not self.shutdown_requested and self.auto_reconnect:
                _logger.info("🔄 Connection lost - starting reconnection process")
                self.reconnect_task = asyncio.create_task(self._attempt_reconnect())
        except Exception as e:
            _logger.error(f"💥 Error in message handler: {e}")
            print(f"💥 Connection error: {e}")
            self.connected = False
            # Trigger reconnection if not shutting down
            if not self.shutdown_requested and self.auto_reconnect:
                _logger.info("🔄 Connection error - starting reconnection process")
                self.reconnect_task = asyncio.create_task(self._attempt_reconnect())
            
    async def _process_server_message(self, data: Dict[str, Any]):
        """Process a message from the server."""
        message_type = data.get("type")
        
        if message_type == "welcome":
            print(f"🔊 {data.get('message')}")
            server_config = data.get("server_config", {})
            available_tools = server_config.get("available_tools", [])
            if available_tools:
                print(f"Available tools: {', '.join(available_tools)}")
                
        elif message_type == "query_started":
            query = data.get("query")
            print(f"🤖 Processing: {query}")
            self.processing = True
            
        elif message_type == "query_response":
            response = data.get("response")
            # Don't print here since we already printed in text_response
            # Don't use local TTS since server will send TTS audio
            self.processing = False
            
        elif message_type == "audio_processed":
            query = data.get("transcribed_text", "")
            if query:
                print(f"🗣️  Heard: {query}")
                
        elif message_type == "activation_detected":
            activation_word = data.get("activation_word", "")
            full_text = data.get("full_text", "")
            print(f"🎯 Activation detected: '{activation_word}' in '{full_text}'")
            
        elif message_type == "tts_audio":
            # Handle TTS audio from server
            audio_data = data.get("audio_data", "")
            text = data.get("text", "")
            if audio_data:
                print(f"🔊 Playing TTS: {text}")
                await self._play_tts_audio(audio_data, data.get("sample_rate", 16000))
                
        elif message_type == "tts_cancel":
            # Stop current TTS playback
            print("🛑 TTS playback cancelled")
            # No local TTS to stop - server handles all TTS
            
        elif message_type == "text_response":
            # Display text immediately (before TTS)
            text = data.get("text", "")
            print(f"💬 Response: {text}")
            
        elif message_type == "tts_error":
            # Handle TTS generation errors
            error = data.get("error", "Unknown TTS error")
            print(f"🔇 TTS Error: {error}")
            
        elif message_type == "tts_status":
            # Handle TTS status messages
            status = data.get("status", "")
            print(f"🔊 TTS Status: {status}")
            
        elif message_type == "sound_file":
            # Handle sound file playback from server
            filename = data.get("filename", "")
            audio_data = data.get("data", "")
            sample_rate = data.get("sample_rate", 16000)
            channels = data.get("channels", 1)
            format_bytes = data.get("format", 2)
            if audio_data:
                print(f"🔊 Playing sound: {filename}")
                await self._play_sound_file(audio_data, sample_rate, channels, format_bytes)
                
        elif message_type == "cancel_sound":
            # Handle sound cancellation from server
            print("🛑 Sound playback cancelled")
            # TODO: Implement sound cancellation if needed
                
        elif message_type == "start_listening":
            print("🎤 Server is listening...")
            
        elif message_type == "stop_listening":
            print("🔇 Server stopped listening")
            
        elif message_type == "exit_command":
            message = data.get("message", "Goodbye!")
            print(f"🤖 {message}")
            # Don't use local TTS - server will send tts_audio message
            self.conversation_active = False
            
        elif message_type == "error":
            error = data.get("error")
            print(f"❌ Error: {error}")
            self.processing = False
            
        elif message_type == "status":
            self._print_status(data)
            
    def _print_status(self, status_data: Dict[str, Any]):
        """Print server status information."""
        print("\n📊 Server Status:")
        print(f"  Conversation length: {status_data.get('conversation_length', 0)}")
        print(f"  Tool results: {status_data.get('tool_results_count', 0)}")
        print(f"  Available tools: {', '.join(status_data.get('available_tools', []))}")
        print(f"  Connected clients: {status_data.get('connected_clients', 0)}")
        
    async def send_text_query(self, query: str):
        """Send a text query to the server."""
        if not self.connected:
            print("❌ Not connected to server")
            return
            
        message = {
            "type": "text_query",
            "query": query
        }
        
        await self.websocket.send(json.dumps(message))
        
    async def send_reset_conversation(self):
        """Reset the conversation on the server."""
        if not self.connected:
            return
            
        message = {"type": "reset_conversation"}
        await self.websocket.send(json.dumps(message))
        
    async def get_server_status(self):
        """Get server status."""
        if not self.connected:
            return
            
        message = {"type": "get_status"}
        await self.websocket.send(json.dumps(message))
        
    async def start_audio_streaming(self):
        """Start streaming audio to the server."""
        if self.streaming_audio:
            return
            
        print("🎤 Starting audio stream...")
        self.streaming_audio = True
        self.audio_streamer.start_streaming()
        
        # Start the streaming task
        self.streaming_task = asyncio.create_task(self._audio_streaming_loop())
        
    async def stop_audio_streaming(self):
        """Stop streaming audio to the server."""
        if not self.streaming_audio:
            return
            
        print("🔇 Stopping audio stream...")
        self.streaming_audio = False
        self.audio_streamer.stop_streaming()
        
        if self.streaming_task:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass
            self.streaming_task = None
            
    async def _audio_streaming_loop(self):
        """Continuously stream audio chunks to the server."""
        try:
            while self.streaming_audio and self.connected:
                audio_chunk = self.audio_streamer.get_audio_chunk()
                if audio_chunk and self.connected:
                    try:
                        # Encode audio chunk as base64 and send to server
                        encoded_audio = base64.b64encode(audio_chunk).decode('utf-8')
                        message = {
                            "type": "audio_chunk",
                            "audio_data": encoded_audio,
                            "sample_rate": self.audio_streamer.rate,
                            "channels": self.audio_streamer.channels,
                            "format": "pcm16"
                        }
                        await self.websocket.send(json.dumps(message))
                    except (websockets.exceptions.ConnectionClosed, ConnectionResetError) as e:
                        _logger.warning(f"Connection lost during audio streaming: {e}")
                        self.connected = False
                        break
                    
                await asyncio.sleep(0.01)  # Small delay to prevent busy loop
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            _logger.error(f"Error in audio streaming loop: {e}")
            
    async def send_start_listening(self):
        """Tell server to start listening for voice commands."""
        if not self.connected:
            return
            
        message = {"type": "start_listening"}
        await self.websocket.send(json.dumps(message))
        
    async def send_stop_listening(self):
        """Tell server to stop listening for voice commands."""
        if not self.connected:
            return
            
        message = {"type": "stop_listening"}
        await self.websocket.send(json.dumps(message))
        
    async def _play_tts_audio(self, encoded_audio: str, sample_rate: int = 16000):
        """Play TTS audio received from server."""
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(encoded_audio)
            
            # Convert bytes to numpy array (int16 format from server)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Convert to float32 for playback
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Play using PyAudio
            stream = self.audio_streamer.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                output=True,
                output_device_index=None,
                frames_per_buffer=1024
            )
            
            # Play the audio
            stream.write(audio_float.tobytes())
            
            # Clean up
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            _logger.error(f"Error playing TTS audio: {e}")
            # Fallback to local TTS with the text
            # Note: We don't have the text here, so this is just error handling
            
    async def _play_sound_file(self, encoded_audio: str, sample_rate: int, channels: int, format_bytes: int):
        """Play sound file received from server."""
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(encoded_audio)
            
            # Determine PyAudio format based on sample width
            if format_bytes == 1:
                pyaudio_format = pyaudio.paUInt8
                dtype = np.uint8
            elif format_bytes == 2:
                pyaudio_format = pyaudio.paInt16
                dtype = np.int16
            elif format_bytes == 4:
                pyaudio_format = pyaudio.paInt32
                dtype = np.int32
            else:
                _logger.warning(f"Unsupported audio format: {format_bytes} bytes per sample")
                return
                
            # Play using PyAudio
            stream = self.audio_streamer.audio.open(
                format=pyaudio_format,
                channels=channels,
                rate=sample_rate,
                output=True,
                output_device_index=None,
                frames_per_buffer=1024
            )
            
            # Play the audio directly
            stream.write(audio_bytes)
            
            # Clean up
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            _logger.error(f"Error playing sound file: {e}")
            
    async def run_interactive_mode(self):
        """Run interactive voice mode."""
        print("\n🎙️  Voice mode activated!")
        print("🎤 Audio streaming is active - say the activation word followed by your command")
        print("📝 Type 'q' to quit, 's' for status, 'r' to reset")
        
        self.conversation_active = True
        
        try:
            # Start audio streaming immediately
            await self.start_audio_streaming()
            print("🔊 Microphone is live - listening for the activation word...")
            
            # Simple control loop - just wait for user commands while audio streams
            while self.conversation_active and self.connected:
                try:
                    # Use asyncio to handle input without blocking
                    user_input = await asyncio.wait_for(
                        asyncio.to_thread(input, ""), 
                        timeout=1.0
                    )
                    
                    user_input = user_input.strip().lower()
                    
                    if user_input == 'q' or user_input == 'quit':
                        break
                    elif user_input == 's' or user_input == 'status':
                        await self.get_server_status()
                    elif user_input == 'r' or user_input == 'reset':
                        await self.send_reset_conversation()
                        print("🔄 Conversation reset")
                    elif user_input:
                        # Send text directly
                        await self.send_text_query(user_input)
                        
                except asyncio.TimeoutError:
                    # No input, continue listening
                    pass
                except EOFError:
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Voice mode interrupted by user")
        finally:
            await self.stop_audio_streaming()
            
    async def run_text_mode(self):
        """Run interactive text mode."""
        print("\n💬 Text mode activated!")
        print("Type your queries or 'exit' to quit.")
        
        try:
            while self.connected:
                try:
                    query = input("\nuser> ").strip()
                    if not query:
                        continue
                        
                    if query.lower() in ['exit', 'quit', 'goodbye']:
                        await self.send_text_query(query)
                        break
                        
                    if query.lower() == 'status':
                        await self.get_server_status()
                        continue
                        
                    if query.lower() == 'reset':
                        await self.send_reset_conversation()
                        print("🔄 Conversation reset")
                        continue
                        
                    await self.send_text_query(query)
                    
                    # Wait for response
                    while self.processing:
                        await asyncio.sleep(0.1)
                        
                except EOFError:
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Text mode interrupted by user")


async def main():
    """Main entry point for the voice client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Computer Client")
    parser.add_argument("--server", type=str, default="ws://localhost:8765", help="Server WebSocket URI")
    parser.add_argument("--text", action="store_true", help="Run in text mode instead of voice mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no-reconnect", action="store_true", help="Disable auto-reconnection")
    parser.add_argument("--reconnect-interval", type=float, default=60.0, help="Reconnection interval in seconds (default: 60)")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Create client
    client = VoiceComputerClient(
        server_uri=args.server,
        auto_reconnect=not args.no_reconnect,
        reconnect_interval=args.reconnect_interval
    )
    
    try:
        # Connect to server
        await client.connect()
        
        # Run appropriate mode
        if args.text:
            await client.run_text_mode()
        else:
            await client.run_interactive_mode()
            
    except KeyboardInterrupt:
        print("\n🛑 Client interrupted by user")
    except Exception as e:
        _logger.error(f"Client error: {e}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())