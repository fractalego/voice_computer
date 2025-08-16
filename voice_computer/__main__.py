"""
Main execution script for the Voice Computer system.

Usage:
    python run_voice_computer.py                    # Run in voice mode
    python run_voice_computer.py --test             # Run in text mode
    python run_voice_computer.py --config=config.json  # Use custom config file
"""

import asyncio
import logging
import argparse
import sys
import signal
import time
import json
import base64
import websockets
from pathlib import Path
from typing import Optional, Dict, Any

from voice_computer.conversation_handler import ConversationHandler
from voice_computer.server_voice_listener import ServerVoiceListener
from voice_computer.server_tts_speaker import ServerTTSSpeaker
from voice_computer.config import load_config, create_example_config_file


def list_audio_devices():
    """List available audio input devices."""
    try:
        import pyaudio
        
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        
        print(f"Available audio input devices ({device_count} total):")
        print("=" * 60)
        
        input_devices = []
        for i in range(device_count):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:  # Only input devices
                input_devices.append((i, info))
                print(f"Device {i}: {info['name']}")
                print(f"  Channels: {info['maxInputChannels']}")
                print(f"  Sample Rate: {info['defaultSampleRate']}")
                print(f"  Host API: {p.get_host_api_info_by_index(info['hostApi'])['name']}")
                print()
        
        if input_devices:
            try:
                default_input = p.get_default_input_device_info()
                print(f"Default input device: {default_input['name']} (index: {default_input['index']})")
                print()
                print("To use a specific device, add this to your config file:")
                print('"listener_model": {')
                print('  "microphone_device_index": <device_index>')
                print('}')
            except Exception as e:
                print(f"Could not get default input device: {e}")
        else:
            print("No input devices found!")
            
        p.terminate()
        return 0
        
    except ImportError:
        print("Error: pyaudio is not installed. Install it with: pip install pyaudio")
        return 1
    except Exception as e:
        print(f"Error listing audio devices: {e}")
        return 1


def setup_logging(level=logging.INFO, log_file="voice_computer.log"):
    """Setup logging configuration."""
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not setup file logging: {e}")
    
    # Suppress verbose websockets debug logging
    websockets_logger = logging.getLogger('websockets')
    websockets_logger.setLevel(logging.WARNING)
    
    # Also suppress other potentially verbose loggers
    logging.getLogger('websockets.server').setLevel(logging.WARNING)
    logging.getLogger('websockets.protocol').setLevel(logging.WARNING)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Voice Computer - Voice-driven assistant with MCP integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run in voice mode with default config
  %(prog)s --test                       # Run in text-only mode for testing
  %(prog)s --config=my_config.json      # Use custom configuration file
  %(prog)s --create-config=config.json  # Create example configuration file
  %(prog)s --list-devices               # List available audio input devices
  %(prog)s --verbose                    # Enable debug logging
        """
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in text-only mode (no voice input/output)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON configuration file"
    )
    
    parser.add_argument(
        "--create-config",
        type=str,
        metavar="PATH",
        help="Create an example configuration file at the specified path and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) logging"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default="voice_computer.log",
        help="Log file path (default: voice_computer.log)"
    )
    
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit"
    )
    
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run as WebSocket server instead of local voice interface"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host address (only used with --server)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Server port (only used with --server)"
    )
    
    return parser.parse_args()


async def run_websocket_server(host: str, port: int, config_path: Optional[str] = None, verbose: bool = False):
    """Run the WebSocket server mode."""
    
    # Load configuration
    config = load_config(config_path)
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Voice Computer Server on {host}:{port}")
    
    # Pre-load models and MCP tools before accepting clients
    logger.info("Pre-loading models and MCP tools...")
    
    # Create a temporary handler to initialize shared resources
    temp_handler = ConversationHandler(config)
    
    # Setup MCP tools
    logger.info("Setting up MCP tools...")
    await temp_handler._setup_mcp_tools()
    
    logger.info("Initializing models...")
    temp_handler._initialize_models()
    
    # Get the shared MCP tools
    shared_mcp_tools = temp_handler.mcp_tools
    shared_tool_handler = temp_handler.tool_handler
    
    logger.info("Server initialization complete - ready to accept clients")
    
    # Track connected clients
    connected_clients = set()
    server = None
    
    async def handle_client(websocket):
        """Handle a new client connection."""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_id}")
        
        connected_clients.add(websocket)
        
        # Create client-specific voice listener (shares models with shared_voice_listener)
        voice_listener = ServerVoiceListener(config)
        
        # Create WebSocket callback for TTS
        async def websocket_send_callback(message_data):
            try:
                await websocket.send(json.dumps(message_data))
            except Exception as e:
                logger.error(f"Error sending message to client: {e}")
        
        tts_speaker = ServerTTSSpeaker(websocket_send_callback, config=config)
        
        # Create handler with server components using shared resources
        handler = ConversationHandler(config, voice_listener=voice_listener, tts_speaker=tts_speaker)
        
        # Use pre-loaded shared MCP tools
        handler.mcp_tools = shared_mcp_tools
        handler.tool_handler = shared_tool_handler
        
        logger.debug(f"Client {client_id} using shared pre-loaded MCP tools")
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "welcome",
                "message": "Connected to Voice Computer Server",
                "server_config": {
                    "streaming_enabled": True,
                    "available_tools": handler.get_available_tool_names()
                }
            }))
            
            # Test TTS by sending a connection sound
            connection_message = "Voice computer connected and ready"
            logger.info(f"Sending connection test TTS: {connection_message}")
            
            # Send the text response first
            await websocket.send(json.dumps({
                "type": "text_response",
                "text": connection_message
            }))
            
            # Generate and send TTS audio
            try:
                tts_speaker.speak(connection_message)
                logger.info("Connection TTS initiated successfully")
            except Exception as e:
                logger.error(f"Error generating connection TTS: {e}")
            
            # Track audio chunks for processing
            chunk_count = 0
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    if message_type == "text_query":
                        await handle_text_query(websocket, handler, data)
                    elif message_type == "audio_chunk":
                        await handle_audio_chunk(websocket, handler, data, chunk_count)
                        chunk_count += 1
                        # Process every ~3 seconds of audio
                        if chunk_count >= 50:
                            chunk_count = 0
                            await process_accumulated_audio(websocket, handler)
                    elif message_type == "reset_conversation":
                        handler._reset_conversation_state()
                        await websocket.send(json.dumps({
                            "type": "conversation_reset",
                            "message": "Conversation history cleared"
                        }))
                    elif message_type == "get_status":
                        await send_status(websocket, handler, len(connected_clients))
                    else:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "error": f"Unknown message type: {message_type}"
                        }))
                        
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": "Invalid JSON format"
                    }))
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "error": f"Server error: {str(e)}"
                    }))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            connected_clients.discard(websocket)
            # No need to cleanup shared MCP connections here since they're shared
            # Only cleanup happens at server shutdown
    
    async def handle_text_query(websocket, handler, data):
        """Handle a text query from client."""
        query = data.get("query", "").strip()
        if not query:
            await websocket.send(json.dumps({
                "type": "error", 
                "error": "Empty query"
            }))
            return
            
        logger.info(f"Processing text query: {query}")
        
        try:
            # Send query started notification
            await websocket.send(json.dumps({
                "type": "query_started",
                "query": query
            }))
            
            # Check for exit command
            if await handler.is_exit_command(query):
                await websocket.send(json.dumps({
                    "type": "exit_command",
                    "message": "Goodbye!"
                }))
                handler._reset_conversation_state()
                return
            
            # Process query
            response = await handler.process_query(query, use_streaming=False)
            
            # Send text response first (for immediate display)
            await websocket.send(json.dumps({
                "type": "text_response",
                "text": response
            }))
            
            # Send final response
            await websocket.send(json.dumps({
                "type": "query_response",
                "query": query,
                "response": response,
                "conversation_length": handler.get_conversation_length()
            }))
            
            # TTS is handled automatically by the ServerTTSSpeaker
            
        except Exception as e:
            logger.error(f"Error processing text query: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "error": f"Error processing query: {str(e)}"
            }))
    
    async def handle_audio_chunk(websocket, handler, data, chunk_count):
        """Handle real-time audio chunk from client."""
        audio_data = data.get("audio_data", "")
        if audio_data:
            try:
                decoded_audio = base64.b64decode(audio_data)
                await handler.add_audio_chunk(decoded_audio)
            except Exception as e:
                logger.error(f"Error decoding audio data: {e}")
    
    async def process_accumulated_audio(websocket, handler):
        """Process accumulated audio for activation words and commands."""
        try:
            result = await handler.process_accumulated_audio()
            
            if not result:
                return
                
            result_type = result.get("type")
            
            if result_type == "activation_detected":
                # Send activation notification
                await websocket.send(json.dumps({
                    "type": "activation_detected",
                    "activation_word": result.get("activation_word"),
                    "full_text": result.get("full_text")
                }))
                
                # If there's a response, send it
                if "response" in result:
                    response_text = result["response"]
                    await websocket.send(json.dumps({
                        "type": "text_response",
                        "text": response_text
                    }))
                    await websocket.send(json.dumps({
                        "type": "query_response",
                        "query": result.get("command", ""),
                        "response": response_text
                    }))
                elif result.get("command") and result.get("command").strip():
                    # Command detected but not processed yet
                    await handle_text_query(websocket, handler, {
                        "query": result["command"]
                    })
                else:
                    # Just activation word or empty command
                    help_text = "How can I help you?"
                    await websocket.send(json.dumps({
                        "type": "text_response",
                        "text": help_text
                    }))
                    await websocket.send(json.dumps({
                        "type": "query_response",
                        "query": "activation",
                        "response": help_text
                    }))
                    
                    # Also trigger TTS for the help text by calling the handler's speak method
                    if handler.tts_speaker:
                        try:
                            handler.tts_speaker.speak(help_text)
                        except Exception as e:
                            logger.error(f"Error generating TTS for help text: {e}")
                    
            elif result_type == "transcription_only":
                # No activation word, just transcription
                await websocket.send(json.dumps({
                    "type": "audio_processed",
                    "transcribed_text": result.get("text", "")
                }))
                
            elif result_type == "error":
                logger.error(f"Audio processing error: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"Error processing accumulated audio: {e}")
    
    async def send_status(websocket, handler, client_count):
        """Send server status information."""
        status = {
            "type": "status",
            "conversation_length": handler.get_conversation_length(),
            "tool_results_count": len(handler.tool_results_queue),
            "failed_tools_count": len(handler.failed_tools_queue),
            "available_tools": handler.get_available_tool_names(),
            "streaming_enabled": True,
            "connected_clients": client_count,
            "audio_buffer_duration": handler.get_buffer_duration()
        }
        await websocket.send(json.dumps(status))
    
    try:
        # Start WebSocket server
        server = await websockets.serve(
            handle_client,
            host,
            port,
            ping_interval=30,
            ping_timeout=10
        )
        
        logger.info(f"Voice Computer Server started on ws://{host}:{port}")
        
        # Keep server running
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        if server:
            server.close()
            await server.wait_closed()
        
        # Cleanup shared resources
        logger.info("Cleaning up shared resources...")
        try:
            await temp_handler._cleanup_mcp_connections()
        except Exception as e:
            logger.error(f"Error cleaning up shared resources: {e}")
        
        logger.info("Voice Computer Server stopped")


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Handle list devices command
    if args.list_devices:
        return list_audio_devices()
    
    # Handle config file creation
    if args.create_config:
        try:
            create_example_config_file(args.create_config)
            print(f"Created example configuration file: {args.create_config}")
            print("Edit this file to customize your MCP servers and settings.")
            return 0
        except Exception as e:
            print(f"Error creating configuration file: {e}")
            return 1
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level, args.log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Voice Computer")
    
    try:
        # Validate config file exists if specified
        if args.config:
            config_path = Path(args.config)
            if not config_path.exists():
                print(f"Error: Configuration file '{args.config}' does not exist")
                return 1
            if not config_path.is_file():
                print(f"Error: Configuration path '{args.config}' is not a file")
                return 1
        
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config or 'default'}")
        
        # Run appropriate mode
        if args.server:
            logger.info("Running in WebSocket server mode")
            await run_websocket_server(args.host, args.port, args.config, args.verbose)
        else:
            # Create handler only for local modes
            handler = ConversationHandler(config)
            
            # Add example MCP servers if none configured
            mcp_servers = config.get_value("mcp_servers") or []
            if not mcp_servers:
                logger.info("No MCP servers configured. Add some to enable tool functionality.")
                logger.info("Example: handler.add_mcp_server('filesystem', 'mcp-server-filesystem', ['--root', '/tmp'])")
            
            if args.test:
                logger.info("Running in text-only mode")
                await handler.run_text_loop()
            else:
                logger.info("Running in voice mode")
                await handler.run_voice_loop()
            
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Error running Voice Computer: {e}", exc_info=True)
        return 1


def cli_main():
    """CLI entry point that handles async execution."""
    
    def exception_handler(loop, context):
        """Handle async exceptions, suppress MCP cleanup errors."""
        exception = context.get('exception')
        if exception:
            # Suppress common MCP cleanup errors
            if ('cancel scope' in str(exception) or 
                'asynchronous generator' in str(exception) or
                'aclose(): asynchronous generator is already running' in str(exception)):
                # Log debug message but don't print to console
                logger = logging.getLogger('asyncio')
                logger.debug(f"Suppressed MCP cleanup error: {exception}")
                return
        
        # For other exceptions, use default handling
        loop.default_exception_handler(context)
    
    try:
        # Set custom exception handler before running
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(exception_handler)
        
        try:
            exit_code = loop.run_until_complete(main())
            sys.exit(exit_code)
        finally:
            loop.close()
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()