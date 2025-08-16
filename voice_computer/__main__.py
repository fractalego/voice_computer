"""
Main entry point for running voice_computer as a module.

Usage:
    python -m voice_computer                    # Run in voice mode
    python -m voice_computer --test             # Run in text mode
    python -m voice_computer --config=config.json  # Use custom config file
"""

import asyncio
import logging
import argparse
import sys
from pathlib import Path

from . import Handler
from .config import load_config, create_example_config_file


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Voice Computer - Voice-driven assistant with MCP integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m voice_computer                          # Run in voice mode with default config
  python -m voice_computer --test                   # Run in text-only mode for testing
  python -m voice_computer --config=my_config.json  # Use custom configuration file
  python -m voice_computer --create-config=config.json  # Create example configuration file
  python -m voice_computer --list-devices           # List available audio input devices
  python -m voice_computer --verbose                # Enable debug logging
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
    
    return parser.parse_args()


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
        
        # Create handler
        handler = Handler(config)
        
        # Add example MCP servers if none configured
        mcp_servers = config.get_value("mcp_servers") or []
        if not mcp_servers:
            logger.info("No MCP servers configured. Add some to enable tool functionality.")
            logger.info("Example: handler.add_mcp_server('filesystem', 'mcp-server-filesystem', ['--root', '/tmp'])")
        
        # Run appropriate mode
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
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()