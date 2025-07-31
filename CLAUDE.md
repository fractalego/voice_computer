# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Voice Computer is a voice-driven assistant that integrates Whisper speech recognition with MCP (Model Context Protocol) tools via Ollama. The system provides a continuous voice interaction loop where users can speak commands and receive spoken responses, with all processing handled through configurable MCP servers.

**IMPORTANT**: This project has been refactored to remove the yaaaf framework dependencies. All functionality is now contained within the `voice_computer` package.

## Development Commands

### Setup and Installation
- **Install dependencies**: `uv add <DEPENDENCY>`
- **dependency file**: `pyproject.toml`


### Running the Application
- **Start voice client**: `python run_voice_computer.py`
- **Test mode (text only)**: `python run_voice_computer.py --test`
- **Custom config**: `python run_voice_computer.py --config=my_config.json`
- **Create example config**: `python run_voice_computer.py --create-config=config.json`
- **Verbose logging**: `python run_voice_computer.py --verbose`
- **Check Ollama connection**: Ensure Ollama is running at `http://localhost:11434`

### Development and Testing
- **Test imports**: `python -c "from voice_computer import VoiceComputerClient; print('Imports working')"`
- **Test MCP tools**: `python -c "from voice_computer.mcp_connector import MCPConnector; print('MCP connector available')"`
- **Test voice interface**: `python -c "from voice_computer.voice_interface import VoiceInterface; print('Voice interface available')"`

## Architecture

### Core Components

**Voice Interface (`voice_computer/voice_interface.py`, `voice_computer/whisper_listener.py`):**
- Handles audio input/output and speech recognition
- Uses Whisper command-line tool for speech-to-text transcription
- Manages voice activation/deactivation with audio feedback
- Simplified implementation using system audio tools

**MCP Integration (`voice_computer/client.py`):**
- `VoiceComputerClient`: Main orchestrator that connects voice input to MCP processing
- Uses `ToolAgent` for all query processing and tool coordination
- Configurable MCP servers for extensible functionality
- Continuous voice interaction loop

**Core Framework Components:**
- `voice_computer/ollama_client.py`: Ollama LLM client for language model interaction
- `voice_computer/tool_agent.py`: MCP tool execution agent and coordinator
- `voice_computer/mcp_connector.py`: MCP protocol integration and tool management
- `voice_computer/data_types.py`: Core data structures (Messages, Utterances, Tools)
- `voice_computer/config.py`: Configuration management and loading

### Configuration
- Config loaded from JSON files or programmatic configuration classes
- Default Ollama model: `qwen2.5:32b` 
- Configurable MCP servers in `mcp_servers` array
- Voice settings: volume threshold, silence timeout, activation sounds
- Use `voice_computer.config.load_config()` for loading configurations

### Data Flow
1. User speaks → `WhisperListener` captures audio using system recording tools
2. Audio transcribed via Whisper command-line tool
3. `VoiceComputerClient` processes text query via `ToolAgent`
4. `ToolAgent` routes to appropriate MCP tools via `MCPConnector`
5. Results processed by Ollama LLM via `OllamaClient`
6. Response spoken back via `VoiceInterface` using system TTS

## Key Files to Understand

- `voice_computer/client.py`: Main voice computer client with voice loop and query processing
- `voice_computer/voice_interface.py`: Voice I/O management with audio feedback
- `voice_computer/whisper_listener.py`: Whisper-based speech recognition using CLI tools
- `voice_computer/tool_agent.py`: MCP tool execution and result processing
- `voice_computer/mcp_connector.py`: MCP protocol implementation and tool management
- `voice_computer/config.py`: Configuration management and loading
- `run_voice_computer.py`: Main execution script with argument parsing and logging

## Prerequisites and Dependencies

### External Services
- **Ollama**: Local LLM server running with compatible model
- **MCP Servers**: Installed and configured MCP protocol servers
- **Audio Hardware**: Working microphone and speakers/headphones

### Python Dependencies
- `requests`: HTTP client for Ollama API
- `pydantic`: Data validation and serialization

### Optional Dependencies
- `openai-whisper` or `faster-whisper`: Speech recognition (required for voice functionality)
- Various MCP servers: `mcp-server-filesystem`, `mcp-server-brave-search`, etc.
- System TTS packages: `espeak`, `festival`, `speech-dispatcher` (Linux)

## Development Notes

- The system uses async/await patterns throughout for real-time voice processing
- Simplified voice interface using system command-line tools for maximum compatibility
- All query processing goes through MCP tools when available - fallback to direct LLM queries
- Configuration is flexible to support different MCP server combinations
- Audio feedback provides clear activation/deactivation cues using system sounds
- Logging configured for both console and file output
- **No external framework dependencies** - all functionality is self-contained

## Configuration Examples

### Basic Configuration (JSON config)
```json
{
  "listener_model": {
    "listener_silence_timeout": 2,
    "listener_volume_threshold": 0.6,
    "listener_hotword_logp": -8
  },
  "activation_hotwords": ["computer"],
  "waking_up_sound": true,
  "deactivate_sound": true,
  "whisper_model": "fractalego/personal-whisper-distilled-model",
  "ollama_host": "http://localhost:11434",
  "ollama_model": "qwen2.5:32b",
  "mcp_servers": [
    {
      "name": "filesystem",
      "path": "mcp-server-filesystem",
      "args": ["--root", "/tmp"]
    }
  ]
}
```

### Multiple MCP Servers (JSON config)
```json
{
  "mcp_servers": [
    {
      "name": "filesystem",
      "path": "mcp-server-filesystem", 
      "args": ["--root", "/home/user/documents"]
    },
    {
      "name": "web-search",
      "path": "mcp-server-brave-search",
      "args": ["--api-key", "your-api-key-here"]
    }
  ]
}
```

### Programmatic Configuration
```python
from voice_computer.config import Config
from voice_computer import VoiceComputerClient

config = Config()
config.set_value("ollama_model", "llama3.2:latest")
config.set_value("mcp_servers", [
    {"name": "filesystem", "path": "mcp-server-filesystem", "args": ["--root", "/tmp"]}
])

client = VoiceComputerClient(config)
```

## Troubleshooting

### Common Issues
- **"No MCP tools available"**: Check MCP server installation and configuration in config file
- **Ollama connection errors**: Verify Ollama is running and model is available
- **Audio issues**: Install required system audio tools (`whisper`, `espeak`/`festival`)
- **Import errors**: Ensure all dependencies installed and Python path correct
- **Voice recognition not working**: Install Whisper (`pip install openai-whisper`)
- **TTS not working**: Install system TTS tools (see requirements.txt)

### Debug Commands
- **Test Ollama**: `curl http://localhost:11434/api/tags`
- **Test MCP server**: Run MCP server directly with `--help` flag
- **Test Whisper**: `whisper --help` (should show whisper command-line options)
- **Test TTS**: `espeak "test"` or `say "test"` (macOS)
- **Check logs**: Review `voice_computer.log` for detailed error information
- **Verbose logging**: Use `--verbose` flag when running

## Voice Commands

- **General queries**: Any natural language request processed through MCP tools
- **Exit commands**: "exit", "quit", "stop", "goodbye", "bye" to end session
- **System responds**: Via text-to-speech with configurable voice settings

## Extending the System

- **Add MCP servers**: Install new MCP servers and add to configuration JSON or programmatically
- **Customize voice settings**: Modify volume thresholds, timeouts in config files
- **Change LLM model**: Update Ollama model in configuration (any model available in Ollama)
- **Custom configurations**: Create config classes extending `voice_computer.config.Config`
- **Add new connectors**: Extend `MCPConnector` class for new MCP server types
- **Modify voice responses**: System TTS settings handled automatically based on available tools