# Voice Computer

A voice-driven assistant that integrates Whisper speech recognition with MCP (Model Context Protocol) tools via Ollama. Speak naturally to interact with various tools and get spoken responses.

## 🚀 Quick Start

### Prerequisites

1. **Ollama** running locally with a model (e.g., `ollama pull qwen2.5:32b`)
2. **Whisper** for speech recognition: `pip install openai-whisper`
3. **System TTS** (Linux: `sudo apt-get install espeak` | macOS: built-in `say`)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd voice_mcp_client

# Install dependencies
pip install -r requirements.txt

# Install Whisper
pip install openai-whisper

# Install MCP servers (example)
pip install mcp-server-filesystem
```

### Basic Usage

```bash
# Run in voice mode
python run_voice_computer.py

# Run in text-only mode (for testing)
python run_voice_computer.py --test

# Create a configuration file
python run_voice_computer.py --create-config=config.json

# Run with custom config
python run_voice_computer.py --config=config.json
```

## 📁 Project Structure

```
voice_mcp_client/
├── voice_computer/              # Main package (replaces yaaaf dependencies)
│   ├── __init__.py             # Package exports
│   ├── client.py               # Main voice computer client
│   ├── voice_interface.py      # Voice input/output handling
│   ├── whisper_listener.py     # Speech recognition using Whisper CLI
│   ├── ollama_client.py        # Ollama API integration
│   ├── mcp_connector.py        # MCP server integration
│   ├── tool_agent.py           # Tool coordination and execution
│   ├── data_types.py           # Core data structures
│   └── config.py               # Configuration management
├── run_voice_computer.py       # Main execution script
├── requirements.txt            # Dependencies
├── CLAUDE.md                   # Development guidance
└── README.md        # This file
```

## 🔧 Configuration

### JSON Configuration

Create a configuration file:

```bash
python run_voice_computer.py --create-config=my_config.json
```

Example configuration:

```json
{
  "listener_model": {
    "listener_silence_timeout": 2,
    "listener_volume_threshold": 1,
    "listener_hotword_logp": -8
  },
  "waking_up_sound": true,
  "deactivate_sound": true,
  "ollama_host": "http://localhost:11434",
  "ollama_model": "qwen2.5:32b",
  "mcp_servers": [
    {
      "name": "filesystem",
      "path": "mcp-server-filesystem",
      "args": ["--root", "/tmp"]
    },
    {
      "name": "web-search",
      "path": "mcp-server-brave-search",
      "args": ["--api-key", "your-api-key"]
    }
  ]
}
```

### Programmatic Configuration

```python
from voice_computer import VoiceComputerClient
from voice_computer.config import Config

# Create custom configuration
config = Config()
config.set_value("ollama_model", "llama3.2:latest")
config.set_value("mcp_servers", [
    {
        "name": "filesystem",
        "path": "mcp-server-filesystem",
        "args": ["--root", "/home/user/documents"]
    }
])

# Initialize client
client = VoiceComputerClient(config)

# Add MCP servers dynamically
client.add_mcp_server("git", "mcp-server-git", ["--repository", "/path/to/repo"])

# Run the voice loop
await client.run_voice_loop()
```

## 🎤 Usage Examples

### Voice Mode
```bash
python run_voice_computer.py
# Speak: "List files in the current directory"
# Speak: "Search the web for Python tutorials"
# Speak: "Exit" to quit
```

### Text Mode (Testing)
```bash
python run_voice_computer.py --test
user> List files in the current directory
bot> I found the following files: ...
user> exit
```

### With Custom Config
```bash
python run_voice_computer.py --config=production.json --verbose
```

## 🛠 MCP Server Integration

### Available MCP Servers

- **Filesystem**: `pip install mcp-server-filesystem`
- **Web Search**: `pip install mcp-server-brave-search`
- **SQLite**: `pip install mcp-server-sqlite`
- **Git**: `pip install mcp-server-git`

### Adding MCP Servers

#### Via Configuration File
```json
{
  "mcp_servers": [
    {
      "name": "my-server",
      "path": "path-to-mcp-server",
      "args": ["--option", "value"]
    }
  ]
}
```

#### Programmatically
```python
client.add_mcp_server("server-name", "server-path", ["--arg1", "value1"])
```

## 🎛 Command Line Options

```bash
python run_voice_computer.py [OPTIONS]

Options:
  --test                    Run in text-only mode
  --config PATH             Use custom JSON configuration file
  --create-config PATH      Create example configuration file
  --verbose, -v             Enable debug logging
  --log-file PATH           Specify log file location
  --help                    Show help message
```

## 🔍 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| No voice recognition | Install Whisper: `pip install openai-whisper` |
| No speech output | Install TTS: `sudo apt-get install espeak` (Linux) |
| Ollama connection failed | Ensure Ollama is running: `ollama serve` |
| No MCP tools available | Check MCP server installation and config |
| Import errors | Install requirements: `pip install -r requirements.txt` |

### Debug Commands

```bash
# Test Ollama connection
curl http://localhost:11434/api/tags

# Test Whisper installation
whisper --help

# Test TTS (Linux)
espeak "Hello world"

# Test TTS (macOS)
say "Hello world"

# Run with verbose logging
python run_voice_computer.py --verbose
```

### Logging

Logs are written to `voice_computer.log` by default. Use `--verbose` for detailed debugging information.

## 🏗 Architecture

### Key Components

- **VoiceComputerClient**: Main orchestrator
- **VoiceInterface**: Handles voice I/O using system tools
- **WhisperListener**: Speech recognition via Whisper CLI
- **OllamaClient**: Communicates with Ollama API
- **ToolAgent**: Coordinates MCP tool execution
- **MCPConnector**: Integrates with MCP servers

### Data Flow

1. Audio captured using system recording tools
2. Transcribed via Whisper command-line tool
3. Query processed by ToolAgent with MCP tools
4. Results returned via Ollama LLM
5. Response spoken using system TTS

## 🚫 What Was Removed

This refactor removed the `yaaaf` framework dependencies:

- ❌ `yaaaf.components.*`
- ❌ `yaaaf.connectors.*`
- ❌ `yaaaf.agents.*`
- ❌ Complex external dependencies (`pyaudio`, `numpy`, `pandas`, `mdpd`)

✅ **Replaced with**: Self-contained `voice_computer` package using system tools

## 🤝 Contributing

1. All functionality is now in the `voice_computer/` directory
2. No external framework dependencies
3. Uses system command-line tools for maximum compatibility
4. Async/await patterns throughout
5. Comprehensive logging and error handling

## 📄 License

[Your license here]

---

**Note**: This is the refactored version without yaaaf dependencies. The original voice interface functionality is preserved but simplified to use system tools directly.