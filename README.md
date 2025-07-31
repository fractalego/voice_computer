# Voice Computer

A voice-driven assistant that integrates Whisper speech recognition with MCP (Model Context Protocol) tools via Ollama. Speak naturally to interact with various tools and get spoken responses.

## 🚀 Quick Start

### Prerequisites

1. **Ollama** running locally with a model (e.g., `ollama pull qwen2.5:32b`)
2. **PyTorch** and **Transformers** for speech recognition (installed automatically)
3. **System TTS** (Linux: `sudo apt-get install espeak` | macOS: built-in `say`)
4. **Audio System** (Linux: `sudo apt-get install portaudio19-dev` for microphone input)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd voice_mcp_client

# Install dependencies
pip install -r requirements.txt
# or use uv: uv sync

# Install MCP servers (example)
pip install mcp-server-filesystem
```

### Basic Usage

```bash
# Run in voice mode (default MCP server starts automatically)
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
│   ├── whisper_listener.py     # Speech recognition using Hugging Face Whisper
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
      "name": "default",
      "type": "sse",
      "url": "http://localhost:8080",
      "args": []
    },
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
# Say: "Computer, list files in the current directory"
# Say: "Computer, search the web for Python tutorials"
# Say: "Computer, exit" to quit
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

### Built-in Default MCP Server

The system includes a **built-in default MCP server** with basic tools:

```bash
# Start the default server (runs on http://localhost:8080)
python start_default_mcp_server.py
```

**Available tools:**
- `add_two_numbers(lhs: int, rhs: int)`: Calculate the sum of two integers

### External MCP Servers

- **Filesystem**: `pip install mcp-server-filesystem`
- **Web Search**: `pip install mcp-server-brave-search`
- **SQLite**: `pip install mcp-server-sqlite`
- **Git**: `pip install mcp-server-git`

### Adding MCP Servers

#### Via Configuration File

**SSE-based servers (like the default):**
```json
{
  "mcp_servers": [
    {
      "name": "my-sse-server",
      "type": "sse",
      "url": "http://localhost:8080",
      "args": []
    }
  ]
}
```

**Stdio-based servers:**
```json
{
  "mcp_servers": [
    {
      "name": "my-stdio-server",
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

## 🎙️ Hotword Configuration

The system uses **"computer"** as the default activation hotword with logp-based detection.

### Default Usage
```bash
# Say "Computer, what's the weather?" 
# The system detects "computer" using probability scores, not text matching
```

### Custom Hotwords
```json
{
  "activation_hotwords": ["computer", "assistant", "ai"],
  "listener_model": {
    "listener_hotword_logp": -8  // Lower = more sensitive, Higher = less sensitive
  }
}
```

### Hotword Detection Features
- **Logp-based**: Uses probability scores from Whisper model, not text transcription
- **Multiple hotwords**: Support for multiple activation words
- **Configurable sensitivity**: Adjust detection threshold via `listener_hotword_logp`
- **Real-time processing**: Instant hotword detection without full transcription

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
| No voice recognition | Install dependencies: `pip install -r requirements.txt` |
| No speech output | Install TTS: `sudo apt-get install espeak` (Linux) |
| Ollama connection failed | Ensure Ollama is running: `ollama serve` |
| No MCP tools available | Check MCP server installation and config |
| Import errors | Install requirements: `pip install -r requirements.txt` |
| BetterTransformer errors | Update dependencies: `pip install -U transformers torch` |
| Audio warnings | Check microphone permissions and audio drivers |

### Debug Commands

```bash
# Test Ollama connection
curl http://localhost:11434/api/tags

# Test Python imports
python -c "from voice_computer import VoiceComputerClient; print('Imports working')"

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
- **WhisperListener**: Speech recognition via Hugging Face Whisper with logp-based hotword detection
- **OllamaClient**: Communicates with Ollama API
- **ToolAgent**: Coordinates MCP tool execution
- **MCPConnector**: Integrates with MCP servers

### Data Flow

1. **Audio Capture**: Real-time microphone input via PyAudio
2. **Voice Activity Detection**: RMS-based volume threshold detection
3. **Hotword Detection**: Logp-based activation word detection ("computer" by default)
4. **Speech Recognition**: Hugging Face Whisper model transcription with PyTorch optimization
5. **Query Processing**: ToolAgent coordinates with MCP tools
6. **LLM Processing**: Results processed via Ollama API
7. **Speech Output**: Response spoken using system TTS (espeak/say)

## 🤝 Contributing

1. All functionality is now in the `voice_computer/` directory
2. No external framework dependencies
3. Uses system command-line tools for maximum compatibility
4. Async/await patterns throughout
5. Comprehensive logging and error handling

## 📄 License

MIT license. See `license.txt` file for details.

---

**Note**: This is the refactored version without yaaaf dependencies. The original voice interface functionality is preserved but simplified to use system tools directly.