# Voice Computer

A voice-driven assistant that integrates Whisper speech recognition with MCP (Model Context Protocol) tools via configurable language models (Ollama or local HuggingFace models). Speak naturally to interact with various tools and get spoken responses.

## 🚀 Quick Start

### Prerequisites

1. **Language Model**: Choose one of:
   - **Ollama** running locally with a model (e.g., `ollama pull qwen2.5:32b`)
   - **HuggingFace models** downloaded locally (e.g., `Qwen/Qwen2.5-32B`)
2. **PyTorch** and **Transformers** for speech recognition and local models (installed automatically)
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

### Language Model Configuration

The system supports two types of language models:

#### Option 1: Ollama (Default)
```json
{
  "llm_client_type": "ollama",
  "ollama_host": "http://localhost:11434",
  "ollama_model": "qwen2.5:32b"
}
```

#### Option 2: Local HuggingFace Models
```json
{
  "llm_client_type": "huggingface",
  "huggingface_model": "Qwen/Qwen2.5-32B",
  "huggingface_quantization": "bfloat16"
}
```

### Multi-Client Configuration

You can use different models for different components:

```json
{
  "llm_client_type": "huggingface",
  "entailer_client_type": "ollama", 
  "extractor_client_type": "huggingface",
  
  "ollama_host": "http://localhost:11434",
  "ollama_model": "qwen2.5:7b",
  "huggingface_model": "Qwen/Qwen2.5-32B"
}
```

**Client Types:**
- `llm_client_type`: Main conversation model
- `entailer_client_type`: Tool selection model (defaults to `llm_client_type` if null)
- `extractor_client_type`: Argument extraction model (defaults to `llm_client_type` if null)

### Complete Configuration Examples

#### Ollama Configuration
```json
{
  "listener_model": {
    "listener_silence_timeout": 0.5,
    "listener_volume_threshold": 0.6,
    "listener_hotword_logp": -8
  },
  "activation_hotwords": ["computer"],
  "waking_up_sound": true,
  "deactivate_sound": true,
  "whisper_model": "fractalego/personal-whisper-distilled-model",
  
  "llm_client_type": "ollama",
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

#### HuggingFace Configuration (Default bfloat16)
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
  
  "llm_client_type": "huggingface",
  "huggingface_model": "Qwen/Qwen2.5-32B",
  "huggingface_quantization": "bfloat16",
  
  "mcp_servers": [
    {
      "name": "math",
      "path": "python",
      "args": ["-m", "voice_computer.mcp_servers.math_mcp_server"]
    },
    {
      "name": "time", 
      "path": "python",
      "args": ["-m", "voice_computer.mcp_servers.time_mcp_server"]
    }
  ]
}
```

#### HuggingFace Configuration (8-bit Quantization)
```json
{
  "llm_client_type": "huggingface",
  "huggingface_model": "Qwen/Qwen2.5-32B",
  "huggingface_quantization": "8bit",
  
  "mcp_servers": [
    {
      "name": "filesystem",
      "path": "mcp-server-filesystem",
      "args": ["--root", "/tmp"]
    }
  ]
}
```

#### HuggingFace Configuration (4-bit Quantization)
```json
{
  "llm_client_type": "huggingface",
  "huggingface_model": "Qwen/Qwen2.5-32B", 
  "huggingface_quantization": "4bit",
  
  "mcp_servers": [
    {
      "name": "math",
      "path": "python",
      "args": ["-m", "voice_computer.mcp_servers.math_mcp_server"]
    }
  ]
}
```

#### Mixed Configuration
```json
{
  "llm_client_type": "huggingface",
  "entailer_client_type": "ollama",
  "extractor_client_type": "huggingface",
  
  "ollama_host": "http://localhost:11434", 
  "ollama_model": "qwen2.5:7b",
  "huggingface_model": "Qwen/Qwen2.5-32B",
  
  "mcp_servers": [
    {
      "name": "filesystem",
      "path": "mcp-server-filesystem", 
      "args": ["--root", "/tmp"]
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

## 🤗 HuggingFace Models

### Supported Models

The system supports any HuggingFace causal language model, including:

- **Qwen Models**: `Qwen/Qwen2.5-32B`, `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-3B`
- **Llama Models**: `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf`
- **Mistral Models**: `mistralai/Mistral-7B-Instruct-v0.1`
- **Code Models**: `bigcode/starcoder2-7b`, `microsoft/DialoGPT-medium`

### Model Loading Process

When using HuggingFace models, the system will:

1. **🚀 Initialize** the local model client
2. **📁 Check local cache** first (offline mode) 
3. **📥 Download if needed** (first-time use or new model)
4. **✅ Load into memory** with optimal device settings
5. **🔧 Ready for inference** completely offline

### First-Time Setup

For first-time model download:

```bash
# Option 1: Let the system download automatically
python run_voice_computer.py --config=hf_config.json

# Option 2: Pre-download manually
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading model...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B')
print('Model cached successfully!')
"
```

### Offline Usage

Once downloaded, models work completely offline:

```bash
# Set offline mode (optional)
export TRANSFORMERS_OFFLINE=1
python run_voice_computer.py --config=hf_config.json
```

### Hardware Requirements

**Memory Requirements (7B Model):**
- **4-bit**: ~4GB VRAM/RAM
- **8-bit**: ~7GB VRAM/RAM
- **bfloat16**: ~14GB VRAM/RAM
- **float32**: ~28GB VRAM/RAM

**Memory Requirements (32B Model):**
- **4-bit**: ~16GB VRAM/RAM
- **8-bit**: ~32GB VRAM/RAM  
- **bfloat16**: ~64GB VRAM/RAM
- **float32**: ~128GB VRAM/RAM

**GPU Support:**
- **CUDA**: Automatically detected and used
- **CPU**: Falls back automatically if no GPU
- **MPS** (Apple Silicon): Supported for M1/M2 Macs

### Model Configuration Options

```json
{
  "llm_client_type": "huggingface",
  "huggingface_model": "Qwen/Qwen2.5-7B",
  
  // Quantization options (default: "bfloat16")
  "huggingface_quantization": "bfloat16",  // "bfloat16", "float16", "float32", "8bit", "4bit"
  
  // Optional: Force specific device
  "huggingface_device": "cuda",  // "cuda", "cpu", "auto"
  
  // Optional: Override precision (not needed for quantization)
  "huggingface_torch_dtype": "bfloat16"  // "bfloat16", "float16", "float32"
}
```

### Quantization Options

**bfloat16 (Default)**: Best balance of performance and memory usage
```json
{
  "huggingface_quantization": "bfloat16"
}
```

**8-bit Quantization**: ~50% memory reduction with minimal quality loss
```json
{
  "huggingface_quantization": "8bit"
}
```

**4-bit Quantization**: ~75% memory reduction for very large models
```json
{
  "huggingface_quantization": "4bit"
}
```

**Memory Comparison (32B Model):**
- **bfloat16**: ~64GB VRAM
- **8bit**: ~32GB VRAM  
- **4bit**: ~16GB VRAM

### Performance Tips

1. **Choose appropriate quantization** for your hardware:
   - **RTX 4090 (24GB)**: Use bfloat16 for 7B models, 8bit for 13B models, 4bit for 32B models
   - **RTX 3080 (10GB)**: Use 8bit for 7B models, 4bit for larger models
   - **CPU only**: Use 4bit quantization to reduce memory usage

2. **Use smaller models** for faster responses:
   - `Qwen/Qwen2.5-7B` instead of `Qwen/Qwen2.5-32B`
   - `microsoft/DialoGPT-medium` for basic conversation

3. **GPU acceleration** with quantization:
   - Install CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
   - Install quantization support: `pip install bitsandbytes accelerate`

4. **Pre-download models** when you have good internet:
   ```bash
   python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B')"
   ```

### Troubleshooting HuggingFace Models

| Issue | Solution |
|-------|----------|
| **DNS/Network errors during download** | Use offline mode or pre-download when you have connectivity |
| **Out of memory errors** | Use 8bit or 4bit quantization: `"huggingface_quantization": "4bit"` |
| **Slow loading** | Models are cached after first download - subsequent loads are faster |
| **CUDA out of memory** | Use quantization or CPU: `"huggingface_quantization": "4bit"` |
| **bitsandbytes not installed** | Install with: `pip install bitsandbytes` |
| **Quantization errors on CPU** | Use bfloat16/float32 instead of 8bit/4bit for CPU |
| **Model not found** | Verify model name on [huggingface.co/models](https://huggingface.co/models) |

### Cache Location

Models are cached in:
- **Linux/Mac**: `~/.cache/huggingface/hub/`
- **Windows**: `%USERPROFILE%\.cache\huggingface\hub\`

You can clear cache if needed:
```bash
rm -rf ~/.cache/huggingface/hub/
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

The system includes **built-in MCP servers** with essential tools:

#### Math Server
```bash
# Test the math server
python -m voice_computer.mcp_servers.math_mcp_server
```

**Available math tools:**
- `add_two_numbers(lhs: number, rhs: number)`: Calculate the sum of two numbers
- `subtract_two_numbers(lhs: number, rhs: number)`: Calculate the difference of two numbers (lhs - rhs)
- `multiply_two_numbers(lhs: number, rhs: number)`: Calculate the product of two numbers
- `divide_two_numbers(lhs: number, rhs: number)`: Calculate the division of two numbers (lhs / rhs)
- `square_root(number: number)`: Calculate the square root of a number

#### Time Server
```bash
# Test the time server
python -m voice_computer.mcp_servers.time_mcp_server
```

**Available time tools:**
- `current_time()`: Get the current time in HH:MM:SS format
- `current_date()`: Get the current date in YYYY-MM-DD format
- `current_day_of_week()`: Get the current day of the week
- `current_datetime()`: Get the current date and time in a readable format

#### Weather Server
```bash
# Set your API key first
export WEATHER_API_KEY="your-weatherapi-key"

# Test the weather server
python -m voice_computer.mcp_servers.weather_mcp_server
```

**Available weather tools:**
- `get_current_weather(location: str)`: Get current weather conditions for a location
- `get_weather_forecast(location: str, days: int = 3)`: Get weather forecast (1-10 days)
- `search_locations(query: str)`: Search for locations to get weather data
- `get_weather_alerts(location: str)`: Get weather alerts for a location

**Setup:**
1. Get a free API key from [weatherapi.com](https://www.weatherapi.com/)
2. Set the `WEATHER_API_KEY` environment variable
3. Add the weather server to your configuration (see examples below)

#### Train Server
```bash
# Set your API credentials first
export TRANSPORT_API_ID="your-transport-api-id"
export TRANSPORT_API_KEY="your-transport-api-key"

# Test the train server
python -m voice_computer.mcp_servers.train_mcp_server
```

**Available train tools:**
- `get_train_timetable_from_postcode(postcode: str, direction: str = "departures", limit: int = 5)`: Get train times from nearest station to a postcode
- `find_stations_near_postcode(postcode: str, distance: int = 1000)`: Find train stations near a postcode
- `get_station_departures(station_code: str, limit: int = 10)`: Get live departures from a station
- `get_station_arrivals(station_code: str, limit: int = 10)`: Get live arrivals at a station
- `search_station_codes(query: str)`: Search for train station codes by name

**Setup:**
1. Register at [transportapi.com](https://www.transportapi.com/) for a free API account
2. Get your API ID and key from the developer dashboard
3. Set the environment variables: `TRANSPORT_API_ID` and `TRANSPORT_API_KEY`
4. Add the train server to your configuration (see examples below)

### External MCP Servers

You can integrate external MCP servers to extend functionality. Supported types include:
- **SSE-based servers**: Use Server-Sent Events for real-time updates
- **Stdio-based servers**: Use standard input/output for communication

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

**Built-in stdio servers:**
```json
{
  "mcp_servers": [
    {
      "name": "math operations",
      "path": "python",
      "args": ["-m", "voice_computer.mcp_servers.math_mcp_server"]
    },
    {
      "name": "time operations", 
      "path": "python",
      "args": ["-m", "voice_computer.mcp_servers.time_mcp_server"]
    },
    {
      "name": "weather operations",
      "path": "python", 
      "args": ["-m", "voice_computer.mcp_servers.weather_mcp_server"],
      "env_vars": {"WEATHER_API_KEY": "your-api-key-here"}
    },
    {
      "name": "train operations",
      "path": "python",
      "args": ["-m", "voice_computer.mcp_servers.train_mcp_server"],
      "env_vars": {
        "TRANSPORT_API_ID": "your-transport-api-id",
        "TRANSPORT_API_KEY": "your-transport-api-key"
      }
    }
  ]
}
```

**External stdio servers:**
```json
{
  "mcp_servers": [
    {
      "name": "filesystem",
      "path": "mcp-server-filesystem",
      "args": ["--root", "/tmp"]
    }
  ]
}
```

**Environment Variables:**
You can pass environment variables to MCP servers:
```json
{
  "mcp_servers": [
    {
      "name": "weather",
      "path": "python",
      "args": ["-m", "voice_computer.mcp_servers.weather_mcp_server"],
      "env_vars": {"WEATHER_API_KEY": "${WEATHER_API_KEY}"}
    },
    {
      "name": "trains",
      "path": "python",
      "args": ["-m", "voice_computer.mcp_servers.train_mcp_server"],
      "env_vars": {
        "TRANSPORT_API_ID": "${TRANSPORT_API_ID}",
        "TRANSPORT_API_KEY": "${TRANSPORT_API_KEY}"
      }
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
| HuggingFace DNS errors | Use offline mode: `export TRANSFORMERS_OFFLINE=1` |
| HuggingFace CUDA OOM | Use smaller model or CPU: `"device": "cpu"` |
| HuggingFace model not found | Verify model name on huggingface.co |
| Long model loading time | First download takes time, subsequent loads are faster |
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

# Test HuggingFace model loading
python -c "
from voice_computer.client.hf_client import HFClient
from voice_computer.data_types import Messages, Utterance
print('Testing HF client...')
# This will test without actually loading a large model
"

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