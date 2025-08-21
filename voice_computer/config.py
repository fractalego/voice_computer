"""
Configuration classes for the voice computer system.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List


class Config:
    """Base configuration class."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        self._config = config_dict or self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            # Voice interface settings
            "listener_model": {
                "listener_silence_timeout": 0.5,      # seconds of silence before stopping recording
                "listener_volume_threshold": 0.6,   # minimum volume to trigger recording
                "listener_hotword_logp": -8,        # hotword detection threshold
                "microphone_device_index": None     # specific microphone device index (None for default)
            },
            "activation_hotwords": ["computer"],    # default activation hotwords
            "waking_up_sound": True,                # play sound when activating
            "deactivate_sound": True,               # play sound when deactivating
            
            # Whisper model settings
            "whisper_model": "fractalego/personal-whisper-distilled-model",  # Hugging Face model ID
            
            # Entailment settings
            "entailer_client_type": None,           # "ollama", "huggingface", or None to use llm_client_type
            "entailer_host": None,                  # Use ollama_host if None
            "entailer_model": None,                 # Use ollama_model/huggingface_model if None  
            "entailer_conversation_history_length": 3,  # Number of recent utterances to include for entailment context
            
            # LLM client configuration
            "llm_client_type": "ollama",            # "ollama" or "huggingface"
            
            # Ollama settings
            "ollama_host": "http://localhost:11434",
            "ollama_model": "qwen2.5:32b",          # or any model you have in Ollama
            
            # HuggingFace local model settings  
            "huggingface_model": "Qwen/Qwen2.5-32B",       # Default HF model
            "huggingface_quantization": "bfloat16",        # "bfloat16", "float16", "float32", "8bit", "4bit"
            "huggingface_device": None,                     # "cuda", "cpu", or None for auto-detection
            "huggingface_torch_dtype": None,                # Override quantization dtype if needed
            
            # Extractor LLM settings (for argument extraction from queries)
            "extractor_client_type": None,          # "ollama", "huggingface", or None to use llm_client_type
            "extractor_host": None,                 # Use ollama_host if None
            "extractor_model": None,                # Use ollama_model/huggingface_model if None
            "extractor_conversation_history_length": 2,  # Number of recent exchanges to include in context
            
            # MCP servers configuration
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
                },
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
                },
                {
                    "name": "tfl",
                    "path": "python",
                    "args": ["-m", "voice_computer.mcp_servers.tfl_mcp_server"]
                },
                {
                    "name": "spotify",
                    "path": "python",
                    "args": ["-m", "voice_computer.mcp_servers.spotify_mcp_server"],
                    "env_vars": {
                        "SPOTIFY_CLIENT_ID": "${SPOTIFY_CLIENT_ID}",
                        "SPOTIFY_CLIENT_SECRET": "${SPOTIFY_CLIENT_SECRET}"
                    }
                }
            ],
            
            # Streaming configuration
            "streaming": {
                "enabled": True,                # Enable streaming output
                "token_batch_size": 6,          # Number of tokens to batch before displaying
                "flush_delay": 1.0              # Delay in seconds between token batch checks
            },
            
            # Bot facts configuration
            "facts": [
                "The name of this chatbot is 'Computer'",  # Facts immediately accessible to the bot
                "You can check London transport status using TFL tools"
            ],
            
            # Tool results queue configuration
            "tool_results_queue_length": 2,  # Maximum number of tool results to keep in queue
            
            # Special sentences configuration - maps sentences to specific MCP server tools
            # Supports two formats:
            # Simple: "sentence": "server.tool" 
            # Extended: "sentence": {"tool": "server.tool", "default_args": {"arg": "value"}}
            "special_sentences": {
                "what time is it": "time.current_time",
                "are the trains running": {
                    "tool": "trains.get_station_departures",
                    "default_args": {
                        "station_code": "KGX",
                        "limit": 5
                    }
                },
            },
            
            # Constant listening mode configuration
            "constant_listening": {
                "enabled": False,                   # Enable constant listening mode
                "command_timeout": 10.0,           # Timeout in seconds for command after hotword
                "command_join_delay": 0.1,         # Delay in seconds to collect rapid commands before joining
                "max_conversation_history": 20     # Maximum conversation history to keep
            }
        }
    
    def get_value(self, key: str) -> Any:
        """Get a configuration value."""
        return self._config.get(key)
    
    def set_value(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._config[key] = value
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._config.copy()


class ExampleConfig(Config):
    """Example configuration with sample MCP servers."""
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration with example MCP servers."""
        config = super()._get_default_config()
        
        # Add whisper model
        config["whisper_model"] = "fractalego/personal-whisper-distilled-model"
        
        # Add default activation hotwords
        config["activation_hotwords"] = ["computer"]
        
        # Add example MCP servers
        config["mcp_servers"] = [
            # Math stdio MCP server
            {
                "name": "math operations",
                "path": "python",
                "args": ["-m", "voice_computer.mcp_servers.math_mcp_server"]
            },
            # Time stdio MCP server
            {
                "name": "time operations",
                "path": "python",
                "args": ["-m", "voice_computer.mcp_servers.time_mcp_server"]
            },
            # Weather MCP server 
            {
                "name": "weather operations",
                "path": "python",
                "args": ["-m", "voice_computer.mcp_servers.weather_mcp_server"],
                "env_vars": {"WEATHER_API_KEY": "${WEATHER_API_KEY}"}
            },
            # Train MCP server
            {
                "name": "train operations",
                "path": "python",
                "args": ["-m", "voice_computer.mcp_servers.train_mcp_server"],
                "env_vars": {
                    "TRANSPORT_API_ID": "${TRANSPORT_API_ID}",
                    "TRANSPORT_API_KEY": "${TRANSPORT_API_KEY}"
                }
            },
            # TFL (Transport for London) MCP server
            {
                "name": "tfl operations",
                "path": "python",
                "args": ["-m", "voice_computer.mcp_servers.tfl_mcp_server"]
            },
        ]
        
        return config


class JSONConfig(Config):
    """Configuration loaded from a JSON file."""
    
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        config_dict = self._load_from_json()
        super().__init__(config_dict)
    
    def _load_from_json(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not self.json_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.json_path}")
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {self.json_path}: {e}")
    
    def save_to_json(self) -> None:
        """Save current configuration to JSON file."""
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2)


def create_example_config_file(path: str) -> None:
    """Create an example configuration JSON file."""
    example_config = {
        "listener_model": {
            "listener_silence_timeout": 2,
            "listener_volume_threshold": 0.6
            ,
            "listener_hotword_logp": -8
        },
        "activation_hotwords": ["computer"],
        "waking_up_sound": True,
        "deactivate_sound": True,
        "whisper_model": "fractalego/personal-whisper-distilled-model",
        "entailer_host": None,
        "entailer_model": None,
        "entailer_conversation_history_length": 3,
        "llm_client_type": "ollama",
        "ollama_host": "http://localhost:11434",
        "ollama_model": "qwen2.5:32b",
        "huggingface_model": "Qwen/Qwen2.5-32B",
        "huggingface_api_key": None,
        "extractor_host": None,
        "extractor_model": None,
        "extractor_conversation_history_length": 2,
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
            },
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
            },
            {
                "name": "tfl",
                "path": "python",
                "args": ["-m", "voice_computer.mcp_servers.tfl_mcp_server"]
            },
            {
                "name": "spotify",
                "path": "python",
                "args": ["-m", "voice_computer.mcp_servers.spotify_mcp_server"],
                "env_vars": {
                    "SPOTIFY_CLIENT_ID": "${SPOTIFY_CLIENT_ID}",
                    "SPOTIFY_CLIENT_SECRET": "${SPOTIFY_CLIENT_SECRET}"
                }
            }
        ],
        "streaming": {
            "enabled": True,
            "token_batch_size": 6,
            "flush_delay": 1.0
        },
        "facts": [
            "The name of this chatbot is 'Computer'",
            "You can check London transport status using TFL tools"
        ],
        "tool_results_queue_length": 2,
        "special_sentences": {
            "what time is it": "time.current_time",
            "are the trains running": {
                "tool": "trains.get_station_departures",
                "default_args": {
                    "station_code": "KGX",
                    "limit": 5
                }
            },
        },
        "constant_listening": {
            "enabled": False,
            "command_timeout": 10.0,
            "command_join_delay": 0.1,
            "max_conversation_history": 20
        }
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(example_config, f, indent=2)


# Convenience function for loading configuration
def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or return default.
    
    Args:
        config_path: Optional path to JSON configuration file
        
    Returns:
        Configuration instance
    """
    if config_path:
        return JSONConfig(config_path)
    else:
        return ExampleConfig()