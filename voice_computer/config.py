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
                "listener_silence_timeout": 2,      # seconds of silence before stopping recording
                "listener_volume_threshold": 0.6,   # minimum volume to trigger recording
                "listener_hotword_logp": -8,        # hotword detection threshold
                "microphone_device_index": None     # specific microphone device index (None for default)
            },
            "activation_hotwords": ["computer"],    # default activation hotwords
            "waking_up_sound": True,                # play sound when activating
            "deactivate_sound": True,               # play sound when deactivating
            
            # Whisper model settings
            "whisper_model": "fractalego/personal-whisper-distilled-model",  # Hugging Face model ID
            
            # Entailment model settings
            "entailment_model": "vectara/hallucination_evaluation_model",  # Hugging Face model ID
            "entailment_device": None,              # Auto-detect device if None (cuda/mps/cpu)
            "entailment_threshold": 0.4,            # Minimum entailment score to run a tool
            
            # Ollama settings
            "ollama_host": "http://localhost:11434",
            "ollama_model": "qwen2.5:32b",          # or any model you have in Ollama
            
            # Extractor LLM settings (for argument extraction from queries)
            "extractor_host": None,                 # Use ollama_host if None
            "extractor_model": None,                # Use ollama_model if None
            
            # MCP servers configuration
            "mcp_servers": [
                {
                    "name": "default",
                    "path": "python",
                    "args": ["-m", "voice_computer.default_mcp_server"]
                }
            ],
            
            # Streaming configuration
            "streaming": {
                "enabled": True,                # Enable streaming output
                "token_batch_size": 4,          # Number of tokens to batch before displaying
                "flush_delay": 0.1              # Delay in seconds between token batch checks
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
            # Default stdio MCP server
            {
                "name": "math operations",
                "path": "python",
                "args": ["-m", "voice_computer.default_mcp_server"]
            },
            # Filesystem MCP server example (commented out)
            # {
            #     "name": "filesystem",
            #     "path": "mcp-server-filesystem",
            #     "args": ["--root", "/tmp"]
            # },
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
            "listener_volume_threshold": 0.6,
            "listener_hotword_logp": -8
        },
        "activation_hotwords": ["computer"],
        "waking_up_sound": True,
        "deactivate_sound": True,
        "whisper_model": "fractalego/personal-whisper-distilled-model",
        "entailment_model": "vectara/hallucination_evaluation_model",
        "entailment_device": None,
        "entailment_threshold": 0.4,
        "ollama_host": "http://localhost:11434",
        "ollama_model": "qwen2.5:32b",
        "extractor_host": None,
        "extractor_model": None,
        "mcp_servers": [
            {
                "name": "default",
                "path": "python",
                "args": ["-m", "voice_computer.default_mcp_server"]
            }
        ],
        "streaming": {
            "enabled": True,
            "token_batch_size": 4,
            "flush_delay": 0.1
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