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
                "listener_volume_threshold": 1,     # minimum volume to trigger recording
                "listener_hotword_logp": -8         # hotword detection threshold
            },
            "waking_up_sound": True,                # play sound when activating
            "deactivate_sound": True,               # play sound when deactivating
            
            # Ollama settings
            "ollama_host": "http://localhost:11434",
            "ollama_model": "qwen2.5:32b",          # or any model you have in Ollama
            
            # MCP servers configuration
            "mcp_servers": []
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
        
        # Add example MCP servers
        config["mcp_servers"] = [
            # Filesystem MCP server example
            {
                "name": "filesystem",
                "path": "mcp-server-filesystem",
                "args": ["--root", "/tmp"]
            },
            
            # Web search MCP server example (requires API key)
            # {
            #     "name": "brave-search",
            #     "path": "mcp-server-brave-search", 
            #     "args": ["--api-key", "YOUR_BRAVE_API_KEY"]
            # },
            
            # SQLite MCP server example
            # {
            #     "name": "sqlite",
            #     "path": "mcp-server-sqlite",
            #     "args": ["--db", "/path/to/your/database.db"]
            # },
            
            # Git MCP server example
            # {
            #     "name": "git",
            #     "path": "mcp-server-git",
            #     "args": ["--repository", "/path/to/your/repo"]
            # }
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
            "listener_volume_threshold": 1,
            "listener_hotword_logp": -8
        },
        "waking_up_sound": True,
        "deactivate_sound": True,
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