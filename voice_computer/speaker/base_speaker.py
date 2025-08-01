"""
Base speaker class for audio output functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseSpeaker(ABC):
    """Abstract base class for all speaker implementations."""
    
    @abstractmethod
    def speak(self, filename: str) -> None:
        """
        Play audio from a file.
        
        Args:
            filename: Path to the audio file to play
        """
        pass
    
    def cleanup(self) -> None:
        """
        Clean up resources. Override if needed.
        """
        pass
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during destruction