"""
Speaker module for audio output functionality.
"""

from .base_speaker import BaseSpeaker
from .sound_file_speaker import SoundFileSpeaker

__all__ = ["BaseSpeaker", "SoundFileSpeaker"]