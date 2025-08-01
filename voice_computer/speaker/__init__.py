"""
Speaker module for audio output functionality.
"""

from .base_speaker import BaseSpeaker
from .sound_file_speaker import SoundFileSpeaker
from .tts_speaker import TTSSpeaker

__all__ = ["BaseSpeaker", "SoundFileSpeaker", "TTSSpeaker"]