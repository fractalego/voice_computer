"""
Speaker module for audio output functionality.
"""

from voice_computer.speaker.base_speaker import BaseSpeaker
from voice_computer.speaker.sound_file_speaker import SoundFileSpeaker
from voice_computer.speaker.tts_speaker import TTSSpeaker

__all__ = ["BaseSpeaker", "SoundFileSpeaker", "TTSSpeaker"]