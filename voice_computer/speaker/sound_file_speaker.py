"""
Sound file speaker implementation using pyaudio and wave.
"""

import logging
import pyaudio
import wave
from typing import Optional

from .base_speaker import BaseSpeaker

_logger = logging.getLogger(__name__)


class SoundFileSpeaker(BaseSpeaker):
    """Speaker implementation for playing sound files using pyaudio and wave."""
    
    _chunk = 1024

    def __init__(self):
        """Initialize the sound file speaker."""
        try:
            self._p = pyaudio.PyAudio()
            _logger.debug("PyAudio initialized successfully")
        except Exception as e:
            _logger.error(f"Failed to initialize PyAudio: {e}")
            raise

    def speak(self, filename: str) -> None:
        """
        Play audio from a sound file.
        
        Args:
            filename: Path to the audio file to play
        """
        try:
            self._play_sound(filename)
        except Exception as e:
            _logger.error(f"Failed to play sound file {filename}: {e}")
            raise

    def _play_sound(self, sound_filename: str) -> None:
        """
        Internal method to play a sound file.
        
        Args:
            sound_filename: Path to the sound file to play
        """
        stream = None
        f = None
        
        try:
            # Open the wave file
            f = wave.open(sound_filename, "rb")
            
            # Open PyAudio stream
            stream = self._p.open(
                format=self._p.get_format_from_width(f.getsampwidth()),
                channels=f.getnchannels(),
                rate=f.getframerate(),
                output=True,
            )
            
            # Read and play audio data
            data = f.readframes(self._chunk)
            while data:
                stream.write(data)
                data = f.readframes(self._chunk)
                
        except FileNotFoundError:
            _logger.error(f"Sound file not found: {sound_filename}")
            raise
        except wave.Error as e:
            _logger.error(f"Wave file error for {sound_filename}: {e}")
            raise
        except Exception as e:
            _logger.error(f"Error playing sound {sound_filename}: {e}")
            raise
        finally:
            # Clean up resources
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception as e:
                    _logger.debug(f"Error closing stream: {e}")
            
            if f is not None:
                try:
                    f.close()
                except Exception as e:
                    _logger.debug(f"Error closing wave file: {e}")

    def cleanup(self) -> None:
        """Clean up PyAudio resources."""
        if hasattr(self, '_p') and self._p is not None:
            try:
                self._p.terminate()
                _logger.debug("PyAudio terminated successfully")
            except Exception as e:
                _logger.debug(f"Error terminating PyAudio: {e}")
            finally:
                self._p = None