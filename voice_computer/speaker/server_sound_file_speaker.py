"""
Server-based sound file speaker implementation that sends audio data via websocket.
"""

import logging
import wave
import base64
import asyncio
from typing import Optional, Callable, Any

from voice_computer.speaker.base_speaker import BaseSpeaker

_logger = logging.getLogger(__name__)


class ServerSoundFileSpeaker(BaseSpeaker):
    """Speaker implementation that sends sound file data to client via websocket."""
    
    def __init__(self, websocket_send_callback: Optional[Callable[[dict], Any]] = None):
        """
        Initialize the server sound file speaker.
        
        Args:
            websocket_send_callback: Callback function to send data via websocket
        """
        self.websocket_send_callback = websocket_send_callback
        self._initialized = False
        _logger.info("ServerSoundFileSpeaker initialized with callback: %s", websocket_send_callback is not None)

    def set_websocket_callback(self, callback: Callable[[dict], Any]) -> None:
        """
        Set the websocket callback after initialization.
        
        Args:
            callback: Function to call with websocket data
        """
        self.websocket_send_callback = callback
        _logger.debug("Websocket callback set for ServerSoundFileSpeaker")

    def initialize(self) -> None:
        """Initialize the speaker (called by conversation handler)."""
        self._initialized = True
        _logger.debug("ServerSoundFileSpeaker initialized")

    def speak(self, filename: str) -> None:
        """
        Send audio file data to client via websocket.
        
        Args:
            filename: Path to the audio file to send
        """
        _logger.info(f"ServerSoundFileSpeaker.speak() called with filename: {filename}")
        
        if not self.websocket_send_callback:
            _logger.error("No websocket callback set for ServerSoundFileSpeaker")
            return
            
        try:
            audio_data = self._read_audio_file(filename)
            
            # Send audio data via websocket
            message = {
                "type": "sound_file",
                "filename": filename,
                "data": audio_data["data"],
                "format": audio_data["format"],
                "sample_rate": audio_data["sample_rate"],
                "channels": audio_data["channels"]
            }
            
            # Handle async websocket callback
            result = self.websocket_send_callback(message)
            if asyncio.iscoroutine(result):
                # Run in event loop if we have one, otherwise create a task
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(result)
                except RuntimeError:
                    # No running loop, run sync
                    asyncio.run(result)
            
            _logger.info(f"Successfully sent sound file data for {filename} via websocket")
            
        except Exception as e:
            _logger.error(f"Failed to send sound file {filename}: {e}")
            raise

    def _read_audio_file(self, filename: str) -> dict:
        """
        Read audio file and return its data and metadata.
        
        Args:
            filename: Path to the audio file
            
        Returns:
            Dictionary containing audio data and metadata
        """
        try:
            with wave.open(filename, "rb") as wave_file:
                frames = wave_file.readframes(-1)  # Read all frames
                
                # Convert to base64 for JSON transmission
                audio_data_b64 = base64.b64encode(frames).decode('utf-8')
                
                return {
                    "data": audio_data_b64,
                    "format": wave_file.getsampwidth(),  # Sample width in bytes
                    "sample_rate": wave_file.getframerate(),
                    "channels": wave_file.getnchannels()
                }
                
        except FileNotFoundError:
            _logger.error(f"Sound file not found: {filename}")
            raise
        except wave.Error as e:
            _logger.error(f"Wave file error for {filename}: {e}")
            raise
        except Exception as e:
            _logger.error(f"Error reading sound file {filename}: {e}")
            raise

    def cancel_playback(self) -> None:
        """
        Cancel current playback (send cancellation message via websocket).
        """
        if not self.websocket_send_callback:
            _logger.warning("No websocket callback set for cancelling playback")
            return
            
        try:
            self.websocket_send_callback({
                "type": "cancel_sound",
                "action": "cancel"
            })
            _logger.debug("Sent sound cancellation via websocket")
            
        except Exception as e:
            _logger.error(f"Error sending sound cancellation: {e}")

    def cleanup(self) -> None:
        """Clean up resources."""
        self._initialized = False
        _logger.debug("ServerSoundFileSpeaker cleaned up")