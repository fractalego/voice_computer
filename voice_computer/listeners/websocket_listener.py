"""
WebSocket-based voice listener that receives audio from WebSocket connections
instead of local microphone input.
"""

import asyncio
import logging
import numpy as np
from typing import Optional

from .base_listener import BaseListener, VoiceInterruptionException

_logger = logging.getLogger(__name__)


class WebSocketListener(BaseListener):
    """Voice listener that receives audio from WebSocket connections instead of microphone."""
    
    def __init__(self, config=None):
        super().__init__(config)

        self._range = 32768 # Range for int16 audio samples
        
        # WebSocket-specific audio buffer
        self.audio_buffer = []
        self.buffer_lock = asyncio.Lock()
        
        _logger.info("WebSocketListener initialized")
    
    def activate(self):
        """Activate the voice listener (no-op for server mode)."""
        self.is_active = True
        self.audio_buffer.clear()
        _logger.debug("WebSocketListener activated")
    
    def deactivate(self):
        """Deactivate the voice listener (no-op for server mode)."""
        self.is_active = False
        _logger.debug("WebSocketListener deactivated")
    
    async def add_audio_chunk(self, audio_data: bytes):
        """
        Add audio chunk received from WebSocket.
        
        Args:
            audio_data: Raw audio bytes (PCM format)
        """
        try:
            # Store raw bytes in buffer, just like MicrophoneListener does
            # The _rms() method will handle the conversion when needed
            async with self.buffer_lock:
                self.audio_buffer.extend(audio_data)
            
        except Exception as e:
            _logger.error(f"Error adding audio chunk: {e}")
    
    async def transcribe_accumulated_audio(self) -> Optional[str]:
        """
        Transcribe all accumulated audio in the buffer.
        
        Returns:
            Transcribed text or None
        """
        async with self.buffer_lock:
            if not self.audio_buffer:
                return None
            
            # Get all buffered audio
            audio_data = np.array(self.audio_buffer)
            self.audio_buffer.clear()
        
        # Transcribe using base class method
        return await self.transcribe_audio(audio_data)

    async def throw_exception_on_voice_activity(self):
        """Monitor for voice activity and throw exception when detected."""
        while True:
            await asyncio.sleep(0)
            async with self.buffer_lock:
                if not self.audio_buffer:
                    await asyncio.sleep(0.1)
                    continue
                rms: float = self._rms(bytes(self.audio_buffer))
                if rms > self.volume_threshold:
                    _logger.debug(f"Detected RMS: {rms}")
                    self.audio_buffer.clear()
                    raise VoiceInterruptionException(f"Voice activity detected in WebSocket audio buffer")
                self.audio_buffer.clear()
            await asyncio.sleep(0.1)

    def _get_input(self) -> bytes:
        frame = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        return frame


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]