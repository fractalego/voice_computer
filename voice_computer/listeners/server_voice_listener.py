"""
Server-based voice listener that receives audio from WebSocket connections
instead of local microphone input.
"""

import asyncio
import logging
import numpy as np
import time
from typing import Optional, List, Dict, Any, Tuple

from .base_listener import BaseListener

_logger = logging.getLogger(__name__)


class ServerVoiceListener(BaseListener):
    """Voice listener that receives audio from WebSocket connections instead of microphone."""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # WebSocket-specific audio buffer
        self.audio_buffer = []
        self.buffer_lock = asyncio.Lock()
        
        _logger.info("ServerVoiceListener initialized")
    
    def activate(self):
        """Activate the voice listener (no-op for server mode)."""
        self.is_active = True
        _logger.debug("ServerVoiceListener activated")
    
    def deactivate(self):
        """Deactivate the voice listener (no-op for server mode)."""
        self.is_active = False
        _logger.debug("ServerVoiceListener deactivated")
    
    async def add_audio_chunk(self, audio_data: bytes):
        """
        Add audio chunk received from WebSocket.
        
        Args:
            audio_data: Raw audio bytes (PCM format)
        """
        try:
            # Convert bytes to numpy array (assuming int16 PCM from client)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            async with self.buffer_lock:
                self.audio_buffer.extend(audio_float)
            
        except Exception as e:
            _logger.error(f"Error adding audio chunk: {e}")
    
    async def listen_for_audio(self, timeout_seconds: float = None) -> Tuple[Optional[np.ndarray], bool]:
        """
        Listen for audio input from WebSocket buffer.
        
        Args:
            timeout_seconds: Maximum time to listen
            
        Returns:
            Tuple of (audio_data, voice_detected)
        """
        if timeout_seconds is None:
            timeout_seconds = self.timeout
        
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            async with self.buffer_lock:
                if len(self.audio_buffer) > self.rate * 2:  # At least 2 seconds of audio
                    # Extract audio from buffer
                    audio_data = np.array(self.audio_buffer)
                    self.audio_buffer.clear()
                    
                    # Check if there's voice activity
                    rms = np.sqrt(np.mean(audio_data ** 2))
                    voice_detected = rms > self.volume_threshold
                    
                    return audio_data, voice_detected
            
            await asyncio.sleep(0.1)  # Check every 100ms
        
        # Timeout reached
        async with self.buffer_lock:
            if self.audio_buffer:
                audio_data = np.array(self.audio_buffer)
                self.audio_buffer.clear()
                rms = np.sqrt(np.mean(audio_data ** 2))
                voice_detected = rms > self.volume_threshold
                return audio_data, voice_detected
        
        return None, False
    
    async def throw_exception_on_voice_activity(self):
        """Monitor for voice activity and throw exception when detected."""
        # In server mode, this is handled differently - WebSocket stream is continuous
        # Just wait forever to avoid interrupting streaming responses
        await asyncio.Future()  # Wait forever
    
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
        
        if len(audio_data) < self.rate * 0.5:  # Less than 0.5 seconds
            return None
        
        # Transcribe using base class method
        return await self.transcribe_audio(audio_data)
    
    async def input(self) -> str:
        """
        Listen for audio input from WebSocket buffer and return transcribed text.
        Uses the same logic and thresholds as WhisperListener.
        
        Returns:
            Transcribed text or empty string if no speech detected
        """
        try:
            # Listen for audio with voice activity detection
            audio_data, voice_detected = await self.listen_for_audio()
            
            if not voice_detected or audio_data is None:
                return ""
            
            # Transcribe the audio
            text = await self.transcribe_audio(audio_data)
            return text if text else ""
            
        except Exception as e:
            _logger.error(f"Error in ServerVoiceListener.input(): {e}")
            return ""