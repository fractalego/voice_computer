"""
WebSocket-based voice listener that receives audio from WebSocket connections
instead of local microphone input.
"""

import asyncio
import logging
import numpy as np
import time
from typing import Optional, List, Dict, Any, Tuple

from .base_listener import BaseListener, VoiceInterruptionException

_logger = logging.getLogger(__name__)


class WebSocketListener(BaseListener):
    """Voice listener that receives audio from WebSocket connections instead of microphone."""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # WebSocket-specific audio buffer
        self.audio_buffer = []
        self.buffer_lock = asyncio.Lock()
        
        _logger.info("WebSocketListener initialized")
    
    def activate(self):
        """Activate the voice listener (no-op for server mode)."""
        self.is_active = True
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
            # Store raw bytes in buffer, just like WhisperListener does
            # The _rms() method will handle the conversion when needed
            async with self.buffer_lock:
                buffer_size_before = len(self.audio_buffer)
                self.audio_buffer.extend(audio_data)
                buffer_size_after = len(self.audio_buffer)
                _logger.debug(f"Audio buffer: {buffer_size_before} -> {buffer_size_after} bytes ({len(audio_data)} bytes received)")
            
        except Exception as e:
            _logger.error(f"Error adding audio chunk: {e}")
    
    
    async def throw_exception_on_voice_activity(self):
        """Monitor for voice activity and throw exception when detected."""
        async with self.buffer_lock:
            if not self.audio_buffer:
                return
            audio_bytes = bytes(self.audio_buffer)
            rms = self._rms(audio_bytes)
            if rms > self.volume_threshold:
                _logger.debug(f"Voice activity detected with RMS={rms:.6f}, throwing exception")
                raise VoiceInterruptionException("Voice activity detected in WebSocket audio buffer")

    
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
    
    async def listen_for_audio(self, timeout_seconds: float = None) -> Tuple[Optional[np.ndarray], bool]:
        """
        Listen for audio input from WebSocket buffer.
        Similar to MicrophoneListener but uses buffered audio from WebSocket instead of microphone.
        
        Args:
            timeout_seconds: Maximum time to listen
            
        Returns:
            Tuple of (audio_data, voice_detected)
        """
        if timeout_seconds is None:
            timeout_seconds = self.timeout
        
        try:
            start_time = time.time()
            silence_start = None
            voice_detected = False
            
            while time.time() - start_time < timeout_seconds:
                async with self.buffer_lock:
                    # Check if we have enough bytes (0.5 seconds = rate * 0.5 samples * 2 bytes per sample)
                    bytes_needed = int(self.rate * 0.5 * 2)  # 2 bytes per int16 sample
                    if len(self.audio_buffer) >= bytes_needed:
                        # Get current buffer as bytes data
                        audio_bytes = bytes(self.audio_buffer)
                        self.audio_buffer.clear()
                        
                        # Convert bytes to numpy array (like MicrophoneListener processes frames)
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        audio_float = audio_array.astype(np.float32) / 32768.0
                        
                        # Check voice activity using the same _rms method as MicrophoneListener
                        rms = self._rms(audio_bytes)
                        if rms > self.volume_threshold:
                            voice_detected = True
                            silence_start = None
                        else:
                            if silence_start is None:
                                silence_start = time.time()
                            elif voice_detected and (time.time() - silence_start) > self.timeout:
                                # End of speech detected
                                break
                        
                        # If we have voice activity or enough audio, return it
                        if voice_detected or len(audio_bytes) >= self.rate * 1.0 * 2:  # 1 second minimum (in bytes)
                            _logger.debug(f"Returning {len(audio_float)} samples, RMS={rms:.6f}, voice_detected={voice_detected}")
                            return audio_float, voice_detected
                
                await asyncio.sleep(0.1)  # Check every 100ms
            
            # Timeout reached - return any buffered audio
            async with self.buffer_lock:
                if self.audio_buffer:
                    audio_bytes = bytes(self.audio_buffer)
                    self.audio_buffer.clear()
                    rms = self._rms(audio_bytes)
                    voice_detected = rms > self.volume_threshold
                    
                    # Convert to numpy array for return
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    audio_float = audio_array.astype(np.float32) / 32768.0
                    
                    _logger.debug(f"Timeout - returning {len(audio_float)} samples, RMS={rms:.6f}, voice_detected={voice_detected}")
                    return audio_float, voice_detected
            
            return None, False
                
        except Exception as e:
            _logger.error(f"Error in listen_for_audio: {e}")
            return None, False