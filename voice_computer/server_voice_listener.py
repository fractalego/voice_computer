"""
Server-based voice listener that receives audio from WebSocket connections
instead of local microphone input.
"""

import asyncio
import logging
import numpy as np
import time
import torch
from typing import Optional, List, Dict, Any
import base64

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from voice_computer.model_factory import get_model_factory
from voice_computer.sound_thresholds import calculate_rms

_logger = logging.getLogger(__name__)


class ServerVoiceListener:
    """Voice listener that receives audio from WebSocket connections instead of microphone."""
    
    def __init__(self, config=None):
        self.config = config
        self.is_active = False
        
        # Audio settings (matching client format)
        self.chunk = 1024
        self.channels = 1
        self.rate = 16000
        self.range = 32768
        
        # Voice activity detection settings
        self.timeout = 2
        self.volume_threshold = 0.6
        self.original_volume_threshold = self.volume_threshold
        self.max_timeout = 4
        self.hotword_threshold = -8  # logp threshold for hotword detection
        
        # Load configuration
        if config:
            listener_config = config.get_value("listener_model") or {}
            self.timeout = listener_config.get("listener_silence_timeout", 2)
            self.volume_threshold = listener_config.get("listener_volume_threshold", 0.6)
            self.original_volume_threshold = self.volume_threshold
            self.hotword_threshold = listener_config.get("listener_hotword_logp", -8)
            
        # Model initialization (will be done in initialize())
        self.model = None
        self.processor = None
        self.device = None
        self.whisper_model_name = None
        
        # Audio buffer for accumulating chunks
        self.audio_buffer = []
        self.buffer_lock = asyncio.Lock()
        
        # Activation words for hotword detection
        self.activation_hotwords = ["computer"]
        if config:
            self.activation_hotwords = config.get_value("activation_hotwords") or ["computer"]
            
        _logger.info("ServerVoiceListener initialized")
        
    def initialize(self):
        """Initialize the Whisper model and processor."""
        try:
            _logger.info("Initializing Whisper model for server...")
            
            # Get model configuration
            if self.config:
                self.whisper_model_name = self.config.get_value("whisper_model") or "fractalego/personal-whisper-distilled-model"
            else:
                self.whisper_model_name = "fractalego/personal-whisper-distilled-model"
                
            # Initialize device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _logger.info(f"Using device: {self.device}")
            
            # Load model and processor
            model_factory = get_model_factory()
            self.processor = WhisperProcessor.from_pretrained(self.whisper_model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.whisper_model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Enable optimizations
            if hasattr(self.model, 'half') and self.device.type == 'cuda':
                self.model = self.model.half()
                
            _logger.info(f"Whisper model '{self.whisper_model_name}' loaded successfully")
            self.is_active = True
            
        except Exception as e:
            _logger.error(f"Failed to initialize Whisper model: {e}")
            raise
            
    async def add_audio_chunk(self, audio_data: bytes):
        """Add an audio chunk from WebSocket to the buffer."""
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            async with self.buffer_lock:
                self.audio_buffer.extend(audio_float)
                
        except Exception as e:
            _logger.error(f"Error adding audio chunk: {e}")
            
    async def get_accumulated_audio(self, clear_buffer: bool = True) -> Optional[np.ndarray]:
        """Get accumulated audio from buffer."""
        async with self.buffer_lock:
            if not self.audio_buffer:
                return None
                
            audio_data = np.array(self.audio_buffer, dtype=np.float32)
            
            if clear_buffer:
                self.audio_buffer.clear()
                
            return audio_data
            
    async def transcribe_accumulated_audio(self) -> str:
        """Transcribe the accumulated audio buffer."""
        audio_data = await self.get_accumulated_audio(clear_buffer=True)
        
        if audio_data is None or len(audio_data) == 0:
            return ""
            
        # Ensure minimum length for transcription
        min_audio_length = self.rate * 0.5  # 0.5 seconds minimum
        if len(audio_data) < min_audio_length:
            _logger.debug("Audio too short for transcription")
            return ""
            
        return await self._transcribe_audio(audio_data)
        
    async def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        """Internal method to transcribe audio using Whisper."""
        if not self.is_active or self.model is None:
            _logger.warning("Whisper model not initialized")
            return ""
            
        try:
            # Ensure audio is the right format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            # Normalize audio to [-1, 1] if needed
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
                
            # Resample to 16kHz if needed (Whisper expects 16kHz)
            if len(audio_data) > 0:
                # Simple check - if we have exactly the expected samples for our duration, assume 16kHz
                expected_samples = int(len(audio_data) / self.rate * 16000)
                if len(audio_data) != expected_samples and abs(len(audio_data) - expected_samples) > 100:
                    # Basic resampling (for more complex cases, you'd use librosa)
                    from scipy import signal
                    if len(audio_data) > expected_samples:
                        # Downsample
                        audio_data = signal.resample(audio_data, expected_samples)
                    
            # Process with Whisper
            inputs = self.processor(
                audio_data, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            
            # Move inputs to device
            input_features = inputs.input_features.to(self.device)
            if hasattr(self.model, 'half') and self.device.type == 'cuda':
                input_features = input_features.half()
                
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=448,
                    num_beams=1,
                    do_sample=False,
                    language="english"
                )
                
            # Decode the transcription
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
            
            # Clean up the transcription
            transcription = transcription.strip()
            
            if transcription:
                _logger.debug(f"Transcribed: '{transcription}'")
                
            return transcription
            
        except Exception as e:
            _logger.error(f"Error during transcription: {e}")
            return ""
            
    async def detect_activation_words(self, text: str) -> Optional[str]:
        """Detect activation words in transcribed text."""
        if not text:
            return None
            
        text_lower = text.lower().strip()
        
        for hotword in self.activation_hotwords:
            if hotword.lower() in text_lower:
                _logger.info(f"Activation word '{hotword}' detected in: '{text}'")
                return hotword
                
        return None
        
    async def extract_command_after_activation(self, text: str, activation_word: str) -> str:
        """Extract command text that comes after the activation word."""
        if not text or not activation_word:
            return ""
            
        text_lower = text.lower().strip()
        activation_lower = activation_word.lower()
        
        # Find the activation word and extract what comes after
        if activation_lower in text_lower:
            parts = text_lower.split(activation_lower, 1)
            if len(parts) > 1:
                command = parts[1].strip()
                # Remove common filler words at the beginning
                filler_words = ['please', 'can you', 'could you', 'would you']
                for filler in filler_words:
                    if command.startswith(filler):
                        command = command[len(filler):].strip()
                return command
                
        return ""
        
    def clear_audio_buffer(self):
        """Clear the audio buffer."""
        asyncio.create_task(self._clear_buffer_async())
        
    async def _clear_buffer_async(self):
        """Async method to clear the buffer."""
        async with self.buffer_lock:
            self.audio_buffer.clear()
            
    def get_buffer_duration(self) -> float:
        """Get the current duration of audio in the buffer (in seconds)."""
        return len(self.audio_buffer) / self.rate if self.audio_buffer else 0.0
        
    async def wait_for_audio(self, min_duration: float = 2.0, max_wait: float = 10.0) -> bool:
        """Wait for sufficient audio to accumulate in buffer."""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if self.get_buffer_duration() >= min_duration:
                return True
            await asyncio.sleep(0.1)
            
        return False