"""
Base listener class with common functionality for all voice listeners.
"""

import logging
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from voice_computer.model_factory import get_model_factory
from voice_computer.listeners.sound_thresholds import calculate_rms

_logger = logging.getLogger(__name__)


class BaseListener(ABC):
    """Base class for voice listeners with common functionality."""
    
    def __init__(self, config=None):
        self.config = config
        self.is_active = False
        
        # Audio settings
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
            self.timeout = listener_config.get("listener_silence_timeout", 0.5)
            self.volume_threshold = listener_config.get("listener_volume_threshold", 0.6)
            self.original_volume_threshold = self.volume_threshold
            self.hotword_threshold = listener_config.get("listener_hotword_logp", -8)
        
        # Whisper model settings
        self.whisper_model_name = "fractalego/personal-whisper-distilled-model"
        if config:
            self.whisper_model_name = config.get_value("whisper_model") or self.whisper_model_name
        
        # Whisper components (loaded lazily)
        self.processor = None
        self.model = None
        self.device = None
        self.initialized = False
        self.hotwords = []
        self.last_audio = None
        
        # Whisper tokenizer tokens
        self._starting_tokens = None
        self._ending_tokens = None
        
        _logger.info(f"BaseListener initialized with model: {self.whisper_model_name}")
    
    def set_hotwords(self, hotwords: List[str]) -> None:
        """Set activation hotwords for detection."""
        if hotwords and not isinstance(hotwords, list):
            hotwords = [hotwords]
        
        if hotwords:
            self.hotwords = [word.lower() for word in hotwords]
            _logger.info(f"Set hotwords: {self.hotwords}")
    
    def add_hotwords(self, hotwords: List[str]) -> None:
        """Add hotwords for detection."""
        if hotwords and not isinstance(hotwords, list):
            hotwords = [hotwords]
        
        if hotwords:
            new_hotwords = [word.lower() for word in hotwords]
            self.hotwords.extend(new_hotwords)
            _logger.info(f"Added hotwords: {new_hotwords}")
    
    def initialize(self):
        """Initialize the Whisper model and processor."""
        try:
            _logger.info("Initializing Whisper model...")
            
            # Initialize device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _logger.info(f"Using device: {self.device}")
            
            # Load model and processor using model factory
            model_factory = get_model_factory()
            self.processor, self.model, self.device = model_factory.get_whisper_model(self.whisper_model_name, self.device)
            
            self.initialized = True
            _logger.info("Whisper model initialized successfully using model factory")
            
        except Exception as e:
            _logger.error(f"Failed to initialize Whisper model {self.whisper_model_name}: {e}")
            raise
    
    def _rms(self, frame: bytes) -> float:
        """Calculate RMS of audio data."""
        if len(frame) == 0:
            return 0.0
        
        return calculate_rms(frame)
    
    async def detect_activation_words(self, text: str) -> Optional[str]:
        """
        Check if text contains any activation words.
        
        Args:
            text: Transcribed text to check
            
        Returns:
            The detected activation word or None
        """
        if not text or not self.hotwords:
            return None
            
        text_lower = text.lower()
        for hotword in self.hotwords:
            if hotword in text_lower:
                _logger.info(f"Activation word '{hotword}' detected in: '{text}'")
                return hotword
        
        return None
    
    async def extract_command_after_activation(self, text: str, activation_word: str) -> Optional[str]:
        """
        Extract command text that comes after the activation word.
        
        Args:
            text: Full transcribed text
            activation_word: The detected activation word
            
        Returns:
            Command text after activation word, or None if no command
        """
        if not text or not activation_word:
            return None
            
        text_lower = text.lower()
        activation_lower = activation_word.lower()
        
        # Find the activation word position
        activation_pos = text_lower.find(activation_lower)
        if activation_pos == -1:
            return None
        
        # Extract everything after the activation word
        command_start = activation_pos + len(activation_lower)
        command = text[command_start:].strip()
        
        # Remove common punctuation at the end
        command = command.rstrip('.,!?;')
        
        return command if command else None
    
    async def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Transcribe audio data using Whisper model.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Transcribed text or None if transcription failed
        """
        if not self.initialized:
            self.initialize()
        
        try:
            # Process with Whisper processor
            inputs = self.processor(
                audio_data,
                sampling_rate=self.rate,
                return_tensors="pt"
            )
            
            # Move inputs to device and match model dtype
            input_features = inputs.input_features.to(self.device)
            
            # Ensure input dtype matches model dtype
            if self.device.type == 'cuda':
                # Get the model's dtype from its parameters
                model_dtype = next(self.model.parameters()).dtype
                input_features = input_features.to(dtype=model_dtype)
                
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    max_length=448,
                    num_beams=1,
                    do_sample=False
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
            else:
                return None
                
        except Exception as e:
            _logger.error(f"Error during transcription: {e}")
            return None
    
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def activate(self):
        """Activate the audio input."""
        pass
    
    @abstractmethod
    def deactivate(self):
        """Deactivate the audio input."""
        pass
    
    @abstractmethod
    async def listen_for_audio(self, timeout_seconds: float = None) -> Tuple[Optional[np.ndarray], bool]:
        """
        Listen for audio input.
        
        Args:
            timeout_seconds: Maximum time to listen
            
        Returns:
            Tuple of (audio_data, voice_detected)
        """
        pass
    
    @abstractmethod
    async def throw_exception_on_voice_activity(self):
        """Monitor for voice activity and throw exception when detected."""
        pass
    
    async def input(self) -> str:
        """
        Listen for audio input and return transcribed text.
        Common implementation that uses listen_for_audio() and transcribe_audio().
        
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
            _logger.error(f"Error in {self.__class__.__name__}.input(): {e}")
            return ""


class VoiceInterruptionException(Exception):
    """Exception raised when voice input is interrupted."""

    def __init__(self, message: str = "Voice input interrupted"):
        super().__init__(message)
        _logger.warning(message)