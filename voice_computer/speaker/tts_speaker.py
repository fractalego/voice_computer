"""
TTS Speaker implementation using microsoft/speecht5_tts with streaming support.
"""

import asyncio
import logging
import time

import torch
import numpy as np
import pyaudio

from typing import Optional

from .base_speaker import BaseSpeaker
from .number_conversion_to_words import convert_numbers_to_words
from ..speaker_embeddings import get_default_speaker_embedding
from ..model_factory import get_model_factory

_logger = logging.getLogger(__name__)


class TTSSpeaker(BaseSpeaker):
    """TTS Speaker implementation with streaming text-to-speech capability."""
    
    def __init__(self, device: Optional[str] = None, model_name: str = "microsoft/speecht5_tts", config=None):
        """
        Initialize the TTS speaker.
        
        Args:
            device: Device to run the model on ('cpu', 'cuda', 'mps', or None for auto-detect)
            model_name: Hugging Face model name for TTS
            config: Configuration object for sound threshold settings
        """
        self.model_name = model_name
        self.device = device or self._get_best_device()
        self.config = config
        self.initialized = False
        self._processor = None
        self._model = None
        self._vocoder = None
        self._speaker_embedding = None
        
        # Audio playback setup
        self._pyaudio = None
        self._sample_rate = 16000  # Default sample rate for SpeechT5
        self._shaved_float_margin = 512
        self._audio_queue = []
        

        _logger.info(f"TTSSpeaker created with model {model_name} on device {self.device}")
    
    def _get_best_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _load_speaker_embedding(self):
        """Load speaker embedding for SpeechT5."""
        try:
            # Try to load from HuggingFace datasets (standard approach)
            from datasets import load_dataset
            
            embeddings_dataset = load_dataset("Matthijs/cmu_arctic_x_vectors_speecht5", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
            
            # Ensure correct dtype and device
            speaker_embeddings = speaker_embeddings.to(dtype=self.torch_dtype, device=self.device)
            _logger.info("Loaded speaker embeddings from CMU Arctic dataset")
            return speaker_embeddings
            
        except Exception as e:
            _logger.warning(f"Failed to load speaker embeddings from dataset: {e}")
            
            # Fallback to our custom embedding
            try:
                embedding = get_default_speaker_embedding()
                embedding = embedding.to(dtype=self.torch_dtype, device=self.device)
                _logger.info("Using fallback speaker embedding")
                return embedding
            except Exception as e2:
                _logger.error(f"Failed to load fallback embedding: {e2}")
                
                # Last resort: create a random embedding
                embedding = torch.randn(1, 512, dtype=self.torch_dtype, device=self.device) * 0.01
                _logger.warning("Using random speaker embedding as last resort")
                return embedding
    
    def initialize(self):
        """Initialize the TTS model and audio system using model factory."""
        if self.initialized:
            return
            
        _logger.info(f"Initializing TTS model {self.model_name} on device {self.device}")
        
        try:
            # Get cached model from factory
            model_factory = get_model_factory()
            self._processor, self._model, self._vocoder, self.device = model_factory.get_tts_model(
                self.model_name, self.device
            )
            
            # Set torch dtype for speaker embedding consistency
            self.torch_dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
            
            # Get speaker embedding using standard approach
            self._speaker_embedding = self._load_speaker_embedding()
            
            # Initialize PyAudio for playback
            self._pyaudio = pyaudio.PyAudio()
            
            self.initialized = True
            _logger.info(f"TTS model {self.model_name} initialized successfully using model factory")
            
        except Exception as e:
            _logger.error(f"Failed to initialize TTS model {self.model_name}: {e}")
            raise
    
    def speak(self, text: str) -> None:
        """
        Synthesize and play speech from text (non-streaming).
        
        Args:
            text: Text to synthesize and speak
        """
        if not self.initialized:
            self.initialize()

        text = convert_numbers_to_words(text)

        try:
            with torch.no_grad():
                # Process text input
                inputs = self._processor(text=text, return_tensors="pt")
                
                # Move inputs to device
                input_ids = inputs["input_ids"].to(self.device)
                
                # Generate speech
                speech = self._model.generate_speech(input_ids, self._speaker_embedding, vocoder=self._vocoder)
                
                # Play the audio (speech is already a tensor)
                self._play_audio(speech.cpu(), self._sample_rate)
                
        except Exception as e:
            _logger.error(f"Error in TTS synthesis: {e}")
            _logger.debug(f"Model dtype: {getattr(self, 'torch_dtype', 'unknown')}")
            _logger.debug(f"Speaker embedding dtype: {self._speaker_embedding.dtype if self._speaker_embedding is not None else 'None'}")
            raise
    
    def _play_audio(self, audio_data, sample_rate: int):
        """Play audio data using PyAudio."""
        try:
            # Convert torch tensor to numpy if needed
            if torch.is_tensor(audio_data):
                audio_data = audio_data.cpu().float().numpy()
            
            # Ensure correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                output=True
            )
            
            # Play audio
            stream.write(audio_data.tobytes())
            
            # Clean up
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            _logger.error(f"Error playing audio: {e}")

    def add_text_batch(self, batch_text):
        """
        Add a batch of text to be synthesized.

        Args:
            batch_text: Text to synthesize
        """
        try:
            batch_text = convert_numbers_to_words(batch_text)
            # Process text input
            if not batch_text.endswith(('.', '!', '?', '...')):
                batch_text += " ..."
            inputs = self._processor(text=batch_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            speech = self._model.generate_speech(input_ids, self._speaker_embedding, vocoder=self._vocoder)
            self._audio_queue.append((speech.cpu()[:-self._shaved_float_margin], batch_text))

        except Exception as e:
            _logger.error(f"Error in TTS synthesis for batch: {e}")
            raise

    async def speak_batch(self):
        while self._audio_queue:
            audio_data, text = self._audio_queue.pop(0)
            try:
                self._play_audio(audio_data, self._sample_rate)
                # Allow other async tasks to run during audio playback
                await asyncio.sleep(0.01)
                    
            except Exception as e:
                _logger.error(f"Error playing audio batch: {e}")
                raise
        await asyncio.sleep(0.2)

        # Clear the queue after processing
        self._audio_queue.clear()
    
    def cleanup(self):
        """Clean up TTS speaker resources."""
        # Clean up PyAudio
        if self._pyaudio:
            try:
                self._pyaudio.terminate()
                self._pyaudio = None
            except Exception as e:
                _logger.debug(f"Error terminating PyAudio: {e}")
        
        # Clear audio queue
        self._audio_queue.clear()
        
        _logger.debug("TTSSpeaker cleanup completed")


