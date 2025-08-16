"""
Server TTS Speaker implementation that sends audio data over WebSocket instead of local playback.
"""

import asyncio
import logging
import time
import json
import base64

import torch
import numpy as np

from typing import Optional, Callable, Any, Dict

from .base_speaker import BaseSpeaker
from .number_conversion_to_words import convert_numbers_to_words
from ..speaker_embeddings import get_default_speaker_embedding
from ..model_factory import get_model_factory

_logger = logging.getLogger(__name__)


class ServerTTSSpeaker(BaseSpeaker):
    """TTS Speaker implementation that sends audio over WebSocket instead of local playback."""
    
    def __init__(self, websocket_send_callback: Callable[[Dict[str, Any]], None], 
                 device: Optional[str] = None, model_name: str = "microsoft/speecht5_tts", config=None):
        """
        Initialize the server TTS speaker.
        
        Args:
            websocket_send_callback: Async function to send messages over WebSocket
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
        
        # WebSocket communication
        self.websocket_send_callback = websocket_send_callback
        
        # Audio settings
        self._sample_rate = 16000  # Default sample rate for SpeechT5
        self._shaved_float_margin = 512
        self._audio_queue = []
        
        # Playback control
        self._playback_cancelled = False
        
        _logger.info(f"ServerTTSSpeaker created with model {model_name} on device {self.device}")
    
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
        embedding = get_default_speaker_embedding()
        embedding = embedding.to(dtype=self.torch_dtype, device=self.device)
        _logger.info("Using fallback speaker embedding")
        return embedding
    
    def initialize(self):
        """Initialize the TTS model using model factory."""
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
            
            self.initialized = True
            _logger.info(f"TTS model {self.model_name} initialized successfully using model factory")
            
        except Exception as e:
            _logger.error(f"Failed to initialize TTS model {self.model_name}: {e}")
            raise
    
    def speak(self, text: str) -> None:
        """
        Synthesize speech from text and send over WebSocket.
        
        Args:
            text: Text to synthesize and send
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
                
                # Send the audio over WebSocket
                asyncio.create_task(self._send_audio_over_websocket(speech.cpu(), text))
                
        except Exception as e:
            _logger.error(f"Error in TTS synthesis: {e}")
            _logger.debug(f"Model dtype: {getattr(self, 'torch_dtype', 'unknown')}")
            _logger.debug(f"Speaker embedding dtype: {self._speaker_embedding.dtype if self._speaker_embedding is not None else 'None'}")
            raise
    
    async def _send_audio_over_websocket(self, audio_data, text: str):
        """Send audio data over WebSocket connection."""
        try:
            # Convert torch tensor to numpy if needed
            if torch.is_tensor(audio_data):
                audio_data = audio_data.cpu().float().numpy()
            
            # Ensure correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Convert float32 audio to int16 for transmission efficiency
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Encode as base64 for JSON transmission
            audio_bytes = audio_int16.tobytes()
            encoded_audio = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Create WebSocket message
            message = {
                "type": "tts_audio",
                "audio_data": encoded_audio,
                "sample_rate": self._sample_rate,
                "channels": 1,
                "format": "pcm16",
                "text": text,
                "timestamp": time.time()
            }
            
            # Send via WebSocket callback
            if self.websocket_send_callback:
                await self.websocket_send_callback(message)
                _logger.debug(f"Sent TTS audio for text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            else:
                _logger.warning("No WebSocket callback available to send audio")
                
        except Exception as e:
            _logger.error(f"Error sending audio over WebSocket: {e}")
    
    def cancel_playback(self):
        """Cancel any ongoing audio synthesis/transmission."""
        self._playback_cancelled = True
        self._audio_queue.clear()
        
        # Send cancellation message over WebSocket
        if self.websocket_send_callback:
            cancel_message = {
                "type": "tts_cancel",
                "timestamp": time.time()
            }
            asyncio.create_task(self.websocket_send_callback(cancel_message))

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
        """Send all queued audio batches over WebSocket."""
        if not self._audio_queue:
            return
            
        # Reset cancellation flag
        self._playback_cancelled = False
        
        try:
            # Send all audio in queue
            while self._audio_queue and not self._playback_cancelled:
                audio_data, text = self._audio_queue.pop(0)
                await self._send_audio_over_websocket(audio_data, text)
                
                # Small delay between batches to prevent overwhelming the client
                await asyncio.sleep(0.01)
                
        except Exception as e:
            _logger.error(f"Error in batch audio transmission: {e}")
        finally:
            # Clear the queue after processing
            self._audio_queue.clear()
    
    async def send_text_message(self, text: str):
        """Send a text-only message over WebSocket (for immediate display before TTS)."""
        try:
            message = {
                "type": "text_response",
                "text": text,
                "timestamp": time.time()
            }
            
            if self.websocket_send_callback:
                await self.websocket_send_callback(message)
                _logger.debug(f"Sent text message: '{text[:100]}{'...' if len(text) > 100 else ''}'")
            else:
                _logger.warning("No WebSocket callback available to send text")
                
        except Exception as e:
            _logger.error(f"Error sending text message over WebSocket: {e}")
    
    async def send_status_message(self, status: str, details: Optional[Dict[str, Any]] = None):
        """Send a status message over WebSocket."""
        try:
            message = {
                "type": "tts_status",
                "status": status,
                "timestamp": time.time()
            }
            
            if details:
                message.update(details)
            
            if self.websocket_send_callback:
                await self.websocket_send_callback(message)
                _logger.debug(f"Sent status message: {status}")
            else:
                _logger.warning("No WebSocket callback available to send status")
                
        except Exception as e:
            _logger.error(f"Error sending status message over WebSocket: {e}")
    
    def update_websocket_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Update the WebSocket send callback."""
        self.websocket_send_callback = callback
        _logger.debug("WebSocket callback updated")
    
    def cleanup(self):
        """Clean up TTS speaker resources."""
        # Clear audio queue
        self._audio_queue.clear()
        
        # Send cleanup notification
        if self.websocket_send_callback:
            cleanup_message = {
                "type": "tts_cleanup",
                "timestamp": time.time()
            }
            try:
                asyncio.create_task(self.websocket_send_callback(cleanup_message))
            except Exception as e:
                _logger.debug(f"Error sending cleanup message: {e}")
        
        _logger.debug("ServerTTSSpeaker cleanup completed")