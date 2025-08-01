"""
TTS Speaker implementation using microsoft/speecht5_tts with streaming support.
"""

import asyncio
import logging
import queue
import threading
import time
import torch
import numpy as np
from typing import Optional, Callable
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import pyaudio

from .base_speaker import BaseSpeaker
from ..speaker_embeddings import get_default_speaker_embedding

_logger = logging.getLogger(__name__)


class TTSSpeaker(BaseSpeaker):
    """TTS Speaker implementation with streaming text-to-speech capability."""
    
    def __init__(self, device: Optional[str] = None, model_name: str = "microsoft/speecht5_tts"):
        """
        Initialize the TTS speaker.
        
        Args:
            device: Device to run the model on ('cpu', 'cuda', 'mps', or None for auto-detect)
            model_name: Hugging Face model name for TTS
        """
        self.model_name = model_name
        self.device = device or self._get_best_device()
        self.initialized = False
        self._processor = None
        self._model = None
        self._vocoder = None
        self._speaker_embedding = None
        
        # Audio playback setup
        self._pyaudio = None
        self._sample_rate = 16000  # Default sample rate for SpeechT5
        
        # Streaming setup
        self._text_queue = queue.Queue()
        self._audio_queue = queue.Queue()
        self._streaming_thread = None
        self._playback_thread = None
        self._stop_streaming = threading.Event()
        self._is_streaming = False
        
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
        """Initialize the TTS model and audio system."""
        if self.initialized:
            return
            
        _logger.info(f"Loading TTS model {self.model_name} on device {self.device}")
        
        try:
            # Initialize SpeechT5 components
            self.torch_dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
            
            # Load processor, model, and vocoder
            self._processor = SpeechT5Processor.from_pretrained(self.model_name)
            self._model = SpeechT5ForTextToSpeech.from_pretrained(self.model_name)
            
            # Load HiFi-GAN vocoder for SpeechT5
            self._vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Move models to device and set dtype
            self._model = self._model.to(self.device)
            self._vocoder = self._vocoder.to(self.device)
            
            if self.device != "cpu":
                self._model = self._model.to(dtype=self.torch_dtype)
                self._vocoder = self._vocoder.to(dtype=self.torch_dtype)
            
            self._model.eval()
            self._vocoder.eval()
            
            # Get speaker embedding using standard approach
            self._speaker_embedding = self._load_speaker_embedding()
            
            # Initialize PyAudio for playback
            self._pyaudio = pyaudio.PyAudio()
            
            self.initialized = True
            _logger.info(f"TTS model {self.model_name} loaded successfully on {self.device}")
            
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
        
        try:
            with torch.no_grad():
                _logger.debug(f"Speaker embedding shape: {self._speaker_embedding.shape}, dtype: {self._speaker_embedding.dtype}")
                
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
    
    def start_streaming_speech(self, text_callback: Optional[Callable[[str], None]] = None):
        """
        Start streaming TTS mode to process text batches as they arrive.
        
        Args:
            text_callback: Optional callback to receive processed text batches
        """
        if not self.initialized:
            self.initialize()
        
        if self._is_streaming:
            _logger.warning("Streaming already active")
            return
        
        self._is_streaming = True
        self._stop_streaming.clear()
        
        # Start processing threads
        self._streaming_thread = threading.Thread(
            target=self._streaming_worker, 
            args=(text_callback,)
        )
        self._playback_thread = threading.Thread(target=self._playback_worker)
        
        self._streaming_thread.start()
        self._playback_thread.start()
        
        _logger.info("Streaming TTS started")
    
    def add_text_batch(self, text_batch: str):
        """
        Add a text batch to the streaming TTS queue.
        
        Args:
            text_batch: Batch of text to synthesize
        """
        if not self._is_streaming:
            _logger.warning("Streaming not active, call start_streaming_speech() first")
            return
        
        if text_batch.strip():  # Only add non-empty batches
            self._text_queue.put(text_batch)
    
    def stop_streaming_speech(self):
        """Stop streaming TTS mode."""
        if not self._is_streaming:
            return
        
        self._is_streaming = False
        self._stop_streaming.set()
        
        # Signal end of text stream
        self._text_queue.put(None)
        self._audio_queue.put(None)
        
        # Wait for threads to finish
        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=5.0)
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=5.0)
        
        _logger.info("Streaming TTS stopped")
    
    def _streaming_worker(self, text_callback: Optional[Callable[[str], None]]):
        """Worker thread for processing streaming text batches."""
        _logger.debug("Streaming worker started")
        
        accumulated_text = ""
        sentence_endings = ['.', '!', '?', '\n']
        
        while not self._stop_streaming.is_set():
            try:
                # Get text batch with timeout
                text_batch = self._text_queue.get(timeout=0.1)
                
                if text_batch is None:  # End of stream signal
                    break
                
                accumulated_text += text_batch
                
                # Call text callback if provided
                if text_callback:
                    text_callback(text_batch)
                
                # Check if we have complete sentences to synthesize
                sentences_to_synthesize = []
                
                # Look for sentence endings
                for ending in sentence_endings:
                    if ending in accumulated_text:
                        parts = accumulated_text.split(ending)
                        # Process all complete sentences
                        for i in range(len(parts) - 1):
                            sentence = parts[i] + ending
                            if sentence.strip():
                                sentences_to_synthesize.append(sentence.strip())
                        
                        # Keep the last incomplete part
                        accumulated_text = parts[-1]
                        break
                
                # Also synthesize if we have enough accumulated text (word boundary)
                if len(accumulated_text) > 100:  # Adjust threshold as needed
                    words = accumulated_text.split()
                    if len(words) > 10:  # Ensure we have complete words
                        # Take most words, leave a few for context
                        words_to_synthesize = words[:-2]
                        sentence = ' '.join(words_to_synthesize)
                        if sentence.strip():
                            sentences_to_synthesize.append(sentence.strip())
                        accumulated_text = ' '.join(words[-2:])
                
                # Synthesize sentences
                for sentence in sentences_to_synthesize:
                    self._synthesize_sentence(sentence)
                    
            except queue.Empty:
                continue
            except Exception as e:
                _logger.error(f"Error in streaming worker: {e}")
        
        # Process any remaining text
        if accumulated_text.strip():
            self._synthesize_sentence(accumulated_text.strip())
        
        _logger.debug("Streaming worker finished")
    
    def _synthesize_sentence(self, text: str):
        """Synthesize a single sentence and queue audio."""
        try:
            with torch.no_grad():
                # Debug logging for dtype issues
                _logger.debug(f"Synthesizing: '{text[:50]}...'")
                _logger.debug(f"Speaker embedding shape: {self._speaker_embedding.shape}, dtype: {self._speaker_embedding.dtype}")
                
                # Process text input
                inputs = self._processor(text=text, return_tensors="pt")
                
                # Move inputs to device
                input_ids = inputs["input_ids"].to(self.device)
                
                # Generate speech
                speech = self._model.generate_speech(input_ids, self._speaker_embedding, vocoder=self._vocoder)
                
                # Queue audio for playback
                self._audio_queue.put({
                    "audio": speech.cpu(),
                    "sample_rate": self._sample_rate,
                    "text": text
                })
                
        except Exception as e:
            _logger.error(f"Error synthesizing sentence '{text}': {e}")
            _logger.debug(f"Model dtype: {getattr(self, 'torch_dtype', 'unknown')}")
            _logger.debug(f"Speaker embedding dtype: {self._speaker_embedding.dtype if self._speaker_embedding is not None else 'None'}")
    
    def _playback_worker(self):
        """Worker thread for playing back synthesized audio."""
        _logger.debug("Playback worker started")
        
        stream = None
        
        try:
            while not self._stop_streaming.is_set():
                try:
                    # Get audio with timeout
                    audio_data = self._audio_queue.get(timeout=0.1)
                    
                    if audio_data is None:  # End of stream signal
                        break
                    
                    # Play the audio
                    self._play_audio(audio_data["audio"], audio_data["sample_rate"])
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    _logger.error(f"Error in playback worker: {e}")
                    
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
        
        _logger.debug("Playback worker finished")
    
    def _play_audio(self, audio_data, sample_rate: int):
        """Play audio data using PyAudio."""
        try:
            # Convert torch tensor to numpy if needed
            if torch.is_tensor(audio_data):
                audio_data = audio_data.cpu().float().numpy()
            
            # Ensure correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Open stream for this audio chunk
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
    
    def cleanup(self):
        """Clean up resources."""
        # Stop streaming if active
        if self._is_streaming:
            self.stop_streaming_speech()
        
        # Clean up PyAudio
        if self._pyaudio is not None:
            try:
                self._pyaudio.terminate()
                _logger.debug("PyAudio terminated successfully")
            except Exception as e:
                _logger.debug(f"Error terminating PyAudio: {e}")
            finally:
                self._pyaudio = None
        
        # Clean up model resources
        for model_name, model in [("processor", self._processor), ("model", self._model), ("vocoder", self._vocoder)]:
            if model is not None:
                try:
                    del model
                    setattr(self, f"_{model_name}", None)
                except Exception as e:
                    _logger.debug(f"Error cleaning up {model_name}: {e}")
        
        # Clear CUDA cache if using GPU
        if self.device != "cpu" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass