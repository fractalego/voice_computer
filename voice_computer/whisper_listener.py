"""
Whisper-based speech recognition using local serving based on WhisperHandler.
"""

import asyncio
import logging
import numpy as np
import pyaudio
import time
import torch
from typing import Optional, List

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from .model_factory import get_model_factory
from .sound_thresholds import calculate_rms

_logger = logging.getLogger(__name__)


class WhisperListener:
    """Whisper listener using local serving with logp-based hotword detection."""
    
    def __init__(self, config=None):
        self.config = config
        self.is_active = False
        
        # Audio settings
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.range = 32768
        
        # Voice activity detection settings
        self.timeout = 2
        self.volume_threshold = 0.6  # Updated default
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
            self.device_index = listener_config.get("microphone_device_index", None)
        else:
            self.device_index = None
        
        # Whisper model settings
        self.whisper_model_name = "fractalego/personal-whisper-distilled-model"
        if config:
            self.whisper_model_name = config.get_value("whisper_model") or self.whisper_model_name
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
        
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
        
        _logger.info(f"WhisperListener initialized with model: {self.whisper_model_name}")
        
        # Debug: List available audio devices in debug mode
        self._log_available_audio_devices()
    
    def test_microphone(self, duration_seconds: int = 5) -> dict:
        """
        Test microphone input for a specified duration and return statistics.
        
        Args:
            duration_seconds: How long to test for
            
        Returns:
            Dictionary with microphone test results
        """
        if not self.is_active:
            self.activate()
        
        _logger.info(f"Testing microphone for {duration_seconds} seconds...")
        samples = []
        rms_values = []
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            try:
                inp = self.stream.read(self.chunk, exception_on_overflow=False)
                rms_val = self._rms(inp)
                samples.append(inp)
                rms_values.append(rms_val)
                time.sleep(0.01)  # Small delay
            except Exception as e:
                _logger.error(f"Error during microphone test: {e}")
                break
        
        if rms_values:
            results = {
                "duration": time.time() - start_time,
                "samples_collected": len(samples),
                "min_rms": min(rms_values),
                "max_rms": max(rms_values),
                "avg_rms": sum(rms_values) / len(rms_values),
                "current_threshold": self.volume_threshold,
                "samples_above_threshold": sum(1 for rms in rms_values if rms > self.volume_threshold),
                "recommended_threshold": max(rms_values) * 0.3  # 30% of max volume
            }
            
            _logger.info(f"Microphone test results:")
            _logger.info(f"  Duration: {results['duration']:.1f}s")
            _logger.info(f"  RMS range: {results['min_rms']:.6f} - {results['max_rms']:.6f}")
            _logger.info(f"  Average RMS: {results['avg_rms']:.6f}")
            _logger.info(f"  Current threshold: {results['current_threshold']:.6f}")
            _logger.info(f"  Samples above threshold: {results['samples_above_threshold']}/{len(rms_values)}")
            _logger.info(f"  Recommended threshold: {results['recommended_threshold']:.6f}")
            
            return results
        else:
            return {"error": "No samples collected"}

    def _log_available_audio_devices(self):
        """Log available audio input devices for debugging."""
        try:
            device_count = self.p.get_device_count()
            _logger.debug(f"Available audio devices ({device_count} total):")
            
            for i in range(device_count):
                info = self.p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:  # Only input devices
                    _logger.debug(f"  Input Device {i}: {info['name']} (channels: {info['maxInputChannels']}, sample rate: {info['defaultSampleRate']})")
                    
            default_input = self.p.get_default_input_device_info()
            _logger.debug(f"Default input device: {default_input['name']} (index: {default_input['index']})")
            
        except Exception as e:
            _logger.debug(f"Error listing audio devices: {e}")

    def initialize_whisper(self):
        """Initialize the Whisper model and processor using model factory."""
        if self.initialized:
            return
        
        try:
            _logger.info(f"Initializing Whisper model: {self.whisper_model_name}")
            
            # Get cached model from factory
            model_factory = get_model_factory()
            self.processor, self.model, self.device = model_factory.get_whisper_model(
                self.whisper_model_name
            )
            
            _logger.info(f"Using device: {self.device}")
            
            # Setup tokenizer tokens
            self._starting_tokens = self.processor.tokenizer.convert_tokens_to_ids(
                ["<|startoftranscript|>", "<|notimestamps|>"]
            )
            self._ending_tokens = self.processor.tokenizer.convert_tokens_to_ids(
                ["<|endoftext|>"]
            )
            
            self.initialized = True
            _logger.info("Whisper model initialized successfully using model factory")
            
        except Exception as e:
            _logger.error(f"Failed to initialize Whisper model: {e}")
            raise RuntimeError(f"Could not initialize Whisper model '{self.whisper_model_name}': {e}")

    def set_timeout(self, timeout: float):
        """Set the silence timeout."""
        self.timeout = timeout

    def set_volume_threshold(self, threshold: float):
        """Set the volume threshold."""
        self.volume_threshold = threshold
        self.original_volume_threshold = threshold

    def set_hotword_threshold(self, threshold: float):
        """Set the hotword detection threshold (logp)."""
        self.hotword_threshold = threshold

    def set_hotwords(self, hotwords):
        """Set hotwords for detection."""
        if hotwords:
            self.hotwords = [word.lower() for word in hotwords]
            _logger.info(f"Set hotwords: {self.hotwords}")

    def add_hotwords(self, hotwords):
        """Add hotwords for detection."""
        if hotwords and not isinstance(hotwords, list):
            hotwords = [hotwords]
        
        if hotwords:
            new_hotwords = [word.lower() for word in hotwords]
            self.hotwords.extend(new_hotwords)
            _logger.info(f"Added hotwords: {new_hotwords}")

    def activate(self):
        """Activate the audio stream."""
        if not self.is_active:
            try:
                # Get device info for debugging
                if self.device_index is not None:
                    try:
                        device_info = self.p.get_device_info_by_index(self.device_index)
                        _logger.debug(f"Using configured microphone device: {device_info['name']} (index: {self.device_index}, channels: {device_info['maxInputChannels']}, sample rate: {device_info['defaultSampleRate']})")
                    except Exception as e:
                        _logger.warning(f"Error getting info for configured device {self.device_index}: {e}. Falling back to default.")
                        self.device_index = None
                
                if self.device_index is None:
                    default_input_device = self.p.get_default_input_device_info()
                    _logger.debug(f"Using default microphone device: {default_input_device['name']} (index: {default_input_device['index']}, channels: {default_input_device['maxInputChannels']}, sample rate: {default_input_device['defaultSampleRate']})")
                
                self.stream = self.p.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.rate,
                    input=True,
                    output=False,
                    frames_per_buffer=self.chunk,
                    input_device_index=self.device_index,  # Use configured device or None for default
                )
                self.is_active = True
                _logger.debug(f"WhisperListener audio stream activated - format: {self.format}, channels: {self.channels}, rate: {self.rate}, chunk: {self.chunk}")
                
                # Test read a small chunk to make sure the stream is working
                try:
                    test_chunk = self.stream.read(self.chunk, exception_on_overflow=False)
                    test_rms = self._rms(test_chunk)
                    _logger.debug(f"Stream test successful - read {len(test_chunk) if test_chunk else 0} bytes, RMS: {test_rms:.6f}")
                except Exception as e:
                    _logger.warning(f"Stream test failed: {e}")
            except Exception as e:
                _logger.error(f"Failed to activate audio stream: {e}")
                raise

    def deactivate(self):
        """Deactivate the audio stream."""
        if self.is_active and self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
                self.is_active = False
                _logger.debug("WhisperListener audio stream deactivated")
            except Exception as e:
                _logger.error(f"Error deactivating audio stream: {e}")

    def _rms(self, frame):
        """Calculate RMS (root mean square) of audio frame using shared utility."""
        try:
            data = np.frombuffer(frame, dtype=np.int16)
            return calculate_rms(data)
            
        except Exception as e:
            _logger.debug(f"Error calculating RMS: {e}")
            return 0.0

    def _record_audio(self, start_with):
        """Record audio starting with the given frame."""
        rec = [start_with]
        
        current = time.time()
        end = time.time() + self.timeout
        upper_limit_end = time.time() + self.max_timeout

        while current <= end and current < upper_limit_end:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                if self._rms(data) >= self.volume_threshold:
                    end = time.time() + self.timeout

                current = time.time()
                rec.append(data)
            except Exception as e:
                _logger.warning(f"Error reading audio data: {e}")
                break

        # Convert to numpy array and normalize
        try:
            audio_bytes = b"".join(rec)
            if len(audio_bytes) == 0:
                return np.array([], dtype=np.float32)
            
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            if len(audio_data) == 0:
                return np.array([], dtype=np.float32)
            
            # Normalize to float32 in range [-1, 1]
            normalized = audio_data.astype(np.float32) / self.range
            
            # Clip to valid range to prevent overflow issues
            normalized = np.clip(normalized, -1.0, 1.0)
            
            return normalized
            
        except Exception as e:
            _logger.warning(f"Error processing audio data: {e}")
            return np.array([], dtype=np.float32)

    async def input(self) -> str:
        """Get voice input and return transcribed text."""
        if not self.is_active:
            self.activate()

        # Initialize Whisper model if not already done
        self.initialize_whisper()

        _logger.debug(f"Starting microphone listening loop, volume threshold: {self.volume_threshold}")
        sample_count = 0
        
        while True:
            await asyncio.sleep(0.01)  # Small sleep to prevent blocking
            
            try:
                # Read initial chunk
                inp = self.stream.read(self.chunk, exception_on_overflow=False)
                
                # Debug: Check if we're actually getting data
                if sample_count == 1:  # Log first sample details
                    _logger.info(f"First microphone sample: length={len(inp) if inp else 0}, type={type(inp)}")
                    if inp and len(inp) > 0:
                        # Convert to numpy to check actual values
                        data_array = np.frombuffer(inp, dtype=np.int16)
                        _logger.info(f"Sample data range: [{data_array.min()}, {data_array.max()}], mean: {data_array.mean():.2f}")
                    else:
                        _logger.warning("First sample is empty or None!")
                        
            except Exception as e:
                _logger.warning(f"Error reading audio: {e}")
                # Try to reactivate stream
                self.deactivate()
                self.activate()
                continue

            rms_val = self._rms(inp)
            sample_count += 1
            
            # Debug logging every 100 samples (about 1 second at 10ms chunks)
            if sample_count % 100 == 0:
                _logger.debug(f"Microphone input: RMS={rms_val:.4f}, threshold={self.volume_threshold:.4f}, data_len={len(inp) if inp else 0}")
            
            if rms_val > self.volume_threshold:
                _logger.debug(f"Audio detected! RMS={rms_val:.4f} > threshold={self.volume_threshold:.4f}")
                
                # Record full audio
                audio_data = self._record_audio(start_with=inp)
                
                # Check if we got valid audio data
                if len(audio_data) == 0:
                    _logger.debug("No audio data recorded, continuing...")
                    continue
                    
                _logger.debug(f"Recorded audio: length={len(audio_data)}, duration={len(audio_data)/16000:.2f}s")
                self.last_audio = audio_data
                
                # Transcribe audio
                result = await self._process_audio(audio_data)
                _logger.debug(f"Transcription result: {result}")
                
                if result and result.get("transcription"):
                    transcription = result["transcription"].strip()
                    if transcription and transcription.lower() != "[unclear]":
                        _logger.info(f"Successfully transcribed: '{transcription}'")
                        return transcription
                    else:
                        _logger.debug(f"Got unclear transcription or empty result: '{transcription}'")
            else:
                # Adjust threshold based on ambient noise
                new_threshold = 2 * rms_val
                self.volume_threshold = max(new_threshold, self.original_volume_threshold)

    async def _process_audio(self, waveform: np.ndarray, hotword: Optional[str] = None) -> dict:
        """
        Process audio waveform and return transcription with optional hotword logp.
        Adapted from WhisperHandler's preprocess, inference, and postprocess methods.
        """
        try:
            # Run processing in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._process_audio_sync, waveform, hotword)
            return result
        except Exception as e:
            _logger.error(f"Error processing audio: {e}")
            return {"transcription": "[unclear]", "score": 0.0, "logp": None}

    def _process_audio_sync(self, waveform: np.ndarray, hotword: Optional[str] = None) -> dict:
        """Synchronous audio processing (adapted from WhisperHandler)."""
        try:
            # Check for valid audio input
            if len(waveform) == 0:
                return {"transcription": "[unclear]", "score": 0.0, "logp": None}
            
            # Check for valid audio values
            if not np.isfinite(waveform).all():
                _logger.warning("Audio contains invalid values, cleaning...")
                waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Preprocess (adapted from WhisperHandler.preprocess)
            input_features = self.processor(
                audio=waveform, 
                return_tensors="pt", 
                sampling_rate=16000
            ).input_features
            
            hotword_tokens = None
            if hotword:
                hotword_tokens = torch.tensor(
                    [
                        item
                        for item in self.processor.tokenizer.encode(f" {hotword}")
                        if item not in set(self._ending_tokens + self._starting_tokens)
                    ],
                    dtype=torch.int,
                ).unsqueeze(0)

            # Prepare processed data with proper tensor handling
            input_features = input_features.to(self.device)
            if self.device == "cuda":
                input_features = input_features.half()
            
            processed_hotword_tokens = None
            if hotword_tokens is not None:
                processed_hotword_tokens = hotword_tokens.to(self.device)
                # Don't apply half() to integer tensors
            
            processed_data = {
                "input_features": input_features,
                "num_beams": 1,  # Default beam search
                "num_tokens": 448,  # Default max tokens
                "hotword_tokens": processed_hotword_tokens,
            }
            
            # Inference (adapted from WhisperHandler.inference)
            with torch.no_grad():
                input_features = processed_data["input_features"]
                num_beams = processed_data["num_beams"]
                num_tokens = processed_data["num_tokens"]
                hotword_tokens = processed_data["hotword_tokens"]
                
                output = self.model.generate(
                    input_features,
                    num_beams=num_beams,
                    return_dict_in_generate=True,
                    max_length=num_tokens,
                )
                
                transcription = self.processor.batch_decode(
                    output.sequences, skip_special_tokens=True
                )[0]
                
                # Handle different versions of transformers output format
                score = 0.0
                if hasattr(output, 'sequences_scores') and output.sequences_scores is not None:
                    score = output.sequences_scores
                elif hasattr(output, 'scores') and output.scores is not None:
                    # Alternative: calculate score from logits if available
                    score = sum(s.max().item() for s in output.scores) / len(output.scores)
                
                logp = None
                if hotword_tokens is not None:
                    logp = self._compute_logp(hotword_tokens, input_features)

                return {
                    "transcription": transcription,
                    "score": float(score),
                    "logp": float(logp) if logp is not None else None,
                }
                
        except Exception as e:
            _logger.error(f"Error in synchronous audio processing: {e}")
            return {"transcription": "[unclear]", "score": 0.0, "logp": None}

    def _compute_logp(self, hotword_tokens, input_features):
        """
        Compute log probability for hotword detection.
        Adapted from WhisperHandler.compute_logp method.
        """
        try:
            input_ids = torch.tensor([self._starting_tokens]).to(self.device)
            
            for _ in range(hotword_tokens.shape[1]):
                logits = self.model(
                    input_features,
                    decoder_input_ids=input_ids,
                ).logits
                new_token = torch.argmax(logits, dim=-1)
                new_token = torch.tensor([[new_token[:, -1]]]).to(self.device)
                input_ids = torch.cat([input_ids, new_token], dim=-1)

            logprobs = torch.log(torch.softmax(logits, dim=-1))
            sum_logp = 0
            for logp, index in zip(logprobs[0][1:], hotword_tokens[0]):
                sum_logp += logp[int(index)]

            return sum_logp
            
        except Exception as e:
            _logger.error(f"Error computing logp: {e}")
            return None

    async def get_hotword_if_present(self) -> str:
        """
        Check if any hotword is present in the last audio using logp threshold.
        """
        if not self.hotwords or self.last_audio is None:
            return ""
        
        for hotword in self.hotwords:
            if await self.hotword_is_present(hotword):
                return hotword
        
        return ""

    async def hotword_is_present(self, hotword: str) -> bool:
        """
        Check if a specific hotword is present using logp threshold.
        This uses the actual logp computation from WhisperHandler.
        """
        if self.last_audio is None:
            return False
        
        try:
            # Process the last audio with the specific hotword
            result = await self._process_audio(self.last_audio, hotword=hotword)
            
            if result.get("logp") is not None:
                logp = result["logp"]
                _logger.debug(f"Hotword '{hotword}' logp: {logp}, threshold: {self.hotword_threshold}")
                return logp > self.hotword_threshold
            
            return False
            
        except Exception as e:
            _logger.error(f"Error checking hotword presence: {e}")
            return False

    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            if self.is_active:
                self.deactivate()
            if hasattr(self, 'p') and self.p:
                self.p.terminate()
        except Exception:
            pass  # Ignore cleanup errors