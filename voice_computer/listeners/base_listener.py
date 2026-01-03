"""
Base listener class with common functionality for all voice listeners.
"""
import asyncio
import logging
import time

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
            self.timeout = listener_config.get("listener_silence_timeout", 2)
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

            # Initialize special tokens for logp computation
            self._initialize_special_tokens()

            # Warmup inference with dummy audio to pre-compile torch operations
            self._warmup_inference()

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

    def _warmup_inference(self) -> None:
        """
        Run a dummy inference to pre-compile torch operations.
        This ensures the first real inference is fast.
        """
        _logger.info("Running warmup inference...")
        try:
            # Create 1 second of silent audio at 16kHz
            dummy_audio = np.zeros(self.rate, dtype=np.float32)

            # Warmup the transcription path (uses .generate())
            inputs = self.processor(
                dummy_audio,
                sampling_rate=self.rate,
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device)

            if self.device.type == 'cuda':
                model_dtype = next(self.model.parameters()).dtype
                input_features = input_features.to(dtype=model_dtype)

            with torch.no_grad():
                # Warmup generate path
                _ = self.model.generate(
                    input_features,
                    max_length=10,
                    num_beams=1,
                    do_sample=False
                )

                # Warmup forward pass path (used by compute_logp)
                input_ids = torch.tensor([self._starting_tokens], dtype=torch.long).to(self.device)
                _ = self.model(input_features, decoder_input_ids=input_ids)

            _logger.info("Warmup inference completed successfully")

        except Exception as e:
            _logger.warning(f"Warmup inference failed (non-fatal): {e}")

    def _initialize_special_tokens(self) -> None:
        """Initialize starting and ending tokens from the Whisper tokenizer."""
        try:
            tokenizer = self.processor.tokenizer

            # Get special token IDs
            # Starting tokens: typically <|startoftranscript|> and language/task tokens
            start_of_transcript = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

            # For English transcription, we use <|en|> and <|transcribe|>
            en_token = tokenizer.convert_tokens_to_ids("<|en|>")
            transcribe_token = tokenizer.convert_tokens_to_ids("<|transcribe|>")
            notimestamps_token = tokenizer.convert_tokens_to_ids("<|notimestamps|>")

            # Build starting tokens sequence
            self._starting_tokens = [start_of_transcript]
            if en_token is not None and en_token != tokenizer.unk_token_id:
                self._starting_tokens.append(en_token)
            if transcribe_token is not None and transcribe_token != tokenizer.unk_token_id:
                self._starting_tokens.append(transcribe_token)
            if notimestamps_token is not None and notimestamps_token != tokenizer.unk_token_id:
                self._starting_tokens.append(notimestamps_token)

            # Ending tokens: <|endoftext|> or <|endoftranscript|>
            end_of_text = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            end_of_transcript = tokenizer.convert_tokens_to_ids("<|endoftranscript|>")

            self._ending_tokens = []
            if end_of_text is not None and end_of_text != tokenizer.unk_token_id:
                self._ending_tokens.append(end_of_text)
            if end_of_transcript is not None and end_of_transcript != tokenizer.unk_token_id:
                self._ending_tokens.append(end_of_transcript)

            _logger.info(f"Initialized special tokens - starting: {self._starting_tokens}, ending: {self._ending_tokens}")

        except Exception as e:
            _logger.warning(f"Failed to initialize special tokens: {e}. Using empty lists.")
            self._starting_tokens = []
            self._ending_tokens = []

    def _tokenize_hotword(self, hotword: str) -> torch.Tensor:
        """
        Tokenize a hotword, removing special tokens.

        Args:
            hotword: The hotword string to tokenize

        Returns:
            Tensor of token IDs for the hotword
        """
        # Encode with leading space (common for subword tokenizers)
        all_tokens = self.processor.tokenizer.encode(f" {hotword}")

        # Filter out ALL special tokens (not just starting/ending)
        special_tokens = set(self.processor.tokenizer.all_special_ids)
        hotword_tokens = [t for t in all_tokens if t not in special_tokens]

        _logger.debug(f"Tokenized '{hotword}': raw={all_tokens}, filtered={hotword_tokens}")

        return torch.tensor(hotword_tokens, dtype=torch.long).unsqueeze(0)

    def compute_logp(self, hotword_tokens: torch.Tensor, input_features: torch.Tensor, max_generate_steps: int = 20) -> Tuple[float, int, List[int]]:
        """
        Compute the log probability of a hotword using sliding window matching.

        This runs the Whisper decoder autoregressively to generate tokens,
        then slides the hotword tokens over the generated sequence to find
        the best matching position.

        Args:
            hotword_tokens: Tensor of shape (1, K) with hotword token IDs
            input_features: Tensor of audio features from the processor
            max_generate_steps: Maximum number of tokens to generate

        Returns:
            Tuple of (best_logp, best_position, all_generated_tokens)
            - best_logp: Best sum of log probabilities for the hotword
            - best_position: Position in generated tokens where hotword best matches
            - all_generated_tokens: All tokens generated from the audio
        """
        if not self.initialized or self.model is None:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        # Start with the starting tokens
        input_ids = torch.tensor([self._starting_tokens], dtype=torch.long).to(self.device)

        # Get model dtype for consistency
        model_dtype = next(self.model.parameters()).dtype
        input_features = input_features.to(device=self.device, dtype=model_dtype)

        hotword_len = hotword_tokens.shape[1]
        eos_token_id = self.processor.tokenizer.eos_token_id

        # Get all special token IDs
        special_token_ids = set(self.processor.tokenizer.all_special_ids)

        with torch.no_grad():
            generated_tokens = []
            all_logprobs = []

            for step in range(max_generate_steps):
                outputs = self.model(
                    input_features,
                    decoder_input_ids=input_ids,
                )
                logits = outputs.logits

                # Store logprobs for the last position (predicting next token)
                step_logprobs = torch.log_softmax(logits[:, -1, :], dim=-1)
                all_logprobs.append(step_logprobs)

                # Greedy decoding: pick the most likely next token
                new_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                token_id = new_token.item()
                generated_tokens.append(token_id)

                input_ids = torch.cat([input_ids, new_token], dim=-1)

                # Stop if we hit EOS token
                if token_id == eos_token_id:
                    break

            _logger.debug(f"compute_logp: Generated {len(generated_tokens)} tokens: {generated_tokens}")
            _logger.debug(f"compute_logp: Decoded: '{self.processor.tokenizer.decode(generated_tokens)}'")

            # Now slide the hotword window over the generated sequence
            # all_logprobs[i] contains the logprobs for predicting token at position i
            num_generated = len(generated_tokens)

            if num_generated < hotword_len:
                _logger.debug(f"compute_logp: Not enough tokens generated ({num_generated} < {hotword_len})")
                return float('-inf'), -1, generated_tokens

            best_logp = float('-inf')
            best_position = -1

            # Decode hotword tokens for logging
            hotword_token_strs = [self.processor.tokenizer.decode([int(t)]) for t in hotword_tokens[0]]
            _logger.info(f"compute_logp: Sliding hotword tokens {hotword_tokens[0].tolist()} ({hotword_token_strs}) over {num_generated} generated tokens")

            # Log generated tokens with their decoded strings
            generated_token_strs = [self.processor.tokenizer.decode([t]) for t in generated_tokens]
            _logger.info(f"compute_logp: Generated tokens: {list(zip(generated_tokens, generated_token_strs))}")

            # Slide window: check positions 0 to (num_generated - hotword_len)
            window_scores = []
            for start_pos in range(num_generated - hotword_len + 1):
                sum_logp = 0.0
                token_scores = []
                for i, token_id in enumerate(hotword_tokens[0]):
                    pos = start_pos + i
                    token_logp = all_logprobs[pos][0, int(token_id)].item()
                    token_scores.append(token_logp)
                    sum_logp += token_logp

                # Show what generated token is at this position vs what hotword token we're checking
                gen_at_pos = generated_tokens[start_pos:start_pos + hotword_len]
                gen_at_pos_strs = [self.processor.tokenizer.decode([t]) for t in gen_at_pos]
                window_scores.append((start_pos, sum_logp, gen_at_pos, gen_at_pos_strs, token_scores))

                if sum_logp > best_logp:
                    best_logp = sum_logp
                    best_position = start_pos

            # Log all window scores
            for pos, score, gen_tokens, gen_strs, token_scores in window_scores:
                match_indicator = " <-- BEST" if pos == best_position else ""
                _logger.info(f"  Position {pos}: score={score:.2f} | generated={gen_strs} vs hotword={hotword_token_strs} | per-token={[f'{s:.2f}' for s in token_scores]}{match_indicator}")

            _logger.info(f"compute_logp: Best match at position {best_position} with logp {best_logp:.2f}")

        return best_logp, best_position, generated_tokens

    async def detect_hotword_by_logp(self, audio_data: np.ndarray) -> Tuple[Optional[str], float, Optional[str]]:
        """
        Detect hotword using sliding window log probability scoring.

        This method generates tokens from the audio, slides the hotword tokens
        over the generated sequence to find the best match, and extracts the
        instruction (tokens after the hotword).

        Args:
            audio_data: Audio data as numpy array (float, normalized)

        Returns:
            Tuple of (detected_hotword, logp_score, instruction)
            - detected_hotword: The hotword if detected, None otherwise
            - logp_score: The best logp score
            - instruction: Text after the hotword (if hotword detected), None otherwise
        """
        if not self.initialized:
            self.initialize()

        if not self.hotwords:
            return None, 0.0, None

        try:
            # Process audio to get input features
            inputs = self.processor(
                audio_data,
                sampling_rate=self.rate,
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device)

            # Ensure dtype matches model
            if self.device.type == 'cuda':
                model_dtype = next(self.model.parameters()).dtype
                input_features = input_features.to(dtype=model_dtype)

            best_hotword = None
            best_logp = float('-inf')
            best_position = -1
            best_generated_tokens = []
            best_hotword_len = 0

            # Check each hotword
            for hotword in self.hotwords:
                hotword_tokens = self._tokenize_hotword(hotword)
                _logger.debug(f"Hotword '{hotword}' tokenized to: {hotword_tokens.tolist()}")

                logp, position, generated_tokens = self.compute_logp(hotword_tokens, input_features)

                _logger.debug(f"Hotword '{hotword}' logp: {logp:.2f} at position {position} (threshold: {self.hotword_threshold})")

                if logp > best_logp:
                    best_logp = logp
                    best_hotword = hotword
                    best_position = position
                    best_generated_tokens = generated_tokens
                    best_hotword_len = hotword_tokens.shape[1]

            # Check if best match exceeds threshold
            if best_logp >= self.hotword_threshold:
                # Extract instruction: tokens after the hotword
                instruction_start = best_position + best_hotword_len
                instruction_tokens = best_generated_tokens[instruction_start:]

                # Filter out special tokens from instruction
                special_token_ids = set(self.processor.tokenizer.all_special_ids)
                instruction_tokens = [t for t in instruction_tokens if t not in special_token_ids]

                instruction = None
                if instruction_tokens:
                    instruction = self.processor.tokenizer.decode(instruction_tokens).strip()
                    # Filter out meaningless instructions (punctuation only, very short, etc.)
                    if instruction and len(instruction) > 1 and not all(c in '.,!?;:\'"' for c in instruction):
                        _logger.info(f"Hotword detected by logp: '{best_hotword}' with score {best_logp:.2f}, instruction: '{instruction}'")
                    else:
                        _logger.debug(f"Filtered out meaningless instruction: '{instruction}'")
                        instruction = None
                        _logger.info(f"Hotword detected by logp: '{best_hotword}' with score {best_logp:.2f}, no instruction")
                else:
                    _logger.info(f"Hotword detected by logp: '{best_hotword}' with score {best_logp:.2f}, no instruction")

                return best_hotword, best_logp, instruction
            else:
                _logger.info(f"Hotword NOT detected by logp. Best: '{best_hotword}' with score {best_logp:.2f} (threshold: {self.hotword_threshold})")
                return None, best_logp, None

        except Exception as e:
            _logger.error(f"Error in detect_hotword_by_logp: {e}")
            return None, 0.0, None

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

    async def listen_for_audio(self, timeout_seconds: float = None) -> Tuple[Optional[np.ndarray], bool]:
        """
        Listen for audio input from WebSocket buffer.
        Similar to MicrophoneListener but uses buffered audio from WebSocket instead of microphone.

        Args:
            timeout_seconds: Maximum time to listen

        Returns:
            Tuple of (audio_data, voice_detected)
        """

        if not self.is_active:
            self.activate()

        if timeout_seconds is None:
            timeout_seconds = self.timeout * 20

        try:
            audio_frames = []
            last_spoken = None
            voice_detected = False
            start_time = None
            while True:
                if start_time is not None:
                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        _logger.debug("Timeout reached, stopping audio collection")
                        break
                try:
                    await asyncio.sleep(0.1)
                    frame = self._get_input()
                    if not frame:
                        continue
                    rms = self._rms(frame)
                    if rms > self.volume_threshold:
                        last_spoken = time.time()
                        voice_detected = True
                        audio_frames.append(frame)
                    else:
                        if last_spoken is not None and (time.time() - last_spoken) > self.timeout:
                            _logger.debug("No voice activity detected for timeout period, stopping audio collection")
                            break
                        elif last_spoken is not None:
                            # Still within timeout, keep collecting frames
                            audio_frames.append(frame)

                    await asyncio.sleep(0.01)  # Small delay to prevent busy loop

                except Exception as e:
                    _logger.error(f"Error reading audio: {e}")
                    break

            if audio_frames:
                audio_data = b''.join(audio_frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16) / self.range
                return audio_array, voice_detected
            else:
                return None, False

        except Exception as e:
            _logger.error(f"Error in listen_for_audio: {e}")
            return None, False

        except Exception as e:
            _logger.error(f"Error in listen_for_audio: {e}")
            return None, False

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
    async def throw_exception_on_voice_activity(self):
        """Monitor for voice activity and throw exception when detected."""
        pass

    @abstractmethod
    def _get_input(self) -> bytes:
        """Get raw audio input data."""
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