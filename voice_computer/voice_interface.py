"""
Simplified voice interface for input and output.
"""

import asyncio
import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Optional

from .whisper_listener import WhisperListener
from .speaker import SoundFileSpeaker, TTSSpeaker

_logger = logging.getLogger(__name__)

COLOR_START = "\033[94m"  # Blue
COLOR_END = "\033[0m"      # Reset
COLOR_GREEN = "\033[92m"   # Green


class VoiceInterface:
    """Simplified voice interface for speech input and output."""
    
    def __init__(self, config=None):
        self.config = config
        self._is_listening = False
        self._bot_has_spoken = False
        self._listener = WhisperListener(config)
        self._tts_speaker = TTSSpeaker(config=config)
        
        # Audio feedback settings
        self._waking_up_sound = True
        self._deactivate_sound = True
        
        # Set up sound file paths
        sounds_dir = Path(__file__).parent / "sounds"
        self._activation_sound_path = sounds_dir / "activation.wav"
        self._deactivation_sound_path = sounds_dir / "deactivation.wav"
        self._deny_sound_path = sounds_dir / "deny.wav"
        self._computer_work_beep_path = sounds_dir / "computer_work_beep.wav"
        self._computer_starting_to_work_path = sounds_dir / "computer_starting_to_work.wav"
        
        # Initialize sound file speaker
        try:
            self._speaker = SoundFileSpeaker()
            _logger.debug("SoundFileSpeaker initialized successfully")
        except Exception as e:
            _logger.warning(f"Failed to initialize SoundFileSpeaker: {e}")
            self._speaker = None
        
        if config:
            self._waking_up_sound = config.get_value("waking_up_sound") or True
            self._deactivate_sound = config.get_value("deactivate_sound") or True
            
            # Set up activation hotwords
            activation_hotwords = config.get_value("activation_hotwords")
            if activation_hotwords:
                self._listener.set_hotwords(activation_hotwords)
                _logger.info(f"Activation hotwords configured: {activation_hotwords}")
            else:
                # Default hotword
                self._listener.set_hotwords(["computer"])
                _logger.info("Using default activation hotword: computer")
        
        # Verify sound files exist
        for sound_file in [self._activation_sound_path, self._deactivation_sound_path, self._computer_work_beep_path, self._computer_starting_to_work_path]:
            if not sound_file.exists():
                _logger.warning(f"Sound file not found: {sound_file}")

    def add_hotwords(self, hotwords):
        """Add hotwords to the listener."""
        self._listener.add_hotwords(hotwords)

    async def output(self, text: str, silent: bool = False, skip_print: bool = False) -> None:
        """
        Output text via speech synthesis.
        
        Args:
            text: The text to speak
            silent: If True, only print to console without speaking
            skip_print: If True, only speak without printing (useful when text was already printed via streaming)
        """
        if not text:
            return

        if silent:
            print(text)
            return

        self._listener.activate()
        
        # Print text unless skip_print is True (e.g., when streaming already displayed it)
        if not skip_print:
            print(COLOR_START + "bot> " + text + COLOR_END)
        
        # Speak the text using system TTS
        await self._speak_text(text)
        self.bot_has_spoken(True)

    async def input(self) -> str:
        """
        Get voice input from the user.
        First waits for hotword activation, then listens for the actual command.
        
        Returns:
            The transcribed text from the user
        """
        # Phase 1: Wait for hotword activation
        print(COLOR_START + "💭 Listening for activation word ('computer')..." + COLOR_END)
        await self._wait_for_hotword()
        
        # Phase 2: Listen for command after hotword detected
        print(COLOR_START + "✨ Hotword detected! Listening for your command..." + COLOR_END)
        text = ""
        while not text:
            text = await self._listener.input()
            if not text or not text.strip() or text.strip() == "[unclear]":
                continue
            
            text = self._remove_activation_word_and_normalize(text)
            break

        # Simple quality check - if text seems too short or unclear, ask for repeat
        while self._is_listening and self._not_good_enough(text):
            print(COLOR_START + "user> " + text + COLOR_END)
            await self.output("Sorry? Can you repeat?")
            text = await self._listener.input()
            text = self._remove_activation_word_and_normalize(text)

        text = text.lower().capitalize()
        print(COLOR_START + "user> " + text + COLOR_END)
        
        return self._remove_unclear(text)
    
    async def _wait_for_hotword(self) -> tuple[str, str]:
        """
        Wait in a loop until a hotword is detected.
        Uses continuous audio monitoring with hotword detection.
        
        Returns:
            Tuple of (detected_hotword, instruction_text)
            If instruction found with hotword, returns both
            If only hotword found, returns (hotword, "")
        """
        _logger.debug("Starting hotword detection loop...")
        
        while True:
            # Listen for audio that might contain hotword
            text = await self._listener.input()
            if not text or not text.strip() or text.strip() == "[unclear]":
                continue
            
            _logger.debug(f"Transcribed audio for hotword check: '{text}'")
            
            # Check if any configured hotword appears with potential instruction
            detected_hotword, instruction = self._check_hotword_with_instruction(text)
            if detected_hotword:
                _logger.info(f"Hotword '{detected_hotword}' detected in transcription!")
                if instruction:
                    _logger.info(f"Instruction detected with hotword: '{instruction}'")
                return (detected_hotword, instruction)
            
            # Also check using the advanced logp-based detection if available
            try:
                hotword = await self._listener.get_hotword_if_present()
                if hotword:
                    _logger.info(f"Hotword '{hotword}' detected via logp analysis!")
                    return (hotword, "")
            except Exception as e:
                _logger.debug(f"Logp-based hotword detection failed: {e}")
            
            # If no hotword detected, continue listening
            _logger.debug(f"No hotword detected in: '{text}', continuing to listen...")
            await asyncio.sleep(0.1)  # Small delay before next listen cycle
    
    def _check_hotword_in_text(self, text: str) -> str:
        """
        Check if any configured hotword appears in the transcribed text.
        
        Args:
            text: The transcribed text to check
            
        Returns:
            The detected hotword, or empty string if none found
        """
        if not text:
            return ""
        
        text_lower = text.lower().strip()
        
        # Get configured hotwords
        activation_hotwords = []
        if self.config:
            activation_hotwords = self.config.get_value("activation_hotwords") or ["computer"]
        else:
            activation_hotwords = ["computer"]
        
        # Check if any hotword appears in the text
        for hotword in activation_hotwords:
            if hotword.lower() in text_lower:
                return hotword
        
        return ""
    
    def _check_hotword_with_instruction(self, text: str) -> tuple[str, str]:
        """
        Check if hotword appears with an instruction in the same text.
        
        Args:
            text: The transcribed text to check
            
        Returns:
            Tuple of (detected_hotword, instruction_text)
            If no hotword found, returns ("", "")
            If hotword found but no instruction, returns (hotword, "")
        """
        if not text:
            return ("", "")
        
        text_lower = text.lower().strip()
        
        # Get configured hotwords
        activation_hotwords = []
        if self.config:
            activation_hotwords = self.config.get_value("activation_hotwords") or ["computer"]
        else:
            activation_hotwords = ["computer"]
        
        # Check if any hotword appears in the text
        for hotword in activation_hotwords:
            hotword_lower = hotword.lower()
            if hotword_lower in text_lower:
                # Find the position of the hotword
                hotword_pos = text_lower.find(hotword_lower)
                
                # Extract text after the hotword
                after_hotword = text[hotword_pos + len(hotword):].strip()
                
                # Remove common filler words at the beginning
                filler_words = ["please", "can you", "could you", "would you"]
                for filler in filler_words:
                    if after_hotword.lower().startswith(filler):
                        after_hotword = after_hotword[len(filler):].strip()
                
                # If there's meaningful text after the hotword, it's an instruction
                if after_hotword and len(after_hotword) > 2:
                    return (hotword, after_hotword)
                else:
                    return (hotword, "")
        
        return ("", "")

    def bot_has_spoken(self, to_set: Optional[bool] = None) -> bool:
        """Get or set whether the bot has spoken."""
        if to_set is not None:
            self._bot_has_spoken = to_set
        return self._bot_has_spoken

    def activate(self):
        """Activate the voice interface."""
        if not self._is_listening:
            self._is_listening = True
            _logger.info("Voice interface activated")

    def deactivate(self):
        """Deactivate the voice interface."""
        if self._is_listening:
            if self._deactivate_sound:
                asyncio.create_task(self._play_deactivation_sound())
            self._is_listening = False
            self._listener.deactivate()
            _logger.info("Voice interface deactivated")
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, '_speaker') and self._speaker is not None:
            try:
                self._speaker.cleanup()
                _logger.debug("Speaker cleanup completed")
            except Exception as e:
                _logger.debug(f"Error during speaker cleanup: {e}")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during destruction

    async def _speak_text(self, text: str) -> None:
        """
        Use system TTS to speak text.
        """
        self._tts_speaker.speak(text)

    async def _play_activation_sound(self) -> None:
        """Play activation sound."""
        if self._activation_sound_path.exists():
            await self._play_sound_file(self._activation_sound_path)
        else:
            _logger.warning("Activation sound file not found")

    async def _play_deactivation_sound(self) -> None:
        """Play deactivation sound."""
        if self._deactivation_sound_path.exists():
            await self._play_sound_file(self._deactivation_sound_path)
        else:
            _logger.warning("Deactivation sound file not found")

    async def _play_deny_sound(self) -> None:
        """Play deny/error sound."""
        if self._deny_sound_path.exists():
            await self._play_sound_file(self._deny_sound_path)
        else:
            _logger.warning("Deny sound file not found")

    async def play_computer_work_beep(self) -> None:
        """Play computer work beep sound when tools are being executed."""
        if self._computer_work_beep_path.exists():
            await self._play_sound_file(self._computer_work_beep_path)
        else:
            _logger.warning("Computer work beep sound file not found")

    async def play_computer_starting_to_work(self) -> None:
        """Play computer starting to work sound when processing user query."""
        if self._computer_starting_to_work_path.exists():
            await self._play_sound_file(self._computer_starting_to_work_path)
        else:
            _logger.warning("Computer starting to work sound file not found")

    async def _play_sound_file(self, sound_file_path: Path) -> None:
        """
        Play a specific sound file using the SoundFileSpeaker.
        """
        if self._speaker is None:
            _logger.warning(f"Speaker not available, cannot play sound: {sound_file_path}")
            return
            
        try:
            sound_file_str = str(sound_file_path)
            
            # Run the speaker in a thread to avoid blocking the async loop
            def play_sound():
                self._speaker.speak(sound_file_str)
            
            # Execute the blocking audio playback in a thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, play_sound)
            
            _logger.debug(f"Successfully played sound: {sound_file_path}")
            
        except Exception as e:
            _logger.error(f"Error playing sound file {sound_file_path}: {e}")

    def _remove_activation_word_and_normalize(self, text: str) -> str:
        """Remove activation words and normalize text."""
        import re
        
        # Simple activation word removal
        activation_patterns = [
            r'^\[.*?\]\s*',  # Remove [word] at start
            r'^(hey|hello|hi)\s+',  # Remove common activation words
        ]
        
        for pattern in activation_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()

    def _not_good_enough(self, text: str) -> bool:
        """Simple check if text quality is not good enough."""
        if not text or len(text.strip()) < 2:
            return True
        if "[unclear]" in text.lower():
            return True
        return False

    def _remove_unclear(self, text: str) -> str:
        """Remove unclear markers from text."""
        import re
        text = re.sub(r'\[unclear\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[.*?\]', '', text)  # Remove any bracketed content
        return text.strip()

    def initialize(self):
        self._listener.initialize_whisper()

    async def throw_exception_on_intelligible_speech(self):
        await self._listener.input()
        raise VoiceInterruptionException("Intelligible speech detected, interrupting voice input.")
    
    async def throw_exception_on_voice_activity(self):
        """
        Monitor audio volume and throw exception immediately when voice activity is detected.
        This is much faster than full speech transcription.
        """
        await self._listener.detect_voice_activity()
        raise VoiceInterruptionException("Voice activity detected, interrupting output.")


class VoiceInterruptionException(Exception):
    """Exception raised when voice input is interrupted."""

    def __init__(self, message: str = "Voice input interrupted"):
        super().__init__(message)
        _logger.warning(message)