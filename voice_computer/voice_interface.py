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

_logger = logging.getLogger(__name__)

COLOR_START = "\033[94m"
COLOR_END = "\033[0m"


class VoiceInterface:
    """Simplified voice interface for speech input and output."""
    
    def __init__(self, config=None):
        self.config = config
        self._is_listening = False
        self._bot_has_spoken = False
        self._listener = WhisperListener(config)
        
        # Audio feedback settings
        self._waking_up_sound = True
        self._deactivate_sound = True
        
        # Set up sound file paths
        sounds_dir = Path(__file__).parent / "sounds"
        self._activation_sound_path = sounds_dir / "activation.wav"
        self._deactivation_sound_path = sounds_dir / "deactivation.wav"
        self._deny_sound_path = sounds_dir / "deny.wav"
        
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
        for sound_file in [self._activation_sound_path, self._deactivation_sound_path]:
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

    async def _speak_text(self, text: str) -> None:
        """
        Use system TTS to speak text.
        """
        try:
            # Try different TTS commands based on the system
            tts_commands = [
                # macOS
                ['say', text],
                # Linux with espeak
                ['espeak', text],
                # Linux with festival
                ['festival', '--tts'],
                # Linux with spd-say
                ['spd-say', text],
            ]

            for cmd in tts_commands:
                try:
                    if cmd[0] == 'festival':
                        # Festival reads from stdin
                        process = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL
                        )
                        await process.communicate(text.encode())
                    else:
                        process = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=asyncio.subprocess.DEVNULL,
                            stderr=asyncio.subprocess.DEVNULL
                        )
                        await process.wait()
                    
                    if process.returncode == 0:
                        return
                        
                except FileNotFoundError:
                    continue
                except Exception as e:
                    _logger.debug(f"TTS command failed: {e}")
                    continue

            _logger.warning("No working TTS command found. Install espeak, festival, or spd-say for speech output.")
            
        except Exception as e:
            _logger.error(f"Error speaking text: {e}")

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

    async def _play_sound_file(self, sound_file_path: Path) -> None:
        """
        Play a specific sound file using available audio players.
        """
        try:
            sound_file_str = str(sound_file_path)
            
            # Try different audio players in order of preference
            audio_players = [
                # macOS
                ['afplay', sound_file_str],
                # Linux with aplay (ALSA)
                ['aplay', sound_file_str],
                # Linux with paplay (PulseAudio)
                ['paplay', sound_file_str],
                # Cross-platform with ffplay (if available)
                ['ffplay', '-nodisp', '-autoexit', sound_file_str],
                # Cross-platform with mpv (if available)
                ['mpv', '--no-video', '--quiet', sound_file_str],
            ]

            for cmd in audio_players:
                try:
                    _logger.debug(f"Trying audio command: {' '.join(cmd)}")
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    await process.wait()
                    
                    if process.returncode == 0:
                        _logger.debug(f"Successfully played sound: {sound_file_path}")
                        return
                        
                except FileNotFoundError:
                    _logger.debug(f"Audio player not found: {cmd[0]}")
                    continue
                except Exception as e:
                    _logger.debug(f"Audio player failed: {cmd[0]} - {e}")
                    continue

            # If no audio player works, log warning
            _logger.warning(f"No working audio player found to play: {sound_file_path}")
            _logger.info("Install an audio player: sudo apt-get install alsa-utils pulseaudio-utils (Linux) or use macOS/Windows built-in players")
            
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