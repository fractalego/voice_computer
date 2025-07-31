"""
Simplified voice interface for input and output.
"""

import asyncio
import logging
import subprocess
import tempfile
import os
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
        
        if config:
            self._waking_up_sound = config.get_value("waking_up_sound") or True
            self._deactivate_sound = config.get_value("deactivate_sound") or True

    def add_hotwords(self, hotwords):
        """Add hotwords to the listener."""
        self._listener.add_hotwords(hotwords)

    async def output(self, text: str, silent: bool = False) -> None:
        """
        Output text via speech synthesis.
        
        Args:
            text: The text to speak
            silent: If True, only print to console without speaking
        """
        if not text:
            return

        if silent:
            print(text)
            return

        self._listener.activate()
        print(COLOR_START + "bot> " + text + COLOR_END)
        
        # Speak the text using system TTS
        await self._speak_text(text)
        self.bot_has_spoken(True)

    async def input(self) -> str:
        """
        Get voice input from the user.
        
        Returns:
            The transcribed text from the user
        """
        text = ""
        while not text:
            text = await self._listener.input()
            if not text or not text.strip() or text.strip() == "[unclear]":
                continue
            
            text = self._remove_activation_word_and_normalize(text)
            hotword = await self._listener.get_hotword_if_present()
            if hotword:
                text = f"[{hotword}] {text}"

        # Simple quality check - if text seems too short or unclear, ask for repeat
        while self._is_listening and self._not_good_enough(text):
            print(COLOR_START + "user> " + text + COLOR_END)
            await self.output("Sorry? Can you repeat?")
            text = await self._listener.input()

        text = text.lower().capitalize()
        print(COLOR_START + "user> " + text + COLOR_END)
        
        return self._remove_unclear(text)

    def bot_has_spoken(self, to_set: Optional[bool] = None) -> bool:
        """Get or set whether the bot has spoken."""
        if to_set is not None:
            self._bot_has_spoken = to_set
        return self._bot_has_spoken

    def activate(self):
        """Activate the voice interface."""
        if not self._is_listening:
            if self._waking_up_sound:
                asyncio.create_task(self._play_activation_sound())
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
        await self._play_system_sound("activation")

    async def _play_deactivation_sound(self) -> None:
        """Play deactivation sound."""
        await self._play_system_sound("deactivation")

    async def _play_system_sound(self, sound_type: str) -> None:
        """
        Play a system sound.
        """
        try:
            # Try different sound playing commands
            sound_commands = [
                # macOS
                ['afplay', '/System/Library/Sounds/Glass.aiff'],
                # Linux with aplay
                ['aplay', '/usr/share/sounds/alsa/Front_Left.wav'],
                # Linux with paplay
                ['paplay', '/usr/share/sounds/alsa/Front_Left.wav'],
                # Generic beep
                ['beep'],
            ]

            for cmd in sound_commands:
                try:
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
                except Exception:
                    continue

            # If no sound command works, just log
            _logger.debug(f"No working sound command found for {sound_type}")
            
        except Exception as e:
            _logger.debug(f"Error playing {sound_type} sound: {e}")

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