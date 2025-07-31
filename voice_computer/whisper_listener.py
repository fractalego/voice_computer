"""
Simplified Whisper-based speech recognition listener.
"""

import asyncio
import logging
import subprocess
import tempfile
import os
from typing import Optional

_logger = logging.getLogger(__name__)


class WhisperListener:
    """Simplified Whisper listener that uses the whisper command-line tool."""
    
    def __init__(self, config=None):
        self.config = config
        self.is_active = False
        self.timeout = 2
        self.volume_threshold = 1
        self.hotword_threshold = -8
        
        if config:
            listener_config = config.get_value("listener_model") or {}
            self.timeout = listener_config.get("listener_silence_timeout", 2)
            self.volume_threshold = listener_config.get("listener_volume_threshold", 1)
            self.hotword_threshold = listener_config.get("listener_hotword_logp", -8)

    def set_timeout(self, timeout: float):
        """Set the silence timeout."""
        self.timeout = timeout

    def set_volume_threshold(self, threshold: float):
        """Set the volume threshold."""
        self.volume_threshold = threshold

    def set_hotword_threshold(self, threshold: float):
        """Set the hotword detection threshold."""
        self.hotword_threshold = threshold

    def add_hotwords(self, hotwords):
        """Add hotwords for detection (simplified implementation)."""
        if hotwords:
            _logger.info(f"Hotwords configured: {hotwords}")

    def activate(self):
        """Activate the listener."""
        self.is_active = True
        _logger.debug("WhisperListener activated")

    def deactivate(self):
        """Deactivate the listener."""
        self.is_active = False
        _logger.debug("WhisperListener deactivated")

    async def input(self) -> str:
        """
        Get voice input and return transcribed text.
        
        This is a simplified implementation that uses the system's
        audio recording capabilities and Whisper for transcription.
        """
        if not self.is_active:
            self.activate()

        try:
            # Record audio using system tools
            audio_file = await self._record_audio()
            
            if audio_file and os.path.exists(audio_file):
                # Transcribe using Whisper
                text = await self._transcribe_audio(audio_file)
                
                # Clean up temporary file
                try:
                    os.unlink(audio_file)
                except OSError:
                    pass
                
                return text or "[unclear]"
            else:
                return "[unclear]"
                
        except Exception as e:
            _logger.error(f"Error during voice input: {e}")
            return "[unclear]"

    async def _record_audio(self) -> Optional[str]:
        """
        Record audio using system recording tools.
        Returns path to recorded audio file.
        """
        try:
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                audio_file = temp_file.name

            # Try different recording methods based on available tools
            recording_commands = [
                # macOS
                ['rec', audio_file, 'silence', '1', '0.1', '2%', '1', '2.0', '2%'],
                # Linux with ALSA
                ['arecord', '-f', 'cd', '-t', 'wav', '-d', str(self.timeout), audio_file],
                # Linux with PulseAudio
                ['parecord', '--format=s16le', '--rate=44100', '--channels=1', f'--record-time={self.timeout}', audio_file],
            ]

            for cmd in recording_commands:
                try:
                    _logger.debug(f"Trying recording command: {' '.join(cmd)}")
                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL
                    )
                    await process.wait()
                    
                    if process.returncode == 0 and os.path.exists(audio_file):
                        _logger.debug(f"Successfully recorded audio to {audio_file}")
                        return audio_file
                        
                except FileNotFoundError:
                    continue
                except Exception as e:
                    _logger.debug(f"Recording command failed: {e}")
                    continue

            _logger.warning("No working audio recording command found")
            return None

        except Exception as e:
            _logger.error(f"Error recording audio: {e}")
            return None

    async def _transcribe_audio(self, audio_file: str) -> Optional[str]:
        """
        Transcribe audio file using Whisper.
        """
        try:
            # Try using whisper command-line tool
            cmd = ['whisper', audio_file, '--output_format', 'txt', '--output_dir', '/tmp']
            
            _logger.debug(f"Running whisper command: {' '.join(cmd)}")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                # Find the output text file
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                txt_file = f"/tmp/{base_name}.txt"
                
                if os.path.exists(txt_file):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    # Clean up text file
                    try:
                        os.unlink(txt_file)
                    except OSError:
                        pass
                    
                    _logger.debug(f"Transcribed text: {text}")
                    return text
            else:
                _logger.error(f"Whisper command failed: {stderr.decode()}")
                
        except FileNotFoundError:
            _logger.error("Whisper command not found. Please install openai-whisper: pip install openai-whisper")
        except Exception as e:
            _logger.error(f"Error transcribing audio: {e}")

        return None

    async def get_hotword_if_present(self) -> str:
        """
        Check if a hotword is present (simplified implementation).
        """
        # This is a placeholder - in a real implementation,
        # you would analyze the audio for hotwords
        return ""

    async def hotword_is_present(self, hotword: str) -> bool:
        """
        Check if a specific hotword is present (simplified implementation).
        """
        # This is a placeholder - in a real implementation,
        # you would analyze the audio for the specific hotword
        return False