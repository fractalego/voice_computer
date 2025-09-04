"""
Microphone-based speech recognition using local microphone input with Whisper.
"""

import asyncio
import logging
import pyaudio

from .base_listener import BaseListener, VoiceInterruptionException

_logger = logging.getLogger(__name__)


class MicrophoneListener(BaseListener):
    """Microphone listener using local microphone input with PyAudio and Whisper."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self._range = 32768  # Range for int16 audio data

        # PyAudio-specific settings
        self.format = pyaudio.paInt16
        self.device_index = None
        
        # Load PyAudio-specific configuration
        if config:
            listener_config = config.get_value("listener_model") or {}
            self.device_index = listener_config.get("microphone_device_index", None)
        
        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        _logger.info(f"MicrophoneListener initialized with PyAudio")
        
        # Debug: List available audio devices in debug mode
        self._log_available_audio_devices()
    
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
                    input_device_index=self.device_index,
                )
                self.is_active = True
                _logger.debug(f"MicrophoneListener audio stream activated - format: {self.format}, channels: {self.channels}, rate: {self.rate}, chunk: {self.chunk}")
                
                # Test read a small chunk
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
        if self.is_active:
            try:
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.stream = None
                self.is_active = False
                _logger.debug("MicrophoneListener audio stream deactivated")
            except Exception as e:
                _logger.error(f"Error deactivating audio stream: {e}")

    async def throw_exception_on_voice_activity(self):
        """Monitor for voice activity and throw exception when detected."""
        if not self.is_active:
            self.activate()
        
        try:
            while True:
                frame = self.stream.read(self.chunk, exception_on_overflow=False)
                rms = self._rms(frame)
                
                if rms > self.volume_threshold:
                    # Voice activity detected
                    raise VoiceInterruptionException("Voice activity detected")
                
                await asyncio.sleep(0.01)
                
        except Exception as e:
            # Re-raise the exception to signal voice activity
            raise e

    def _get_input(self) -> bytes:
        return self.stream.read(self.chunk, exception_on_overflow=False)

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.deactivate()
            if hasattr(self, 'p'):
                self.p.terminate()
        except:
            pass