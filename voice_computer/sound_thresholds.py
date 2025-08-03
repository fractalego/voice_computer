"""
Sound threshold utilities for voice activity detection.

Provides shared RMS calculation and threshold detection functionality
used across the voice computer system for interrupting TTS playback
and detecting voice activity.
"""

import logging
import numpy as np

_logger = logging.getLogger(__name__)


def check_audio_input_threshold(config, stream) -> bool:
    """
    Listen to one chunk of audio input and check if it exceeds the volume threshold.
    
    Args:
        config: Configuration object containing volume threshold
        stream: PyAudio stream (must have input=True)
        
    Returns:
        True if audio input exceeds threshold, False otherwise
    """
    if not config or not stream:
        return False
    
    # Get threshold from config
    listener_config = config.get_value("listener_model") or {}
    threshold = listener_config.get("listener_volume_threshold", 0.6)
    
    try:
        # Read one chunk of audio data
        data = stream.read(1024, exception_on_overflow=False)
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = calculate_rms(audio_data)
        print(f"RMS level: {rms}, Threshold: {threshold}")  # Debug output
        return rms >= threshold
        
    except Exception as e:
        _logger.debug(f"Error checking audio input: {e}")
        return False


def calculate_rms(audio_data: np.ndarray) -> float:
    """
    Calculate RMS (Root Mean Square) level of audio data.
    
    Args:
        audio_data: Audio data as numpy array
        
    Returns:
        RMS level as float between 0 and 1
    """
    if len(audio_data) == 0:
        return 0.0

    data = np.frombuffer(audio_data, dtype=np.int16)
    if len(data) == 0:
        return 0.0

    rms = np.mean(np.sqrt(data.astype(np.float64) ** 2)) / data.shape[0]

    # Handle NaN or infinite values
    if not np.isfinite(rms):
        return 0.0

    return float(rms)


def get_volume_threshold_from_config(config) -> float:
    """
    Extract volume threshold from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Volume threshold value
    """
    if not config:
        return 0.6  # default threshold
    
    listener_config = config.get_value("listener_model") or {}
    return listener_config.get("listener_volume_threshold", 0.6)