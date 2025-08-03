"""
Sound threshold utilities for voice activity detection.

Provides shared RMS calculation and threshold detection functionality
used across the voice computer system for interrupting TTS playback
and detecting voice activity.
"""

import logging
import numpy as np

_logger = logging.getLogger(__name__)



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

    rms = np.mean(np.sqrt(audio_data.astype(np.float64) ** 2)) / audio_data.shape[0]

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