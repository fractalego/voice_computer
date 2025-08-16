"""
Test for MicrophoneListener transcription accuracy.
"""

import asyncio
import unittest
import numpy as np
import wave
from pathlib import Path
import sys
import os

# Add the parent directory to the path so we can import voice_computer
sys.path.insert(0, str(Path(__file__).parent.parent))

from voice_computer.listeners import MicrophoneListener
from voice_computer.config import Config


class TestMicrophoneListener(unittest.TestCase):
    """Test cases for MicrophoneListener transcription accuracy."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(__file__).parent
        self.audio_file = self.test_dir / "quick_brown_fox.wav"
        
        # Verify test audio file exists
        self.assertTrue(self.audio_file.exists(), 
                       f"Test audio file not found: {self.audio_file}")
        
        # Create a test configuration
        self.config = Config()
        # Use a smaller/faster model for testing if available
        self.config.set_value("whisper_model", "openai/whisper-tiny")
        
        # Initialize MicrophoneListener
        self.listener = MicrophoneListener(self.config)
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self.listener, 'p') and self.listener.p:
            self.listener.p.terminate()
    
    def load_wav_file(self, file_path: Path) -> np.ndarray:
        """
        Load a WAV file and return it as a numpy array normalized to [-1, 1].
        
        Args:
            file_path: Path to the WAV file
            
        Returns:
            Normalized audio data as numpy array
        """
        with wave.open(str(file_path), 'rb') as wav_file:
            # Get audio parameters
            frames = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()
            framerate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            
            # Read audio data
            audio_bytes = wav_file.readframes(frames)
            
            # Convert to numpy array based on sample width
            if sample_width == 1:
                # 8-bit unsigned
                audio_data = np.frombuffer(audio_bytes, dtype=np.uint8)
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
            elif sample_width == 2:
                # 16-bit signed
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif sample_width == 4:
                # 32-bit signed
                audio_data = np.frombuffer(audio_bytes, dtype=np.int32)
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Convert stereo to mono if necessary
            if channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
            # Resample to 16kHz if necessary
            if framerate != 16000:
                # More robust resampling using scipy if available, otherwise simple interpolation
                try:
                    from scipy import signal
                    # Use scipy for better resampling
                    target_length = int(len(audio_data) * 16000 / framerate)
                    audio_data = signal.resample(audio_data, target_length)
                except ImportError:
                    # Fallback to simple interpolation
                    target_length = int(len(audio_data) * 16000 / framerate)
                    audio_data = np.interp(
                        np.linspace(0, len(audio_data), target_length),
                        np.arange(len(audio_data)),
                        audio_data
                    )
                    
            # Ensure audio is not empty and has reasonable amplitude
            if len(audio_data) == 0:
                raise ValueError("Audio file produced empty data")
            if audio_data.max() - audio_data.min() < 0.01:
                # Very quiet audio, might be problematic
                pass  # Don't fail, but could log a warning
            
            return audio_data
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison by removing punctuation and converting to lowercase.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        import re
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def test_transcription_accuracy(self):
        """Test that MicrophoneListener correctly transcribes the quick brown fox audio."""
        expected_text = "the quick brown fox jumps over the lazy dog"
        
        # Load the test audio file
        audio_data = self.load_wav_file(self.audio_file)
        
        # Initialize Whisper model before processing
        try:
            self.listener.initialize()
        except Exception as e:
            self.skipTest(f"Could not initialize Whisper model: {e}")
        
        # Check if model was properly initialized
        if not self.listener.initialized or self.listener.processor is None:
            self.skipTest("Whisper model not properly initialized - this might be due to missing model files or dependencies")
        
        # Process the audio through MicrophoneListener
        result = asyncio.run(self.listener.transcribe_audio(audio_data))
        
        # Check that we got a result
        self.assertIsInstance(result, str, "Expected string result from audio transcription")
        
        transcription = result
        self.assertNotEqual(transcription.strip(), "", "Expected non-empty transcription")
        
        # Normalize both texts for comparison
        normalized_transcription = self.normalize_text(transcription)
        normalized_expected = self.normalize_text(expected_text)
        
        print(f"\nExpected: '{normalized_expected}'")
        print(f"Got: '{normalized_transcription}'")
        
        # Check for exact match first
        if normalized_transcription == normalized_expected:
            return  # Perfect match!
        
        # If not exact match, check for high similarity
        # Calculate word-level similarity
        expected_words = set(normalized_expected.split())
        transcribed_words = set(normalized_transcription.split())
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(expected_words.intersection(transcribed_words))
        union = len(expected_words.union(transcribed_words))
        similarity = intersection / union if union > 0 else 0
        
        print(f"Word similarity: {similarity:.2%}")
        print(f"Expected words: {expected_words}")
        print(f"Transcribed words: {transcribed_words}")
        
        # Assert high similarity (at least 70% word overlap)
        self.assertGreater(similarity, 0.7, 
                          f"Transcription similarity too low: {similarity:.2%}. "
                          f"Expected: '{normalized_expected}', Got: '{normalized_transcription}'")
        
        # Check that at least the key words are present
        key_words = {"quick", "brown", "fox", "jumps", "lazy", "dog"}
        transcribed_words_set = set(normalized_transcription.split())
        key_words_found = key_words.intersection(transcribed_words_set)
        
        self.assertGreater(len(key_words_found), len(key_words) * 0.5,
                          f"Too few key words found. Expected at least half of {key_words}, "
                          f"but only found {key_words_found}")
    
    def test_empty_audio(self):
        """Test handling of empty audio data."""
        empty_audio = np.array([], dtype=np.float32)
        result = asyncio.run(self.listener.transcribe_audio(empty_audio))
        
        # Empty audio should return None, empty string, or [unclear]
        self.assertTrue(result is None or result == "" or result == "[unclear]", 
                       f"Expected None, empty string, or '[unclear]', got: {result}")
    
    def test_silence_audio(self):
        """Test handling of silent audio."""
        # Create 1 second of silence
        silence = np.zeros(16000, dtype=np.float32)
        result = asyncio.run(self.listener.transcribe_audio(silence))
        
        # Silent audio should produce minimal transcription, [unclear], or None
        if result:
            # Accept [unclear] or very short text
            self.assertTrue(result == "[unclear]" or len(result.strip()) <= 10, 
                           f"Silent audio should produce '[unclear]' or short text, got: '{result}'")


class TestMicrophoneListenerIntegration(unittest.TestCase):
    """Integration tests for MicrophoneListener with configuration."""
    
    def test_listener_initialization(self):
        """Test that MicrophoneListener initializes correctly with config."""
        config = Config()
        listener = MicrophoneListener(config)
        
        self.assertIsNotNone(listener.config)
        self.assertEqual(listener.is_active, False)
        self.assertIsNotNone(listener.p)  # PyAudio instance
        
        # Clean up
        listener.p.terminate()
    
    def test_config_loading(self):
        """Test that configuration values are loaded correctly."""
        config = Config()
        config.set_value("listener_model", {
            "listener_silence_timeout": 3,
            "listener_volume_threshold": 0.8,
            "listener_hotword_logp": -5,
            "microphone_device_index": 1
        })
        
        listener = MicrophoneListener(config)
        
        self.assertEqual(listener.timeout, 3)
        self.assertEqual(listener.volume_threshold, 0.8)
        self.assertEqual(listener.hotword_threshold, -5)
        self.assertEqual(listener.device_index, 1)
        
        # Clean up
        listener.p.terminate()


if __name__ == "__main__":
    # Set up test environment
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)