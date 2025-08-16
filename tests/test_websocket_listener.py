"""
Test for WebSocketListener transcription accuracy and WebSocket audio handling.
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

from voice_computer.listeners import WebSocketListener
from voice_computer.config import Config


class TestWebSocketListener(unittest.TestCase):
    """Test WebSocketListener with simulated WebSocket audio data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.listener = WebSocketListener(self.config)
        
        # Path to test audio file
        self.audio_file = Path(__file__).parent / "quick_brown_fox.wav"
        
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self.listener, 'cleanup'):
            self.listener.cleanup()
    
    def load_wav_file(self, file_path):
        """Load a WAV file and return audio data as numpy array."""
        if not file_path.exists():
            self.skipTest(f"Test audio file not found: {file_path}")
        
        with wave.open(str(file_path), 'rb') as wav_file:
            # Get audio parameters
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            
            # Read audio data
            audio_bytes = wav_file.readframes(frames)
            
            # Convert to numpy array
            if sample_width == 1:
                dtype = np.uint8
            elif sample_width == 2:
                dtype = np.int16
            elif sample_width == 4:
                dtype = np.int32
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            audio_data = np.frombuffer(audio_bytes, dtype=dtype)
            
            # Convert to float32 and normalize
            audio_float = audio_data.astype(np.float32)
            if dtype != np.float32:
                audio_float = audio_float / np.iinfo(dtype).max
            
            # Handle stereo by taking the first channel
            if channels > 1:
                audio_float = audio_float[::channels]
            
            return audio_float
    
    async def simulate_client_audio_stream(self, audio_data, chunk_size=1024):
        """
        Simulate how a client would send audio data in chunks over WebSocket.
        
        Args:
            audio_data: numpy array of audio data
            chunk_size: size of each chunk to send
        """
        # Convert float32 audio back to int16 bytes (as client would send)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        
        # Send audio in chunks
        for i in range(0, len(audio_bytes), chunk_size * 2):  # *2 because int16 is 2 bytes
            chunk = audio_bytes[i:i + chunk_size * 2]
            if len(chunk) > 0:
                await self.listener.add_audio_chunk(chunk)
                # Small delay to simulate real-time streaming
                await asyncio.sleep(0.01)
    
    def normalize_text(self, text):
        """Normalize text for comparison by removing punctuation and converting to lowercase."""
        import re
        # Remove punctuation and extra whitespace
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = ' '.join(normalized.split())
        return normalized
    
    def test_transcription_accuracy(self):
        """Test that WebSocketListener correctly transcribes the quick brown fox audio."""
        # Load test audio file
        audio_data = self.load_wav_file(self.audio_file)
        
        # Initialize Whisper model before processing
        try:
            self.listener.initialize()
        except Exception as e:
            self.skipTest(f"Could not initialize Whisper model: {e}")
        
        # Check if model was properly initialized
        if not self.listener.initialized or self.listener.processor is None:
            self.skipTest("Whisper model not properly initialized - this might be due to missing model files or dependencies")
        
        async def run_test():
            # Simulate client sending audio data
            await self.simulate_client_audio_stream(audio_data)
            
            # Give a moment for audio to be processed
            await asyncio.sleep(0.1)
            
            # Get transcription using the input() method (which uses listen_for_audio internally)
            result = await self.listener.input()
            return result
        
        # Run the async test
        result = asyncio.run(run_test())
        
        # Check that we got a result
        self.assertIsInstance(result, str, "Expected string result from audio transcription")
        
        transcription = result
        self.assertNotEqual(transcription.strip(), "", "Expected non-empty transcription")
        
        # Normalize both texts for comparison
        normalized_transcription = self.normalize_text(transcription)
        expected_text = "the quick brown fox jumps over the lazy dog"
        normalized_expected = self.normalize_text(expected_text)
        
        print(f"Expected: '{expected_text}'")
        print(f"Got: '{transcription}'")
        
        # Check if the transcription contains key words from the expected text
        key_words = ["quick", "brown", "fox", "jumps", "over", "lazy", "dog"]
        key_words_found = []
        
        for word in key_words:
            if word in normalized_transcription:
                key_words_found.append(word)
        
        # We should find at least half of the key words
        self.assertGreater(len(key_words_found), len(key_words) * 0.5,
                          f"Too few key words found. Expected at least half of {key_words}, "
                          f"but only found {key_words_found}")
    
    def test_empty_audio(self):
        """Test handling of empty audio data."""
        async def run_test():
            # Don't send any audio data
            await asyncio.sleep(0.1)
            
            # Try to get transcription
            result = await asyncio.wait_for(self.listener.input(), timeout=1.0)
            return result
        
        # Empty audio should return empty string quickly
        try:
            result = asyncio.run(run_test())
            # Empty audio should return empty string, None, or [unclear]
            self.assertTrue(result == "" or result is None or result == "[unclear]", 
                           f"Expected empty string, None, or '[unclear]', got: {result}")
        except asyncio.TimeoutError:
            # Timeout is also acceptable for empty audio
            pass
    
    def test_silence_audio(self):
        """Test handling of silent audio."""
        # Create 1 second of silence
        silence = np.zeros(16000, dtype=np.float32)
        
        async def run_test():
            # Send silence to the listener
            await self.simulate_client_audio_stream(silence)
            await asyncio.sleep(0.1)
            
            # Try to get transcription
            result = await asyncio.wait_for(self.listener.input(), timeout=2.0)
            return result
        
        try:
            result = asyncio.run(run_test())
            
            # Silent audio should produce minimal transcription, [unclear], or None
            if result:
                # Accept [unclear] or very short text
                self.assertTrue(result == "[unclear]" or len(result.strip()) <= 10, 
                               f"Silent audio should produce '[unclear]' or short text, got: '{result}'")
        except asyncio.TimeoutError:
            # Timeout is also acceptable for silent audio
            pass
    
    def test_chunked_audio_processing(self):
        """Test that audio sent in multiple chunks is processed correctly."""
        # Create test audio - a simple sine wave
        duration = 2.0  # seconds
        sample_rate = 16000
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.5
        
        async def run_test():
            # Send audio in very small chunks to test buffering
            await self.simulate_client_audio_stream(audio_data, chunk_size=256)
            await asyncio.sleep(0.1)
            
            # The listener should have received and buffered all chunks
            async with self.listener.buffer_lock:
                buffer_length = len(self.listener.audio_buffer)
            
            # We should have approximately the right amount of audio
            expected_samples = len(audio_data)
            self.assertGreater(buffer_length, expected_samples * 0.8, 
                             f"Buffer should contain most of the sent audio. Expected ~{expected_samples}, got {buffer_length}")
        
        asyncio.run(run_test())
    
    def test_voice_activity_detection(self):
        """Test that voice activity detection works with buffered audio."""
        # Create audio with clear voice activity (loud signal)
        duration = 1.0
        sample_rate = 16000
        
        # Create a signal that should trigger voice activity detection  
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # We need a signal with RMS > 0.6. Account for int16 quantization loss
        # Use louder amplitude and clip to prevent overflow
        loud_audio = np.random.normal(0, 1.2, int(sample_rate * duration)).astype(np.float32)  # White noise with high RMS
        loud_audio = np.clip(loud_audio, -1.0, 1.0)  # Clip to valid range
        
        async def run_test():
            # Send the loud audio
            await self.simulate_client_audio_stream(loud_audio)
            await asyncio.sleep(0.1)
            
            # Test the listen_for_audio method directly
            audio_data, voice_detected = await self.listener.listen_for_audio(timeout_seconds=1.0)
            
            return audio_data, voice_detected
        
        audio_data, voice_detected = asyncio.run(run_test())
        
        # Should detect voice activity in loud signal
        self.assertTrue(voice_detected, "Should detect voice activity in loud audio signal")
        self.assertIsNotNone(audio_data, "Should return audio data when voice is detected")
        self.assertGreater(len(audio_data), 0, "Audio data should not be empty")


class TestWebSocketListenerIntegration(unittest.TestCase):
    """Integration tests for WebSocketListener with configuration."""
    
    def test_listener_initialization(self):
        """Test that WebSocketListener initializes correctly with config."""
        config = Config()
        config_dict = {
            "whisper_model": "fractalego/personal-whisper-distilled-model",
            "listener_model": {
                "listener_silence_timeout": 3,
                "listener_volume_threshold": 0.7,
                "listener_hotword_logp": -10
            }
        }
        
        for key, value in config_dict.items():
            config.set_value(key, value)
        
        listener = WebSocketListener(config)
        
        # Check that configuration was loaded correctly
        self.assertEqual(listener.timeout, 3)
        self.assertEqual(listener.volume_threshold, 0.7)
        self.assertEqual(listener.hotword_threshold, -10)
        self.assertEqual(listener.whisper_model_name, "fractalego/personal-whisper-distilled-model")
    
    def test_config_loading(self):
        """Test that configuration values are loaded correctly."""
        config = Config()
        
        # Set custom configuration
        listener_config = {
            "listener_silence_timeout": 5,
            "listener_volume_threshold": 0.8,
            "listener_hotword_logp": -12
        }
        config.set_value("listener_model", listener_config)
        config.set_value("whisper_model", "fractalego/personal-whisper-distilled-model")
        
        listener = WebSocketListener(config)
        
        # Verify configuration was applied
        self.assertEqual(listener.timeout, 5, "Timeout should be set from config")
        self.assertEqual(listener.volume_threshold, 0.8, "Volume threshold should be set from config")
        self.assertEqual(listener.hotword_threshold, -12, "Hotword threshold should be set from config")
        self.assertEqual(listener.whisper_model_name, "fractalego/personal-whisper-distilled-model", 
                        "Whisper model should be set from config")
    
    def test_concurrent_audio_chunks(self):
        """Test handling of concurrent audio chunk additions."""
        config = Config()
        listener = WebSocketListener(config)
        
        async def add_chunks_concurrently():
            # Create multiple tasks that add audio chunks simultaneously
            chunk_data = np.random.randint(-32768, 32767, 1024, dtype=np.int16).tobytes()
            
            tasks = []
            for i in range(10):
                task = asyncio.create_task(listener.add_audio_chunk(chunk_data))
                tasks.append(task)
            
            # Wait for all chunks to be added
            await asyncio.gather(*tasks)
            
            # Check buffer size
            async with listener.buffer_lock:
                buffer_size = len(listener.audio_buffer)
            
            return buffer_size
        
        buffer_size = asyncio.run(add_chunks_concurrently())
        
        # Should have received all chunks (10 chunks * 1024 samples * 2 bytes per sample = 20480 bytes)
        expected_size = 10 * 1024 * 2
        self.assertEqual(buffer_size, expected_size, 
                        f"Buffer should contain all concurrent chunks. Expected {expected_size}, got {buffer_size}")


if __name__ == '__main__':
    unittest.main()