"""
Unit tests for TTSSpeaker functionality.
"""

import unittest
import logging
import sys
import os
import time

# Add the parent directory to the path so we can import voice_computer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from voice_computer.speaker import TTSSpeaker

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class TestTTSSpeaker(unittest.TestCase):
    """Unit tests for TTSSpeaker class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tts_speaker = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def tearDown(self):
        """Clean up after each test method."""
        if self.tts_speaker is not None:
            try:
                # Stop streaming if active
                if hasattr(self.tts_speaker, '_is_streaming') and self.tts_speaker._is_streaming:
                    self.tts_speaker.stop_streaming_speech()
                
                # Clean up resources
                self.tts_speaker.cleanup()
                self.logger.info("TTSSpeaker cleaned up successfully")
            except Exception as e:
                self.logger.warning(f"Error during cleanup: {e}")
            finally:
                self.tts_speaker = None
    
    def test_tts_speaker_initialization(self):
        """Test TTSSpeaker initialization."""
        self.logger.info("Testing TTSSpeaker initialization...")
        
        # Create and initialize TTSSpeaker
        self.tts_speaker = TTSSpeaker()
        self.assertIsNotNone(self.tts_speaker)
        self.assertFalse(self.tts_speaker.initialized)
        
        # Initialize the speaker
        self.tts_speaker.initialize()
        self.assertTrue(self.tts_speaker.initialized)
        
        # Check that components are loaded
        self.assertIsNotNone(self.tts_speaker._processor)
        self.assertIsNotNone(self.tts_speaker._model)
        self.assertIsNotNone(self.tts_speaker._vocoder)
        self.assertIsNotNone(self.tts_speaker._speaker_embedding)
        self.assertIsNotNone(self.tts_speaker._pyaudio)
        
        self.logger.info("TTSSpeaker initialization test passed")
    
    def test_tts_speaker_hello_synthesis(self):
        """Test TTSSpeaker basic speech synthesis with 'Hello there'."""
        self.logger.info("Testing TTSSpeaker speech synthesis...")
        
        # Create and initialize TTSSpeaker
        self.tts_speaker = TTSSpeaker()
        self.tts_speaker.initialize()
        
        # Test synthesis - this should not raise an exception
        try:
            self.tts_speaker.speak("Hello there")
            self.logger.info("Speech synthesis completed successfully")
        except Exception as e:
            self.fail(f"Speech synthesis failed: {e}")
        
        self.logger.info("TTSSpeaker speech synthesis test passed")
    
    def test_tts_speaker_streaming_mode(self):
        """Test TTSSpeaker streaming functionality."""
        self.logger.info("Testing TTSSpeaker streaming mode...")
        
        # Create and initialize TTSSpeaker
        self.tts_speaker = TTSSpeaker()
        self.tts_speaker.initialize()
        
        # Test starting streaming mode
        self.assertFalse(self.tts_speaker._is_streaming)
        
        self.tts_speaker.start_streaming_speech()
        self.assertTrue(self.tts_speaker._is_streaming)
        
        # Add text batches
        test_phrases = ["Hello", " there!", " This is a streaming test."]
        for phrase in test_phrases:
            self.tts_speaker.add_text_batch(phrase)
        
        # Wait for processing
        time.sleep(2)
        
        # Stop streaming
        self.tts_speaker.stop_streaming_speech()
        self.assertFalse(self.tts_speaker._is_streaming)
        
        self.logger.info("TTSSpeaker streaming mode test passed")
    
    def test_tts_speaker_multiple_synthesis(self):
        """Test multiple speech synthesis calls."""
        self.logger.info("Testing multiple speech synthesis calls...")
        
        # Create and initialize TTSSpeaker
        self.tts_speaker = TTSSpeaker()
        self.tts_speaker.initialize()
        
        # Test multiple synthesis calls
        test_phrases = [
            "Hello there",
            "This is a test",
            "Multiple synthesis works"
        ]
        
        for i, phrase in enumerate(test_phrases):
            try:
                self.logger.info(f"Synthesizing phrase {i+1}: '{phrase}'")
                self.tts_speaker.speak(phrase)
            except Exception as e:
                self.fail(f"Multiple synthesis failed on phrase {i+1}: {e}")
        
        self.logger.info("Multiple speech synthesis test passed")
    
    def test_tts_speaker_device_detection(self):
        """Test device detection functionality."""
        self.logger.info("Testing device detection...")
        
        self.tts_speaker = TTSSpeaker()
        
        # Test device detection
        device = self.tts_speaker._get_best_device()
        self.assertIn(device, ['cpu', 'cuda', 'mps'])
        
        self.logger.info(f"Detected device: {device}")
        self.logger.info("Device detection test passed")
    
    def test_tts_speaker_speaker_embedding_loading(self):
        """Test speaker embedding loading."""
        self.logger.info("Testing speaker embedding loading...")
        
        self.tts_speaker = TTSSpeaker()
        
        # Test speaker embedding loading
        embedding = self.tts_speaker._load_speaker_embedding()
        self.assertIsNotNone(embedding)
        
        # Check embedding shape (should be [1, 512] for SpeechT5)
        self.assertEqual(len(embedding.shape), 2)
        self.assertEqual(embedding.shape[0], 1)
        self.assertEqual(embedding.shape[1], 512)
        
        self.logger.info(f"Speaker embedding shape: {embedding.shape}")
        self.logger.info("Speaker embedding loading test passed")
    
    def test_tts_speaker_cleanup(self):
        """Test proper cleanup of TTSSpeaker resources."""
        self.logger.info("Testing TTSSpeaker cleanup...")
        
        # Create and initialize TTSSpeaker
        self.tts_speaker = TTSSpeaker()
        self.tts_speaker.initialize()
        
        # Verify components exist
        self.assertIsNotNone(self.tts_speaker._processor)
        self.assertIsNotNone(self.tts_speaker._model)
        self.assertIsNotNone(self.tts_speaker._vocoder)
        self.assertIsNotNone(self.tts_speaker._pyaudio)
        
        # Test cleanup
        self.tts_speaker.cleanup()
        
        # Verify cleanup (components should be None after cleanup)
        self.assertIsNone(self.tts_speaker._processor)
        self.assertIsNone(self.tts_speaker._model)
        self.assertIsNone(self.tts_speaker._vocoder)
        self.assertIsNone(self.tts_speaker._pyaudio)
        
        # Set to None to prevent tearDown from trying to clean up again
        self.tts_speaker = None
        
        self.logger.info("TTSSpeaker cleanup test passed")


class TestTTSSpeakerIntegration(unittest.TestCase):
    """Integration tests for TTSSpeaker."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.tts_speaker = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def tearDown(self):
        """Clean up after integration tests."""
        if self.tts_speaker is not None:
            try:
                if hasattr(self.tts_speaker, '_is_streaming') and self.tts_speaker._is_streaming:
                    self.tts_speaker.stop_streaming_speech()
                self.tts_speaker.cleanup()
            except Exception as e:
                self.logger.warning(f"Error during integration test cleanup: {e}")
            finally:
                self.tts_speaker = None
    
    def test_full_tts_workflow(self):
        """Test complete TTS workflow from initialization to speech."""
        self.logger.info("Testing full TTS workflow...")
        
        # Complete workflow test
        self.tts_speaker = TTSSpeaker()
        
        # Step 1: Initialize
        self.tts_speaker.initialize()
        self.assertTrue(self.tts_speaker.initialized)
        
        # Step 2: Speak simple phrase
        self.tts_speaker.speak("Hello there")
        
        # Step 3: Test streaming
        self.tts_speaker.start_streaming_speech()
        self.tts_speaker.add_text_batch("This is a complete workflow test.")
        time.sleep(1)
        self.tts_speaker.stop_streaming_speech()
        
        # Step 4: Speak another phrase
        self.tts_speaker.speak("Workflow completed successfully")
        
        self.logger.info("Full TTS workflow test passed")


def run_tests():
    """Run all TTSSpeaker tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTTSSpeaker))
    suite.addTests(loader.loadTestsFromTestCase(TestTTSSpeakerIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 80)
    print("TTSSpeaker Unit Test Suite")
    print("=" * 80)
    
    success = run_tests()
    
    print("\n" + "=" * 80)
    if success:
        print("All tests passed successfully! 🎉")
    else:
        print("Some tests failed! ❌")
    print("=" * 80)
    
    sys.exit(0 if success else 1)