"""
Test cases for Voice Computer Server-Client communication.
"""

import unittest
import asyncio
import json
import websockets
import tempfile
import subprocess
import time
import threading
from pathlib import Path
import logging
import signal
import os

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


class TestVoiceServerClient(unittest.TestCase):
    """Test cases for server-client communication."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.server_port = 18765  # Use different port for testing
        cls.server_uri = f"ws://localhost:{cls.server_port}"
        cls.server_process = None
        cls.test_timeout = 30
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if cls.server_process:
            cls._stop_server()
            
    @classmethod
    def _start_server(cls):
        """Start the voice server for testing."""
        try:
            # Start server in background
            cmd = [
                "python", "voice_server.py",
                "--host", "localhost", 
                "--port", str(cls.server_port),
                "--verbose"
            ]
            
            cls.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent.parent
            )
            
            # Wait for server to start
            time.sleep(3)
            
            # Check if server is running
            if cls.server_process.poll() is not None:
                stdout, stderr = cls.server_process.communicate()
                raise Exception(f"Server failed to start: {stderr.decode()}")
                
            _logger.info(f"Test server started on port {cls.server_port}")
            return True
            
        except Exception as e:
            _logger.error(f"Failed to start test server: {e}")
            return False
            
    @classmethod
    def _stop_server(cls):
        """Stop the voice server."""
        if cls.server_process:
            try:
                cls.server_process.terminate()
                cls.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.server_process.kill()
                cls.server_process.wait()
            finally:
                cls.server_process = None
                _logger.info("Test server stopped")
                
    def setUp(self):
        """Set up each test."""
        if not self.server_process:
            self.assertTrue(self._start_server(), "Failed to start test server")
            
    async def _connect_to_server(self):
        """Helper to connect to test server."""
        try:
            websocket = await websockets.connect(self.server_uri)
            
            # Wait for welcome message
            welcome_msg = await asyncio.wait_for(websocket.recv(), timeout=5)
            welcome_data = json.loads(welcome_msg)
            
            self.assertEqual(welcome_data["type"], "welcome")
            return websocket
            
        except Exception as e:
            self.fail(f"Failed to connect to server: {e}")
            
    def test_server_connection(self):
        """Test basic server connection."""
        async def _test():
            websocket = await self._connect_to_server()
            await websocket.close()
            
        asyncio.run(_test())
        
    def test_text_query(self):
        """Test sending a text query to the server."""
        async def _test():
            websocket = await self._connect_to_server()
            
            # Send test query
            test_query = {
                "type": "text_query",
                "query": "Hello, this is a test"
            }
            
            await websocket.send(json.dumps(test_query))
            
            # Wait for query_started response
            response1 = await asyncio.wait_for(websocket.recv(), timeout=10)
            data1 = json.loads(response1)
            self.assertEqual(data1["type"], "query_started")
            
            # Wait for query_response 
            response2 = await asyncio.wait_for(websocket.recv(), timeout=30)
            data2 = json.loads(response2)
            self.assertEqual(data2["type"], "query_response")
            self.assertIn("response", data2)
            
            await websocket.close()
            
        asyncio.run(_test())
        
    def test_server_status(self):
        """Test getting server status."""
        async def _test():
            websocket = await self._connect_to_server()
            
            # Request status
            status_query = {"type": "get_status"}
            await websocket.send(json.dumps(status_query))
            
            # Wait for status response
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            data = json.loads(response)
            
            self.assertEqual(data["type"], "status")
            self.assertIn("conversation_length", data)
            self.assertIn("available_tools", data)
            
            await websocket.close()
            
        asyncio.run(_test())
        
    def test_conversation_reset(self):
        """Test conversation reset functionality."""
        async def _test():
            websocket = await self._connect_to_server()
            
            # Send a query first
            test_query = {
                "type": "text_query", 
                "query": "Remember this test message"
            }
            await websocket.send(json.dumps(test_query))
            
            # Wait for responses
            await asyncio.wait_for(websocket.recv(), timeout=10)  # query_started
            await asyncio.wait_for(websocket.recv(), timeout=30)  # query_response
            
            # Reset conversation
            reset_msg = {"type": "reset_conversation"}
            await websocket.send(json.dumps(reset_msg))
            
            # Wait for reset confirmation
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            data = json.loads(response)
            self.assertEqual(data["type"], "conversation_reset")
            
            await websocket.close()
            
        asyncio.run(_test())
        
    def test_audio_streaming_protocol(self):
        """Test audio streaming message protocol."""
        async def _test():
            websocket = await self._connect_to_server()
            
            # Test start listening
            start_msg = {"type": "start_listening"}
            await websocket.send(json.dumps(start_msg))
            
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            data = json.loads(response)
            self.assertEqual(data["type"], "start_listening")
            
            # Test audio chunk (with dummy data)
            import base64
            dummy_audio = b'\x00' * 1024  # Silent audio data
            encoded_audio = base64.b64encode(dummy_audio).decode('utf-8')
            
            audio_msg = {
                "type": "audio_chunk",
                "audio_data": encoded_audio,
                "sample_rate": 16000,
                "channels": 1,
                "format": "pcm16"
            }
            await websocket.send(json.dumps(audio_msg))
            
            # Test stop listening
            stop_msg = {"type": "stop_listening"}
            await websocket.send(json.dumps(stop_msg))
            
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            data = json.loads(response)
            self.assertEqual(data["type"], "stop_listening")
            
            await websocket.close()
            
        asyncio.run(_test())
        
    def test_invalid_message_handling(self):
        """Test server handling of invalid messages."""
        async def _test():
            websocket = await self._connect_to_server()
            
            # Send invalid message type
            invalid_msg = {"type": "invalid_message_type"}
            await websocket.send(json.dumps(invalid_msg))
            
            # Should get error response
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            data = json.loads(response)
            self.assertEqual(data["type"], "error")
            
            # Send malformed JSON
            await websocket.send("invalid json")
            
            # Should get error response
            response = await asyncio.wait_for(websocket.recv(), timeout=5)
            data = json.loads(response)
            self.assertEqual(data["type"], "error")
            
            await websocket.close()
            
        asyncio.run(_test())
        
    def test_multiple_clients(self):
        """Test multiple clients connecting simultaneously."""
        async def _test():
            # Connect multiple clients
            websocket1 = await self._connect_to_server()
            websocket2 = await self._connect_to_server()
            
            # Send queries from both
            query1 = {"type": "text_query", "query": "Client 1 query"}
            query2 = {"type": "text_query", "query": "Client 2 query"}
            
            await websocket1.send(json.dumps(query1))
            await websocket2.send(json.dumps(query2))
            
            # Both should get responses (order may vary)
            responses = []
            for _ in range(4):  # 2 started + 2 responses
                try:
                    resp1 = await asyncio.wait_for(websocket1.recv(), timeout=1)
                    responses.append(("client1", json.loads(resp1)))
                except asyncio.TimeoutError:
                    pass
                    
                try:
                    resp2 = await asyncio.wait_for(websocket2.recv(), timeout=1)
                    responses.append(("client2", json.loads(resp2)))
                except asyncio.TimeoutError:
                    pass
            
            # Should have received responses for both clients
            started_count = sum(1 for _, resp in responses if resp["type"] == "query_started")
            response_count = sum(1 for _, resp in responses if resp["type"] == "query_response")
            
            self.assertGreaterEqual(started_count, 2)
            self.assertGreaterEqual(response_count, 2)
            
            await websocket1.close()
            await websocket2.close()
            
        asyncio.run(_test())


class TestClientAudioStreaming(unittest.TestCase):
    """Test client audio streaming functionality."""
    
    def test_audio_streamer_initialization(self):
        """Test AudioStreamer can be initialized."""
        # Import the client module
        import sys
        client_path = Path(__file__).parent.parent / "client"
        sys.path.insert(0, str(client_path))
        
        try:
            from voice_client import AudioStreamer
            
            # Test initialization
            streamer = AudioStreamer()
            self.assertEqual(streamer.rate, 16000)
            self.assertEqual(streamer.channels, 1)
            self.assertFalse(streamer.streaming)
            
            # Test cleanup
            streamer.cleanup()
            
        except ImportError as e:
            self.skipTest(f"Client dependencies not available: {e}")
            
    def test_tts_player_initialization(self):
        """Test TTSPlayer can be initialized."""
        import sys
        client_path = Path(__file__).parent.parent / "client"
        sys.path.insert(0, str(client_path))
        
        try:
            from voice_client import TTSPlayer
            
            # Test initialization
            tts = TTSPlayer()
            self.assertIsNone(tts.current_process)
            
            # Test stop (should not crash)
            tts.stop()
            
        except ImportError as e:
            self.skipTest(f"Client dependencies not available: {e}")


def run_server_tests():
    """Run server-specific tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVoiceServerClient)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def run_client_tests():
    """Run client-specific tests."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestClientAudioStreaming)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def run_all_tests():
    """Run all tests."""
    # Run client tests first (no server required)
    print("=" * 60)
    print("Running Client Tests")
    print("=" * 60)
    client_result = run_client_tests()
    
    print("\n" + "=" * 60)
    print("Running Server-Client Integration Tests")
    print("=" * 60)
    print("Note: These tests require Ollama to be running")
    
    # Ask user if they want to run server tests
    try:
        response = input("Run server tests? (requires Ollama) [y/N]: ")
        if response.lower() in ['y', 'yes']:
            server_result = run_server_tests()
        else:
            print("Skipping server tests")
            server_result = None
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return False
        
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Client tests: {'PASSED' if client_result.wasSuccessful() else 'FAILED'}")
    if server_result:
        print(f"Server tests: {'PASSED' if server_result.wasSuccessful() else 'FAILED'}")
    else:
        print("Server tests: SKIPPED")
        
    return client_result.wasSuccessful() and (server_result is None or server_result.wasSuccessful())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Voice Computer Server-Client Tests")
    parser.add_argument("--server-only", action="store_true", help="Run only server tests")
    parser.add_argument("--client-only", action="store_true", help="Run only client tests")
    
    args = parser.parse_args()
    
    if args.server_only:
        result = run_server_tests()
        exit(0 if result.wasSuccessful() else 1)
    elif args.client_only:
        result = run_client_tests()
        exit(0 if result.wasSuccessful() else 1)
    else:
        success = run_all_tests()
        exit(0 if success else 1)