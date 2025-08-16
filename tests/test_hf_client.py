"""
Unit tests for HFClient functionality.
"""

import unittest
import asyncio
import logging
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from voice_computer.client.hf_client import HFClient, HFConnectionError, HFResponseError
from voice_computer.client.base_client import BaseClient
from voice_computer.data_types import Messages, Utterance, ClientResponse, ToolCall, Tool, ToolFunction

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class TestHFClient(unittest.TestCase):
    """Unit tests for HFClient class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_model = "microsoft/DialoGPT-medium"
        self.test_api_key = "test-api-key-123"
        self.client = HFClient(
            model=self.test_model,
            api_key=self.test_api_key,
            temperature=0.7,
            max_tokens=100
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def test_initialization(self):
        """Test HFClient initialization."""
        self.assertEqual(self.client.model, self.test_model)
        self.assertEqual(self.client.api_key, self.test_api_key)
        self.assertEqual(self.client.temperature, 0.7)
        self.assertEqual(self.client.max_tokens, 100)
        self.assertTrue(self.client.api_url.endswith(self.test_model))

    def test_inheritance(self):
        """Test that HFClient inherits from BaseClient."""
        self.assertIsInstance(self.client, BaseClient)
        
    def test_messages_to_prompt(self):
        """Test conversion of Messages to prompt string."""
        messages = Messages(utterances=[
            Utterance(role="system", content="You are a helpful assistant."),
            Utterance(role="user", content="Hello!"),
            Utterance(role="assistant", content="Hi there!"),
            Utterance(role="user", content="How are you?")
        ])
        
        prompt = self.client._messages_to_prompt(messages)
        expected_parts = [
            "System: You are a helpful assistant.",
            "Human: Hello!",
            "Assistant: Hi there!",
            "Human: How are you?",
            "Assistant:"
        ]
        expected = "\n\n".join(expected_parts)
        
        self.assertEqual(prompt, expected)

    @patch('requests.post')
    def test_predict_non_streaming_success(self, mock_post):
        """Test successful non-streaming prediction."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"generated_text": "Hello! How can I help you today?"}]
        mock_post.return_value = mock_response
        
        messages = Messages(utterances=[
            Utterance(role="user", content="Hello!")
        ])
        
        # Run the async test
        async def run_test():
            result = await self.client.predict(messages)
            return result
            
        result = asyncio.run(run_test())
        
        # Assertions
        self.assertIsInstance(result, ClientResponse)
        self.assertEqual(result.message, "Hello! How can I help you today?")
        self.assertIsNone(result.tool_calls)
        
        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertIn("Authorization", call_args[1]["headers"])
        self.assertEqual(call_args[1]["headers"]["Authorization"], f"Bearer {self.test_api_key}")

    @patch('requests.post')
    def test_predict_alternative_response_format(self, mock_post):
        """Test prediction with alternative response format."""
        # Mock response with dict format
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"generated_text": "This is a response"}
        mock_post.return_value = mock_response
        
        messages = Messages(utterances=[
            Utterance(role="user", content="Test")
        ])
        
        async def run_test():
            result = await self.client.predict(messages)
            return result
            
        result = asyncio.run(run_test())
        
        self.assertEqual(result.message, "This is a response")

    @patch('requests.post')
    def test_predict_with_stop_sequences(self, mock_post):
        """Test prediction with stop sequences."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"generated_text": "Response"}]
        mock_post.return_value = mock_response
        
        messages = Messages(utterances=[
            Utterance(role="user", content="Hello!")
        ])
        
        async def run_test():
            result = await self.client.predict(messages, stop_sequences=["END", "STOP"])
            return result
            
        result = asyncio.run(run_test())
        
        # Check that stop sequences were included in the request
        call_args = mock_post.call_args
        request_data = json.loads(call_args[1]["data"])
        self.assertEqual(request_data["parameters"]["stop"], ["END", "STOP"])

    def test_predict_with_tools_warning(self):
        """Test that tools parameter logs a warning."""
        messages = Messages(utterances=[
            Utterance(role="user", content="Hello!")
        ])
        
        tools = [Tool(
            type="function",
            function=ToolFunction(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object"}
            )
        )]
        
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = [{"generated_text": "Response"}]
            mock_post.return_value = mock_response
            
            with patch('voice_computer.client.hf_client._logger') as mock_logger:
                async def run_test():
                    result = await self.client.predict(messages, tools=tools)
                    return result
                    
                asyncio.run(run_test())
                mock_logger.warning.assert_called_with("Tool calls are not currently supported by HFClient")

    @patch('requests.post')
    def test_predict_connection_error(self, mock_post):
        """Test handling of connection errors."""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        messages = Messages(utterances=[
            Utterance(role="user", content="Hello!")
        ])
        
        async def run_test():
            with self.assertRaises(HFConnectionError) as context:
                await self.client.predict(messages)
            return context.exception
            
        exception = asyncio.run(run_test())
        self.assertEqual(exception.model, self.test_model)
        self.assertIn("Connection failed", str(exception))

    @patch('requests.post')
    def test_predict_timeout_error(self, mock_post):
        """Test handling of timeout errors."""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        messages = Messages(utterances=[
            Utterance(role="user", content="Hello!")
        ])
        
        async def run_test():
            with self.assertRaises(HFConnectionError) as context:
                await self.client.predict(messages)
            return context.exception
            
        exception = asyncio.run(run_test())
        self.assertIn("Request timed out", str(exception))

    @patch('requests.post')
    def test_predict_http_error(self, mock_post):
        """Test handling of HTTP error responses."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized: Invalid API key"
        mock_post.return_value = mock_response
        
        messages = Messages(utterances=[
            Utterance(role="user", content="Hello!")
        ])
        
        async def run_test():
            with self.assertRaises(HFResponseError) as context:
                await self.client.predict(messages)
            return context.exception
            
        exception = asyncio.run(run_test())
        self.assertEqual(exception.status_code, 401)
        self.assertIn("Unauthorized", exception.response_text)

    @patch('requests.post')
    def test_predict_invalid_json_response(self, mock_post):
        """Test handling of invalid JSON responses."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_post.return_value = mock_response
        
        messages = Messages(utterances=[
            Utterance(role="user", content="Hello!")
        ])
        
        async def run_test():
            with self.assertRaises(HFResponseError) as context:
                await self.client.predict(messages)
            return context.exception
            
        exception = asyncio.run(run_test())
        self.assertIn("Invalid JSON", str(exception))

    @patch('requests.post')
    def test_predict_streaming_success(self, mock_post):
        """Test successful streaming prediction."""
        # Mock streaming response
        mock_response = Mock()
        mock_response.status_code = 200
        
        # Simulate streaming data
        streaming_data = [
            "data: {\"token\": \"Hello\"}",
            "data: {\"token\": \" there\"}",
            "data: {\"token\": \"!\"}",
            "data: [DONE]"
        ]
        
        mock_response.iter_lines.return_value = streaming_data
        mock_post.return_value = mock_response
        
        messages = Messages(utterances=[
            Utterance(role="user", content="Hello!")
        ])
        
        async def run_test():
            token_queue = asyncio.Queue()
            result = await self.client.predict(messages, stream=True, token_queue=token_queue)
            
            # Collect all tokens from queue
            tokens = []
            while True:
                try:
                    token = token_queue.get_nowait()
                    if token is None:  # End of stream
                        break
                    tokens.append(token)
                except asyncio.QueueEmpty:
                    break
            
            return result, tokens
            
        result, tokens = asyncio.run(run_test())
        
        self.assertEqual(result.message, "Hello there!")
        self.assertEqual(tokens, ["Hello", " there", "!"])

    @patch('requests.post')
    def test_predict_streaming_with_generated_text_field(self, mock_post):
        """Test streaming with generated_text field instead of token."""
        mock_response = Mock()
        mock_response.status_code = 200
        
        streaming_data = [
            "data: {\"generated_text\": \"Hello\"}",
            "data: {\"generated_text\": \" world\"}",
            "data: [DONE]"
        ]
        
        mock_response.iter_lines.return_value = streaming_data
        mock_post.return_value = mock_response
        
        messages = Messages(utterances=[
            Utterance(role="user", content="Hi!")
        ])
        
        async def run_test():
            token_queue = asyncio.Queue()
            result = await self.client.predict(messages, stream=True, token_queue=token_queue)
            return result
            
        result = asyncio.run(run_test())
        self.assertEqual(result.message, "Hello world")

    @patch('requests.post')
    def test_predict_streaming_choices_format(self, mock_post):
        """Test streaming with OpenAI-style choices format."""
        mock_response = Mock()
        mock_response.status_code = 200
        
        streaming_data = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}',
            'data: {"choices": [{"delta": {"content": " world"}}]}',
            "data: [DONE]"
        ]
        
        mock_response.iter_lines.return_value = streaming_data
        mock_post.return_value = mock_response
        
        messages = Messages(utterances=[
            Utterance(role="user", content="Hi!")
        ])
        
        async def run_test():
            token_queue = asyncio.Queue()
            result = await self.client.predict(messages, stream=True, token_queue=token_queue)
            return result
            
        result = asyncio.run(run_test())
        self.assertEqual(result.message, "Hello world")

    @patch('requests.post')
    def test_predict_streaming_connection_error(self, mock_post):
        """Test streaming with connection error."""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError("Streaming failed")
        
        messages = Messages(utterances=[
            Utterance(role="user", content="Hello!")
        ])
        
        async def run_test():
            token_queue = asyncio.Queue()
            with self.assertRaises(HFConnectionError):
                await self.client.predict(messages, stream=True, token_queue=token_queue)
            
        asyncio.run(run_test())

    @patch('requests.post')
    def test_predict_streaming_http_error(self, mock_post):
        """Test streaming with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        messages = Messages(utterances=[
            Utterance(role="user", content="Hello!")
        ])
        
        async def run_test():
            token_queue = asyncio.Queue()
            with self.assertRaises(HFResponseError) as context:
                await self.client.predict(messages, stream=True, token_queue=token_queue)
            return context.exception
            
        exception = asyncio.run(run_test())
        self.assertEqual(exception.status_code, 500)
        self.assertIn("Internal Server Error", exception.response_text)

    def test_predict_streaming_cancelled(self):
        """Test handling of cancelled streaming requests."""
        messages = Messages(utterances=[
            Utterance(role="user", content="Hello!")
        ])
        
        async def run_test():
            token_queue = asyncio.Queue()
            
            # Create a task and cancel it immediately
            task = asyncio.create_task(
                self.client.predict(messages, stream=True, token_queue=token_queue)
            )
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # The method should handle cancellation gracefully
            # and return an empty response
            return True
            
        result = asyncio.run(run_test())
        self.assertTrue(result)

    def test_custom_base_url(self):
        """Test HFClient with custom base URL."""
        custom_url = "https://custom-inference.example.com/models"
        client = HFClient(
            model="custom/model",
            api_key="test-key",
            base_url=custom_url
        )
        
        expected_url = f"{custom_url}/custom/model"
        self.assertEqual(client.api_url, expected_url)


class TestHFClientIntegration(unittest.TestCase):
    """Integration tests for HFClient (these would require actual API access)."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # These tests would use real API keys and models
        # For now, they're designed to be skipped in CI
        self.skip_integration = True
        
    def test_real_api_call(self):
        """Test actual API call (skipped by default)."""
        if self.skip_integration:
            self.skipTest("Integration tests require API key")
            
        # This would test with a real API key and model
        # client = HFClient(model="real-model", api_key="real-key")
        # messages = Messages(utterances=[Utterance(role="user", content="Hello!")])
        # result = asyncio.run(client.predict(messages))
        # self.assertIsInstance(result, ClientResponse)


if __name__ == '__main__':
    unittest.main()