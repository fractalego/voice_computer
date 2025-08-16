"""
Test for local HuggingFace client functionality.
"""

import unittest
from unittest.mock import Mock, patch
from voice_computer.client.hf_client import HFClient, HFModelError
from voice_computer.data_types import Messages, Utterance


class TestLocalHFClient(unittest.TestCase):
    """Test local HuggingFace client."""

    @patch('voice_computer.client.hf_client.AutoTokenizer')
    @patch('voice_computer.client.hf_client.AutoModelForCausalLM')
    @patch('torch.cuda.is_available')
    def test_hf_client_initialization_cpu(self, mock_cuda, mock_model, mock_tokenizer):
        """Test HF client initialization on CPU."""
        # Mock CUDA not available
        mock_cuda.return_value = False
        
        # Mock tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Initialize client
        client = HFClient("test-model")
        
        # Verify initialization
        self.assertEqual(client.model, "test-model")
        self.assertEqual(client.device, "cpu")
        self.assertIsNotNone(client.tokenizer)
        self.assertIsNotNone(client.model_instance)
        
        # Verify model loading was called
        mock_tokenizer.from_pretrained.assert_called_once_with("test-model")
        mock_model.from_pretrained.assert_called_once()

    @patch('voice_computer.client.hf_client.AutoTokenizer')
    @patch('voice_computer.client.hf_client.AutoModelForCausalLM')
    @patch('torch.cuda.is_available')
    def test_hf_client_initialization_cuda(self, mock_cuda, mock_model, mock_tokenizer):
        """Test HF client initialization on CUDA."""
        # Mock CUDA available
        mock_cuda.return_value = True
        
        # Mock tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        # Initialize client
        client = HFClient("test-model")
        
        # Verify device is set to CUDA
        self.assertEqual(client.device, "cuda")

    def test_messages_to_prompt(self):
        """Test message to prompt conversion."""
        # Mock the model loading to avoid actual loading
        with patch('voice_computer.client.hf_client.AutoTokenizer'), \
             patch('voice_computer.client.hf_client.AutoModelForCausalLM'), \
             patch('torch.cuda.is_available', return_value=False):
            
            client = HFClient("test-model")
            
            # Create test messages
            messages = Messages(utterances=[
                Utterance(role="system", content="You are an assistant"),
                Utterance(role="user", content="Hello"),
                Utterance(role="assistant", content="Hi there"),
                Utterance(role="user", content="How are you?")
            ])
            
            # Convert to prompt
            prompt = client._messages_to_prompt(messages)
            
            # Verify prompt format
            expected = ("System: You are an assistant\n\n"
                       "Human: Hello\n\n"
                       "Assistant: Hi there\n\n"
                       "Human: How are you?\n\n"
                       "Assistant:")
            
            self.assertEqual(prompt, expected)

    @patch('voice_computer.client.hf_client.AutoTokenizer')
    @patch('voice_computer.client.hf_client.AutoModelForCausalLM')
    def test_model_loading_error(self, mock_model, mock_tokenizer):
        """Test model loading error handling."""
        # Mock tokenizer loading to fail
        mock_tokenizer.from_pretrained.side_effect = Exception("Model not found")
        
        # Should raise HFModelError
        with self.assertRaises(HFModelError) as context:
            HFClient("invalid-model")
        
        self.assertIn("invalid-model", str(context.exception))


if __name__ == '__main__':
    unittest.main()