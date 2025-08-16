"""
HuggingFace client for running local models using transformers library.
"""

import torch
import logging
import asyncio
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

from voice_computer.client.base_client import BaseClient
from voice_computer.data_types import Messages, Tool, ClientResponse, ToolCall

_logger = logging.getLogger(__name__)


class HFModelError(Exception):
    """Exception raised when there's an error with the local HuggingFace model."""

    def __init__(self, model: str, original_error: Exception):
        self.model = model
        self.original_error = original_error
        super().__init__(
            f"Failed to load or run local HuggingFace model '{model}': {original_error}"
        )


class HFClient(BaseClient):
    """Client for local HuggingFace models using transformers library."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        quantization: Optional[str] = None,
    ):
        super().__init__(model, temperature, max_tokens)
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Set quantization mode
        self.quantization = quantization or "bfloat16"  # Default to bfloat16
        
        # Set data type based on quantization
        if torch_dtype is not None:
            # Explicit dtype specified
            if torch_dtype == "bfloat16":
                self.torch_dtype = torch.bfloat16
            elif torch_dtype == "float16":
                self.torch_dtype = torch.float16
            elif torch_dtype == "float32":
                self.torch_dtype = torch.float32
            else:
                self.torch_dtype = torch.bfloat16
        else:
            # Auto-select based on device and quantization
            if self.quantization in ["8bit", "4bit"]:
                self.torch_dtype = None  # Quantization handles dtype
            elif self.device == "cuda":
                self.torch_dtype = torch.bfloat16  # Default to bfloat16 for CUDA
            else:
                self.torch_dtype = torch.float32  # CPU fallback

        _logger.info(f"Initializing HFClient for local model '{model}' on device '{self.device}' with quantization '{self.quantization}'")
        
        # Initialize model and tokenizer
        self.model_instance = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the HuggingFace model and tokenizer locally."""
        try:
            _logger.info(f"🚀 Starting HuggingFace local model initialization for '{self.model}'")
            
            # Try offline first, then fallback to online if needed
            try:
                _logger.info(f"📁 Attempting to load from local cache (offline mode)...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model, local_files_only=True)
                
                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                _logger.info(f"📁 Loading model '{self.model}' from local cache on device '{self.device}' with {self.quantization} precision...")
                self.model_instance = self._load_model_with_quantization(local_files_only=True)
                _logger.info(f"✅ Successfully loaded model from local cache!")
                
            except (OSError, ValueError) as offline_error:
                _logger.warning(f"⚠️ Model not found in local cache: {offline_error}")
                _logger.info(f"📥 Downloading/loading tokenizer... (this may take a moment)")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model)
                
                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                _logger.info(f"📥 Downloading/loading model '{self.model}' on device '{self.device}' with {self.quantization} precision...")
                _logger.info(f"⏳ This may take several minutes for large models or first-time downloads...")
                self.model_instance = self._load_model_with_quantization(local_files_only=False)
            
            
            if self.device == "cpu":
                self.model_instance = self.model_instance.to(self.device)
                
            _logger.info(f"✅ Successfully loaded local HuggingFace model '{self.model}' on device '{self.device}'")
            _logger.info(f"🔧 Model is now ready for local inference (no internet required)")
            
        except Exception as e:
            _logger.error(f"❌ Failed to load model '{self.model}': {e}")
            _logger.error(f"💡 Tip: If you're getting network errors, try using a smaller model or ensure internet connectivity")
            raise HFModelError(self.model, e)

    def _load_model_with_quantization(self, local_files_only: bool = False):
        """Load model with specified quantization configuration."""
        try:
            # Prepare model loading arguments
            model_kwargs = {
                "trust_remote_code": True,
                "local_files_only": local_files_only,
            }
            
            if self.quantization == "8bit":
                _logger.info("🔧 Loading with 8-bit quantization (requires bitsandbytes)")
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False,
                    )
                    model_kwargs.update({
                        "quantization_config": quantization_config,
                        "device_map": "auto",
                    })
                except ImportError:
                    _logger.error("❌ bitsandbytes not installed. Install with: pip install bitsandbytes")
                    raise
                    
            elif self.quantization == "4bit":
                _logger.info("🔧 Loading with 4-bit quantization (requires bitsandbytes)")
                try:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    model_kwargs.update({
                        "quantization_config": quantization_config,
                        "device_map": "auto",
                    })
                except ImportError:
                    _logger.error("❌ bitsandbytes not installed. Install with: pip install bitsandbytes")
                    raise
                    
            else:
                # Standard precision loading (bfloat16, float16, float32)
                _logger.info(f"🔧 Loading with {self.quantization} precision")
                model_kwargs.update({
                    "torch_dtype": self.torch_dtype,
                    "device_map": self.device if self.device != "cpu" else None,
                })
                
            # Load the model
            model = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)
            
            # For non-quantized models on CPU, move explicitly
            if self.quantization not in ["8bit", "4bit"] and self.device == "cpu":
                model = model.to(self.device)
                
            return model
            
        except Exception as e:
            _logger.error(f"❌ Failed to load model with {self.quantization} quantization: {e}")
            raise

    def _messages_to_prompt(self, messages: Messages) -> str:
        """
        Convert Messages to a single prompt string for HF models that don't support chat format.
        """
        prompt_parts = []
        for msg in messages.utterances:
            role = msg.role
            content = msg.content
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

    async def predict(
        self,
        messages: Messages,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Tool]] = None,
        stream: bool = False,
        token_queue: Optional[asyncio.Queue] = None,
    ) -> ClientResponse:
        """
        Generate a response using the local HuggingFace model.
        
        Args:
            messages: The conversation messages
            stop_sequences: Optional list of stop sequences
            tools: Optional list of tools available to the model
            stream: Whether to stream the response
            token_queue: Asyncio queue for streaming tokens in real-time
            
        Returns:
            ClientResponse containing the response message and any tool calls
        """
        _logger.debug(f"Generating response with local HuggingFace model '{self.model}'")

        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages)
        
        # Note: Tools are not directly supported by most HF models
        # This would require model-specific formatting or fine-tuned models
        if tools:
            _logger.warning("Tool calls are not currently supported by HFClient")

        try:
            if stream and token_queue:
                return await self._handle_streaming_generation(prompt, stop_sequences, token_queue)
            else:
                return await self._handle_non_streaming_generation(prompt, stop_sequences)
        except asyncio.exceptions.CancelledError:
            _logger.debug("HuggingFace generation was cancelled")
            return ClientResponse(message="", tool_calls=None)
        except Exception as e:
            error_msg = f"Error generating response with local model {self.model}: {e}"
            _logger.error(error_msg)
            raise HFModelError(self.model, e)

    async def _handle_non_streaming_generation(
        self, 
        prompt: str, 
        stop_sequences: Optional[List[str]] = None
    ) -> ClientResponse:
        """
        Generate a complete response without streaming.
        
        Args:
            prompt: The input prompt
            stop_sequences: Optional list of stop sequences
            
        Returns:
            ClientResponse with complete message
        """
        def _generate():
            """Run generation in a thread."""
            try:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model_instance.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        stop_strings=stop_sequences,
                    )
                
                # Decode response
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                return ClientResponse(message=response_text, tool_calls=None)
                
            except Exception as e:
                raise HFModelError(self.model, e)
        
        # Run generation in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(executor, _generate)
            return await future

    async def _handle_streaming_generation(
        self, 
        prompt: str, 
        stop_sequences: Optional[List[str]] = None,
        token_queue: Optional[asyncio.Queue] = None
    ) -> ClientResponse:
        """
        Generate a streaming response token by token.
        
        Args:
            prompt: The input prompt
            stop_sequences: Optional list of stop sequences
            token_queue: Asyncio queue to put tokens into as they arrive
            
        Returns:
            ClientResponse with complete message
        """
        def _stream_generate(loop):
            """Run streaming generation in a thread."""
            try:
                # Tokenize input
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Custom streamer that puts tokens in queue
                class AsyncStreamer(TextStreamer):
                    def __init__(self, tokenizer, token_queue, loop):
                        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
                        self.token_queue = token_queue
                        self.loop = loop
                        self.complete_text = ""

                    def on_finalized_text(self, text: str, stream_end: bool = False):
                        self.complete_text += text
                        if text and not stream_end:
                            # Put token in queue for immediate display
                            asyncio.run_coroutine_threadsafe(
                                self.token_queue.put(text), 
                                self.loop
                            )
                        elif stream_end:
                            # Signal completion
                            asyncio.run_coroutine_threadsafe(
                                self.token_queue.put(None),
                                self.loop
                            )
                
                streamer = AsyncStreamer(self.tokenizer, token_queue, loop)
                
                # Generate response with streaming
                with torch.no_grad():
                    outputs = self.model_instance.generate(
                        **inputs,
                        max_new_tokens=self.max_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        streamer=streamer,
                        stop_strings=stop_sequences,
                    )
                
                return ClientResponse(message=streamer.complete_text, tool_calls=None)
                
            except Exception as e:
                # Signal error by putting None in queue
                asyncio.run_coroutine_threadsafe(
                    token_queue.put(None),
                    loop
                )
                raise HFModelError(self.model, e)
        
        # Run streaming generation in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            future = loop.run_in_executor(executor, _stream_generate, loop)
            return await future