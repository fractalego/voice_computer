"""
Model factory with caching for efficient model loading and reuse.

This factory caches loaded models to prevent redundant loading across different
components, significantly improving startup and query response times.
"""

import logging
import torch
from typing import Dict, Any, Optional, Tuple
from transformers import (
    WhisperProcessor, WhisperForConditionalGeneration,
    SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan,
    AutoModelForSequenceClassification, AutoTokenizer
)
from voice_computer.client import OllamaClient
from voice_computer.client.hf_client import HFClient

_logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating and caching models to avoid redundant loading."""
    
    def __init__(self):
        self._whisper_cache: Dict[str, Tuple[WhisperProcessor, WhisperForConditionalGeneration, str]] = {}
        self._tts_cache: Dict[str, Tuple[SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, str]] = {}
        self._entailment_cache: Dict[str, Tuple[AutoModelForSequenceClassification, AutoTokenizer, str]] = {}
        self._ollama_cache: Dict[str, OllamaClient] = {}
        self._hf_cache: Dict[str, HFClient] = {}
        
    def get_whisper_model(self, model_name: str, device: Optional[str] = None) -> Tuple[WhisperProcessor, WhisperForConditionalGeneration, str]:
        """
        Get cached Whisper model or load if not cached.
        
        Args:
            model_name: HuggingFace model name
            device: Target device (auto-detected if None)
            
        Returns:
            Tuple of (processor, model, device)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        cache_key = f"{model_name}:{device}"
        
        if cache_key not in self._whisper_cache:
            _logger.info(f"Loading Whisper model: {model_name} on device: {device}")
            
            # Load processor and model
            processor = WhisperProcessor.from_pretrained(model_name)
            model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
            # Move model to device and optimize
            model = model.to(device)
            
            # Use half precision for better performance (if using CUDA)
            if device == "cuda":
                model = model.half()
            
            # Compile model for better performance (PyTorch 2.0+)
            try:
                model = torch.compile(model, mode="reduce-overhead")
                _logger.info("Whisper model compiled successfully")
            except Exception as e:
                _logger.warning(f"Whisper model compilation failed, continuing without: {e}")
            
            # Set to eval mode
            model.eval()
            
            self._whisper_cache[cache_key] = (processor, model, device)
            _logger.info(f"Whisper model {model_name} cached successfully")
        else:
            _logger.debug(f"Using cached Whisper model: {model_name}")
            
        return self._whisper_cache[cache_key]
    
    def get_tts_model(self, model_name: str = "microsoft/speecht5_tts", device: Optional[str] = None) -> Tuple[SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, str]:
        """
        Get cached TTS model or load if not cached.
        
        Args:
            model_name: HuggingFace model name
            device: Target device (auto-detected if None)
            
        Returns:
            Tuple of (processor, model, vocoder, device)
        """
        if device is None:
            device = self._get_best_device()
            
        cache_key = f"{model_name}:{device}"
        
        if cache_key not in self._tts_cache:
            _logger.info(f"Loading TTS model: {model_name} on device: {device}")
            
            torch_dtype = torch.bfloat16 if device != "cpu" else torch.float32
            
            # Load processor, model, and vocoder
            processor = SpeechT5Processor.from_pretrained(model_name)
            model = SpeechT5ForTextToSpeech.from_pretrained(model_name)
            vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Move models to device and set dtype
            model = model.to(device)
            vocoder = vocoder.to(device)
            
            if device != "cpu":
                model = model.to(dtype=torch_dtype)
                vocoder = vocoder.to(dtype=torch_dtype)
            
            model.eval()
            vocoder.eval()
            
            self._tts_cache[cache_key] = (processor, model, vocoder, device)
            _logger.info(f"TTS model {model_name} cached successfully")
        else:
            _logger.debug(f"Using cached TTS model: {model_name}")
            
        return self._tts_cache[cache_key]
    
    def get_entailment_model(self, model_name: str, device: Optional[str] = None) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, str]:
        """
        Get cached entailment model or load if not cached.
        
        Args:
            model_name: HuggingFace model name
            device: Target device (auto-detected if None)
            
        Returns:
            Tuple of (model, tokenizer, device)
        """
        if device is None:
            device = self._get_best_device()
            
        cache_key = f"{model_name}:{device}"
        
        if cache_key not in self._entailment_cache:
            _logger.info(f"Loading entailment model: {model_name} on device: {device}")
            
            # Load model
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # Handle tokenizer loading (special case for vectara model)
            tokenizer = None
            if "vectara/hallucination_evaluation_model" in model_name:
                try:
                    # Try to get tokenizer from model object
                    if hasattr(model, 'tokenizer'):
                        tokenizer = model.tokenizer
                    elif hasattr(model, 'config') and hasattr(model.config, 'tokenizer'):
                        tokenizer = model.config.tokenizer
                    else:
                        # Fallback to T5 tokenizer
                        from transformers import T5Tokenizer
                        tokenizer = T5Tokenizer.from_pretrained("t5-base")
                        _logger.warning("Using T5 base tokenizer as fallback for vectara model")
                except Exception as e:
                    _logger.error(f"Failed to load tokenizer for vectara model: {e}")
                    # Final fallback - use a basic BERT tokenizer
                    from transformers import BertTokenizer
                    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                    _logger.warning("Using BERT base tokenizer as final fallback for vectara model")
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
            
            # Move model to device and set to eval mode
            model = model.to(device)
            if hasattr(model, 'bfloat16') and device != "cpu":
                model = model.bfloat16()
            model.eval()
            
            self._entailment_cache[cache_key] = (model, tokenizer, device)
            _logger.info(f"Entailment model {model_name} cached successfully")
        else:
            _logger.debug(f"Using cached entailment model: {model_name}")
            
        return self._entailment_cache[cache_key]
    
    def get_ollama_client(self, model: str, host: str = "http://localhost:11434", **kwargs) -> OllamaClient:
        """
        Get cached Ollama client or create if not cached.
        
        Args:
            model: Ollama model name
            host: Ollama host URL
            **kwargs: Additional OllamaClient parameters
            
        Returns:
            OllamaClient instance
        """
        # Create cache key from all relevant parameters
        cache_key = f"{model}:{host}:{hash(frozenset(kwargs.items()))}"
        
        if cache_key not in self._ollama_cache:
            _logger.info(f"Creating Ollama client for model: {model} at host: {host}")
            
            client = OllamaClient(
                model=model,
                host=host,
                **kwargs
            )
            
            self._ollama_cache[cache_key] = client
            _logger.info(f"Ollama client for {model} cached successfully")
        else:
            _logger.debug(f"Using cached Ollama client for model: {model}")
            
        return self._ollama_cache[cache_key]
    
    def get_hf_client(self, model: str, **kwargs) -> HFClient:
        """
        Get cached HuggingFace client or create if not cached.
        
        Args:
            model: HuggingFace model name
            **kwargs: Additional HFClient parameters
            
        Returns:
            HFClient instance
        """
        # Create cache key from all relevant parameters
        cache_key = f"{model}:{hash(frozenset(kwargs.items()))}"
        
        if cache_key not in self._hf_cache:
            _logger.info(f"Creating local HuggingFace client for model: {model}")
            
            client = HFClient(
                model=model,
                **kwargs
            )
            
            self._hf_cache[cache_key] = client
            _logger.info(f"Local HuggingFace client for {model} cached successfully")
        else:
            _logger.debug(f"Using cached local HuggingFace client for model: {model}")
            
        return self._hf_cache[cache_key]
    
    def get_llm_client(self, config_dict: Dict[str, Any]):
        """
        Get the appropriate LLM client based on configuration.
        
        Args:
            config_dict: Configuration dictionary containing client type and settings
            
        Returns:
            OllamaClient or HFClient instance
        """
        client_type = config_dict.get("llm_client_type", "ollama")
        
        if client_type == "ollama":
            model = config_dict.get("ollama_model", "qwen2.5:32b")
            host = config_dict.get("ollama_host", "http://localhost:11434")
            return self.get_ollama_client(model=model, host=host)
        
        elif client_type == "huggingface":
            model = config_dict.get("huggingface_model", "Qwen/Qwen2.5-32B")
            device = config_dict.get("huggingface_device")
            torch_dtype = config_dict.get("huggingface_torch_dtype")
            quantization = config_dict.get("huggingface_quantization")
            return self.get_hf_client(
                model=model, 
                device=device, 
                torch_dtype=torch_dtype,
                quantization=quantization
            )
        
        else:
            raise ValueError(f"Unknown LLM client type: {client_type}. Must be 'ollama' or 'huggingface'")
    
    def _get_best_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def clear_cache(self, model_type: Optional[str] = None):
        """
        Clear cached models.
        
        Args:
            model_type: Type of models to clear ('whisper', 'tts', 'entailment', 'ollama', 'hf', or None for all)
        """
        if model_type is None or model_type == "whisper":
            self._whisper_cache.clear()
            _logger.info("Cleared Whisper model cache")
            
        if model_type is None or model_type == "tts":
            self._tts_cache.clear()
            _logger.info("Cleared TTS model cache")
            
        if model_type is None or model_type == "entailment":
            self._entailment_cache.clear()
            _logger.info("Cleared entailment model cache")
            
        if model_type is None or model_type == "ollama":
            self._ollama_cache.clear()
            _logger.info("Cleared Ollama client cache")
            
        if model_type is None or model_type == "hf":
            self._hf_cache.clear()
            _logger.info("Cleared HuggingFace client cache")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get information about cached models."""
        return {
            "whisper_models": len(self._whisper_cache),
            "tts_models": len(self._tts_cache),
            "entailment_models": len(self._entailment_cache),
            "ollama_clients": len(self._ollama_cache),
            "hf_clients": len(self._hf_cache)
        }


# Global model factory instance
_model_factory = ModelFactory()


def get_model_factory() -> ModelFactory:
    """Get the global model factory instance."""
    return _model_factory