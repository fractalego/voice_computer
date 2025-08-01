"""
Entailment model for judging whether one text entails another.

Uses HuggingFace transformers to load and run entailment models.
"""

import logging
import torch
from typing import Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as transformers_logging
from .config import Config

_logger = logging.getLogger(__name__)


class Entailer:
    """Entailment model for judging text entailment relationships."""
    
    def __init__(self, config: Optional[Config] = None, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the entailer with a specified model.
        
        Args:
            config: Configuration object containing model settings
            model_name: HuggingFace model name for entailment evaluation (overrides config)
            device: Device to run the model on (overrides config, auto-detected if None)
        """
        if config:
            self.model_name = model_name or config.get_value("entailment_model") or "vectara/hallucination_evaluation_model"
            self.device = device or config.get_value("entailment_device") or self._get_best_device()
        else:
            self.model_name = model_name or "vectara/hallucination_evaluation_model"
            self.device = device or self._get_best_device()
        
        self.model = None
        self.tokenizer = None
        self.initialized = False
        
    def _get_best_device(self) -> str:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def initialize(self):
        """Load the model and tokenizer."""
        if self.initialized:
            return
            
        _logger.info(f"Loading entailment model {self.model_name} on device {self.device}")
        
        # Temporarily suppress transformers warnings about unsupported generation flags
        original_verbosity = transformers_logging.get_verbosity()
        transformers_logging.set_verbosity_error()
        
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            # Handle special case for vectara model which has custom tokenizer
            if "vectara/hallucination_evaluation_model" in self.model_name:
                # For vectara model, we need to access the tokenizer differently
                # The model contains the tokenizer, not as a separate component
                try:
                    # Try to get tokenizer from model object
                    if hasattr(self.model, 'tokenizer'):
                        self.tokenizer = self.model.tokenizer
                    elif hasattr(self.model, 'config') and hasattr(self.model.config, 'tokenizer'):
                        self.tokenizer = self.model.config.tokenizer
                    else:
                        # For this model, we might need to manually create the tokenizer
                        # Let's use T5 tokenizer as base since it's likely T5-based
                        from transformers import T5Tokenizer
                        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
                        _logger.warning("Using T5 base tokenizer as fallback for vectara model")
                except Exception as e:
                    _logger.error(f"Failed to load tokenizer for vectara model: {e}")
                    # Final fallback - use a basic BERT tokenizer
                    from transformers import BertTokenizer
                    self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                    _logger.warning("Using BERT base tokenizer as final fallback for vectara model")
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
            
            # Move model to device and set to eval mode
            self.model = self.model.to(self.device)
            if hasattr(self.model, 'bfloat16') and self.device != "cpu":
                self.model = self.model.bfloat16()
            self.model.eval()
            
            self.initialized = True
            _logger.info(f"Entailment model {self.model_name} loaded successfully on {self.device}")
            
        except Exception as e:
            _logger.error(f"Failed to load entailment model {self.model_name}: {e}")
            raise
        finally:
            # Restore original transformers verbosity
            transformers_logging.set_verbosity(original_verbosity)
    
    def judge(self, lhs: str, rhs: str) -> float:
        """
        Judge whether the left-hand side entails the right-hand side.
        
        Args:
            lhs: The premise text (left-hand side)
            rhs: The hypothesis text (right-hand side)
            
        Returns:
            Float score between 0 and 1, where higher values indicate stronger entailment
        """
        if not self.initialized:
            self.initialize()
        
        try:
            # Special handling for vectara model
            if "vectara/hallucination_evaluation_model" in self.model_name:
                # For vectara model, use the specific format it expects
                if hasattr(self.model.config, 'prompt'):
                    input_text = self.model.config.prompt.format(text1=lhs, text2=rhs)
                else:
                    # Default format for hallucination evaluation
                    input_text = f"{lhs}\n{rhs}"
                
                inputs = self.tokenizer(
                    [input_text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Run inference for vectara model
                with torch.no_grad():
                    if hasattr(self.model, 't5'):
                        # T5-based vectara model
                        outputs = self.model.t5(**inputs)
                        logits = outputs.logits[:, 0, :]
                    else:
                        # Direct model call
                        outputs = self.model(**inputs)
                        logits = outputs.logits
                    
                    probs = torch.softmax(logits, dim=-1)
                    score = float(probs[0, 1]) if probs.shape[-1] >= 2 else float(probs[0, 0])
                    return score
            else:
                # Standard entailment models
                # Prepare the input based on the model's expected format
                if hasattr(self.model.config, 'prompt'):
                    # Use model-specific prompt format if available
                    input_text = self.model.config.prompt.format(text1=lhs, text2=rhs)
                    inputs = self.tokenizer(
                        [input_text],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                else:
                    # Standard entailment format: premise [SEP] hypothesis
                    inputs = self.tokenizer(
                        lhs, rhs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    
                    # Convert logits to probabilities
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Return entailment score (assuming index 1 is entailment class)
                    if probs.shape[-1] >= 2:
                        score = float(probs[0, 1])  # Entailment probability
                    else:
                        score = float(probs[0, 0])  # Single output case
                    
                    return score
                
        except Exception as e:
            _logger.error(f"Error during entailment judgment: {e}")
            raise
    
    def batch_judge(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Judge multiple entailment pairs in batch.
        
        Args:
            pairs: List of (lhs, rhs) tuples
            
        Returns:
            List of entailment scores
        """
        if not self.initialized:
            self.initialize()
        
        scores = []
        for lhs, rhs in pairs:
            score = self.judge(lhs, rhs)
            scores.append(score)
        
        return scores
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer