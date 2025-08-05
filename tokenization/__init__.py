"""
Tokenization module for HRM code generation.

This module wraps the HuggingFace GPT-2 tokenizer for use in code generation tasks,
providing convenience methods for encoding/decoding, handling special tokens,
and preparing inputs for training and generation.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer


_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "tokenizer")
os.makedirs(_CACHE_DIR, exist_ok=True)

# Global tokenizer instance
_tokenizer = None


def get_tokenizer(force_reload: bool = False) -> PreTrainedTokenizer:
    """
    Returns the GPT-2 tokenizer, loading it if necessary.
    
    Args:
        force_reload: If True, reload the tokenizer even if cached
        
    Returns:
        The configured GPT-2 tokenizer
    """
    global _tokenizer
    
    if _tokenizer is None or force_reload:
        # Check for cached tokenizer
        if os.path.exists(os.path.join(_CACHE_DIR, "tokenizer.json")) and not force_reload:
            _tokenizer = AutoTokenizer.from_pretrained(_CACHE_DIR)
        else:
            # Load from HuggingFace and save locally
            _tokenizer = AutoTokenizer.from_pretrained("gpt2", add_bos_token=True)
            _tokenizer.save_pretrained(_CACHE_DIR)
            
    return _tokenizer


def encode(
    text: Union[str, List[str]],
    add_bos: bool = True,
    add_eos: bool = False,
    max_length: Optional[int] = None,
    padding: bool = False,
    truncation: bool = True,
    return_tensors: Optional[str] = None,
) -> Dict[str, Union[List[int], torch.Tensor]]:
    """
    Encodes text using the GPT-2 tokenizer.
    
    Args:
        text: Input text or list of texts
        add_bos: Whether to add the beginning-of-sequence token
        add_eos: Whether to add the end-of-sequence token
        max_length: Maximum sequence length (will truncate if needed)
        padding: Whether to pad sequences to max_length
        truncation: Whether to truncate sequences to max_length
        return_tensors: Output format ('pt' for PyTorch tensors, None for lists)
        
    Returns:
        Dictionary with 'input_ids' and optionally 'attention_mask'
    """
    tokenizer = get_tokenizer()
    
    # Handle single strings
    if isinstance(text, str):
        text = [text]
        
    # Encode with appropriate special tokens
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors=return_tensors,
    )
    
    return encoded


def create_training_batch(
    prompts: List[str],
    completions: List[str],
    max_length: int = 512,
    return_tensors: str = "pt",
) -> Dict[str, torch.Tensor]:
    """
    Creates a training batch with shifted labels for causal language modeling.
    
    Args:
        prompts: List of prompt texts
        completions: List of completion texts (solutions)
        max_length: Maximum sequence length
        return_tensors: Output format ('pt' for PyTorch tensors)
        
    Returns:
        Dictionary with 'input_ids' and 'labels' tensors
    """
    tokenizer = get_tokenizer()
    batch = {"input_ids": [], "labels": []}
    
    for prompt, completion in zip(prompts, completions):
        # Encode prompt and completion
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)
        
        # Combine prompt and completion
        input_ids = prompt_ids + completion_ids
        
        # Truncate if necessary
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        
        # Create labels: -100 for prompt tokens (ignored in loss), shifted for completion
        labels = [-100] * len(prompt_ids) + completion_ids
        
        # Truncate labels if necessary
        if len(labels) > max_length:
            labels = labels[:max_length]
            
        # Pad to max_length if needed
        if len(input_ids) < max_length:
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
            
        batch["input_ids"].append(input_ids)
        batch["labels"].append(labels)
    
    # Convert to tensors if requested
    if return_tensors == "pt":
        batch["input_ids"] = torch.tensor(batch["input_ids"])
        batch["labels"] = torch.tensor(batch["labels"])
    
    return batch


def decode(
    token_ids: Union[List[int], torch.Tensor],
    skip_special_tokens: bool = True,
) -> str:
    """
    Decodes token IDs back to text.
    
    Args:
        token_ids: List or tensor of token IDs
        skip_special_tokens: Whether to skip special tokens in the decoded output
        
    Returns:
        Decoded text
    """
    tokenizer = get_tokenizer()
    
    # Convert tensor to list if needed
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
        
    # Handle batch inputs (first element only)
    if isinstance(token_ids[0], list) or (
        isinstance(token_ids, torch.Tensor) and token_ids.dim() > 1
    ):
        token_ids = token_ids[0]
        
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def get_vocab_size() -> int:
    """Returns the vocabulary size of the tokenizer."""
    return get_tokenizer().vocab_size


def validate_tokens(token_ids: List[int]) -> bool:
    """
    Validates that all tokens are within the vocabulary.
    
    Args:
        token_ids: List of token IDs to validate
        
    Returns:
        True if all tokens are valid, False otherwise
    """
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size
    
    return all(0 <= token_id < vocab_size for token_id in token_ids)


# Export main functions
__all__ = [
    "get_tokenizer",
    "encode",
    "decode",
    "create_training_batch",
    "get_vocab_size",
    "validate_tokens",
]
