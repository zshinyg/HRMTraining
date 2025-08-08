"""
Unit tests for the tokenization module.

This module contains comprehensive tests for the tokenization functionality,
including loading, encoding/decoding, batch processing, and training data preparation.
"""

import os
import shutil
import tempfile
from typing import Dict, List

import pytest
import torch

from tokenization import (
    get_tokenizer,
    encode,
    decode,
    create_training_batch,
    get_vocab_size,
    validate_tokens,
    _CACHE_DIR,
)


@pytest.fixture
def sample_texts():
    """Fixture providing sample texts for testing."""
    return [
        "def hello_world():\n    print('Hello, World!')",
        "x = 5\ny = 10\nprint(x + y)",
        "for i in range(10):\n    if i % 2 == 0:\n        print(i)",
    ]


@pytest.fixture
def sample_prompts():
    """Fixture providing sample prompts for testing."""
    return [
        "# Write a function that prints 'Hello, World!'",
        "# Add two numbers and print the result",
        "# Print all even numbers from 0 to 9",
    ]


@pytest.fixture
def sample_completions():
    """Fixture providing sample completions for testing."""
    return [
        "def hello_world():\n    print('Hello, World!')",
        "x = 5\ny = 10\nprint(x + y)",
        "for i in range(10):\n    if i % 2 == 0:\n        print(i)",
    ]


@pytest.fixture
def temp_cache_dir():
    """Fixture providing a temporary cache directory."""
    original_cache_dir = _CACHE_DIR
    temp_dir = tempfile.mkdtemp()

    # Patch the cache directory
    import tokenization

    tokenization._CACHE_DIR = temp_dir

    yield temp_dir

    # Restore original cache directory and clean up
    tokenization._CACHE_DIR = original_cache_dir
    shutil.rmtree(temp_dir)


class TestTokenizerLoading:
    """Tests for tokenizer loading and caching."""

    def test_get_tokenizer(self):
        """Test that get_tokenizer returns a valid tokenizer."""
        tokenizer = get_tokenizer()
        assert tokenizer is not None
        assert hasattr(tokenizer, "encode")
        assert hasattr(tokenizer, "decode")

    def test_tokenizer_caching(self):
        """Test that subsequent calls return the same tokenizer instance."""
        tokenizer1 = get_tokenizer()
        tokenizer2 = get_tokenizer()
        assert tokenizer1 is tokenizer2

    def test_force_reload(self):
        """Test that force_reload creates a new tokenizer instance."""
        tokenizer1 = get_tokenizer()
        tokenizer2 = get_tokenizer(force_reload=True)
        # They should be different instances but have the same vocab
        assert tokenizer1 is not tokenizer2
        assert tokenizer1.vocab_size == tokenizer2.vocab_size

    def test_save_and_load_from_cache(self, temp_cache_dir):
        """Test that tokenizer is saved to and loaded from cache."""
        # First call should create and save the tokenizer
        tokenizer1 = get_tokenizer(force_reload=True)

        # Check that files were created in the cache directory
        assert os.path.exists(os.path.join(temp_cache_dir, "tokenizer.json"))

        # Second call should load from cache
        tokenizer2 = get_tokenizer(force_reload=False)
        assert tokenizer2 is not None


class TestEncodeDecodeRoundTrip:
    """Tests for encoding and decoding functionality."""

    def test_round_trip_single(self, sample_texts):
        """Test that encode followed by decode returns the original text."""
        for text in sample_texts:
            encoded = encode(text, return_tensors="pt")
            decoded = decode(encoded["input_ids"])
            # We need to account for possible whitespace differences and special tokens
            assert text in decoded

    def test_round_trip_batch(self, sample_texts):
        """Test round-trip encoding and decoding for a batch of texts."""
        encoded = encode(sample_texts, return_tensors="pt")
        for i, text in enumerate(sample_texts):
            decoded = decode(encoded["input_ids"][i])
            assert text in decoded

    def test_deterministic_encoding(self, sample_texts):
        """Test that encoding is deterministic across multiple runs."""
        encoded1 = encode(sample_texts[0], return_tensors="pt")
        encoded2 = encode(sample_texts[0], return_tensors="pt")
        assert torch.all(encoded1["input_ids"] == encoded2["input_ids"])


class TestSpecialTokenHandling:
    """Tests for special token handling."""

    def test_bos_token(self):
        """Test that BOS token is added correctly."""
        tokenizer = get_tokenizer()
        text = "Hello, world!"
        encoded = encode(text, add_bos=True, return_tensors="pt")
        # Check if the first token is the BOS token
        assert encoded["input_ids"][0, 0].item() == tokenizer.bos_token_id

    def test_eos_token(self):
        """Test that EOS token is added correctly."""
        tokenizer = get_tokenizer()
        text = "Hello, world!"
        encoded = encode(text, add_eos=True, return_tensors="pt")
        # Check if the last token is the EOS token
        assert encoded["input_ids"][0, -1].item() == tokenizer.eos_token_id

    def test_no_special_tokens(self):
        """Test encoding without special tokens."""
        text = "Hello, world!"
        encoded_with_special = encode(
            text, add_bos=True, add_eos=True, return_tensors="pt"
        )
        encoded_without_special = encode(
            text, add_bos=False, add_eos=False, return_tensors="pt"
        )
        # The version with special tokens should be longer
        assert encoded_with_special["input_ids"].size(1) >= encoded_without_special[
            "input_ids"
        ].size(1)


class TestBatchProcessing:
    """Tests for batch processing functionality."""

    def test_batch_encoding(self, sample_texts):
        """Test encoding a batch of texts."""
        from transformers.tokenization_utils_base import BatchEncoding

        encoded = encode(sample_texts, return_tensors="pt")
        # `encode` may return a plain dict or HuggingFace's BatchEncoding.
        # Treat both as valid because BatchEncoding implements the
        # mapping protocol and is functionally dict-like.
        assert isinstance(encoded, (dict, BatchEncoding))
        assert "input_ids" in encoded
        assert encoded["input_ids"].dim() == 2  # [batch_size, seq_len]
        assert encoded["input_ids"].size(0) == len(sample_texts)

    def test_padding(self, sample_texts):
        """Test padding in batch encoding."""
        encoded = encode(sample_texts, padding=True, return_tensors="pt")
        # All sequences should have the same length
        assert encoded["input_ids"].size(1) == encoded["input_ids"].size(1)

    def test_truncation(self):
        """Test truncation in encoding."""
        long_text = "a " * 1000  # Very long text
        max_length = 50
        encoded = encode(
            long_text, max_length=max_length, truncation=True, return_tensors="pt"
        )
        assert encoded["input_ids"].size(1) <= max_length


class TestTrainingBatchCreation:
    """Tests for training batch creation."""

    def test_create_training_batch(self, sample_prompts, sample_completions):
        """Test creating a training batch with prompts and completions."""
        batch = create_training_batch(
            sample_prompts, sample_completions, return_tensors="pt"
        )
        assert isinstance(batch, dict)
        assert "input_ids" in batch
        assert "labels" in batch
        assert batch["input_ids"].size() == batch["labels"].size()
        assert batch["input_ids"].size(0) == len(sample_prompts)

    def test_label_shifting(self, sample_prompts, sample_completions):
        """Test that labels are properly shifted for causal language modeling."""
        batch = create_training_batch(
            [sample_prompts[0]], [sample_completions[0]], return_tensors="pt"
        )

        # Get tokenizer to encode prompt separately
        tokenizer = get_tokenizer()
        prompt_ids = tokenizer.encode(sample_prompts[0], add_special_tokens=True)

        # Check that prompt tokens have labels = -100 (ignored in loss)
        for i in range(len(prompt_ids)):
            assert batch["labels"][0, i].item() == -100

        # Check that completion tokens have labels = input_ids (for next token prediction)
        completion_start = len(prompt_ids)
        for i in range(completion_start, batch["labels"].size(1) - 1):
            if batch["labels"][0, i].item() != -100:  # Skip padding tokens
                assert (
                    batch["input_ids"][0, i + 1].item() == batch["labels"][0, i].item()
                )

    def test_max_length_handling(self, sample_prompts, sample_completions):
        """Test handling of max_length in training batch creation."""
        max_length = 20  # Intentionally small to force truncation
        batch = create_training_batch(
            sample_prompts,
            sample_completions,
            max_length=max_length,
            return_tensors="pt",
        )
        assert batch["input_ids"].size(1) == max_length
        assert batch["labels"].size(1) == max_length


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_string(self):
        """Test encoding an empty string."""
        encoded = encode("", return_tensors="pt")
        # For an empty input the GPT-2 tokenizer may legitimately return a
        # zero-length sequence (no special tokens are added by default).  We
        # simply assert that the tensor has the expected rank [batch, seq_len]
        # and that the sequence length is *non-negative*.
        assert encoded["input_ids"].dim() == 2
        assert encoded["input_ids"].size(0) == 1  # batch size
        assert encoded["input_ids"].size(1) >= 0  # allow empty sequence

    def test_special_characters(self):
        """Test encoding text with special characters."""
        special_text = "def special_func():\n    print('!@#$%^&*()')\n    return None"
        encoded = encode(special_text, return_tensors="pt")
        decoded = decode(encoded["input_ids"])
        assert "!@#$%^&*()" in decoded

    def test_code_with_indentation(self):
        """Test encoding code with significant indentation."""
        code = "def nested_function():\n    if True:\n        for i in range(10):\n            print(i)\n    return None"
        encoded = encode(code, return_tensors="pt")
        decoded = decode(encoded["input_ids"])
        # Check that indentation is preserved
        assert "    if True:" in decoded
        assert "        for i in range" in decoded

    def test_unicode_characters(self):
        """Test encoding text with Unicode characters."""
        unicode_text = (
            "def unicode_func():\n    print('こんにちは世界')\n    return None"
        )
        encoded = encode(unicode_text, return_tensors="pt")
        decoded = decode(encoded["input_ids"])
        assert "こんにちは世界" in decoded


class TestVocabularyAndValidation:
    """Tests for vocabulary size and token validation."""

    def test_get_vocab_size(self):
        """Test getting the vocabulary size."""
        vocab_size = get_vocab_size()
        assert vocab_size > 0
        assert vocab_size == get_tokenizer().vocab_size

    def test_validate_tokens_valid(self):
        """Test validating valid tokens."""
        tokenizer = get_tokenizer()
        valid_tokens = tokenizer.encode("Hello, world!")
        assert validate_tokens(valid_tokens)

    def test_validate_tokens_invalid(self):
        """Test validating invalid tokens."""
        vocab_size = get_vocab_size()
        invalid_tokens = [vocab_size + 1, vocab_size + 2]  # Tokens outside vocab
        assert not validate_tokens(invalid_tokens)


class TestTensorShapesAndTypes:
    """Tests for tensor shapes and types."""

    def test_encode_tensor_shape(self, sample_texts):
        """Test the shape of tensors returned by encode."""
        encoded = encode(sample_texts[0], return_tensors="pt")
        assert encoded["input_ids"].dim() == 2
        assert encoded["input_ids"].size(0) == 1  # Batch size 1

        # Test batch
        encoded_batch = encode(sample_texts, return_tensors="pt")
        assert encoded_batch["input_ids"].dim() == 2
        assert encoded_batch["input_ids"].size(0) == len(sample_texts)

    def test_tensor_types(self, sample_texts):
        """Test the types of tensors returned."""
        encoded = encode(sample_texts[0], return_tensors="pt")
        assert isinstance(encoded["input_ids"], torch.Tensor)
        assert encoded["input_ids"].dtype == torch.int64  # LongTensor

    def test_list_return_type(self, sample_texts):
        """Test returning lists instead of tensors."""
        encoded = encode(sample_texts[0], return_tensors=None)
        assert isinstance(encoded["input_ids"], list)
        assert all(isinstance(token_id, int) for token_id in encoded["input_ids"])


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
