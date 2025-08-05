"""
Unit tests for the MBPP dataset loader.

This module contains comprehensive tests for the MBPP dataset loader functionality,
including loading, processing, validation, and integration with PyTorch DataLoader.
"""

import json
import os
import tempfile
from typing import Dict, List, Tuple

import pytest
import torch
from torch.utils.data import DataLoader

from datasets.mbpp_loader import MBPPDataset, MBPPConfig, get_mbpp_dataloader, get_mbpp_statistics


@pytest.fixture
def sample_mbpp_data():
    """Fixture providing sample MBPP data for testing."""
    return [
        {
            "task_id": 1,
            "text": "Write a function to add two numbers.",
            "code": "def add_numbers(a, b):\n    return a + b",
            "test_list": ["assert add_numbers(1, 2) == 3", "assert add_numbers(-1, 1) == 0"]
        },
        {
            "task_id": 2,
            "text": "Write a function to check if a number is prime.",
            "code": "def is_prime(n):\n    if n <= 1:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
            "test_list": ["assert is_prime(2) == True", "assert is_prime(4) == False"]
        },
        {
            "task_id": 3,
            "text": "Write a function to reverse a string.",
            "code": "def reverse_string(s):\n    return s[::-1]",
            "test_list": ["assert reverse_string('hello') == 'olleh'", "assert reverse_string('') == ''"]
        }
    ]


@pytest.fixture
def temp_data_dir(sample_mbpp_data):
    """Fixture providing a temporary directory with sample MBPP data files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create train and test files
        train_path = os.path.join(temp_dir, "train_raw.json")
        test_path = os.path.join(temp_dir, "test_raw.json")
        
        # Write sample data to files
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(sample_mbpp_data[:2], f)  # First 2 samples for train
            
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(sample_mbpp_data[2:], f)  # Last sample for test
            
        yield temp_dir


@pytest.fixture
def mbpp_config(temp_data_dir):
    """Fixture providing a MBPPConfig instance with the temporary data directory."""
    return MBPPConfig(
        data_dir=temp_data_dir,
        max_seq_length=128,
        validate_data=True,
        dev_mode=False
    )


@pytest.fixture
def mbpp_dataset(mbpp_config):
    """Fixture providing a MBPPDataset instance with the test configuration."""
    return MBPPDataset(split="train", config=mbpp_config)


class TestDatasetLoading:
    """Tests for basic dataset loading functionality."""
    
    def test_dataset_initialization(self, mbpp_config):
        """Test that the dataset initializes without errors."""
        dataset = MBPPDataset(split="train", config=mbpp_config)
        assert dataset is not None
        assert isinstance(dataset, MBPPDataset)
        
    def test_train_test_split(self, mbpp_config):
        """Test that train and test splits load the correct data."""
        train_dataset = MBPPDataset(split="train", config=mbpp_config)
        test_dataset = MBPPDataset(split="test", config=mbpp_config)
        
        # Check sample counts match expected
        assert len(train_dataset) == 2  # 2 samples in train
        assert len(test_dataset) == 1   # 1 sample in test
        
    def test_file_not_found(self, temp_data_dir):
        """Test that appropriate error is raised when files are not found."""
        # Create config with non-existent file
        with pytest.raises(FileNotFoundError):
            MBPPConfig(
                data_dir=temp_data_dir,
                train_file="nonexistent.json"
            )
            
    def test_empty_dataset_handling(self, temp_data_dir):
        """Test handling of empty dataset files."""
        # Create empty train file
        empty_train_path = os.path.join(temp_data_dir, "empty_train.json")
        with open(empty_train_path, "w", encoding="utf-8") as f:
            json.dump([], f)
            
        # Create config with empty file
        empty_config = MBPPConfig(
            data_dir=temp_data_dir,
            train_file="empty_train.json",
            test_file="test_raw.json"
        )
        
        # Should initialize without error, but have 0 samples
        empty_dataset = MBPPDataset(split="train", config=empty_config)
        assert len(empty_dataset) == 0


class TestDataStructure:
    """Tests for data structure and content validation."""
    
    def test_sample_structure(self, mbpp_dataset):
        """Test that dataset samples have the expected structure."""
        sample = mbpp_dataset[0]
        
        # Check required fields
        assert "input_ids" in sample
        assert "labels" in sample
        assert "task_id" in sample
        
        # Check types
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert isinstance(sample["labels"], torch.Tensor)
        assert isinstance(sample["task_id"], int)
        
    def test_sample_content(self, mbpp_dataset, sample_mbpp_data):
        """Test that sample content matches the source data."""
        sample = mbpp_dataset[0]
        
        # Get tokenizer to decode and check content
        from tokenization import decode, get_tokenizer
        
        # Decode input_ids
        decoded_text = decode(sample["input_ids"])
        
        # Check that the prompt and code are in the decoded text
        assert sample_mbpp_data[0]["text"] in decoded_text
        
        # Check task_id matches
        assert sample["task_id"] == sample_mbpp_data[0]["task_id"]
        
    def test_data_integrity(self, mbpp_dataset):
        """Test that all samples have valid data."""
        for i in range(len(mbpp_dataset)):
            sample = mbpp_dataset[i]
            
            # Check tensor dimensions
            assert sample["input_ids"].dim() == 1
            assert sample["labels"].dim() == 1
            
            # Check matching lengths
            assert sample["input_ids"].size(0) == sample["labels"].size(0)
            
            # Check valid task_id
            assert sample["task_id"] > 0


class TestPromptFormatting:
    """Tests for prompt formatting with different templates."""
    
    def test_default_prompt_template(self, temp_data_dir, sample_mbpp_data):
        """Test the default prompt template."""
        config = MBPPConfig(
            data_dir=temp_data_dir,
            prompt_template="# {text}\n\n",
            include_tests_in_prompt=False
        )
        
        dataset = MBPPDataset(split="train", config=config)
        
        # Get first sample and decode
        from tokenization import decode
        sample = dataset[0]
        decoded = decode(sample["input_ids"])
        
        # Check that the prompt follows the template
        expected_prompt_start = f"# {sample_mbpp_data[0]['text']}"
        assert expected_prompt_start in decoded
        
    def test_custom_prompt_template(self, temp_data_dir):
        """Test a custom prompt template."""
        custom_template = "Problem: {text}\nSolution:\n"
        
        config = MBPPConfig(
            data_dir=temp_data_dir,
            prompt_template=custom_template,
            include_tests_in_prompt=False
        )
        
        dataset = MBPPDataset(split="train", config=config)
        
        # Get first sample and decode
        from tokenization import decode
        sample = dataset[0]
        decoded = decode(sample["input_ids"])
        
        # Check that the prompt follows the custom template
        assert "Problem:" in decoded
        assert "Solution:" in decoded
        
    def test_include_tests_in_prompt(self, temp_data_dir, sample_mbpp_data):
        """Test including test cases in the prompt."""
        config = MBPPConfig(
            data_dir=temp_data_dir,
            include_tests_in_prompt=True,
            test_template="# Test cases:\n# {test}\n\n"
        )
        
        dataset = MBPPDataset(split="train", config=config)
        
        # Get first sample and decode
        from tokenization import decode
        sample = dataset[0]
        decoded = decode(sample["input_ids"])
        
        # Check that test cases are included
        assert "# Test cases:" in decoded
        assert sample_mbpp_data[0]["test_list"][0] in decoded


class TestLabelShifting:
    """Tests for label shifting in the dataset."""
    
    def test_prompt_tokens_masked(self, mbpp_dataset):
        """Test that prompt tokens are masked with -100 in labels."""
        sample = mbpp_dataset[0]
        
        # Find the first non-masked label
        first_valid_idx = (sample["labels"] != -100).nonzero()[0].item()
        
        # Check that all tokens before this are masked
        assert torch.all(sample["labels"][:first_valid_idx] == -100)
        
    def test_completion_tokens_not_masked(self, mbpp_dataset):
        """Test that completion tokens are not masked in labels."""
        sample = mbpp_dataset[0]
        
        # Find the first non-masked label
        first_valid_idx = (sample["labels"] != -100).nonzero()[0].item()
        
        # Check that there are non-masked tokens
        assert first_valid_idx < len(sample["labels"])
        
        # Get a slice of the non-masked region
        non_masked = sample["labels"][first_valid_idx:]
        
        # There should be at least some non-masked tokens
        assert torch.any(non_masked != -100)
        
    def test_labels_shifted_correctly(self, mbpp_dataset):
        """Test that labels are correctly shifted for next-token prediction."""
        sample = mbpp_dataset[0]
        
        # Find the first non-masked label
        valid_indices = (sample["labels"] != -100).nonzero().squeeze()
        
        # Check at least one valid position
        assert len(valid_indices) > 0
        
        # For each valid position, the label should match the next input token
        for i in valid_indices[:-1]:  # Skip the last one as it might be EOS
            idx = i.item()
            # The label at position i should be the input_id at position i+1
            if idx + 1 < len(sample["input_ids"]) and sample["labels"][idx] != -100:
                assert sample["labels"][idx] == sample["input_ids"][idx + 1]


class TestSequenceLengthHandling:
    """Tests for sequence length handling and truncation."""
    
    def test_max_sequence_length(self, temp_data_dir):
        """Test that sequences are limited to max_seq_length."""
        # Create a config with small max_seq_length
        config = MBPPConfig(
            data_dir=temp_data_dir,
            max_seq_length=50  # Intentionally small
        )
        
        dataset = MBPPDataset(split="train", config=config)
        
        # Check all samples
        for i in range(len(dataset)):
            sample = dataset[i]
            assert sample["input_ids"].size(0) <= config.max_seq_length
            assert sample["labels"].size(0) <= config.max_seq_length
            
    def test_padding_to_max_length(self, temp_data_dir):
        """Test that sequences are padded to max_seq_length."""
        # Create a config with a max_seq_length that is larger than any sample
        config = MBPPConfig(
            data_dir=temp_data_dir,
            max_seq_length=200  # Intentionally large to force padding
        )

        dataset = MBPPDataset(split="train", config=config)

        # Verify that every sample is padded up to the configured length
        for idx in range(len(dataset)):
            sample = dataset[idx]
            assert sample["input_ids"].size(0) == config.max_seq_length
            assert sample["labels"].size(0) == config.max_seq_length


class TestDataLoaderIntegration:
    """Tests for integration with PyTorch DataLoader."""
    
    def test_dataloader_creation(self, mbpp_config):
        """Test creating a DataLoader from the dataset."""
        dataloader = get_mbpp_dataloader(
            split="train",
            batch_size=2,
            shuffle=False,
            config=mbpp_config,
            num_workers=0  # Use 0 for testing
        )
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 2
        
    def test_dataloader_iteration(self, mbpp_config):
        """Test iterating through the DataLoader."""
        dataloader = get_mbpp_dataloader(
            split="train",
            batch_size=1,
            shuffle=False,
            config=mbpp_config,
            num_workers=0  # Use 0 for testing
        )
        
        # Iterate through the dataloader
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            
            # Check batch structure
            assert "input_ids" in batch
            assert "labels" in batch
            assert "task_id" in batch
            
            # Check batch dimensions
            assert batch["input_ids"].dim() == 2  # [batch_size, seq_len]
            assert batch["labels"].dim() == 2     # [batch_size, seq_len]
            
        # Check that we got the expected number of batches
        assert batch_count == 2  # 2 samples with batch_size=1
        
    def test_batch_collation(self, mbpp_config):
        """Test that batches are correctly collated."""
        dataloader = get_mbpp_dataloader(
            split="train",
            batch_size=2,  # Batch both samples
            shuffle=False,
            config=mbpp_config,
            num_workers=0  # Use 0 for testing
        )
        
        # Get the first batch
        batch = next(iter(dataloader))
        
        # Check batch dimensions
        assert batch["input_ids"].size(0) == 2  # batch_size
        assert batch["labels"].size(0) == 2     # batch_size
        assert len(batch["task_id"]) == 2       # batch_size


class TestDataValidation:
    """Tests for data validation functionality."""
    
    def test_validation_enabled(self, temp_data_dir):
        """Test that validation is performed when enabled."""
        # Create config with validation enabled
        config = MBPPConfig(
            data_dir=temp_data_dir,
            validate_data=True
        )
        
        # Should initialize without error
        dataset = MBPPDataset(split="train", config=config)
        assert dataset is not None
        
    def test_validation_disabled(self, temp_data_dir):
        """Test that validation is skipped when disabled."""
        # Create config with validation disabled
        config = MBPPConfig(
            data_dir=temp_data_dir,
            validate_data=False
        )
        
        # Should initialize without error
        dataset = MBPPDataset(split="train", config=config)
        assert dataset is not None
        
    def test_invalid_data_detection(self, temp_data_dir):
        """Test detection of invalid data."""
        # Create a sample with invalid Python code
        invalid_sample = {
            "task_id": 999,
            "text": "Write a function with syntax errors.",
            "code": "def invalid_function(:\n    print('Missing parenthesis'",
            "test_list": ["assert True"]
        }
        
        # Write to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([invalid_sample], f)
            temp_file = f.name
            
        try:
            # Create a config with validation enabled
            config = MBPPConfig(
                data_dir=os.path.dirname(temp_file),
                train_file=os.path.basename(temp_file),
                test_file=os.path.basename(temp_file),
                validate_data=True
            )
            
            # Should initialize with a warning about invalid samples
            dataset = MBPPDataset(split="train", config=config)
            assert dataset is not None
            
            # The sample should still be included (we don't filter out invalid samples)
            assert len(dataset) == 1
        finally:
            # Clean up
            os.unlink(temp_file)


class TestDevelopmentMode:
    """Tests for development mode sampling."""
    
    def test_dev_mode_enabled(self, temp_data_dir, sample_mbpp_data):
        """Test that dev_mode limits the number of samples."""
        # Create a larger dataset
        large_dataset = sample_mbpp_data * 10  # 30 samples
        
        # Write to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(large_dataset, f)
            temp_file = f.name
            
        try:
            # Create a config with dev_mode enabled
            config = MBPPConfig(
                data_dir=os.path.dirname(temp_file),
                train_file=os.path.basename(temp_file),
                test_file=os.path.basename(temp_file),
                dev_mode=True,
                dev_samples=5  # Limit to 5 samples
            )
            
            dataset = MBPPDataset(split="train", config=config)
            
            # Check that the dataset is limited to dev_samples
            assert len(dataset) == 5
        finally:
            # Clean up
            os.unlink(temp_file)
            
    def test_dev_mode_disabled(self, temp_data_dir, sample_mbpp_data):
        """Test that dev_mode=False uses all samples."""
        # Create a larger dataset
        large_dataset = sample_mbpp_data * 3  # 9 samples
        
        # Write to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(large_dataset, f)
            temp_file = f.name
            
        try:
            # Create a config with dev_mode disabled
            config = MBPPConfig(
                data_dir=os.path.dirname(temp_file),
                train_file=os.path.basename(temp_file),
                test_file=os.path.basename(temp_file),
                dev_mode=False,
                dev_samples=5  # Should be ignored
            )
            
            dataset = MBPPDataset(split="train", config=config)
            
            # Check that the dataset uses all samples
            assert len(dataset) == 9
        finally:
            # Clean up
            os.unlink(temp_file)
            
    def test_dev_mode_seed(self, temp_data_dir, sample_mbpp_data):
        """Test that dev_mode sampling is deterministic with seed."""
        # Create a larger dataset
        large_dataset = sample_mbpp_data * 10  # 30 samples
        
        # Write to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(large_dataset, f)
            temp_file = f.name
            
        try:
            # Create two configs with the same seed
            config1 = MBPPConfig(
                data_dir=os.path.dirname(temp_file),
                train_file=os.path.basename(temp_file),
                test_file=os.path.basename(temp_file),
                dev_mode=True,
                dev_samples=5,
                seed=42
            )
            
            config2 = MBPPConfig(
                data_dir=os.path.dirname(temp_file),
                train_file=os.path.basename(temp_file),
                test_file=os.path.basename(temp_file),
                dev_mode=True,
                dev_samples=5,
                seed=42
            )
            
            dataset1 = MBPPDataset(split="train", config=config1)
            dataset2 = MBPPDataset(split="train", config=config2)
            
            # Check that the datasets have the same task_ids (same samples)
            task_ids1 = [dataset1[i]["task_id"] for i in range(len(dataset1))]
            task_ids2 = [dataset2[i]["task_id"] for i in range(len(dataset2))]
            
            assert task_ids1 == task_ids2
        finally:
            # Clean up
            os.unlink(temp_file)


class TestStatisticsReporting:
    """Tests for statistics calculation and reporting."""
    
    def test_get_mbpp_statistics(self, temp_data_dir):
        """Test getting MBPP statistics."""
        stats = get_mbpp_statistics(data_dir=temp_data_dir)
        
        # Check that the statistics have the expected keys
        assert "train_samples" in stats
        assert "test_samples" in stats
        assert "total_samples" in stats
        assert "vocab_size" in stats
        
        # Check values
        assert stats["train_samples"] == 2
        assert stats["test_samples"] == 1
        assert stats["total_samples"] == 3
        assert stats["vocab_size"] > 0
        
    def test_dataset_statistics_logging(self, mbpp_dataset, capfd):
        """Test that dataset statistics are logged during initialization."""
        # The `mbpp_dataset` fixture has already been created *before* we start
        # capturing stdout, so its log messages are lost.  To properly test the
        # logging behaviour we create *another* dataset instance **after**
        # enabling capture.
        _ = MBPPDataset(split="train", config=mbpp_dataset.config)

        captured = capfd.readouterr()

        # Check that the output now contains the expected statistics
        assert "MBPP train dataset loaded with" in captured.out
        assert "Sequence length statistics" in captured.out
        assert "Prompt: min=" in captured.out
        assert "Completion: min=" in captured.out
        assert "Total: min=" in captured.out


class TestTensorShapes:
    """Tests for tensor shapes and types."""
    
    def test_input_ids_shape(self, mbpp_dataset, mbpp_config):
        """Test that input_ids have the expected shape."""
        sample = mbpp_dataset[0]
        
        # Check shape
        assert sample["input_ids"].dim() == 1  # [seq_len]
        assert sample["input_ids"].size(0) == mbpp_config.max_seq_length
        
    def test_labels_shape(self, mbpp_dataset, mbpp_config):
        """Test that labels have the expected shape."""
        sample = mbpp_dataset[0]
        
        # Check shape
        assert sample["labels"].dim() == 1  # [seq_len]
        assert sample["labels"].size(0) == mbpp_config.max_seq_length
        
    def test_tensor_types(self, mbpp_dataset):
        """Test that tensors have the expected types."""
        sample = mbpp_dataset[0]
        
        # Check types
        assert sample["input_ids"].dtype == torch.int64  # LongTensor
        assert sample["labels"].dtype == torch.int64     # LongTensor
        
    def test_batch_shapes(self, mbpp_config):
        """Test that batched tensors have the expected shapes."""
        dataloader = get_mbpp_dataloader(
            split="train",
            batch_size=2,
            shuffle=False,
            config=mbpp_config,
            num_workers=0
        )
        
        # Get a batch
        batch = next(iter(dataloader))
        
        # Check shapes
        assert batch["input_ids"].dim() == 2  # [batch_size, seq_len]
        assert batch["input_ids"].size(0) == 2  # batch_size
        assert batch["input_ids"].size(1) == mbpp_config.max_seq_length
        
        assert batch["labels"].dim() == 2  # [batch_size, seq_len]
        assert batch["labels"].size(0) == 2  # batch_size
        assert batch["labels"].size(1) == mbpp_config.max_seq_length


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
