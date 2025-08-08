"""
MBPP (Mostly Basic Python Programming) dataset loader.

This module provides a PyTorch Dataset for loading and processing the MBPP dataset
for code generation tasks, with support for configurable prompts, data validation,
and statistics reporting.
"""

import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

# Import our tokenization module
from tokenization import (create_training_batch, encode, get_tokenizer,
                          get_vocab_size)


@dataclass
class MBPPConfig:
    """Configuration for MBPP dataset loading and processing."""

    # Data paths
    data_dir: str = "data/mbpp"
    train_file: str = "train_raw.json"
    test_file: str = "test_raw.json"

    # Processing options
    max_seq_length: int = 512
    prompt_template: str = "# {text}\n\n"
    include_tests_in_prompt: bool = False
    test_template: str = "# Test cases:\n# {test}\n\n"

    # Sampling options
    dev_mode: bool = False
    dev_samples: int = 100
    seed: int = 42

    # Validation
    validate_data: bool = True

    # Output formatting
    # When True, return `task_id` as a tensor for easier batching; when False, return int
    return_task_id_tensor: bool = False

    def __post_init__(self):
        """Validate paths after initialization."""
        self.train_path = os.path.join(self.data_dir, self.train_file)
        self.test_path = os.path.join(self.data_dir, self.test_file)

        # Validate file existence
        if not os.path.exists(self.train_path):
            raise FileNotFoundError(f"Training file not found: {self.train_path}")
        if not os.path.exists(self.test_path):
            raise FileNotFoundError(f"Test file not found: {self.test_path}")


class MBPPDataset(Dataset):
    """
    PyTorch Dataset for the MBPP (Mostly Basic Python Programming) dataset.

    This dataset loads MBPP problems, formats them with configurable prompts,
    and returns tokenized inputs suitable for causal language modeling training.
    """

    def __init__(
        self,
        split: str = "train",
        config: Optional[MBPPConfig] = None,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the MBPP dataset.

        Args:
            split: Dataset split, either "train" or "test"
            config: Configuration for dataset loading and processing
            transform: Optional transform to apply to samples
        """
        self.split = split
        self.config = config or MBPPConfig()
        self.transform = transform

        # Load and process data
        self.samples = self._load_data()

        # Validate data if requested
        if self.config.validate_data:
            self._validate_data()

        # Print statistics
        self._log_statistics()

    def _load_data(self) -> List[Dict]:
        """
        Load and process the dataset from JSON files.

        Returns:
            List of processed samples with prompts and completions
        """
        # Determine file path based on split
        file_path = (
            self.config.train_path if self.split == "train" else self.config.test_path
        )

        # Load raw data
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # Apply development mode sampling if enabled
        if self.config.dev_mode:
            random.seed(self.config.seed)
            raw_data = random.sample(
                raw_data, min(self.config.dev_samples, len(raw_data))
            )

        # Process each sample
        processed_samples = []
        for item in raw_data:
            # Extract fields
            task_id = item.get("task_id", 0)
            text = item.get("text", "")
            code = item.get("code", "")
            test_list = item.get("test_list", [])

            # Skip samples with missing data
            if not text or not code:
                continue

            # Format prompt
            prompt = self.config.prompt_template.format(text=text)

            # Add test cases to prompt if configured
            if self.config.include_tests_in_prompt and test_list:
                test_str = "\n".join(test_list)
                prompt += self.config.test_template.format(test=test_str)

            # Store processed sample
            processed_samples.append(
                {
                    "task_id": task_id,
                    "prompt": prompt,
                    "completion": code,
                    "test_list": test_list,
                    "full_text": text,
                }
            )

        return processed_samples

    def _validate_data(self):
        """Validate dataset samples for common issues."""
        invalid_count = 0
        for sample in self.samples:
            # Check for empty prompts or completions
            if not sample["prompt"] or not sample["completion"]:
                invalid_count += 1
                continue

            # Check if completion is valid Python (simple syntax check)
            try:
                compile(sample["completion"], "<string>", "exec")
            except SyntaxError:
                invalid_count += 1

        if invalid_count > 0:
            print(
                f"Warning: Found {invalid_count} potentially invalid samples in {self.split} split"
            )

    def _log_statistics(self):
        """Log dataset statistics."""
        # Basic counts
        print(f"MBPP {self.split} dataset loaded with {len(self.samples)} samples")

        # Calculate sequence length statistics
        if self.samples:
            tokenizer = get_tokenizer()
            prompt_lengths = []
            completion_lengths = []
            total_lengths = []

            # Sample up to 100 examples for statistics
            stat_samples = random.sample(self.samples, min(100, len(self.samples)))

            for sample in stat_samples:
                prompt_ids = tokenizer.encode(sample["prompt"], add_special_tokens=True)
                completion_ids = tokenizer.encode(
                    sample["completion"], add_special_tokens=False
                )

                prompt_lengths.append(len(prompt_ids))
                completion_lengths.append(len(completion_ids))
                total_lengths.append(len(prompt_ids) + len(completion_ids))

            # Report statistics
            print(f"Sequence length statistics (based on {len(stat_samples)} samples):")
            print(
                f"  Prompt: min={min(prompt_lengths)}, max={max(prompt_lengths)}, avg={sum(prompt_lengths)/len(prompt_lengths):.1f}"
            )
            print(
                f"  Completion: min={min(completion_lengths)}, max={max(completion_lengths)}, avg={sum(completion_lengths)/len(completion_lengths):.1f}"
            )
            print(
                f"  Total: min={min(total_lengths)}, max={max(total_lengths)}, avg={sum(total_lengths)/len(total_lengths):.1f}"
            )
            print(
                f"  Samples exceeding max length ({self.config.max_seq_length}): {sum(1 for l in total_lengths if l > self.config.max_seq_length)}"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a dataset sample by index.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'input_ids' and 'labels' tensors
        """
        sample = self.samples[idx]

        # Create training batch with prompt and completion
        batch = create_training_batch(
            prompts=[sample["prompt"]],
            completions=[sample["completion"]],
            max_length=self.config.max_seq_length,
            return_tensors="pt",
        )

        # Extract single item from batch (remove batch dimension)
        item = {
            "input_ids": batch["input_ids"][0],
            "labels": batch["labels"][0],
            "attention_mask": batch["attention_mask"][0],
            "task_id": torch.tensor(sample["task_id"], dtype=torch.long)
            if self.config.return_task_id_tensor
            else sample["task_id"],
        }

        # Apply transform if provided
        if self.transform is not None:
            item = self.transform(item)

        return item

    # ---- Convenience accessors for evaluation / tests -----------------------
    def get_prompt(self, idx: int) -> str:
        """Return the raw prompt text for a given sample index."""
        return self.samples[idx]["prompt"]

    def get_test_cases(self, idx: int) -> List[str]:
        """Return the list of test-case strings for a given sample index."""
        return self.samples[idx].get("test_list", [])


def get_mbpp_dataloader(
    split: str = "train",
    batch_size: int = 8,
    shuffle: bool = True,
    config: Optional[MBPPConfig] = None,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create a DataLoader for the MBPP dataset.

    Args:
        split: Dataset split, either "train" or "test"
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the dataset
        config: Configuration for dataset loading
        num_workers: Number of worker processes for data loading

    Returns:
        PyTorch DataLoader for the MBPP dataset
    """
    dataset = MBPPDataset(split=split, config=config)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_mbpp_statistics(data_dir: str = "data/mbpp") -> Dict:
    """
    Compute and return statistics about the MBPP dataset.

    Args:
        data_dir: Directory containing MBPP data files

    Returns:
        Dictionary with dataset statistics
    """
    train_path = os.path.join(data_dir, "train_raw.json")
    test_path = os.path.join(data_dir, "test_raw.json")

    # Load data
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Compute statistics
    stats = {
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "total_samples": len(train_data) + len(test_data),
        "vocab_size": get_vocab_size(),
    }

    return stats


# Example usage
if __name__ == "__main__":
    # Print dataset statistics
    stats = get_mbpp_statistics()
    print(f"MBPP Dataset Statistics:")
    print(f"  Train samples: {stats['train_samples']}")
    print(f"  Test samples: {stats['test_samples']}")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Vocabulary size: {stats['vocab_size']}")

    # Create a small development dataset
    config = MBPPConfig(dev_mode=True, dev_samples=10)
    dataset = MBPPDataset(split="train", config=config)

    # Print a sample
    sample = dataset[0]
    print("\nSample input_ids shape:", sample["input_ids"].shape)
    print("Sample labels shape:", sample["labels"].shape)
