#!/usr/bin/env python
"""
MBPP Dataset Converter for HRM-CodeGen

This script downloads and preprocesses the MBPP (Mostly Basic Python Problems) dataset
for training the Hierarchical Reasoning Model (HRM) on code generation tasks.

The script:
1. Downloads the MBPP dataset from Hugging Face
2. Extracts task descriptions, code solutions, and test cases
3. Creates training examples with context and target
4. Tokenizes the data using a tokenizer (GPT-2 by default)
5. Handles train/validation/test splits
6. Converts to binary format for efficient loading
7. Creates vocabulary and tokenizer files
8. Supports data augmentation (optional)

Usage:
    python convert_mbpp.py --split train --output-dir data/mbpp
    python convert_mbpp.py --split all --output-dir data/mbpp --tokenizer gpt2
    python convert_mbpp.py --split test --output-dir data/mbpp --augment --aug-factor 3
"""

import argparse
import json
import logging
import os
import pickle
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    GPT2Tokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MBPPExample:
    """Class for holding a single MBPP example."""

    task_id: int
    text: str  # Task description
    code: str  # Solution code
    test_list: List[str]  # Test cases
    test_setup_code: Optional[str] = None  # Setup code for tests
    challenge_test_list: Optional[List[str]] = None  # Additional test cases

    def __post_init__(self):
        """Clean and validate the example data."""
        # Clean task description
        self.text = self.text.strip()

        # Clean code
        self.code = self.code.strip()

        # Clean test cases
        self.test_list = [test.strip() for test in self.test_list]

        # Clean challenge test list if present
        if self.challenge_test_list:
            self.challenge_test_list = [
                test.strip() for test in self.challenge_test_list
            ]

    def get_prompt(self) -> str:
        """Get the prompt for the model."""
        prompt = f"# {self.text}\n\n"

        # Add test cases as comments
        prompt += "# Test cases:\n"
        for test in self.test_list:
            prompt += f"# {test}\n"

        # Add function signature if we can extract it
        func_signature = self.extract_function_signature()
        if func_signature:
            prompt += f"\n{func_signature}\n    "

        return prompt

    def get_completion(self) -> str:
        """Get the completion (solution) for the model."""
        # If we extracted a function signature, remove it from the code
        func_signature = self.extract_function_signature()
        completion = self.code

        if func_signature and completion.startswith(func_signature):
            # Remove the function signature and any initial whitespace
            completion = completion[len(func_signature) :].lstrip()

            # Ensure the completion starts with 4 spaces for indentation
            if not completion.startswith("    "):
                completion = "    " + completion

        return completion

    def extract_function_signature(self) -> Optional[str]:
        """Extract the function signature from the code."""
        # Match "def function_name(params):" pattern
        match = re.match(r"(def\s+\w+\s*\([^)]*\)\s*:)", self.code)
        if match:
            return match.group(1)
        return None

    def to_dict(self) -> Dict:
        """Convert the example to a dictionary."""
        return {
            "task_id": self.task_id,
            "text": self.text,
            "code": self.code,
            "test_list": self.test_list,
            "test_setup_code": self.test_setup_code,
            "challenge_test_list": self.challenge_test_list,
            "prompt": self.get_prompt(),
            "completion": self.get_completion(),
        }


class MBPPDatasetProcessor:
    """Processor for the MBPP dataset."""

    def __init__(
        self,
        dataset_name: str = "google-research-datasets/mbpp",
        dataset_subset: str = "all",
        cache_dir: Optional[str] = None,
        tokenizer_name: str = "gpt2",
        max_length: int = 1024,
    ):
        """
        Initialize the MBPP dataset processor.

        Args:
            dataset_name: Name of the dataset on Hugging Face.
            dataset_subset: Subset of the dataset to use ('full' or 'sanitized').
            cache_dir: Directory to cache the dataset.
            tokenizer_name: Name of the tokenizer to use.
            max_length: Maximum sequence length for tokenization.
        """
        self.dataset_name = dataset_name
        self.dataset_subset = dataset_subset
        self.cache_dir = cache_dir
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length

        # Load dataset
        logger.info(f"Loading dataset: {dataset_name}")
        try:
            if dataset_subset == "all":
                self.dataset = load_dataset(dataset_name, cache_dir=cache_dir)
            else:
                self.dataset = load_dataset(
                    dataset_name, dataset_subset, cache_dir=cache_dir
                )
            logger.info(f"Dataset loaded successfully: {len(self.dataset)} splits")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        # Load tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, use_fast=True
            )

            # Ensure the tokenizer has padding and unknown tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(
                f"Tokenizer loaded successfully: {self.tokenizer.__class__.__name__}"
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def process_example(self, example: Dict) -> MBPPExample:
        """
        Process a single example from the dataset.

        Args:
            example: Raw example from the dataset.

        Returns:
            Processed MBPP example.
        """
        return MBPPExample(
            task_id=example["task_id"],
            text=example["text"],
            code=example["code"],
            test_list=example["test_list"],
            test_setup_code=example.get("test_setup_code", ""),
            challenge_test_list=example.get("challenge_test_list", []),
        )

    def get_split_indices(
        self,
        split: str,
        dataset_size: int,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> List[int]:
        """
        Get indices for the specified split.

        Args:
            split: Split to get indices for ('train', 'validation', 'test').
            dataset_size: Total size of the dataset.
            train_ratio: Ratio of examples to use for training.
            val_ratio: Ratio of examples to use for validation.

        Returns:
            List of indices for the specified split.
        """
        # MBPP has predefined splits based on task_id ranges
        if split == "train":
            # Task IDs 1-400 for training
            return list(range(0, int(dataset_size * train_ratio)))
        elif split == "validation":
            # Task IDs 401-500 for validation
            return list(
                range(
                    int(dataset_size * train_ratio),
                    int(dataset_size * (train_ratio + val_ratio)),
                )
            )
        elif split == "test":
            # Task IDs 501-1000 for testing
            return list(
                range(int(dataset_size * (train_ratio + val_ratio)), dataset_size)
            )
        else:
            raise ValueError(f"Invalid split: {split}")

    def create_prompt_completion_pair(self, example: MBPPExample) -> Tuple[str, str]:
        """
        Create a prompt-completion pair from an MBPP example.

        Args:
            example: MBPP example.

        Returns:
            Tuple of (prompt, completion).
        """
        prompt = example.get_prompt()
        completion = example.get_completion()

        return prompt, completion

    def tokenize_example(
        self, prompt: str, completion: str
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Tokenize a prompt-completion pair.

        Args:
            prompt: Prompt text.
            completion: Completion text.

        Returns:
            Tuple of (input_ids, attention_mask, labels).
        """
        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length // 2,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize completion
        completion_tokens = self.tokenizer(
            completion,
            truncation=True,
            max_length=self.max_length // 2,
            padding="max_length",
            return_tensors="pt",
        )

        # Combine prompt and completion
        input_ids = torch.cat(
            [prompt_tokens["input_ids"], completion_tokens["input_ids"][:, 1:]], dim=1
        )
        attention_mask = torch.cat(
            [
                prompt_tokens["attention_mask"],
                completion_tokens["attention_mask"][:, 1:],
            ],
            dim=1,
        )

        # Create labels (shift right)
        labels = input_ids.clone()
        labels[:, : len(prompt_tokens["input_ids"][0]) - 1] = (
            -100
        )  # Ignore prompt tokens in loss

        return input_ids[0].tolist(), attention_mask[0].tolist(), labels[0].tolist()

    def augment_example(
        self, example: MBPPExample, factor: int = 3
    ) -> List[MBPPExample]:
        """
        Augment an example by creating variations.

        Args:
            example: MBPP example to augment.
            factor: Number of augmentations to create.

        Returns:
            List of augmented examples.
        """
        augmented_examples = [example]

        # Simple augmentations:
        # 1. Vary the prompt wording
        prompt_variations = [
            f"Write a Python function that {example.text.lower()}",
            f"Implement a function to {example.text.lower()}",
            f"Create a Python function for the following task: {example.text}",
            f"Define a function that {example.text.lower()}",
        ]

        # 2. Vary the test case presentation
        for i in range(min(factor - 1, len(prompt_variations))):
            new_example = MBPPExample(
                task_id=example.task_id,
                text=prompt_variations[i],
                code=example.code,
                test_list=example.test_list,
                test_setup_code=example.test_setup_code,
                challenge_test_list=example.challenge_test_list,
            )
            augmented_examples.append(new_example)

        return augmented_examples

    def process_split(
        self,
        split: str,
        output_dir: str,
        augment: bool = False,
        aug_factor: int = 3,
    ) -> None:
        """
        Process a split of the dataset.

        Args:
            split: Split to process ('train', 'validation', 'test', 'all').
            output_dir: Directory to save processed data.
            augment: Whether to augment the data.
            aug_factor: Factor by which to augment the data.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Process all splits if requested
        if split == "all":
            for s in ["train", "validation", "test"]:
                self.process_split(s, output_dir, augment, aug_factor)
            return

        # Get the appropriate dataset split
        if (
            "train" in self.dataset
            and "validation" in self.dataset
            and "test" in self.dataset
        ):
            # Dataset already has splits
            if split == "train":
                data = self.dataset["train"]
            elif split == "validation":
                data = self.dataset["validation"]
            elif split == "test":
                data = self.dataset["test"]
            else:
                raise ValueError(f"Invalid split: {split}")
        else:
            # Create splits from the dataset
            data = self.dataset["test"]  # MBPP only has a test split
            indices = self.get_split_indices(split, len(data))
            data = data.select(indices)

        logger.info(f"Processing {split} split: {len(data)} examples")

        # Process examples
        processed_examples = []
        for example in tqdm(data, desc=f"Processing {split} split"):
            mbpp_example = self.process_example(example)

            if augment and split == "train":
                # Augment training examples
                augmented_examples = self.augment_example(mbpp_example, aug_factor)
                processed_examples.extend(augmented_examples)
            else:
                processed_examples.append(mbpp_example)

        logger.info(f"Processed {len(processed_examples)} examples for {split} split")

        # Tokenize examples
        tokenized_examples = []
        for example in tqdm(processed_examples, desc=f"Tokenizing {split} split"):
            prompt, completion = self.create_prompt_completion_pair(example)
            input_ids, attention_mask, labels = self.tokenize_example(
                prompt, completion
            )

            tokenized_examples.append(
                {
                    "task_id": example.task_id,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "prompt": prompt,
                    "completion": completion,
                }
            )

        # Save processed data
        output_file = os.path.join(output_dir, f"{split}.bin")
        with open(output_file, "wb") as f:
            pickle.dump(tokenized_examples, f)

        logger.info(f"Saved {len(tokenized_examples)} examples to {output_file}")

        # Save raw examples for reference
        raw_output_file = os.path.join(output_dir, f"{split}_raw.json")
        with open(raw_output_file, "w") as f:
            json.dump(
                [example.to_dict() for example in processed_examples],
                f,
                indent=2,
            )

        logger.info(f"Saved raw examples to {raw_output_file}")

    def save_tokenizer(self, output_dir: str) -> None:
        """
        Save the tokenizer to the output directory.

        Args:
            output_dir: Directory to save the tokenizer.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save tokenizer files
        self.tokenizer.save_pretrained(output_dir)

        # Save vocabulary
        vocab_file = os.path.join(output_dir, "vocab.json")
        with open(vocab_file, "w") as f:
            json.dump(self.tokenizer.get_vocab(), f, indent=2)

        logger.info(f"Saved tokenizer and vocabulary to {output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert MBPP dataset for HRM-CodeGen")

    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "validation", "test", "all"],
        help="Dataset split to process",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/mbpp",
        help="Directory to save processed data",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="google-research-datasets/mbpp",
        help="Name of the dataset on Hugging Face",
    )

    parser.add_argument(
        "--dataset-subset",
        type=str,
        default="full",
        choices=["full", "sanitized"],
        help="Subset of the dataset to use",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer to use",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache the dataset",
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augment the data",
    )

    parser.add_argument(
        "--aug-factor",
        type=int,
        default=3,
        help="Factor by which to augment the data",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create processor
    processor = MBPPDatasetProcessor(
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        cache_dir=args.cache_dir,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
    )

    # Process split
    processor.process_split(
        split=args.split,
        output_dir=args.output_dir,
        augment=args.augment,
        aug_factor=args.aug_factor,
    )

    # Save tokenizer
    processor.save_tokenizer(args.output_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
