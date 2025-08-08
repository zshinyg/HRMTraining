"""
Unit tests for HRM code generation training system.

This module contains comprehensive tests for the training system,
including configuration, training loops, optimization, evaluation,
checkpointing, and integration with the MBPP dataset.
"""

import os
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple, Union, Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import our training components
from training.trainer import Trainer, TrainingConfig, TrainingState
from hrm_codegen.mock_model import MockHRMModel, MockHRMConfig
from datasets.mbpp_loader import MBPPDataset
from tokenization import get_tokenizer, encode, decode


@pytest.fixture
def device():
    """Fixture for the compute device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def mock_config():
    """Fixture for mock HRM configuration."""
    return MockHRMConfig(
        vocab_size=50257,  # GPT-2 vocabulary size
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=128,
        causal=True,
    )


@pytest.fixture
def mock_model(mock_config, device):
    """Fixture for mock HRM model."""
    model = MockHRMModel(mock_config)
    model.to(device)
    return model


@pytest.fixture
def training_config():
    """Fixture for training configuration."""
    return TrainingConfig(
        output_dir="test_checkpoints",
        train_path="data/mbpp/train_raw.json",
        val_path="data/mbpp/test_raw.json",
        max_seq_len=128,
        batch_size=2,
        eval_batch_size=2,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        learning_rate=5e-5,
        max_steps=10,
        gradient_accumulation_steps=2,
        eval_every=5,
        save_every=5,
        log_every=1,
        fp16=False,  # Disable mixed precision for testing
        use_wandb=False,
        use_tensorboard=False,
        save_total_limit=2,
    )


@pytest.fixture
def temp_dir():
    """Fixture for temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mbpp_dataset():
    """Fixture for MBPP dataset."""
    from datasets.mbpp_loader import MBPPConfig

    # Build a config that mirrors the previous explicit args
    mbpp_cfg = MBPPConfig(
        max_seq_length=128,
        validate_data=True,
        dev_mode=True,  # Enable small-sample mode for fast unit tests
        dev_samples=10,
        include_tests_in_prompt=True,
    )

    # Return the training split using the constructed config
    return MBPPDataset(split="train", config=mbpp_cfg)


@pytest.fixture
def trainer(mock_model, training_config, mbpp_dataset, temp_dir):
    """Fixture for trainer."""
    # Override output directory to use temporary directory
    config = training_config
    config.output_dir = temp_dir

    # Create trainer with mock model and dataset
    trainer = Trainer(
        model=mock_model,
        config=config,
        train_dataset=mbpp_dataset,
        eval_dataset=mbpp_dataset,  # Use same dataset for eval to simplify testing
    )

    return trainer


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_config(self):
        """Test default configuration creation."""
        config = TrainingConfig()
        assert config.output_dir == "checkpoints/codegen"
        assert config.seed == 42
        assert config.learning_rate == 5e-5
        assert config.weight_decay == 0.01
        assert config.max_steps == 10000

    def test_custom_config(self):
        """Test custom configuration creation."""
        config = TrainingConfig(
            output_dir="custom_dir", learning_rate=1e-4, batch_size=16, max_steps=5000
        )
        assert config.output_dir == "custom_dir"
        assert config.learning_rate == 1e-4
        assert config.batch_size == 16
        assert config.max_steps == 5000

    def test_incompatible_settings(self):
        """Test validation of incompatible settings."""
        with pytest.raises(ValueError):
            TrainingConfig(fp16=True, bf16=True)

    def test_output_dir_creation(self):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "new_dir")
            config = TrainingConfig(output_dir=output_dir)
            assert os.path.exists(output_dir)

    def test_device_selection(self):
        """Test device selection logic."""
        # Force CPU for testing
        config = TrainingConfig(device="cpu")
        assert config.device == "cpu"

        # Test MPS fallback on Mac if available
        if hasattr(torch, "has_mps") and torch.has_mps:
            config = TrainingConfig(device="cuda")
            if not torch.cuda.is_available():
                assert config.device == "mps"

    def test_warmup_steps_from_ratio(self):
        """Test warmup steps calculation from ratio."""
        config = TrainingConfig(max_steps=1000, warmup_steps=0, warmup_ratio=0.1)
        assert config.warmup_steps == 100


class TestTrainerInitialization:
    """Tests for Trainer initialization."""

    def test_trainer_init(self, mock_model, training_config, mbpp_dataset):
        """Test trainer initialization."""
        trainer = Trainer(
            model=mock_model,
            config=training_config,
            train_dataset=mbpp_dataset,
            eval_dataset=mbpp_dataset,
        )

        assert trainer.model is mock_model
        assert trainer.config is training_config
        assert trainer.train_dataset is mbpp_dataset
        assert trainer.eval_dataset is mbpp_dataset
        assert isinstance(trainer.train_dataloader, DataLoader)
        assert isinstance(trainer.eval_dataloader, DataLoader)
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.LambdaLR)
        assert trainer.scaler is None  # fp16 is disabled in test config

    def test_dataset_loading(self, mock_model, training_config):
        """Test dataset loading from paths."""
        trainer = Trainer(model=mock_model, config=training_config)

        assert isinstance(trainer.train_dataset, MBPPDataset)
        assert isinstance(trainer.eval_dataset, MBPPDataset)
        assert len(trainer.train_dataset) > 0
        assert len(trainer.eval_dataset) > 0

    def test_optimizer_creation(self, trainer):
        """Test optimizer creation."""
        optimizer = trainer.optimizer

        # Check optimizer type
        assert isinstance(optimizer, torch.optim.AdamW)

        # Check parameter groups (should have weight decay and no weight decay groups)
        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["weight_decay"] > 0
        assert optimizer.param_groups[1]["weight_decay"] == 0

        # Check learning rate
        warmup_lr = (
            0.0 if trainer.config.warmup_steps > 0 else trainer.config.learning_rate
        )
        assert optimizer.param_groups[0]["lr"] == warmup_lr

    def test_scheduler_creation(self, trainer):
        """Test scheduler creation."""
        scheduler = trainer.scheduler

        # Check scheduler type
        assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

        # Check initial learning rate
        warmup_lr = (
            0.0 if trainer.config.warmup_steps > 0 else trainer.config.learning_rate
        )
        assert scheduler.get_last_lr()[0] == warmup_lr


class TestTrainingStep:
    """Tests for training step functionality."""

    def test_single_training_step(self, trainer, device):
        """Test a single training step."""
        # Get a batch from the dataloader
        batch = next(iter(trainer.train_dataloader))

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Perform training step
        metrics = trainer.train_step(batch)

        # Check metrics
        assert "loss" in metrics
        assert "perplexity" in metrics
        assert not torch.isnan(torch.tensor(metrics["loss"]))
        assert not torch.isinf(torch.tensor(metrics["loss"]))
        assert metrics["perplexity"] > 0

    def test_loss_computation(self, trainer, device):
        """Test loss computation."""
        # Create dummy logits and labels
        batch_size = 2
        seq_len = 10
        vocab_size = trainer.model.config.vocab_size

        logits = torch.randn(batch_size, seq_len, vocab_size, device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        # Compute loss
        loss = trainer.compute_loss(logits, labels)

        # Check loss
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() > 0

    def test_gradient_flow(self, trainer, device):
        """Test gradient flow."""
        # Get a batch from the dataloader
        batch = next(iter(trainer.train_dataloader))

        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Zero gradients
        trainer.optimizer.zero_grad()

        # Perform training step
        metrics = trainer.train_step(batch)

        # Check if gradients are computed
        grad_norm = 0.0
        for param in trainer.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm**0.5

        assert grad_norm > 0, "No gradients were computed"

    def test_optimizer_step(self, trainer):
        """Test optimizer step."""
        # Get a batch from the dataloader
        batch = next(iter(trainer.train_dataloader))

        # Move batch to device
        batch = {k: v.to(trainer.config.device) for k, v in batch.items()}

        # Zero gradients
        trainer.optimizer.zero_grad()

        # Perform training step
        trainer.train_step(batch)

        # Get parameter snapshot before optimizer step
        params_before = [param.clone() for param in trainer.model.parameters()]

        # Perform optimizer step
        lr, grad_norm = trainer.optimizer_step()

        # Check if parameters changed
        params_changed = any(
            not torch.allclose(pb, pa)
            for pb, pa in zip(params_before, trainer.model.parameters())
        )

        # Parameters must change once LR is positive; during LR==0 warm-up, allow no change
        if lr > 0.0:
            assert (
                params_changed
            ), "Parameters did not change after optimizer step with positive LR"

        # LR should be non-negative
        assert lr >= 0, "Learning rate should be non-negative"
        assert grad_norm >= 0, "Gradient norm should be non-negative"


class TestCheckpointing:
    """Tests for checkpointing functionality."""

    def test_save_checkpoint(self, trainer, temp_dir):
        """Test saving checkpoint."""
        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint()

        # Check if checkpoint file exists
        assert os.path.exists(checkpoint_path)

        # Check if checkpoint can be loaded
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Check checkpoint contents
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "scheduler_state_dict" in checkpoint
        assert "config" in checkpoint
        assert "state" in checkpoint

    def test_load_checkpoint(self, trainer, temp_dir):
        """Test loading checkpoint."""
        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint()

        # Modify model parameters to verify loading
        for param in trainer.model.parameters():
            param.data.fill_(0.0)

        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)

        # Check if parameters were loaded
        zero_params = True
        for param in trainer.model.parameters():
            if torch.any(param != 0.0):
                zero_params = False
                break

        assert not zero_params, "Parameters were not loaded from checkpoint"

    def test_cleanup_checkpoints(self, trainer, temp_dir):
        """Test cleanup of old checkpoints."""
        # Set save limit to 2
        trainer.config.save_total_limit = 2

        # Save 3 checkpoints
        for i in range(3):
            trainer.state.global_step = i
            trainer.save_checkpoint()

        # Check if only 2 checkpoints remain
        checkpoints = [
            f
            for f in os.listdir(temp_dir)
            if f.startswith("step-") and f.endswith(".pt")
        ]
        assert len(checkpoints) == 2

        # Check if the oldest checkpoint was removed
        steps = [int(f.split("-")[1].split(".")[0]) for f in checkpoints]
        assert 0 not in steps, "Oldest checkpoint was not removed"

    def test_save_best_model(self, trainer, temp_dir):
        """Test saving best model."""
        # Set initial best metric
        trainer.state.best_metric = float("inf")

        # Update evaluation metrics with better value
        trainer.state.update_eval_metrics({"eval_loss": 1.0})

        # Check if model should be saved
        should_save = trainer.state.should_save_best(
            "eval_loss", higher_is_better=False
        )
        assert should_save, "Model should be saved as best"

        # Save best model
        best_path = trainer.save_checkpoint(is_best=True)

        # Check if best model file exists
        assert os.path.exists(best_path)
        assert "best.pt" in best_path

        # Update with worse value
        trainer.state.update_eval_metrics({"eval_loss": 2.0})

        # Check if model should not be saved
        should_save = trainer.state.should_save_best(
            "eval_loss", higher_is_better=False
        )
        assert not should_save, "Model should not be saved as best with worse metric"


class TestEvaluation:
    """Tests for evaluation functionality."""

    def test_evaluation(self, trainer):
        """Test evaluation loop."""
        # Perform evaluation
        metrics = trainer.evaluate()

        # Check metrics
        assert "eval_loss" in metrics
        assert "eval_perplexity" in metrics
        assert not torch.isnan(torch.tensor(metrics["eval_loss"]))
        assert not torch.isinf(torch.tensor(metrics["eval_loss"]))
        assert metrics["eval_perplexity"] > 0

    def test_evaluate_solutions(self, trainer):
        """Test solution evaluation."""
        # Create test solutions and test cases
        solutions = [
            "def add(a, b):\n    return a + b",
            "def add(a, b):\n    return a - b",  # Wrong implementation
            "def add(a, b):\n    return a + b",
        ]

        test_cases = ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]

        # Evaluate solutions
        results = trainer.evaluate_solutions(solutions, test_cases)

        # Check results
        assert len(results) == 3
        assert results[0] is True, "Correct solution should pass"
        assert results[1] is False, "Wrong solution should fail"
        assert results[2] is True, "Correct solution should pass"

    def test_evaluate_pass_k(self, trainer, monkeypatch):
        """Test Pass@k metrics computation."""

        # Mock evaluate_solutions to return predetermined results
        def mock_evaluate_solutions(self, solutions, test_cases):
            # First solution fails, second passes
            return [False, True, False, False, True]

        monkeypatch.setattr(Trainer, "evaluate_solutions", mock_evaluate_solutions)

        # Mock dataset to return test cases
        def mock_get_test_cases(self, idx):
            return ["assert True"]

        monkeypatch.setattr(MBPPDataset, "get_test_cases", mock_get_test_cases)

        # Set Pass@k values
        trainer.config.pass_k_values = [1, 3, 5]
        trainer.config.eval_max_samples = 2  # Only evaluate 2 samples for speed

        # Evaluate Pass@k
        metrics = trainer.evaluate_pass_k()

        # Check metrics
        assert "pass@1" in metrics
        assert "pass@3" in metrics
        assert "pass@5" in metrics

        # With our mock (2/5 solutions pass), we expect:
        # pass@1 = 0.0 (first solution fails)
        # pass@3 = 0.5 (one sample passes within first 3)
        # pass@5 = 1.0 (both samples pass within all 5)
        assert metrics["pass@1"] == 0.0
        assert metrics["pass@3"] == 0.5
        assert metrics["pass@5"] == 1.0


class TestTrainingLoop:
    """Tests for training loop functionality."""

    def test_training_loop(self, trainer, temp_dir):
        """Test full training loop."""
        # Set small number of steps for testing
        trainer.config.max_steps = 6
        trainer.config.gradient_accumulation_steps = 2
        trainer.config.eval_every = 4
        trainer.config.save_every = 4

        # Run training
        final_metrics = trainer.train()

        # Check if training completed
        assert trainer.state.global_step >= 3  # 6 steps with accumulation of 2

        # Check if metrics were computed
        assert "loss" in final_metrics
        assert "eval_loss" in final_metrics
        assert "training_time" in final_metrics

        # Check if checkpoints were saved
        checkpoints = [f for f in os.listdir(temp_dir) if f.endswith(".pt")]
        assert len(checkpoints) > 0

    def test_gradient_accumulation(self, trainer):
        """Test training with gradient accumulation."""
        # Set accumulation steps
        trainer.config.gradient_accumulation_steps = 2

        # Zero gradients
        trainer.optimizer.zero_grad()

        # Get a batch
        batch = next(iter(trainer.train_dataloader))
        batch = {k: v.to(trainer.config.device) for k, v in batch.items()}

        # First step (should not call optimizer)
        trainer.state.step = 0
        metrics1 = trainer.train_step(batch)

        # Check if optimizer was not called (global step should not change)
        assert trainer.state.global_step == 0

        # Second step (should call optimizer)
        trainer.state.step = 1
        metrics2 = trainer.train_step(batch)

        # Manually call optimizer step to simulate training loop
        trainer.optimizer_step()

        # Check if optimizer was called (global step should change)
        assert trainer.state.global_step == 1

    def test_early_stopping(self, trainer, monkeypatch):
        """Test training with early stopping."""

        # Create mock evaluate method that always returns the same metrics
        def mock_evaluate():
            return {"eval_loss": 1.0}

        monkeypatch.setattr(trainer, "evaluate", mock_evaluate)

        # Set up early stopping conditions
        trainer.config.max_steps = 100  # Large number to ensure we stop early

        # Create a callback to stop training after 5 steps
        original_train = trainer.train

        def train_with_early_stop(*args, **kwargs):
            # Override the global step check to stop after 5 steps
            original_should_continue = lambda: (
                trainer.state.global_step < trainer.config.max_steps
                and (
                    trainer.config.max_epochs is None
                    or trainer.state.epoch < trainer.config.max_epochs
                )
            )

            def early_stop():
                return trainer.state.global_step < 5 and original_should_continue()

            # Temporarily replace the condition
            import types

            trainer._should_continue = types.MethodType(early_stop, trainer)

            result = original_train(*args, **kwargs)

            # Restore original method
            trainer._should_continue = types.MethodType(
                original_should_continue, trainer
            )

            return result

        monkeypatch.setattr(trainer, "train", train_with_early_stop)

        # Run training
        final_metrics = trainer.train()

        # Check if training stopped early
        assert trainer.state.global_step == 5

    def test_resume_from_checkpoint(self, trainer, temp_dir):
        """Test resuming from checkpoint."""
        # Train for a few steps
        trainer.config.max_steps = 4
        trainer.train()

        # Save the state
        global_step = trainer.state.global_step
        loss_before = trainer.state.train_loss

        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint()

        # Reset trainer
        new_trainer = Trainer(
            model=trainer.model,
            config=trainer.config,
            train_dataset=trainer.train_dataset,
            eval_dataset=trainer.eval_dataset,
        )

        # Verify state is reset
        assert new_trainer.state.global_step == 0

        # Resume training
        new_trainer.load_checkpoint(checkpoint_path)

        # Check if state was restored
        assert new_trainer.state.global_step == global_step
        assert new_trainer.state.train_loss == loss_before

        # Train for a few more steps
        new_trainer.config.max_steps = global_step + 4
        final_metrics = new_trainer.train()

        # Check if training continued
        assert new_trainer.state.global_step > global_step


class TestIntegration:
    """Tests for integration with MBPP dataset and mock model."""

    def test_mbpp_integration(self, trainer):
        """Test integration with MBPP dataset."""
        # Get a sample from the dataset
        sample = trainer.train_dataset[0]

        # Check sample structure
        assert "input_ids" in sample
        assert "labels" in sample
        assert "attention_mask" in sample

        # Check if model can process the sample
        batch = {k: v.unsqueeze(0).to(trainer.config.device) for k, v in sample.items()}
        outputs = trainer.model(batch["input_ids"])

        # Check outputs
        assert "logits" in outputs
        assert outputs["logits"].shape[0] == 1
        assert outputs["logits"].shape[1] == batch["input_ids"].shape[1]
        assert outputs["logits"].shape[2] == trainer.model.config.vocab_size

    def test_mock_model_generation(self, mock_model):
        """Test mock model generation capabilities."""
        # Generate text from prompt
        prompt = "# Write a function to add two numbers\n\ndef add(a, b):"
        generated_text = mock_model.generate(
            prompt=prompt, max_length=50, temperature=0.8, do_sample=True
        )

        # Check generated text
        assert isinstance(generated_text, str)
        assert prompt in generated_text
        assert len(generated_text) > len(prompt)

        # Test batch generation
        prompts = [
            "# Write a function to add two numbers\n\ndef add(a, b):",
            "# Write a function to check if a number is prime\n\ndef is_prime(n):",
        ]

        generated_texts = mock_model.generate(
            prompt=prompts, max_length=50, temperature=0.8, do_sample=True
        )

        # Check generated texts
        assert isinstance(generated_texts, list)
        assert len(generated_texts) == len(prompts)
        for prompt, text in zip(prompts, generated_texts):
            assert prompt in text
            assert len(text) > len(prompt)

    def test_end_to_end_training(self, mock_model, mbpp_dataset, temp_dir):
        """Test end-to-end training with mock model and MBPP dataset."""
        # Create training config
        config = TrainingConfig(
            output_dir=temp_dir,
            max_steps=4,
            gradient_accumulation_steps=1,
            eval_every=2,
            save_every=2,
            batch_size=2,
            eval_batch_size=2,
            num_workers=0,
            fp16=False,
            use_wandb=False,
            use_tensorboard=False,
        )

        # Create trainer
        trainer = Trainer(
            model=mock_model,
            config=config,
            train_dataset=mbpp_dataset,
            eval_dataset=mbpp_dataset,
        )

        # Train model
        final_metrics = trainer.train()

        # Check training results
        assert trainer.state.global_step == 4
        assert "loss" in final_metrics
        assert "eval_loss" in final_metrics
        assert "training_time" in final_metrics

        # Check if checkpoints were saved
        checkpoints = [f for f in os.listdir(temp_dir) if f.endswith(".pt")]
        assert len(checkpoints) > 0

        # Test generation with trained model
        prompt = "# Write a function to add two numbers\n\ndef add(a, b):"
        generated_text = mock_model.generate(
            prompt=prompt, max_length=50, temperature=0.8, do_sample=True
        )

        # Check generated text
        assert isinstance(generated_text, str)
        assert prompt in generated_text
        assert len(generated_text) > len(prompt)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
