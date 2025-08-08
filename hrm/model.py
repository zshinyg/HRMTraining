"""
Main model implementation for the Hierarchical Reasoning Model (HRM) adapted for code generation.

This module implements the full HRM architecture, combining:
- High-level module for abstract planning (slow timescale)
- Low-level module for detailed computation (fast timescale)
- Token embeddings and output projections
- Hierarchical temporal coordination

The model is designed to generate code by reasoning at multiple levels of abstraction,
with the high-level module planning the overall structure and the low-level module
filling in the details.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hrm.config import HRMConfig, ModelConfig
from hrm.layers import (
    HighLevelModule,
    LearnablePositionalEncoding,
    LowLevelModule,
    PositionalEncoding,
    compute_temporal_indices,
    get_hierarchical_mask,
)


class HRMModel(nn.Module):
    """
    Hierarchical Reasoning Model for code generation.

    This model uses a hierarchical structure with a high-level module for planning
    and a low-level module for execution, inspired by the original HRM architecture
    but adapted for code generation tasks.
    """

    def __init__(self, config: Union[HRMConfig, ModelConfig]):
        """
        Initialize the HRM model.

        Args:
            config: Model configuration.
        """
        super().__init__()

        # Extract model config if full config is provided
        if isinstance(config, HRMConfig):
            self.config = config.model
        else:
            self.config = config

        # Token embeddings
        self.token_embeddings = nn.Embedding(
            self.config.vocab_size, self.config.hidden_dim
        )

        # Positional encodings
        self.pos_encoding = LearnablePositionalEncoding(
            self.config.hidden_dim,
            self.config.max_position_embeddings,
            self.config.dropout,
        )

        # High-level module (planner)
        self.high_level = HighLevelModule(
            input_dim=self.config.hidden_dim,
            hidden_dim=self.config.high_level_dim,
            num_layers=self.config.high_level_layers,
            dropout=self.config.dropout,
            activation=self.config.activation,
        )

        # Low-level module (executor)
        self.low_level = LowLevelModule(
            input_dim=self.config.hidden_dim,
            hidden_dim=self.config.low_level_dim,
            high_dim=self.config.high_level_dim,
            num_layers=self.config.low_level_layers,
            dropout=self.config.dropout,
            activation=self.config.activation,
            num_heads=self.config.num_heads,
        )

        # Output projection
        self.output_projection = nn.Linear(
            self.config.low_level_dim, self.config.vocab_size, bias=False
        )

        # Tie weights if specified
        if self.config.tie_word_embeddings:
            self.output_projection.weight = self.token_embeddings.weight

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.config.low_level_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights of the model.

        Args:
            module: Module to initialize.
        """
        if isinstance(module, nn.Linear):
            # Initialize linear layers with normal distribution
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Initialize embeddings with normal distribution
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            # Initialize layer normalization
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self) -> nn.Embedding:
        """
        Get the input embeddings module.

        Returns:
            Input embeddings module.
        """
        return self.token_embeddings

    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        """
        Set the input embeddings module.

        Args:
            embeddings: New input embeddings module.
        """
        self.token_embeddings = embeddings

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        high_level_states: Optional[List[List[Tensor]]] = None,
        low_level_states: Optional[List[List[Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[Tensor, ...], Dict[str, Tensor]]:
        """
        Forward pass of the HRM model.

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len].
            attention_mask: Attention mask of shape [batch_size, seq_len].
            labels: Optional labels for computing loss of shape [batch_size, seq_len].
            high_level_states: Optional list of high-level hidden states.
            low_level_states: Optional list of low-level hidden states.
            use_cache: Whether to return the hidden states for incremental decoding.
            output_attentions: Whether to return attention weights.
            output_hidden_states: Whether to return all hidden states.
            return_dict: Whether to return a dictionary or tuple.

        Returns:
            Model outputs, including logits, loss (if labels provided), and hidden states.
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        # Get token embeddings
        token_embeds = self.token_embeddings(input_ids)

        # Add positional encodings
        token_embeds = self.pos_encoding(token_embeds)

        # Compute temporal indices for high-level and low-level modules
        high_indices, low_indices = compute_temporal_indices(
            self.config.high_level_steps,
            self.config.low_level_steps,
            self.config.timing_ratio,
        )

        # Initialize hidden states if not provided
        if high_level_states is None:
            high_level_states = [
                [
                    torch.zeros(batch_size, self.config.high_level_dim, device=device)
                    for _ in range(self.config.high_level_layers)
                ]
                for _ in range(len(high_indices))
            ]

        if low_level_states is None:
            low_level_states = [
                [
                    torch.zeros(batch_size, self.config.low_level_dim, device=device)
                    for _ in range(self.config.low_level_layers)
                ]
                for _ in range(len(low_indices))
            ]

        # Initialize outputs
        logits = torch.zeros(batch_size, seq_len, self.config.vocab_size, device=device)

        # Initialize hidden state histories
        high_state_history = torch.zeros(
            batch_size, len(high_indices), self.config.high_level_dim, device=device
        )

        # Create a copy of the initial states to avoid modifying the input
        high_level_states_copy = [
            [state.clone() for state in states] for states in high_level_states
        ]
        low_level_states_copy = [
            [state.clone() for state in states] for states in low_level_states
        ]

        # Process the sequence
        for t in range(seq_len):
            # Current token embedding
            current_embed = token_embeds[:, t]

            # Determine which high-level and low-level steps to execute
            current_low_step = t % self.config.low_level_steps
            current_high_step = current_low_step // self.config.timing_ratio

            # Execute high-level module if needed
            if (
                current_low_step % self.config.timing_ratio == 0
                and current_high_step < len(high_indices)
            ):
                high_output, new_high_states = self.high_level(
                    current_embed,
                    high_level_states_copy[current_high_step],
                    timestep=current_high_step,
                )

                # Update high-level states (create new list instead of in-place modification)
                # Fix for Line ~245: high_level_states[current_high_step] = new_high_states
                high_level_states_copy[current_high_step] = [
                    state.clone() for state in new_high_states
                ]

                # Store high-level state in history (create new tensor instead of in-place modification)
                # Fix for Line ~248: high_state_history[:, current_high_step] = high_output
                high_state_history = torch.cat(
                    [
                        high_state_history[:, :current_high_step],
                        high_output.unsqueeze(1),
                        high_state_history[:, current_high_step + 1 :],
                    ],
                    dim=1,
                )

            # Get the most recent high-level state
            current_high_state = high_state_history[:, : current_high_step + 1]

            # Execute low-level module
            low_output, new_low_states = self.low_level(
                current_embed,
                high_state_history[:, current_high_step],
                current_high_state,
                low_level_states_copy[current_low_step],
                timestep=current_low_step,
            )

            # Update low-level states (create new list instead of in-place modification)
            # Fix for Line ~262: low_level_states[current_low_step] = new_low_states
            low_level_states_copy[current_low_step] = [
                state.clone() for state in new_low_states
            ]

            # Apply layer normalization
            low_output = self.layer_norm(low_output)

            # Project to vocabulary
            token_logits = self.output_projection(low_output)

            # Store logits (create new tensor instead of in-place modification)
            # Fix for Line ~270: logits[:, t] = token_logits
            logits_t = logits.clone()
            logits_t[:, t, :] = token_logits
            logits = logits_t

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Flatten the tensors
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)

            # Compute loss (ignoring padding tokens)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits, shift_labels)

        # Prepare outputs
        if return_dict:
            outputs = {
                "logits": logits,
                "high_level_states": high_level_states_copy,
                "low_level_states": low_level_states_copy,
            }
            if loss is not None:
                outputs["loss"] = loss

            return outputs
        else:
            outputs = (logits,)
            if loss is not None:
                outputs = (loss,) + outputs

            return outputs

    def generate(
        self,
        input_ids: Tensor,
        max_length: int = 100,
        min_length: int = 0,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        num_beams: int = 1,
        num_return_sequences: int = 1,
        attention_mask: Optional[Tensor] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        """
        Generate sequences of token IDs.

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len].
            max_length: Maximum length of the generated sequences.
            min_length: Minimum length of the generated sequences.
            temperature: Temperature for sampling.
            top_k: Number of highest probability tokens to keep for top-k sampling.
            top_p: Cumulative probability for nucleus sampling.
            repetition_penalty: Penalty for repeating tokens.
            do_sample: Whether to sample or use greedy decoding.
            num_beams: Number of beams for beam search.
            num_return_sequences: Number of sequences to return.
            attention_mask: Attention mask of shape [batch_size, seq_len].
            pad_token_id: ID of the padding token.
            eos_token_id: ID of the end-of-sequence token.

        Returns:
            Generated token IDs of shape [batch_size, seq_len].
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Set default values
        if pad_token_id is None:
            pad_token_id = 0
        if eos_token_id is None:
            eos_token_id = 0

        # Initialize generated sequences with input_ids
        generated_ids = input_ids.clone()

        # Initialize high-level and low-level states
        high_level_states = None
        low_level_states = None

        # Initialize attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Track which sequences are finished
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Generate tokens up to max_length
        for step in range(input_ids.size(1), max_length):
            # Forward pass
            outputs = self.forward(
                generated_ids,
                attention_mask=attention_mask,
                high_level_states=high_level_states,
                low_level_states=low_level_states,
                use_cache=True,
                return_dict=True,
            )

            # Get logits for the next token
            next_token_logits = outputs["logits"][:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty

            # Mask finished sequences
            if torch.any(finished_sequences):
                next_token_logits[finished_sequences] = torch.full_like(
                    next_token_logits[0], -float("inf")
                )
                next_token_logits[finished_sequences, pad_token_id] = 0.0

            # Apply min_length constraint
            if step < min_length:
                next_token_logits[:, eos_token_id] = -float("inf")

            # Sample next tokens
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(
                        next_token_logits, top_k, dim=-1
                    )
                    next_token_logits = torch.full_like(
                        next_token_logits, -float("inf")
                    )
                    next_token_logits.scatter_(-1, top_k_indices, top_k_values)

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True, dim=-1
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                        :, :-1
                    ].clone()
                    sorted_indices_to_remove[:, 0] = 0

                    for i in range(batch_size):
                        indices_to_remove = sorted_indices[i][
                            sorted_indices_to_remove[i]
                        ]
                        next_token_logits[i, indices_to_remove] = -float("inf")

                # Convert logits to probabilities
                probs = F.softmax(next_token_logits, dim=-1)

                # Sample from the distribution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)

            # Update finished sequences
            finished_sequences = finished_sequences | (next_tokens == eos_token_id)
            if torch.all(finished_sequences):
                break

            # Append next tokens to generated_ids
            generated_ids = torch.cat(
                [generated_ids, next_tokens.unsqueeze(-1)], dim=-1
            )

            # Update attention mask
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), device=device)], dim=-1
            )

            # Update hidden states
            high_level_states = outputs.get("high_level_states")
            low_level_states = outputs.get("low_level_states")

        return generated_ids

    def beam_search(
        self,
        input_ids: Tensor,
        max_length: int = 100,
        num_beams: int = 5,
        early_stopping: bool = True,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        attention_mask: Optional[Tensor] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> Tensor:
        """
        Generate sequences using beam search.

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len].
            max_length: Maximum length of the generated sequences.
            num_beams: Number of beams.
            early_stopping: Whether to stop when all beams have finished.
            temperature: Temperature for sampling.
            repetition_penalty: Penalty for repeating tokens.
            attention_mask: Attention mask of shape [batch_size, seq_len].
            pad_token_id: ID of the padding token.
            eos_token_id: ID of the end-of-sequence token.

        Returns:
            Generated token IDs of shape [batch_size, seq_len].
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Set default values
        if pad_token_id is None:
            pad_token_id = 0
        if eos_token_id is None:
            eos_token_id = 0

        # Initialize beam scores
        beam_scores = torch.zeros((batch_size, num_beams), device=device)

        # Expand input_ids for beam search
        input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1)
        input_ids = input_ids.contiguous().view(batch_size * num_beams, -1)

        # Expand attention mask for beam search
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, num_beams, -1
            )
            attention_mask = attention_mask.contiguous().view(
                batch_size * num_beams, -1
            )
        else:
            attention_mask = torch.ones_like(input_ids)

        # Initialize high-level and low-level states
        high_level_states = None
        low_level_states = None

        # Track which sequences are finished
        finished_sequences = torch.zeros(
            batch_size * num_beams, dtype=torch.bool, device=device
        )

        # Track the best sequences for each batch
        best_sequences = torch.zeros(
            (batch_size, max_length), dtype=torch.long, device=device
        )
        best_scores = torch.full((batch_size,), -float("inf"), device=device)

        # Generate tokens up to max_length
        for step in range(input_ids.size(1), max_length):
            # Forward pass
            outputs = self.forward(
                input_ids,
                attention_mask=attention_mask,
                high_level_states=high_level_states,
                low_level_states=low_level_states,
                use_cache=True,
                return_dict=True,
            )

            # Get logits for the next token
            next_token_logits = outputs["logits"][:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for token_id in set(input_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty

            # Convert logits to log probabilities
            next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)

            # Reshape for beam search
            next_token_log_probs = next_token_log_probs.view(batch_size, num_beams, -1)

            # Add beam scores
            next_scores = next_token_log_probs + beam_scores.unsqueeze(-1)

            # Flatten for top-k selection
            next_scores = next_scores.view(batch_size, -1)

            # Get the top-k scores and indices
            topk_scores, topk_indices = torch.topk(next_scores, num_beams, dim=-1)

            # Convert indices to token IDs and beam indices
            topk_token_ids = topk_indices % self.config.vocab_size
            topk_beam_indices = topk_indices // self.config.vocab_size

            # Prepare for the next step
            next_input_ids = torch.zeros(
                (batch_size * num_beams, input_ids.size(1) + 1),
                dtype=torch.long,
                device=device,
            )
            next_attention_mask = torch.zeros(
                (batch_size * num_beams, attention_mask.size(1) + 1),
                dtype=torch.long,
                device=device,
            )

            # Update beam scores
            beam_scores = topk_scores

            # Update input_ids and attention_mask
            for batch_idx in range(batch_size):
                for beam_idx in range(num_beams):
                    # Get the token ID and beam index
                    token_id = topk_token_ids[batch_idx, beam_idx]
                    beam_index = topk_beam_indices[batch_idx, beam_idx]

                    # Get the global indices
                    global_beam_index = batch_idx * num_beams + beam_idx
                    global_source_index = batch_idx * num_beams + beam_index

                    # Copy the previous sequence
                    next_input_ids[global_beam_index, :-1] = input_ids[
                        global_source_index
                    ]
                    next_attention_mask[global_beam_index, :-1] = attention_mask[
                        global_source_index
                    ]

                    # Add the new token
                    next_input_ids[global_beam_index, -1] = token_id
                    next_attention_mask[global_beam_index, -1] = 1

                    # Check if the sequence is finished
                    if token_id == eos_token_id:
                        finished_sequences[global_beam_index] = True

                        # Update best sequences
                        if beam_scores[batch_idx, beam_idx] > best_scores[batch_idx]:
                            best_scores[batch_idx] = beam_scores[batch_idx, beam_idx]
                            best_sequences[batch_idx, : next_input_ids.size(1)] = (
                                next_input_ids[global_beam_index]
                            )

            # Update input_ids and attention_mask
            input_ids = next_input_ids
            attention_mask = next_attention_mask

            # Update hidden states
            high_level_states = outputs.get("high_level_states")
            low_level_states = outputs.get("low_level_states")

            # Check if all sequences are finished
            if early_stopping and torch.all(
                finished_sequences.view(batch_size, num_beams)
            ):
                break

        # Return the best sequences
        return best_sequences

    def compute_loss(
        self, logits: Tensor, labels: Tensor, ignore_index: int = -100
    ) -> Tensor:
        """
        Compute the loss for the model.

        Args:
            logits: Logits of shape [batch_size, seq_len, vocab_size].
            labels: Labels of shape [batch_size, seq_len].
            ignore_index: Index to ignore in the loss computation.

        Returns:
            Loss tensor.
        """
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Flatten the tensors
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        # Compute loss (ignoring padding tokens)
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss = loss_fct(shift_logits, shift_labels)

        return loss

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model to the specified directory.

        Args:
            save_directory: Directory to save the model.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save config
        config_path = os.path.join(save_directory, "config.json")
        if hasattr(self, "config") and hasattr(self.config, "to_dict"):
            config_dict = self.config.to_dict()
            with open(config_path, "w") as f:
                import json

                json.dump(config_dict, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_path: str) -> HRMModel:
        """
        Load a model from the specified directory.

        Args:
            model_path: Path to the saved model.

        Returns:
            Loaded model.
        """
        # Load config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            import json

            with open(config_path, "r") as f:
                config_dict = json.load(f)

            from hrm.config import HRMConfig, ModelConfig

            if "model" in config_dict:
                config = HRMConfig.from_dict(config_dict)
            else:
                config = ModelConfig(**config_dict)
        else:
            raise ValueError(f"Config file not found at {config_path}")

        # Create model
        model = cls(config)

        # Load weights
        model_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            model.load_state_dict(torch.load(model_file, map_location="cpu"))
        else:
            raise ValueError(f"Model weights not found at {model_file}")

        return model

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """
        Resize the token embeddings.

        Args:
            new_num_tokens: New number of tokens.

        Returns:
            Resized token embeddings.
        """
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._resize_token_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        # Update config
        self.config.vocab_size = new_num_tokens

        # Tie weights if necessary
        if self.config.tie_word_embeddings:
            self.output_projection.weight = self.token_embeddings.weight

        return self.get_input_embeddings()

    def _resize_token_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: int
    ) -> nn.Embedding:
        """
        Resize token embeddings.

        Args:
            old_embeddings: Old embeddings.
            new_num_tokens: New number of tokens.

        Returns:
            New embeddings.
        """
        if new_num_tokens == old_embeddings.num_embeddings:
            return old_embeddings

        # Create new embeddings
        new_embeddings = nn.Embedding(
            new_num_tokens,
            old_embeddings.embedding_dim,
            padding_idx=old_embeddings.padding_idx,
        )
        new_embeddings.to(old_embeddings.weight.device)

        # Copy weights for existing tokens
        num_tokens_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[
            :num_tokens_to_copy, :
        ]

        return new_embeddings


def create_hrm_model(config: HRMConfig) -> HRMModel:
    """
    Create an HRM model from the given configuration.

    Args:
        config: Model configuration.

    Returns:
        HRM model.
    """
    model = HRMModel(config)
    return model


if __name__ == "__main__":
    # Example usage
    from hrm.config import get_default_mbpp_config

    # Get default config
    config = get_default_mbpp_config()

    # Create model
    model = create_hrm_model(config)

    # Print model summary
    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Test forward pass
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.model.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)

    print(f"Output logits shape: {outputs['logits'].shape}")
