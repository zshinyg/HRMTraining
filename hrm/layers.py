"""
Core layers and components for the Hierarchical Reasoning Model (HRM).

This module implements the fundamental building blocks of the HRM architecture:
- HighLevelModule: The slow, abstract planning RNN that operates on longer timescales
- LowLevelModule: The fast, detailed computation RNN that operates on shorter timescales
- HierarchicalAttention: Custom attention mechanism for hierarchical communication
- PositionalEncoding: Position embeddings for sequences

The implementation is based on the original HRM architecture but adapted for code generation tasks.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from hrm.config import ActivationType, ModelConfig


def get_activation_fn(activation_type: ActivationType) -> Callable[[Tensor], Tensor]:
    """
    Returns the activation function corresponding to the given activation type.

    Args:
        activation_type: The type of activation function to use.

    Returns:
        The activation function.
    """
    if activation_type == ActivationType.RELU:
        return F.relu
    elif activation_type == ActivationType.GELU:
        return F.gelu
    elif activation_type == ActivationType.SWISH:
        return F.silu
    elif activation_type == ActivationType.SILU:
        return F.silu
    elif activation_type == ActivationType.TANH:
        return torch.tanh
    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")


class PositionalEncoding(nn.Module):
    """
    Positional encoding module that adds positional information to input embeddings.

    This implementation uses sine and cosine functions of different frequencies.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize the positional encoding module.

        Args:
            d_model: The dimension of the model.
            max_len: The maximum length of the input sequences.
            dropout: The dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to the input.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Output tensor with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding module that adds positional information to input embeddings.

    This implementation uses learned position embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize the learnable positional encoding module.

        Args:
            d_model: The dimension of the model.
            max_len: The maximum length of the input sequences.
            dropout: The dropout probability.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize the position embeddings."""
        nn.init.normal_(self.position_embeddings, mean=0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to the input.

        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model].

        Returns:
            Output tensor with positional encoding added.
        """
        x = x + self.position_embeddings[:, : x.size(1)]
        return self.dropout(x)


class TimestepEncoding(nn.Module):
    """
    Timestep encoding module that adds temporal information to hidden states.

    This is used to encode the current timestep in the recurrent modules.
    """

    def __init__(self, d_model: int, max_steps: int = 100):
        """
        Initialize the timestep encoding module.

        Args:
            d_model: The dimension of the model.
            max_steps: The maximum number of timesteps.
        """
        super().__init__()
        self.timestep_embeddings = nn.Embedding(max_steps, d_model)

    def forward(self, x: Tensor, timestep: int) -> Tensor:
        """
        Add timestep encoding to the input.

        Args:
            x: Input tensor of shape [batch_size, d_model].
            timestep: The current timestep.

        Returns:
            Output tensor with timestep encoding added.
        """
        # Create timestep tensor
        timestep_tensor = torch.full(
            (x.size(0),), timestep, dtype=torch.long, device=x.device
        )

        # Get timestep embeddings
        timestep_emb = self.timestep_embeddings(timestep_tensor)

        # Add timestep embeddings to input
        return x + timestep_emb


class HierarchicalAttention(nn.Module):
    """
    Custom attention mechanism for hierarchical communication between high-level and low-level modules.

    This module allows the low-level module to attend to the high-level module's hidden states.
    """

    def __init__(
        self,
        high_dim: int,
        low_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize the hierarchical attention module.

        Args:
            high_dim: The dimension of the high-level module.
            low_dim: The dimension of the low-level module.
            num_heads: The number of attention heads.
            dropout: The dropout probability.
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = low_dim // num_heads
        assert (
            self.head_dim * num_heads == low_dim
        ), "low_dim must be divisible by num_heads"

        # Projection matrices
        self.query_proj = nn.Linear(low_dim, low_dim)
        self.key_proj = nn.Linear(high_dim, low_dim)
        self.value_proj = nn.Linear(high_dim, low_dim)

        # Output projection
        self.output_proj = nn.Linear(low_dim, low_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, low_state: Tensor, high_states: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Apply hierarchical attention.

        Args:
            low_state: The low-level state of shape [batch_size, low_dim].
            high_states: The high-level states of shape [batch_size, num_high_states, high_dim].
            mask: Optional attention mask of shape [batch_size, num_high_states].

        Returns:
            Output tensor after attention of shape [batch_size, low_dim].
        """
        batch_size = low_state.size(0)

        # Project queries, keys, and values
        # NOTE: use .reshape instead of .view to avoid inadvertent in-place view
        q = self.query_proj(low_state).reshape(
            batch_size, 1, self.num_heads, self.head_dim
        )
        k = self.key_proj(high_states).reshape(
            batch_size, -1, self.num_heads, self.head_dim
        )
        v = self.value_proj(high_states).reshape(
            batch_size, -1, self.num_heads, self.head_dim
        )

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, num_heads, 1, head_dim]
        k = k.transpose(1, 2)  # [batch_size, num_heads, num_high_states, head_dim]
        v = v.transpose(1, 2)  # [batch_size, num_heads, num_high_states, head_dim]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, num_high_states]
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(
            attn_weights, v
        )  # [batch_size, num_heads, 1, head_dim]

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2).contiguous()
            # Use reshape instead of view to avoid potential in-place view issues
            .reshape(batch_size, 1, self.num_heads * self.head_dim)
        )
        output = self.output_proj(attn_output).squeeze(1)

        return output


class HighLevelModule(nn.Module):
    """
    High-level module responsible for slow, abstract planning.

    This module operates on a slower timescale and provides guidance to the low-level module.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: ActivationType = ActivationType.GELU,
    ):
        """
        Initialize the high-level module.

        Args:
            input_dim: The dimension of the input.
            hidden_dim: The dimension of the hidden state.
            num_layers: The number of recurrent layers.
            dropout: The dropout probability.
            activation: The activation function to use.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Recurrent layers
        self.recurrent_layers = nn.ModuleList(
            [
                HighLevelRecurrentLayer(hidden_dim, hidden_dim, dropout, activation)
                for _ in range(num_layers)
            ]
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Timestep encoding
        self.timestep_encoding = TimestepEncoding(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        hidden_states: Optional[List[Tensor]] = None,
        timestep: int = 0,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass of the high-level module.

        Args:
            x: Input tensor of shape [batch_size, input_dim].
            hidden_states: Optional list of hidden states for each layer.
            timestep: The current timestep.

        Returns:
            Tuple of (output, hidden_states).
        """
        batch_size = x.size(0)

        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]

        # Project input
        x = self.input_proj(x)

        # Add timestep encoding
        x = self.timestep_encoding(x, timestep)

        # Apply recurrent layers
        new_hidden_states = []
        for i, layer in enumerate(self.recurrent_layers):
            x, h = layer(x, hidden_states[i])
            new_hidden_states.append(h)

        # Apply layer normalization
        output = self.layer_norm(x)

        return output, new_hidden_states


class HighLevelRecurrentLayer(nn.Module):
    """
    Recurrent layer for the high-level module.

    This implements a GRU-like recurrent layer with additional mechanisms for code generation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: ActivationType = ActivationType.GELU,
    ):
        """
        Initialize the high-level recurrent layer.

        Args:
            input_dim: The dimension of the input.
            hidden_dim: The dimension of the hidden state.
            dropout: The dropout probability.
            activation: The activation function to use.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Get activation function
        self.activation_fn = get_activation_fn(activation)

        # GRU-like gates
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the high-level recurrent layer.

        Args:
            x: Input tensor of shape [batch_size, input_dim].
            hidden: Hidden state tensor of shape [batch_size, hidden_dim].

        Returns:
            Tuple of (output, new_hidden).
        """
        # Concatenate input and hidden state
        combined = torch.cat([x, hidden], dim=1)

        # Compute gates
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))

        # Compute candidate hidden state
        combined_reset = torch.cat([x, reset * hidden], dim=1)
        candidate = self.activation_fn(self.candidate(combined_reset))

        # Update hidden state
        new_hidden = (1 - update) * hidden + update * candidate

        # Apply dropout
        output = self.dropout(new_hidden)

        return output, new_hidden


class LowLevelModule(nn.Module):
    """
    Low-level module responsible for fast, detailed computation.

    This module operates on a faster timescale and handles token-level code generation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        high_dim: int,
        num_layers: int = 4,
        dropout: float = 0.1,
        activation: ActivationType = ActivationType.GELU,
        num_heads: int = 8,
    ):
        """
        Initialize the low-level module.

        Args:
            input_dim: The dimension of the input.
            hidden_dim: The dimension of the hidden state.
            high_dim: The dimension of the high-level module.
            num_layers: The number of recurrent layers.
            dropout: The dropout probability.
            activation: The activation function to use.
            num_heads: The number of attention heads for hierarchical attention.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.high_dim = high_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Hierarchical attention
        self.hierarchical_attn = HierarchicalAttention(
            high_dim, hidden_dim, num_heads, dropout
        )

        # Recurrent layers
        self.recurrent_layers = nn.ModuleList(
            [
                LowLevelRecurrentLayer(hidden_dim, hidden_dim, dropout, activation)
                for _ in range(num_layers)
            ]
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Timestep encoding
        self.timestep_encoding = TimestepEncoding(hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        high_state: Tensor,
        high_states_history: Tensor,
        hidden_states: Optional[List[Tensor]] = None,
        timestep: int = 0,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass of the low-level module.

        Args:
            x: Input tensor of shape [batch_size, input_dim].
            high_state: Current high-level state of shape [batch_size, high_dim].
            high_states_history: History of high-level states of shape [batch_size, num_high_states, high_dim].
            hidden_states: Optional list of hidden states for each layer.
            timestep: The current timestep.

        Returns:
            Tuple of (output, hidden_states).
        """
        batch_size = x.size(0)

        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]

        # Project input
        x = self.input_proj(x)

        # Add timestep encoding
        x = self.timestep_encoding(x, timestep)

        # Apply hierarchical attention to incorporate high-level guidance
        high_context = self.hierarchical_attn(x, high_states_history)
        x = x + high_context

        # Apply recurrent layers
        new_hidden_states = []
        for i, layer in enumerate(self.recurrent_layers):
            x, h = layer(x, hidden_states[i])
            new_hidden_states.append(h)

        # Apply layer normalization
        output = self.layer_norm(x)

        return output, new_hidden_states


class LowLevelRecurrentLayer(nn.Module):
    """
    Recurrent layer for the low-level module.

    This implements a GRU-like recurrent layer with additional mechanisms for code generation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        activation: ActivationType = ActivationType.GELU,
    ):
        """
        Initialize the low-level recurrent layer.

        Args:
            input_dim: The dimension of the input.
            hidden_dim: The dimension of the hidden state.
            dropout: The dropout probability.
            activation: The activation function to use.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Get activation function
        self.activation_fn = get_activation_fn(activation)

        # GRU-like gates
        self.update_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.candidate = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Code-specific enhancements
        self.syntax_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the low-level recurrent layer.

        Args:
            x: Input tensor of shape [batch_size, input_dim].
            hidden: Hidden state tensor of shape [batch_size, hidden_dim].

        Returns:
            Tuple of (output, new_hidden).
        """
        # Concatenate input and hidden state
        combined = torch.cat([x, hidden], dim=1)

        # Compute gates
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        syntax = torch.sigmoid(self.syntax_gate(combined))

        # Compute candidate hidden state
        combined_reset = torch.cat([x, reset * hidden], dim=1)
        candidate = self.activation_fn(self.candidate(combined_reset))

        # Update hidden state with syntax awareness
        new_hidden = (1 - update) * hidden + update * (
            syntax * candidate + (1 - syntax) * hidden
        )

        # Apply dropout
        output = self.dropout(new_hidden)

        return output, new_hidden


def get_timing_signal(
    length: int, channels: int, min_timescale: float = 1.0, max_timescale: float = 1.0e4
) -> Tensor:
    """
    Create timing signal (sinusoidal positional encoding).

    Args:
        length: The length of the signal.
        channels: The number of channels.
        min_timescale: The minimum timescale.
        max_timescale: The maximum timescale.

    Returns:
        Timing signal tensor of shape [1, length, channels].
    """
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    num_timescales = channels // 2
    log_timescale_increment = math.log(max_timescale / min_timescale) / (
        num_timescales - 1
    )

    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
    ).unsqueeze(0)

    scaled_time = position * inv_timescales
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    # Handle odd channels
    if channels % 2 == 1:
        signal = torch.cat([signal, torch.zeros(length, 1)], dim=1)

    return signal.unsqueeze(0)


def compute_temporal_indices(
    high_steps: int, low_steps: int, timing_ratio: int
) -> Tuple[List[int], List[int]]:
    """
    Compute the temporal indices for synchronizing high-level and low-level modules.

    Args:
        high_steps: The number of high-level steps.
        low_steps: The number of low-level steps.
        timing_ratio: The ratio of low-level to high-level steps.

    Returns:
        Tuple of (high_indices, low_indices).
    """
    # Ensure low_steps is a multiple of timing_ratio
    assert low_steps % timing_ratio == 0, "low_steps must be a multiple of timing_ratio"

    # Compute expected high-level steps
    expected_high_steps = low_steps // timing_ratio

    # Adjust high_steps if necessary
    high_steps = min(high_steps, expected_high_steps)

    # Compute indices
    high_indices = [i for i in range(high_steps)]
    low_indices = [i for i in range(low_steps)]

    return high_indices, low_indices


def get_hierarchical_mask(high_steps: int, low_steps: int, timing_ratio: int) -> Tensor:
    """
    Create a mask for hierarchical attention.

    This mask ensures that the low-level module only attends to high-level states
    from the past and present, not the future.

    Args:
        high_steps: The number of high-level steps.
        low_steps: The number of low-level steps.
        timing_ratio: The ratio of low-level to high-level steps.

    Returns:
        Mask tensor of shape [low_steps, high_steps].
    """
    mask = torch.zeros(low_steps, high_steps)

    for low_idx in range(low_steps):
        # Determine which high-level steps are visible to this low-level step
        high_idx = low_idx // timing_ratio
        mask[low_idx, : high_idx + 1] = 1.0

    return mask
