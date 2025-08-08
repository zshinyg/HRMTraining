"""
Mock HRM model for testing training pipeline.

This module provides a lightweight mock implementation of the HRM model
that mimics the interface of the real model but doesn't require heavy
dependencies like flash-attn. It's useful for testing the training
pipeline and other infrastructure while waiting for the full model
dependencies to be resolved.
"""

import math
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import tokenization utilities
from tokenization import get_tokenizer, encode, decode


class MockHRMConfig:
    """
    Mock configuration for HRM model.

    This class mimics the interface of the real HRM configuration
    but with simplified parameters for testing purposes.
    """

    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocabulary size
        hidden_size: int = 256,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 4,
        intermediate_size: int = 1024,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        eos_token_id: int = 50256,
        bos_token_id: int = 50256,
        causal: bool = True,
        **kwargs,
    ):
        """
        Initialize mock HRM configuration.

        Args:
            vocab_size: Size of the vocabulary
            hidden_size: Dimension of hidden layers
            num_hidden_layers: Number of hidden layers
            num_attention_heads: Number of attention heads
            intermediate_size: Dimension of feedforward layers
            hidden_dropout_prob: Dropout probability for hidden layers
            attention_probs_dropout_prob: Dropout probability for attention
            max_position_embeddings: Maximum sequence length
            initializer_range: Range for weight initialization
            layer_norm_eps: Epsilon for layer normalization
            pad_token_id: ID of padding token
            eos_token_id: ID of end-of-sequence token
            bos_token_id: ID of beginning-of-sequence token
            causal: Whether to use causal attention
            **kwargs: Additional parameters
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.causal = causal

        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MockHRMConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration parameters

        Returns:
            MockHRMConfig instance
        """
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary with configuration parameters
        """
        return {k: v for k, v in self.__dict__.items()}


class MockAttention(nn.Module):
    """
    Mock attention module for testing.

    This is a simplified attention implementation that mimics the interface
    of the real attention module but doesn't require flash-attn.
    """

    def __init__(self, config: MockHRMConfig):
        """
        Initialize mock attention module.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.causal = config.causal

        # Check if head dimension divides hidden size evenly
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size {self.hidden_size} is not divisible by num_heads {self.num_heads}"
            )

        # Create query, key, value projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Create dropout
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for attention module.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project query, key, value
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(
                    seq_len, seq_len, dtype=torch.bool, device=hidden_states.device
                ),
                diagonal=1,
            )
            attn_scores.masked_fill_(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask to proper shape: [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert 0s to -inf and 1s to 0s
            attention_mask = (1.0 - attention_mask) * torch.finfo(attn_scores.dtype).min
            attn_scores = attn_scores + attention_mask

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back to [batch_size, seq_len, hidden_size]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )

        # Project to output dimension
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output


class MockMLP(nn.Module):
    """
    Mock MLP module for testing.

    This is a simplified MLP implementation that mimics the interface
    of the real MLP module.
    """

    def __init__(self, config: MockHRMConfig):
        """
        Initialize mock MLP module.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for MLP module.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class MockTransformerLayer(nn.Module):
    """
    Mock transformer layer for testing.

    This is a simplified transformer layer implementation that mimics the interface
    of the real transformer layer.
    """

    def __init__(self, config: MockHRMConfig):
        """
        Initialize mock transformer layer.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.attention = MockAttention(config)
        self.mlp = MockMLP(config)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for transformer layer.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual connection
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MockHRMModel(nn.Module):
    """
    Mock HRM model for testing training pipeline.

    This is a lightweight mock implementation of the HRM model that mimics
    the interface of the real model but doesn't require heavy dependencies
    like flash-attn.
    """

    def __init__(self, config: MockHRMConfig):
        """
        Initialize mock HRM model.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.config = config

        # Create embeddings
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Create transformer layers
        self.layers = nn.ModuleList(
            [MockTransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # Create output layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Create output projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights with embedding
        self.lm_head.weight = self.wte.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights for the model.

        Args:
            module: Module to initialize
        """
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass for the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            return_dict: Whether to return a dictionary or tensor

        Returns:
            Model outputs, either as dictionary or tensor
        """
        batch_size, seq_len = input_ids.shape

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=input_ids.device)

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = (
                torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )

        # Get embeddings
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)

        # Add position embeddings
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.emb_dropout(hidden_states)

        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Apply final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Return outputs
        if return_dict:
            return {"logits": logits, "hidden_states": hidden_states}
        else:
            return logits

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        prompt: Optional[Union[str, List[str]]] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = True,
        **kwargs,
    ) -> Union[torch.Tensor, str, List[str]]:
        """
        Hugging-Face style generation helper.

        Args:
            input_ids: Tensor of token ids `[batch, seq]` that represents the prompt.
            max_length: Maximum length of generated sequence (prompt + new tokens)
            temperature: Sampling temperature
            top_k: Number of top tokens to consider
            top_p: Cumulative probability threshold
            do_sample: Whether to sample or use greedy decoding
            **kwargs: Ignored extra keyword arguments (for API parity)

        Returns:
            Tensor of generated ids `[batch, seq']`
        """
        device = next(self.parameters()).device

        used_prompt: Optional[Union[str, List[str]]] = None
        if prompt is not None:
            used_prompt = prompt
            if isinstance(prompt, str):
                batch_prompts = [prompt]
                single = True
            else:
                batch_prompts = prompt
                single = False
            enc = encode(batch_prompts, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
        else:
            assert input_ids is not None, "Either prompt or input_ids must be provided"
            single = input_ids.dim() == 1
            if single:
                input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(device)

        generated_ids = input_ids.clone()

        seq_len_limit = max_length

        # Autoregressive loop
        with torch.no_grad():
            for _ in range(seq_len_limit - input_ids.shape[1]):
                outputs = self.forward(generated_ids)
                next_token_logits = outputs["logits"][:, -1, :]

                # temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature

                # top-k
                if top_k > 0:
                    kth_logits = torch.topk(next_token_logits, k=top_k)[0][
                        :, -1
                    ].unsqueeze(-1)
                    mask = kth_logits <= next_token_logits
                    next_token_logits = next_token_logits.masked_fill(
                        ~mask, float("-inf")
                    )

                # top-p
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    remove_mask = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits = next_token_logits.masked_fill(
                        remove_mask, float("-inf")
                    )

                # sample / greedy
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # append token
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # early stop if all finished (multi-batch safe)
                eos_mask = next_token.squeeze(-1) == self.config.eos_token_id
                if eos_mask.dim() > 0 and torch.all(eos_mask):
                    break
        # decode if prompt was provided
        if used_prompt is not None:
            texts: List[str] = []
            for seq in generated_ids:
                texts.append(decode(seq))
            return texts[0] if isinstance(used_prompt, str) else texts
        else:
            return generated_ids

    @classmethod
    def from_config(
        cls, config: Union[Dict[str, Any], MockHRMConfig]
    ) -> "MockHRMModel":
        """
        Create model from configuration.

        Args:
            config: Configuration dictionary or object

        Returns:
            MockHRMModel instance
        """
        if isinstance(config, dict):
            config = MockHRMConfig.from_dict(config)

        return cls(config)

    def save_pretrained(self, save_directory: str):
        """
        Save model to directory.

        Args:
            save_directory: Directory to save model
        """
        import os
        import json

        os.makedirs(save_directory, exist_ok=True)

        # Save configuration
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

    @classmethod
    def from_pretrained(cls, model_path: str) -> "MockHRMModel":
        """
        Load model from directory.

        Args:
            model_path: Directory with saved model

        Returns:
            MockHRMModel instance
        """
        import os
        import json

        # Load configuration
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        config = MockHRMConfig.from_dict(config_dict)

        # Create model
        model = cls(config)

        # Load model weights
        model_path = os.path.join(model_path, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        return model
