"""
HRM model adaptation for code generation.

This module adapts the Hierarchical Reasoning Model (HRM) from Sapient
for code generation tasks, modifying the architecture to support
causal language modeling and autoregressive text generation.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
from torch import nn

# Add the external directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
external_dir = os.path.join(root_dir, "external")
if external_dir not in sys.path:
    sys.path.append(external_dir)
    sys.path.append(root_dir)

# Import original HRM components
from sapient_hrm.models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1_Inner,
    HierarchicalReasoningModel_ACTV1Block,
    HierarchicalReasoningModel_ACTV1Config,
    HierarchicalReasoningModel_ACTV1Carry,
    HierarchicalReasoningModel_ACTV1InnerCarry,
    HierarchicalReasoningModel_ACTV1ReasoningModule,
)
from sapient_hrm.models.layers import Attention, CastedLinear, RotaryEmbedding

# Import our tokenization utilities
from tokenization import get_tokenizer, encode, decode

# Import patched causal utilities
from hrm_codegen.layers import make_hrm_blocks_causal

# Import configuration
from .config import CodeGenConfig


class HRMCodeGeneratorInner(HierarchicalReasoningModel_ACTV1_Inner):
    """
    Adapted HRM inner model for code generation.
    
    Key modifications:
    - Removes puzzle embedding logic
    - Sets puzzle_emb_len = 0 unconditionally
    - Modifies input handling to work with input_ids
    - Optionally disables Q-head for pure language modeling
    """
    
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config, enable_q_head: bool = False):
        super().__init__(config)
        
        # Override puzzle_emb_len to be 0 unconditionally for code generation
        self.puzzle_emb_len = 0
        
        # Store enable_q_head flag
        self.enable_q_head = enable_q_head
        
        # If Q-head is disabled, set it to None to avoid unnecessary computation
        if not enable_q_head:
            self.q_head = None
    
    def _input_embeddings(self, input_ids: torch.Tensor):
        """
        Modified input embedding method that only handles token embeddings.
        
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len]
            
        Returns:
            Embedded representation [batch_size, seq_len, hidden_size]
        """
        # Token embedding
        embedding = self.embed_tokens(input_ids.to(torch.int32))
        
        # Position embeddings
        if self.config.pos_encodings == "learned":
            # Scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
        
        # Scale
        return self.embed_scale * embedding
    
    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with modified input handling for code generation.
        
        Args:
            carry: State carried between forward passes
            batch: Dictionary containing input_ids
            
        Returns:
            Tuple of (new_carry, output_logits, q_values)
        """
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        
        # Input encoding - modified to use input_ids directly
        input_embeddings = self._input_embeddings(batch["input_ids"])
        
        # Forward iterations
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            
            for _H_step in range(self.config.H_cycles):
                for _L_step in range(self.config.L_cycles):
                    if not ((_H_step == self.config.H_cycles - 1) and (_L_step == self.config.L_cycles - 1)):
                        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                
                if not (_H_step == self.config.H_cycles - 1):
                    z_H = self.H_level(z_H, z_L, **seq_info)
        
        assert not z_H.requires_grad and not z_L.requires_grad
        
        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)
        
        # LM Outputs
        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)  # No slicing needed since puzzle_emb_len = 0
        
        # Q head (if enabled)
        if self.enable_q_head and self.q_head is not None:
            q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
            q_values = (q_logits[..., 0], q_logits[..., 1])
        else:
            # Return dummy values when Q-head is disabled
            q_values = (torch.zeros(z_H.size(0), device=z_H.device), 
                       torch.zeros(z_H.size(0), device=z_H.device))
        
        return new_carry, output, q_values


class HRMCodeGenerator(nn.Module):
    """
    HRM model adapted for code generation tasks.
    
    This class wraps the original HRM architecture with modifications
    for causal language modeling and autoregressive text generation.
    """
    
    def __init__(self, config_dict: Dict[str, Any], enable_q_head: bool = False):
        """
        Initialize the HRM code generator.
        
        Args:
            config_dict: Configuration dictionary for HRM
            enable_q_head: Whether to enable the Q-head for adaptive computation
        """
        super().__init__()
        
        # Create HRM config
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        
        # Create our modified inner model
        self.inner = HRMCodeGeneratorInner(self.config, enable_q_head=enable_q_head)
        
        # Store generation parameters
        self.max_length = 512  # Default max generation length
        self.temperature = 0.8  # Default sampling temperature

        # Make sure every attention block is set to causal
        make_hrm_blocks_causal(self.inner, causal=True)
    
    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        """
        Create initial carry state for the model.
        
        Args:
            batch: Dictionary with input_ids
            
        Returns:
            Initial carry state
        """
        batch_size = batch["input_ids"].shape[0]
        
        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=batch["input_ids"].device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=batch["input_ids"].device),
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
    
    def forward(self, input_ids: torch.Tensor, carry: Optional[HierarchicalReasoningModel_ACTV1Carry] = None) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        """
        Forward pass for training and inference.
        
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len]
            carry: Optional carry state from previous forward pass
            
        Returns:
            Tuple of (new_carry, outputs)
        """
        # Create batch dictionary
        batch = {"input_ids": input_ids}
        
        # Initialize carry if not provided
        if carry is None:
            carry = self.initial_carry(batch)
        
        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)
        
        new_current_data = {k: torch.where(
            carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), 
            batch[k], 
            v
        ) for k, v in carry.current_data.items()}
        
        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)
        
        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        # In code generation mode, we always use a single step
        # (no adaptive computation time)
        new_steps = new_steps + 1
        halted = torch.ones_like(new_steps, dtype=torch.bool)
        
        # Create new carry
        new_carry = HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=new_inner_carry,
            steps=new_steps,
            halted=halted,
            current_data=new_current_data
        )
        
        return new_carry, outputs
    
    def generate(
        self, 
        prompt: Union[str, List[str]],
        max_length: Optional[int] = None,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: Text prompt or list of prompts
            max_length: Maximum length of generated text (including prompt)
            temperature: Sampling temperature (higher = more random)
            top_k: Number of highest probability tokens to keep for sampling
            top_p: Cumulative probability threshold for nucleus sampling
            num_return_sequences: Number of sequences to return per prompt
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Token ID for padding
            eos_token_id: Token ID for end of sequence
            **kwargs: Additional arguments
            
        Returns:
            List of generated text sequences
        """
        # Set defaults
        max_length = max_length or self.max_length
        temperature = temperature or self.temperature
        
        # Get tokenizer
        tokenizer = get_tokenizer()
        pad_token_id = pad_token_id or tokenizer.pad_token_id
        eos_token_id = eos_token_id or tokenizer.eos_token_id
        
        # Encode prompt
        is_single_prompt = isinstance(prompt, str)
        if is_single_prompt:
            prompt = [prompt]
        
        # Encode prompts
        encoded_prompts = encode(prompt, return_tensors="pt")
        input_ids = encoded_prompts["input_ids"]
        
        # Move to device
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)
        
        # Generate tokens
        generated_ids = self._generate_tokens(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        # Decode generated tokens
        generated_texts = []
        for ids in generated_ids:
            text = decode(ids, skip_special_tokens=True)
            generated_texts.append(text)
        
        # If single prompt, return just the first sequence if num_return_sequences=1
        if is_single_prompt and num_return_sequences == 1:
            return generated_texts[0]
        
        return generated_texts
    
    def _generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        num_return_sequences: int,
        do_sample: bool,
        pad_token_id: int,
        eos_token_id: int,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: K for top-k sampling
            top_p: P for nucleus sampling
            num_return_sequences: Number of sequences to return per input
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Token ID for padding
            eos_token_id: Token ID for end of sequence
            **kwargs: Additional arguments
            
        Returns:
            List of generated token sequences
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Expand input_ids for multiple return sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size
        
        # Initialize generation
        unfinished_sequences = torch.ones(effective_batch_size, dtype=torch.bool, device=device)
        cur_len = input_ids.shape[1]
        
        # Initialize carry
        carry = self.initial_carry({"input_ids": input_ids})
        
        # Store all generated token ids
        generated_ids = input_ids.clone()
        
        # Generate tokens until max_length or all sequences finished
        while cur_len < max_length and torch.any(unfinished_sequences):
            # Forward pass
            carry, outputs = self.forward(input_ids=generated_ids, carry=carry)
            
            # Get next token logits
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            if temperature != 1.0 and temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                indices_to_remove = torch.topk(next_token_logits, k=top_k, dim=-1)[0][..., -1, None] <= next_token_logits
                next_token_logits = torch.where(indices_to_remove, next_token_logits, torch.tensor(-float("Inf"), device=device))
            
            # Apply top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
                next_token_logits = torch.where(indices_to_remove, torch.tensor(-float("Inf"), device=device), next_token_logits)
            
            # Sample or greedy decode
            if do_sample:
                # Sample from the filtered distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Update unfinished sequences
            if eos_token_id is not None:
                # Tokens that have finished (generated EOS)
                tokens_finished = next_tokens == eos_token_id
                unfinished_sequences = unfinished_sequences & ~tokens_finished
            
            # Only keep unfinished sequences
            next_tokens = next_tokens.masked_fill(~unfinished_sequences, pad_token_id)
            
            # Append next tokens to generated_ids
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update cur_len
            cur_len = generated_ids.shape[1]
            
            # Update input_ids for next iteration
            input_ids = generated_ids
        
        # Return generated tokens
        return [seq for seq in generated_ids]
    
    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[Dict[str, Any]] = None) -> "HRMCodeGenerator":
        """
        Load a pretrained model from a directory.
        
        Args:
            model_path: Path to the model directory
            config: Optional configuration dictionary
            
        Returns:
            Loaded model
        """
        # Load configuration if not provided
        if config is None:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Configuration not found at {config_path}")
        
        # Create model
        model = cls(config_dict=config)
        
        # Load state dict
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            raise ValueError(f"Model weights not found at {state_dict_path}")
        
        return model
    
    def save_pretrained(self, save_directory: str, save_config: bool = True):
        """
        Save the model to a directory.
        
        Args:
            save_directory: Directory to save the model
            save_config: Whether to save the configuration
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Save configuration
        if save_config:
            config_path = os.path.join(save_directory, "config.json")
            import json
            with open(config_path, "w") as f:
                # Convert config to dict
                config_dict = {k: v for k, v in self.config.dict().items()}
                json.dump(config_dict, f, indent=2)
    
    @staticmethod
    def from_config(config: CodeGenConfig) -> "HRMCodeGenerator":
        """
        Create a model from a CodeGenConfig.
        
        Args:
            config: CodeGenConfig instance
            
        Returns:
            HRMCodeGenerator instance
        """
        # Convert config to HRM format
        hrm_config = config.to_hrm_config()
        
        # Get enable_q_head flag
        enable_q_head = config.model.get("enable_q_head", False)
        
        # Create model
        return HRMCodeGenerator(config_dict=hrm_config, enable_q_head=enable_q_head)
