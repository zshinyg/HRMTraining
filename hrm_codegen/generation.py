"""
Generation utilities for HRM code generation.

This module provides utilities and helper functions for generating code
with the HRM model, including sampling strategies, prompt processing,
and batch generation capabilities.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import torch
import torch.nn.functional as F

# Import our tokenization utilities
from tokenization import get_tokenizer, encode, decode

# Avoid circular imports by using type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import HRMCodeGenerator


def generate_code(
    model: "HRMCodeGenerator",
    prompt: Union[str, List[str]],
    max_length: Optional[int] = None,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    num_return_sequences: int = 1,
    do_sample: bool = True,
    stop_tokens: Optional[List[str]] = None,
    clean_output: bool = True,
    **kwargs,
) -> Union[str, List[str]]:
    """
    High-level function to generate code from a prompt.

    Args:
        model: HRMCodeGenerator model
        prompt: Text prompt or list of prompts
        max_length: Maximum length of generated text (including prompt)
        temperature: Sampling temperature (higher = more random)
        top_k: Number of highest probability tokens to keep for sampling
        top_p: Cumulative probability threshold for nucleus sampling
        num_return_sequences: Number of sequences to return per prompt
        do_sample: Whether to sample or use greedy decoding
        stop_tokens: List of strings that signal the end of generation
        clean_output: Whether to clean and format the generated code
        **kwargs: Additional arguments for generation

    Returns:
        Generated code string or list of strings
    """
    # Format prompt if needed
    if isinstance(prompt, str) and not prompt.strip().startswith("#"):
        prompt = format_prompt(prompt)
    elif isinstance(prompt, list):
        prompt = [
            format_prompt(p) if not p.strip().startswith("#") else p for p in prompt
        ]

    # Generate text
    generated_text = model.generate(
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        **kwargs,
    )

    # Apply stop tokens if provided
    if stop_tokens:
        if isinstance(generated_text, str):
            generated_text = apply_stop_tokens(generated_text, stop_tokens)
        else:
            generated_text = [
                apply_stop_tokens(text, stop_tokens) for text in generated_text
            ]

    # Clean and format output if requested
    if clean_output:
        if isinstance(generated_text, str):
            generated_text = extract_code(generated_text)
        else:
            generated_text = [extract_code(text) for text in generated_text]

    return generated_text


def format_prompt(prompt: str) -> str:
    """
    Format a raw prompt into a code generation prompt.

    Args:
        prompt: Raw text prompt

    Returns:
        Formatted prompt
    """
    # If prompt already starts with a comment, leave it as is
    if prompt.strip().startswith("#"):
        return prompt

    # Otherwise, format as a comment
    return f"# {prompt.strip()}\n\n"


def apply_stop_tokens(text: str, stop_tokens: List[str]) -> str:
    """
    Truncate generated text at the first occurrence of any stop token.

    Args:
        text: Generated text
        stop_tokens: List of strings that signal the end of generation

    Returns:
        Truncated text
    """
    for stop_token in stop_tokens:
        if stop_token in text:
            text = text[: text.index(stop_token)]

    return text


def extract_code(text: str) -> str:
    """
    Extract and clean Python code from generated text.

    Args:
        text: Generated text that may contain code and comments

    Returns:
        Cleaned Python code
    """
    # Split by lines
    lines = text.split("\n")

    # Find the first non-comment, non-empty line (start of code)
    code_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            code_start = i
            break

    # Extract code (including comments within the code)
    code_lines = lines[code_start:]

    # Remove trailing empty lines
    while code_lines and not code_lines[-1].strip():
        code_lines.pop()

    # Join back into a string
    code = "\n".join(code_lines)

    return code


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    do_sample: bool = True,
) -> torch.Tensor:
    """
    Sample next tokens from logits using various sampling strategies.

    Args:
        logits: Token logits [batch_size, vocab_size]
        temperature: Sampling temperature
        top_k: Number of highest probability tokens to keep
        top_p: Cumulative probability threshold for nucleus sampling
        do_sample: Whether to sample or use greedy decoding

    Returns:
        Sampled token indices [batch_size]
    """
    # Apply temperature
    if temperature > 0:
        logits = logits / temperature

    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, -float("inf"))

    # Apply top-p (nucleus) filtering
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("inf"))

    # Sample or greedy decode
    if do_sample:
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    else:
        # Greedy decoding
        next_tokens = torch.argmax(logits, dim=-1)

    return next_tokens


def batch_generate(
    model: "HRMCodeGenerator",
    prompts: List[str],
    batch_size: int = 8,
    **generate_kwargs,
) -> List[str]:
    """
    Generate code for a large list of prompts in batches.

    Args:
        model: HRMCodeGenerator model
        prompts: List of text prompts
        batch_size: Batch size for generation
        **generate_kwargs: Additional arguments for generate_code

    Returns:
        List of generated code strings
    """
    results = []

    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_results = generate_code(model, batch_prompts, **generate_kwargs)
        results.extend(batch_results)

    return results


def generate_with_timeout(
    model: "HRMCodeGenerator",
    prompt: str,
    timeout_seconds: float = 5.0,
    fallback_result: str = "",
    **generate_kwargs,
) -> str:
    """
    Generate code with a timeout to prevent hanging.

    Args:
        model: HRMCodeGenerator model
        prompt: Text prompt
        timeout_seconds: Maximum time to wait for generation
        fallback_result: Result to return if timeout occurs
        **generate_kwargs: Additional arguments for generate_code

    Returns:
        Generated code or fallback result if timeout occurs
    """
    import threading
    import queue

    result_queue = queue.Queue()

    def _generate():
        try:
            result = generate_code(model, prompt, **generate_kwargs)
            result_queue.put(result)
        except Exception as e:
            result_queue.put(f"Error: {str(e)}")

    # Start generation in a separate thread
    thread = threading.Thread(target=_generate)
    thread.daemon = True
    thread.start()

    # Wait for result or timeout
    try:
        result = result_queue.get(timeout=timeout_seconds)
        return result
    except queue.Empty:
        return fallback_result


def evaluate_pass_at_k(
    model: "HRMCodeGenerator",
    prompt: str,
    test_cases: List[str],
    k: int = 1,
    temperature: float = 0.8,
    timeout_seconds: float = 5.0,
    execution_timeout: float = 3.0,
) -> Tuple[bool, List[str]]:
    """
    Evaluate if any of k generated solutions pass the test cases.

    Args:
        model: HRMCodeGenerator model
        prompt: Text prompt
        test_cases: List of test case strings to evaluate
        k: Number of solutions to generate
        temperature: Sampling temperature
        timeout_seconds: Maximum time to wait for generation
        execution_timeout: Maximum time to wait for test execution

    Returns:
        Tuple of (passed, generated_solutions)
    """
    # Generate k solutions
    solutions = generate_code(
        model,
        prompt,
        num_return_sequences=k,
        temperature=temperature,
        clean_output=True,
    )

    if isinstance(solutions, str):
        solutions = [solutions]

    # Test each solution
    for solution in solutions:
        if _execute_tests(solution, test_cases, timeout=execution_timeout):
            return True, solutions

    return False, solutions


def _execute_tests(code: str, test_cases: List[str], timeout: float = 3.0) -> bool:
    """
    Execute test cases against generated code.

    Args:
        code: Generated Python code
        test_cases: List of test case strings to evaluate
        timeout: Maximum time to wait for execution

    Returns:
        True if all tests pass, False otherwise
    """
    import subprocess
    import tempfile

    # Create a temporary file with the code and test cases
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code + "\n\n")
        f.write("if __name__ == '__main__':\n")
        for test in test_cases:
            f.write(f"    {test}\n")
        temp_path = f.name

    try:
        # Run the code with test cases
        result = subprocess.run(
            [sys.executable, temp_path], capture_output=True, text=True, timeout=timeout
        )

        # Check if execution was successful
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def format_test_cases(test_cases: List[str]) -> str:
    """
    Format test cases for inclusion in a prompt.

    Args:
        test_cases: List of test case strings

    Returns:
        Formatted test cases as a string
    """
    if not test_cases:
        return ""

    formatted = "# Test cases:\n"
    for test in test_cases:
        formatted += f"# {test}\n"

    return formatted + "\n"


def generate_with_test_cases(
    model: "HRMCodeGenerator",
    description: str,
    test_cases: List[str],
    **generate_kwargs,
) -> str:
    """
    Generate code with test cases included in the prompt.

    Args:
        model: HRMCodeGenerator model
        description: Problem description
        test_cases: List of test case strings
        **generate_kwargs: Additional arguments for generate_code

    Returns:
        Generated code
    """
    # Format description as a comment if needed
    if not description.strip().startswith("#"):
        description = f"# {description.strip()}\n"

    # Format test cases
    test_cases_str = format_test_cases(test_cases)

    # Combine into a prompt
    prompt = description + test_cases_str

    # Generate code
    return generate_code(model, prompt, **generate_kwargs)


def stream_tokens(
    model: "HRMCodeGenerator",
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.8,
    callback: Optional[Callable[[str], None]] = None,
    **generate_kwargs,
) -> str:
    """
    Stream generated tokens one by one with a callback.

    Args:
        model: HRMCodeGenerator model
        prompt: Text prompt
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        callback: Function to call with each new token
        **generate_kwargs: Additional arguments for generation

    Returns:
        Complete generated text
    """
    # Get tokenizer
    tokenizer = get_tokenizer()

    # Encode prompt
    input_ids = encode(prompt, return_tensors="pt")["input_ids"].to(
        next(model.parameters()).device
    )

    # Initialize generation
    generated_ids = input_ids.clone()
    generated_text = prompt

    # Initialize carry
    carry = model.initial_carry({"input_ids": input_ids})

    # Generate tokens one by one
    for _ in range(max_length - input_ids.shape[1]):
        # Forward pass
        carry, outputs = model.forward(input_ids=generated_ids, carry=carry)

        # Get next token logits
        next_token_logits = outputs["logits"][0, -1, :]

        # Sample next token
        next_token = sample_tokens(
            next_token_logits.unsqueeze(0), temperature=temperature, **generate_kwargs
        )[0]

        # Check if EOS token
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Append next token
        generated_ids = torch.cat(
            [generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1
        )

        # Decode new token
        new_text = decode(next_token.unsqueeze(0), skip_special_tokens=True)
        generated_text += new_text

        # Call callback if provided
        if callback:
            callback(new_text)

    return generated_text
