"""
HRM Code Generation Adaptation Package

This package contains the adaptation of Sapient's Hierarchical Reasoning Model (HRM)
for code generation tasks. The original puzzle-solving architecture has been modified
to support causal language modeling for generating Python code.

Key modifications:
- Causal attention masking for autoregressive generation
- Removal of puzzle-specific embedding components
- Addition of code generation-specific configuration
- Integration with the MBPP dataset and evaluation

Usage:
    from hrm_codegen import HRMCodeGenerator, generate_code
    
    model = HRMCodeGenerator.from_pretrained("checkpoints/codegen_base")
    code = generate_code(model, "Write a function to sort a list of integers")
"""

# Package version
__version__ = "0.1.0"

# Import main components
# NOTE: The model and generation utilities are commented out until fully implemented
# from .model import HRMCodeGenerator
# from .generation import generate_code, sample_tokens

# --------------------------------------------------------------------------- #
# Temporary import route
# --------------------------------------------------------------------------- #
# The original configuration helper (`hrm_codegen.config`) depends on the full
# Sapient-HRM codebase and native Flash-Attention bindings.  Until those heavy
# dependencies are available in the execution environment we expose a
# lightweight “stand-alone” drop-in that mimics the public API but avoids the
# optional imports.  Once all native dependencies are satisfied (or replaced
# with pure-Python fall-backs) the import below can be reverted to::
#     from .config import ...
# --------------------------------------------------------------------------- #
from .config_standalone import (  # type: ignore
    CodeGenConfig,
    load_config,
    get_default_config,
    merge_configs,
    create_runtime_config,
)

# Public exports
__all__ = [
    # Config functionality (currently working)
    "CodeGenConfig",
    "load_config",
    "get_default_config",
    "merge_configs",
    "create_runtime_config",
    
    # Model and generation (to be uncommented when implemented)
    # "HRMCodeGenerator",
    # "generate_code",
    # "sample_tokens"
]

# Note: When hrm_codegen.model and hrm_codegen.generation are fully operational,
# uncomment their imports at the top of the file and in the __all__ list.
