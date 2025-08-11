#!/usr/bin/env python3
"""
CI guard: ensure Sapient HRM is available and our HRM codegen layer imports.

Exits non-zero if `sapient_hrm` cannot be imported or if `hrm_codegen.model`
cannot be imported (which depends on Sapient HRM presence).

Usage:
  python scripts/check_hrm_usage.py
"""
from __future__ import annotations

import importlib.util
import sys


def main() -> int:
    # Check for Sapient HRM availability
    spec = importlib.util.find_spec("sapient_hrm")
    if spec is None:
        print(
            "Sapient HRM not found (module 'sapient_hrm' missing). "
            "Ensure external/sapient-hrm exists or install the package.",
            file=sys.stderr,
        )
        return 2

    # Check our HRM codegen adapter can import (relies on Sapient HRM)
    try:
        import hrm_codegen.model as model_mod  # noqa: F401
    except Exception as e:
        print(
            f"Failed to import hrm_codegen.model (depends on sapient_hrm): {e}",
            file=sys.stderr,
        )
        return 3

    print("Sapient HRM available and hrm_codegen.model import succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
