"""
Lean Proof Benchmark

A simple benchmark package for evaluating LLM agents on Lean 4 theorem proving.
"""

import os

# Must be set before torch is imported anywhere below: torch 2.13's
# `torch._native` op-dispatch system silently reroutes some matmul shapes
# (e.g. Qwen3's rotary-embedding outer product) to a JIT-compiled Triton
# kernel by default. Compiling that kernel invokes gcc against Python.h,
# which this environment's python3.12 install doesn't ship - the model
# still loads fine, but the very first forward pass crashes. Disabling
# torch's native-JIT op overrides falls back to the regular aten/cuBLAS
# implementation and avoids ever needing gcc/Python.h.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

__version__ = "0.1.0"

from math_llm.lean_server import LeanServer, LeanResult, MATHLIB_VERSION, LEAN_TOOLCHAIN
from math_llm.data import Problem, load_data, list_datasets
from math_llm.agents import SimpleAgent

__all__ = [
    # Versions
    "MATHLIB_VERSION",
    "LEAN_TOOLCHAIN",
    # Core
    "LeanServer",
    "LeanResult",
    "Problem",
    "load_data",
    "list_datasets",
    "SimpleAgent",
]
