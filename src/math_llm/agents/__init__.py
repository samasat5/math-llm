"""
Lean proof agents for benchmarking.

Agent types:
- SimpleAgent: Direct single-shot proof generation (autoregressive models)

Plus an optional rescue agent:
- Autoformalizer: translates a prover's own (still-sound) plan into Lean 4
  tactics when the prover itself loops instead of writing them
"""

from math_llm.agents.simple import SimpleAgent
from math_llm.agents.autoformalizer import Autoformalizer

__all__ = ["SimpleAgent", "Autoformalizer"]
