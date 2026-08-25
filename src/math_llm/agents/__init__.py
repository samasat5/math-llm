"""
Lean proof agents for benchmarking.

Two agent types:
- SimpleAgent: Direct single-shot proof generation
- ToolAgent: Iterative proof with lean tool calls

Plus an optional rescue agent:
- Autoformalizer: translates a prover's own (still-sound) plan into Lean 4
  tactics when the prover itself loops instead of writing them
"""

from math_llm.agents.simple import SimpleAgent
from math_llm.agents.tool import ToolAgent
from math_llm.agents.autoformalizer import Autoformalizer

__all__ = ["SimpleAgent", "ToolAgent", "Autoformalizer"]
