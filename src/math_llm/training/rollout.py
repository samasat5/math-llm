"""
Rollout data structures.

Experience holds one group (G completions) for a single problem.
"""

from dataclasses import dataclass

import torch

from math_llm.data import Problem


@dataclass
class Experience:
    problem: Problem
    prompt: str
    completions: list[str]
    rewards: torch.Tensor        # [G]
    advantages: torch.Tensor     # [G]
    input_ids: torch.Tensor      # [G, seq_len]
    attention_mask: torch.Tensor # [G, seq_len]
    response_mask: torch.Tensor  # [G, seq_len]  1 = response token
    log_probs: torch.Tensor      # [G, seq_len-1]  at generation time


def compute_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize rewards within the group (GRPO baseline).
    When all rewards are identical (std≈0), return zeros — no signal, no update.
    """
    if rewards.std() < eps: # adv is having G shape
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (rewards.std() + eps)
