"""
Reward functions.

Add new rewards by subclassing RewardFunction and implementing __call__.
"""

from abc import ABC, abstractmethod

from math_llm.data import Problem
from math_llm.lean_server import LeanServer

class RewardFunction(ABC):
    @abstractmethod
    def __call__(self, problem: Problem, completion: str) -> float:
        ...


class LeanReward(RewardFunction):
    """Binary reward: 1.0 if Lean accepts the proof, 0.0 otherwise."""

    def __init__(self, lean_server=None):
        if lean_server is None:
            lean_server = LeanServer()
            lean_server.start()
        self.lean_server = lean_server

    def __call__(self, problem: Problem, completion: str) -> float:
        result = self.lean_server.check_proof(problem.statement, completion)
        return 1.0 if result.complete else 0.0


class LengthPenalizedLeanReward(RewardFunction):
    """
    Lean reward with a soft penalty for overly long proofs.

    """

    def __init__(self, lean_server, max_tokens: int = 512, penalty: float = 0.5):
        self.lean_server = lean_server
        self.max_tokens = max_tokens
        self.penalty = penalty

    def __call__(self, problem: Problem, completion: str) -> float:
        result = self.lean_server.check_proof(problem.statement, completion)
        if not result.complete:
            return 0.0
        if len(completion.split()) > self.max_tokens:
            return 1.0 - self.penalty
        return 1.0
