from abc import ABC, abstractmethod

import torch

from math_llm.training.rollout import Experience


class BaseLoss(ABC):
    """
    All loss functions implement this interface.

    compute() receives a batch of Experience objects and the live Policy,
    and returns a (loss_tensor, metrics_dict) pair.
    The trainer calls loss.backward() on the returned tensor.
    """

    @abstractmethod
    def compute(
        self,
        experiences: list[Experience],
        policy,
    ) -> tuple[torch.Tensor, dict]:
        ...

    def __call__(self, experiences, policy):
        return self.compute(experiences, policy)
