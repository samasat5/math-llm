"""
GRPO loss (Group Relative Policy Optimization).

For each problem group:
  - recompute log probs under the current policy
  - compute importance-sampling ratio vs the reference model
  - clip the ratio (PPO-style)
  - weight by the pre-computed group-relative advantages
  - add a KL penalty term
"""

import torch

from math_llm.training.losses.base import BaseLoss
from math_llm.training.rollout import Experience


class GRPOLoss(BaseLoss):
    def __init__(self, clip_eps: float = 0.2, kl_coef: float = 0.01):
        self.clip_eps = clip_eps
        self.kl_coef = kl_coef

    def compute(
        self,
        experiences: list[Experience],
        policy,
    ) -> tuple[torch.Tensor, dict]:
        total_loss = torch.tensor(0.0, device=policy.device)
        pg_loss_sum = 0.0
        kl_sum = 0.0

        for exp in experiences:
            lp = policy.log_probs(
                exp.input_ids, exp.attention_mask, exp.response_mask
            )  # [G, L-1] per-token
            ref_lp = policy.ref_log_probs(
                exp.input_ids, exp.attention_mask, exp.response_mask
            )
    
            seq_lp = lp.sum(-1)                # [G]
            seq_old_lp = exp.log_probs.sum(-1) # [G] log probs at rollout time
            seq_ref = ref_lp.sum(-1) 
            # Importance-sampling ratio pi/old_pi
            ratio = torch.exp(seq_lp - seq_old_lp)  # [G] # k1 estimator lead to negative kl 
            log_ratio = seq_ref - seq_lp 

            # policy gradient, clipped 
            adv = exp.advantages.to(policy.device)   # [G]
            pg = -torch.min(
                ratio * adv,
                torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv,
            ).mean()

            # K3 KL penalty  ( E[π/π_ref - log π/π_ref -1])
            kl = (torch.exp(log_ratio) - log_ratio - 1).mean()

            total_loss = total_loss + pg + self.kl_coef * kl
            pg_loss_sum += pg.item()
            kl_sum += kl.item()

        n = len(experiences)
        loss = total_loss / n
        return loss, {
            "loss": loss.item(),
            "pg_loss": pg_loss_sum / n,
            "kl": kl_sum / n,
        }
