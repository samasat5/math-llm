"""
SFT trainer: supervised fine-tuning on (statement, proof) pairs.

Loss is cross-entropy on proof tokens only (response_mask).
Run this before GRPO to teach the model Lean 4 tactic syntax.
"""

import random
import torch
from tqdm import tqdm

from math_llm.training.policy import Policy
from math_llm.training.config import TrainingConfig
from math_llm.training.sft_data import ProofPair


class SFTTrainer:
    def __init__(
        self,
        policy: Policy,
        config: TrainingConfig,
        prompt_fn,          # ProofPair -> str  (formats the statement as a prompt)
    ):
        self.policy = policy
        self.config = config
        self.prompt_fn = prompt_fn
        self.optimizer = torch.optim.AdamW(
            policy.trainable_parameters(),
            lr=config.learning_rate,
        )

    def _loss(self, pairs: list[ProofPair]) -> torch.Tensor:
        tokenized = [
            self.policy.tokenize(self.prompt_fn(p), p.proof, max_len=self.config.max_seq_len)
            for p in pairs
        ]

        max_len = max(t["input_ids"].shape[0] for t in tokenized)
        n = len(tokenized)
        input_ids    = torch.zeros(n, max_len, dtype=torch.long)
        attention_mask = torch.zeros(n, max_len, dtype=torch.long)
        response_mask  = torch.zeros(n, max_len, dtype=torch.float)

        for i, t in enumerate(tokenized):
            L = t["input_ids"].shape[0]
            input_ids[i, :L]     = t["input_ids"]
            attention_mask[i, :L] = 1
            response_mask[i, :L]  = t["response_mask"]

        dev = self.policy.device
        input_ids      = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)
        response_mask  = response_mask.to(dev)

        lp = self.policy.log_probs(input_ids, attention_mask, response_mask)  # [B, L-1]
        n_tokens = response_mask[:, 1:].sum().clamp(min=1)
        return -lp.sum() / n_tokens

    def train(self, dataset: list[ProofPair], n_epochs: int = 1):
        for epoch in range(n_epochs):
            random.shuffle(dataset)
            batches = [
                dataset[i:i + self.config.batch_size]
                for i in range(0, len(dataset), self.config.batch_size)
            ]
            pbar = tqdm(batches, desc=f"sft epoch {epoch+1}/{n_epochs}")
            for batch in pbar:
                self.optimizer.zero_grad()
                loss = self._loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
