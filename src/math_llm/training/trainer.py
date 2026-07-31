"""
Trainer: orchestrates rollout → loss → optimizer step.

Usage:
    config = TrainingConfig(model_name="Qwen/Qwen2.5-7B-Instruct", group_size=8)
    policy = Policy(config.model_name)
    reward = LeanReward(lean_server)
    loss_fn = GRPOLoss()
    trainer = Trainer(policy, reward, loss_fn, config, prompt_fn=build_prompt)
    trainer.train(problems)
"""

import os
from time import time
import random
from concurrent.futures import ThreadPoolExecutor
import torch
from tqdm import tqdm
from math_llm.data import Problem
from math_llm.training.config import TrainingConfig
from math_llm.training.policy import Policy
from math_llm.training.reward import RewardFunction
from math_llm.training.losses.base import BaseLoss
from math_llm.training.rollout import Experience, compute_advantages


class Trainer:
    def __init__(
        self,
        policy: Policy,
        reward_fn: RewardFunction,
        loss_fn: BaseLoss,
        config: TrainingConfig,
        prompt_fn,           # Problem -> str
        completion_fn=None,  # str -> str, post-process completions (e.g. extract_proof)
    ):
        self.policy = policy
        self.reward_fn = reward_fn
        self.loss_fn = loss_fn
        self.config = config
        self.prompt_fn = prompt_fn
        self.completion_fn = completion_fn or (lambda x: x)

        self.optimizer = torch.optim.AdamW(
            policy.trainable_parameters(),
            lr=config.learning_rate,
        )
        self.step = 0

    def rollout(self, problems: list[Problem]) -> list[Experience]:
        """Sample G completions per problem, score them, build Experience objects."""
        prompts = [self.prompt_fn(p) for p in problems]
        
        all_completions = self.policy.generate(
            prompts,
            n_samples=self.config.group_size,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
        )

        experiences = []
        for problem, prompt, completions in zip(problems, prompts, all_completions):
            completions = [self.completion_fn(c) for c in completions]
            t1= time()
            with ThreadPoolExecutor(max_workers=self.config.group_size) as ex:
                reward_vals = list(ex.map(lambda c: self.reward_fn(problem, c), completions))
            rewards = torch.tensor(reward_vals, dtype=torch.float)
            if rewards.max() > 0 or self.step % 10 == 0:
                print(f"[rollout] step={self.step} completion: {completions[0][:120]!r}")
                print(f"[rollout] rewards: {rewards.tolist()}")
            t_verify= time()-t1
            advantages = compute_advantages(rewards) # G

            tokenized = [
                self.policy.tokenize(prompt, c, max_len=self.config.max_seq_len)
                for c in completions
            ]
            input_ids, attention_mask, response_mask = self._pad(tokenized)

            with torch.no_grad():
                lp = self.policy.log_probs(input_ids, attention_mask, response_mask)

            experiences.append(Experience(
                problem=problem,
                prompt=prompt,
                completions=completions,
                rewards=rewards,
                advantages=advantages,
                input_ids=input_ids,
                attention_mask=attention_mask,
                response_mask=response_mask,
                log_probs=lp,
            ))
            print(f"t verify:{t_verify}")

        return experiences

    def _pad(self, tokenized: list[dict]):
        max_len = max(t["input_ids"].shape[0] for t in tokenized)
        n = len(tokenized)

        input_ids = torch.zeros(n, max_len, dtype=torch.long)
        attention_mask = torch.zeros(n, max_len, dtype=torch.long)
        response_mask = torch.zeros(n, max_len, dtype=torch.float)

        for i, t in enumerate(tokenized):
            seq_len = t["input_ids"].shape[0]
            input_ids[i, :seq_len] = t["input_ids"]
            attention_mask[i, :seq_len] = 1
            response_mask[i, :seq_len] = t["response_mask"]

        dev = self.policy.device
        return input_ids.to(dev), attention_mask.to(dev), response_mask.to(dev)

    def train_step(self, problems: list[Problem]) -> dict:
        t0= time() 
        experiences = self.rollout(problems)
        t_gen   = time()-t0

        self.optimizer.zero_grad()
        t3=time()
        loss, metrics = self.loss_fn(experiences, self.policy)
        loss.backward() # d(pg)/d(seq_lp_i) = -adv_i
        torch.nn.utils.clip_grad_norm_(
            self.policy.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        t_train = time()-t3
        print(f"t_train: {t_train:.2f}s, t_gen: {t_gen:.2f}s")
        metrics["reward_mean"] = sum(e.rewards.mean().item() for e in experiences) / len(experiences)
        metrics["reward_max"] = max(e.rewards.max().item() for e in experiences)
        # each experience: G shape
        all_adv = torch.cat([e.advantages for e in experiences]) # over all the batch => GxB (Gx|promptsxbatchsize|)
        metrics["adv_mean"] = all_adv.mean().item() # mean over all G and all batch 
        metrics["adv_std"] = all_adv.std().item()

        total_lp = sum(e.log_probs.sum().item() for e in experiences)
        total_tokens = sum(e.response_mask.sum().item() for e in experiences)
        metrics["entropy"] = -total_lp / (total_tokens + 1e-8)

        return metrics

    def train(self, dataset: list[Problem]):
        for epoch in range(self.config.n_epochs):
            random.shuffle(dataset)
            batches = [
                dataset[i:i + self.config.batch_size]
                for i in range(0, len(dataset), self.config.batch_size)
            ]
            pbar = tqdm(batches, desc=f"epoch {epoch+1}/{self.config.n_epochs}")
            for batch in pbar:
                metrics = self.train_step(batch)
                self.step += 1
                pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

                if self.step % self.config.log_every == 0:
                    m = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    print(f"[step {self.step}] {m}")

                if self.step % self.config.save_every == 0:
                    self._save(f"step_{self.step}")

            self._save(f"epoch_{epoch+1}")

    def _save(self, name: str):
        path = os.path.join(self.config.output_dir, name)
        os.makedirs(path, exist_ok=True)
        self.policy.model.save_pretrained(path)
        self.policy.tokenizer.save_pretrained(path)
        print(f"[trainer] Saved checkpoint to {path}")

