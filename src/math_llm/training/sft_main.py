"""
SFT entry point: fine-tune on Mathlib proofs, then save for GRPO/DAPO/ ... .
"""

import logging
import os
from datetime import datetime

from math_llm.training.config import TrainingConfig
from math_llm.training.policy import Policy
from math_llm.training.sft import SFTTrainer
from math_llm.training.sft_data import load_mathlib_proofs, ProofPair
from math_llm.agents.simple import SYSTEM_PROMPT


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"sft_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    print("[sft] Loading Mathlib proofs...")
    pairs = load_mathlib_proofs(max_samples=50_000)
    print(f"[sft] Loaded {len(pairs)} (statement, proof) pairs")
    print(f"one pair example : {pairs[0]}")

    config = TrainingConfig(
        batch_size=4,
        learning_rate=2e-5,     # higher than GRPO, standard SFT rate
        max_grad_norm=1.0,
        output_dir=os.path.join(output_dir, "checkpoints"),
    )

    policy = Policy(config.model_name)

    def prompt_fn(pair: ProofPair) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Prove: {pair.statement}"},
        ]
        return policy.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    trainer = SFTTrainer(policy=policy, config=config, prompt_fn=prompt_fn)
    trainer.train(pairs, n_epochs=1)

    model_path = os.path.join(output_dir, "model")
    policy.model.save_pretrained(model_path)
    policy.tokenizer.save_pretrained(model_path)
    print(f"[sft] Model saved to {model_path}")
