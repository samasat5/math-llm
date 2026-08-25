import json
import logging
import os
from datetime import datetime
from math_llm.data import load_minif2f, load_numina
from math_llm.training.config import TrainingConfig
from math_llm.training.policy import Policy
from math_llm.training.reward import LeanReward
from math_llm.training.losses.grpo import GRPOLoss
from math_llm.training.trainer import Trainer
from math_llm.agents.simple import SYSTEM_PROMPT, extract_proof
from math_llm.lean_server import LeanServerPool



def evaluate(trainer: Trainer, problems, reward) -> dict:
    """Run one rollout on problems and return pass@1 stats."""
    results = []
    for p in problems:
        prompt = trainer.prompt_fn(p)
        completions = trainer.policy.generate([prompt], n_samples=1, temperature=0.1)[0]
        proof = extract_proof(completions[0])
        r = reward(p, proof)
        results.append({"id": p.id, "proof": proof, "reward": r})
    pass_at_1 = sum(r["reward"] for r in results) / len(results)
    return {"pass_at_1": pass_at_1, "results": results}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", run_id)
    os.makedirs(output_dir, exist_ok=True)

    problems = load_minif2f(n_samples=300, test=False)
    test_problems = load_minif2f(n_samples=10, test=True)
    # problems = load_numina(n_samples=2000, test=False)
    # test_problems = load_numina(n_samples=100, test=True)

    config = TrainingConfig(
        model_name="Qwen/Qwen2.5-14B-Instruct",
        group_size=10,
        temperature=0.9,
        batch_size=1,
        n_epochs=1,
        log_every=1,
        save_every=100,
        output_dir=os.path.join(output_dir, "checkpoints"),
    )
    policy = Policy(config.model_name)
    lean_pool = LeanServerPool(n_workers=config.group_size)
    lean_pool.start()
    reward = LeanReward(lean_pool)

    def prompt_fn(p):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Prove: {p.statement}"},
        ]
        return policy.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    trainer = Trainer(
        policy=policy,
        reward_fn=reward,
        loss_fn=GRPOLoss(),
        config=config,
        prompt_fn=prompt_fn,
        completion_fn=extract_proof,
    )

    # Evaluate before training
    eval_before = evaluate(trainer, test_problems, reward)
    print(f"[eval] before training pass@1={eval_before['pass_at_1']:.3f}")

    trainer.train(problems)
    
    model_path = os.path.join(output_dir, "model")
    policy.model.save_pretrained(model_path)
    policy.tokenizer.save_pretrained(model_path)
    print(f"[main] Model saved to {model_path}")

    # Evaluate after training
    eval_after = evaluate(trainer, test_problems, reward)
    print(f"[eval] after training pass@1={eval_after['pass_at_1']:.3f}")

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "config": config.__dict__,
            "eval_before": eval_before,
            "eval_after": eval_after,
        }, f, indent=2)
    print(f"[main] Results saved to {results_path}")
