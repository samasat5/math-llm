"""One-off: prefix-conditioned generation for aime_1987_p5, one run per
model, using SimpleAgent.generate_proofs()'s prefix= param.

Contrast with the earlier "hint" runs (aime_1987_p5_hint_prompt_*.json):
pasting a candidate proof into the prompt as text ("The answer is: ...")
only *tells* the model about it - both Pythagoras and Goedel were free to
(and did) ignore or rewrite it there. A prefix instead becomes part of the
already-generated sequence before the first new token is sampled, so the
model's own KV cache is seeded with it - the continuation is conditioned on
it, not just shown it.

The two models need different prefixes because they answer in different raw
formats (see SimpleAgent.generate_proofs's docstring):
- Goedel emits bare tactics with no code fence, so the answer proof text
  works directly as the prefix.
- Pythagoras emits prose reasoning then a ```lean4 fence with the restated
  theorem; extract_pythagoras_proof() takes the LAST such fence. So the
  prefix here opens that final fence with the theorem statement's `:= sorry`
  replaced by the answer proof, and deliberately leaves the fence UNCLOSED -
  the model's cheapest continuation is to just close it out, rather than
  start over with a new fence (which would become the new "last" one and
  win extraction instead).

Usage: python prefix_conditioned_aime_1987_p5.py
"""

import json
import os
import textwrap
import time
from pathlib import Path

from math_llm.agents.case_study.temp_sweep_aime_1987_p5 import resolve_local_snapshot
from math_llm.agents.simple import (
    PYTHAGORAS_HEADER,
    PYTHAGORAS_TEMPERATURE,
    PYTHAGORAS_TOP_K,
    PYTHAGORAS_TOP_P,
    SimpleAgent,
    is_pythagoras_model,
)
from math_llm.agents.autoformalizer import extract_clean_plan
from math_llm.data import load_data
from math_llm.lean_server import LeanServer

MAX_NEW_TOKENS = 20000
PROBLEM_ID = "minif2f-lean4/aime_1987_p5"

ANSWER_PROOF = """have h1 : 3 * (x ^ 2 * y ^ 2) = 30 * x ^ 2 + 517 - y^2 := by linarith
have h2 : y^2 ≤ 517 := by
  nlinarith [sq_nonneg (x * y), sq_nonneg (x * y - 17), sq_nonneg (y - 17), sq_nonneg (x - 2), sq_nonneg (x + 2)]
have h3 : y ≤ 22 := by nlinarith [h2]
have h4 : y ≥ -22 := by nlinarith [sq_nonneg (y + 22), sq_nonneg (y + 21), sq_nonneg (y + 20), sq_nonneg (y + 19), sq_nonneg (y + 18), sq_nonneg (y + 17), sq_nonneg (y + 16), sq_nonneg (y + 15), sq_nonneg (y + 14), sq_nonneg (y + 13), sq_nonneg (y + 12), sq_nonneg (y + 11), sq_nonneg (y + 10), sq_nonneg (y + 9), sq_nonneg (y + 8), sq_nonneg (y + 7), sq_nonneg (y + 6), sq_nonneg (y + 5), sq_nonneg (y + 4), sq_nonneg (y + 3), sq_nonneg (y + 2), sq_nonneg (y + 1), sq_nonneg (y)]
interval_cases y <;>
  try {
    simp_all
    <;>
    (
      ring_nf
      <;>
      omega
    )
  }"""

# Verified against Kimina-Prover-Preview's own solved proof for this problem
# (~/testmath/Kimina-Prover-Preview/minif2f_test_solved/minif2f-test-solved.jsonl)
# - correct, but check_proof()'s default Mathlib environment has no
# `set_option maxHeartbeats 0` (the source file's header sets it, but that
# header is never sent - only the theorem+proof body is), and this proof's
# 45-way `interval_cases y <;> simp_all <;> ring_nf <;> omega` needs more
# than the 200000 default. Bump it per-call rather than touching LeanServer
# globally (this is the one proof in this repo that needs it so far).
MAX_HEARTBEATS = 1_000_000


def check_proof_with_heartbeats(lean_server: LeanServer, statement: str, proof: str, max_heartbeats: int):
    proof = proof.strip()
    if proof.startswith("by "):
        proof = proof[3:].strip()
    indented = "\n".join("  " + line for line in proof.splitlines())
    code = f"set_option maxHeartbeats {max_heartbeats} in\n" + statement.replace(":= sorry", f":= by\n{indented}")
    resp = lean_server._send_command({"cmd": code, "env": lean_server._env_id})
    return lean_server._parse_response(resp, execution_time=0.0)


def run_one(model: str, prefix: str, output_file: Path, temperature: float | None = None):
    problems = load_data("minif2f-lean4", None, min_tier=1)
    problem = next(p for p in problems if p.id == PROBLEM_ID)
    print(f"[prefix] Problem: {problem.id}  model: {model}")

    lean_server = LeanServer()
    print("[prefix] Starting Lean server...")
    lean_server.start()

    local_model_path = resolve_local_snapshot(model)
    print(f"[prefix] Using local model snapshot: {local_model_path}")

    seed = int.from_bytes(os.urandom(8), "big")
    print(f"[prefix] seed={seed}")

    kwargs = dict(model_name=local_model_path, lean_server=lean_server,
                  max_new_tokens=MAX_NEW_TOKENS, k=1, seed=seed)
    if temperature is not None:
        kwargs["temperature"] = temperature
    agent = SimpleAgent(**kwargs)
    agent.load_model()

    t0 = time.time()
    proof, response = next(agent.generate_proofs(problem, k=1, prefix=prefix))
    gen_time = time.time() - t0

    clean_plan = extract_clean_plan(response)
    lean_result = check_proof_with_heartbeats(lean_server, problem.statement, proof, MAX_HEARTBEATS)

    # Pythagoras ignores SimpleAgent's constructor temperature= entirely and
    # always samples with the hardcoded PYTHAGORAS_TEMPERATURE/TOP_P/TOP_K
    # module constants (see generate_proofs) - report what was actually used.
    if is_pythagoras_model(model):
        actual_temperature, actual_top_p, actual_top_k = PYTHAGORAS_TEMPERATURE, PYTHAGORAS_TOP_P, PYTHAGORAS_TOP_K
    else:
        actual_temperature, actual_top_p, actual_top_k = agent.temperature, None, None

    output = {
        "experiment": "aime_1987_p5_prefix_conditioned",
        "problem_id": problem.id,
        "model": model,
        "variant": "prefix_answer_v1",
        "answer_proof_used_as_prefix": ANSWER_PROOF,
        "prefix": prefix,
        "temperature": actual_temperature,
        "top_p": actual_top_p,
        "top_k": actual_top_k,
        "k": 1,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": seed,
        "raw_response": response,
        "clean_nl_reasoning": clean_plan,
        "extracted_proof": proof,
        "prefix_preserved_verbatim": proof.strip().startswith(ANSWER_PROOF.split("\n")[0].strip()),
        "success": lean_result.success,
        "complete": lean_result.complete,
        "error": lean_result.errors if lean_result.errors else None,
        "generation_time_seconds": gen_time,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[prefix] success={lean_result.success} complete={lean_result.complete} time={gen_time:.1f}s")
    print(f"[prefix] Saved to {output_file}")

    lean_server.stop()


def main():
    problems = load_data("minif2f-lean4", None, min_tier=1)
    problem = next(p for p in problems if p.id == PROBLEM_ID)

    # --- Goedel: raw tactics, no fence - the answer text is the prefix as-is.
    run_one(
        model="Goedel-LM/Goedel-Prover-V2-8B",
        prefix=ANSWER_PROOF + "\n",
        output_file=Path("outputs/aime_1987_p5_prefix_conditioned_prompt_Goedel-Prover-V2-8B_k1.json"),
        temperature=0.1,  # matches the original benchmark's config, same as the baseline/hint runs
    )

    # --- Pythagoras: open the model's own final ```lean4 fence with the
    # theorem statement's `:= sorry` replaced by the answer, fence left open.
    statement_with_proof = problem.statement.rsplit(":= sorry", 1)[0].rstrip() + " := by\n" + textwrap.indent(ANSWER_PROOF, "  ")
    formal_statement = PYTHAGORAS_HEADER + "\n" + statement_with_proof + "\n"
    pythagoras_prefix = "### Complete Lean 4 Proof\n\n```lean4\n" + formal_statement
    run_one(
        model="Pythagoras-LM/Pythagoras-Prover-4B",
        prefix=pythagoras_prefix,
        output_file=Path("outputs/aime_1987_p5_prefix_conditioned_prompt_Pythagoras-Prover-4B_k1.json"),
    )


if __name__ == "__main__":
    main()
