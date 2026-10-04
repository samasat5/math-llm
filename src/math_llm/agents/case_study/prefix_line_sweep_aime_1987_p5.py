"""One-off: prefix LINE sweep for aime_1987_p5, one run per (model, n) where
n = number of leading lines of the verified correct proof
(prefix_conditioned_aime_1987_p5.ANSWER_PROOF, 15 lines) handed to the model
as a prefix (see SimpleAgent.generate_proofs's prefix= docstring). For each
n = 1..15, the model completes the remaining tail itself; the FULL 15-line
result is re-verified with Lean every time (not just the model's tail), so a
"success" at small n means the model correctly reconstructed the rest of a
real, working proof on its own - not just repeated what it was given (that
was already confirmed separately: prefix_preserved_verbatim in the sibling
prefix_conditioned script).

Answers: "how much of the proof can we cut off before completion stops
being reliable" - for both provers.

Seed: FIXED across every (model, n) pair in this sweep (drawn once, reused
via SimpleAgent(seed=...) - see simple.py, seeds per sample index i so k=1
always draws with exactly this value) - varying only n, not the RNG draw,
so any change in outcome across n is attributable to the prefix length, not
resampling luck. Contrast with the earlier baseline/hint/prefix runs, which
each drew a fresh random seed and are NOT directly comparable to each other
on that axis.

max_new_tokens is capped at 6000 here (vs. 20000 elsewhere in this
experiment) - deliberately: the earlier full-prefix Goedel run took 822s
because it fell into a period-2 repetition loop
(LineRepetitionStoppingCriteria only catches 3 IDENTICAL consecutive lines,
not an alternating pattern) that the 20000-token budget let run to
completion. 30 sub-runs at that worst case would make this sweep
impractically slow; 6000 is still generous for what should mostly be short
completions once past the first few n, and any run that still needs more
than that to close out a 15-line proof is itself a meaningful data point.

Usage: python prefix_line_sweep_aime_1987_p5.py
"""

import json
import os
import textwrap
import time
from pathlib import Path

from math_llm.agents.case_study.temp_sweep_aime_1987_p5 import resolve_local_snapshot
from math_llm.agents.case_study.prefix_conditioned_aime_1987_p5 import (
    ANSWER_PROOF,
    MAX_HEARTBEATS,
    check_proof_with_heartbeats,
)
from math_llm.agents.simple import (
    PYTHAGORAS_HEADER,
    PYTHAGORAS_TEMPERATURE,
    PYTHAGORAS_TOP_K,
    PYTHAGORAS_TOP_P,
    SimpleAgent,
    is_pythagoras_model,
)
from math_llm.data import load_data
from math_llm.lean_server import LeanServer

MAX_NEW_TOKENS = 6000
PROBLEM_ID = "minif2f-lean4/aime_1987_p5"
ANSWER_LINES = ANSWER_PROOF.splitlines()
N_SEEDS = 8

OUTPUT_DIR = Path("outputs/experiment_aime_1987_p5")


def goedel_prefix(n: int, statement: str) -> str:
    return "\n".join(ANSWER_LINES[:n]) + "\n"


def pythagoras_prefix(n: int, statement: str) -> str:
    partial = "\n".join(ANSWER_LINES[:n])
    statement_with_partial = statement.rsplit(":= sorry", 1)[0].rstrip() + " := by\n" + textwrap.indent(partial, "  ")
    formal_statement = PYTHAGORAS_HEADER + "\n" + statement_with_partial + "\n"
    return "### Complete Lean 4 Proof\n\n```lean4\n" + formal_statement


def run_sweep(model: str, prefix_fn, output_file: Path, seeds: list[int], temperature: float | None = None):
    problems = load_data("minif2f-lean4", None, min_tier=1)
    problem = next(p for p in problems if p.id == PROBLEM_ID)
    print(f"[sweep] model={model} seeds={seeds}")

    lean_server = LeanServer()
    print("[sweep] Starting Lean server...")
    lean_server.start()

    local_model_path = resolve_local_snapshot(model)
    print(f"[sweep] Using local model snapshot: {local_model_path}")

    kwargs = dict(model_name=local_model_path, lean_server=lean_server,
                  max_new_tokens=MAX_NEW_TOKENS, k=1)
    if temperature is not None:
        kwargs["temperature"] = temperature
    agent = SimpleAgent(**kwargs)
    agent.load_model()

    if is_pythagoras_model(model):
        actual_temperature, actual_top_p, actual_top_k = PYTHAGORAS_TEMPERATURE, PYTHAGORAS_TOP_P, PYTHAGORAS_TOP_K
    else:
        actual_temperature, actual_top_p, actual_top_k = agent.temperature, None, None

    results = []
    for n in range(1, len(ANSWER_LINES) + 1):
        prefix = prefix_fn(n, problem.statement)
        given_text = "\n".join(ANSWER_LINES[:n])

        for seed in seeds:
            agent.seed = seed
            print(f"\n{'='*60}\n[sweep] n={n}/{len(ANSWER_LINES)} lines given, seed={seed}\n{'='*60}")

            t0 = time.time()
            proof, response = next(agent.generate_proofs(problem, k=1, prefix=prefix))
            gen_time = time.time() - t0

            lean_result = check_proof_with_heartbeats(lean_server, problem.statement, proof, MAX_HEARTBEATS)
            preserved = given_text.strip() in proof

            print(f"[sweep] n={n} seed={seed}: success={lean_result.success} complete={lean_result.complete} "
                  f"preserved_given_lines={preserved} time={gen_time:.1f}s")

            results.append({
                "n_lines_given": n,
                "lines_given": ANSWER_LINES[:n],
                "prefix": prefix,
                "seed": seed,
                "temperature": actual_temperature,
                "top_p": actual_top_p,
                "top_k": actual_top_k,
                "max_new_tokens": MAX_NEW_TOKENS,
                "raw_response": response,
                "extracted_proof": proof,
                "given_lines_preserved": preserved,
                "success": lean_result.success,
                "complete": lean_result.complete,
                "error": lean_result.errors if lean_result.errors else None,
                "generation_time_seconds": gen_time,
            })

            output = {
                "experiment": "aime_1987_p5_prefix_line_sweep",
                "problem_id": problem.id,
                "model": model,
                "answer_proof_full": ANSWER_PROOF,
                "n_total_lines": len(ANSWER_LINES),
                "seeds": seeds,
                "results": results,
            }
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"[sweep] Saved progress ({len(results)}/{len(ANSWER_LINES) * len(seeds)}) to {output_file}")

    lean_server.stop()


def main():
    seeds = [int.from_bytes(os.urandom(8), "big") for _ in range(N_SEEDS)]
    print(f"[sweep] {len(ANSWER_LINES)} lines, seeds={seeds}")

    run_sweep(
        model="Pythagoras-LM/Pythagoras-Prover-4B",
        prefix_fn=pythagoras_prefix,
        output_file=OUTPUT_DIR / "aime_1987_p5_prefix_line_sweep_Pythagoras-Prover-4B_k1_8seeds.json",
        seeds=seeds,
    )
    run_sweep(
        model="Goedel-LM/Goedel-Prover-V2-8B",
        prefix_fn=goedel_prefix,
        output_file=OUTPUT_DIR / "aime_1987_p5_prefix_line_sweep_Goedel-Prover-V2-8B_k1_8seeds.json",
        seeds=seeds,
        temperature=0.1,
    )


if __name__ == "__main__":
    main()
