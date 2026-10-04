"""One-off: prefix INDENTATION sweep for aime_1987_p5, one run per
(model, n, seed) where n = number of leading TOP-LEVEL (indentation depth 0)
statement blocks of the verified correct proof
(prefix_conditioned_aime_1987_p5.ANSWER_PROOF) handed to the model as a
prefix - each block is a depth-0 line plus every subsequent line indented
under it, so a cut point never lands inside a nested `by`/`{...}` body (see
the sibling prefix_line_sweep_aime_1987_p5.py, which cuts by raw line count
instead and can therefore hand over a syntactically-dangling partial nested
block).

For this proof (see ANSWER_PROOF) that's 5 blocks, not 15 lines:
  n=1: have h1 ... := by linarith
  n=2: + have h2 : y^2 <= 517 := by / nlinarith [...]
  n=3: + have h3 : y <= 22 := by nlinarith [h2]
  n=4: + have h4 : y >= -22 := by nlinarith [...]
  n=5: + interval_cases y <;> try { ... }              (= full proof)

For each n = 1..5, the model completes the remaining tail itself; the FULL
proof result is re-verified with Lean every time (not just the model's
tail) - same as the line sweep.

Seeds: 8 FIXED seeds (drawn once, reused via SimpleAgent(seed=...) - see
simple.py, seeds per sample index i so k=1 always draws with exactly this
value), same 8 seeds reused across every (model, n) pair so seed is an
independent, comparable axis - varying only n or seed lets you attribute any
change in outcome to that one axis, not resampling luck.

max_new_tokens is capped at 6000 (matches prefix_line_sweep_aime_1987_p5.py)
- generous for what should mostly be short completions, and any run that
still needs more than that to close out the proof is itself a meaningful
data point (Goedel's `<;> omega` repetition loop reliably fills this exact
budget every time it triggers, independent of n or seed - see that sweep's
results).

Usage: python prefix_indent_sweep_aime_1987_p5.py
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
N_SEEDS = 8

OUTPUT_DIR = Path("outputs/experiment_aime_1987_p5")


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def compute_indent_groups(lines: list[str]) -> list[list[str]]:
    """Split proof lines into blocks at every depth-0 line - each block is
    one top-level statement plus everything indented under it."""
    groups: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if _indent_of(line) == 0 and current:
            groups.append(current)
            current = []
        current.append(line)
    if current:
        groups.append(current)
    return groups


ANSWER_LINES = ANSWER_PROOF.splitlines()
INDENT_GROUPS = compute_indent_groups(ANSWER_LINES)
N_GROUPS = len(INDENT_GROUPS)


def lines_through_group(n: int) -> list[str]:
    """Flatten INDENT_GROUPS[:n] back into a flat list of lines."""
    lines: list[str] = []
    for group in INDENT_GROUPS[:n]:
        lines.extend(group)
    return lines


def goedel_prefix(n: int, statement: str) -> str:
    return "\n".join(lines_through_group(n)) + "\n"


def pythagoras_prefix(n: int, statement: str) -> str:
    partial = "\n".join(lines_through_group(n))
    statement_with_partial = statement.rsplit(":= sorry", 1)[0].rstrip() + " := by\n" + textwrap.indent(partial, "  ")
    formal_statement = PYTHAGORAS_HEADER + "\n" + statement_with_partial + "\n"
    return "### Complete Lean 4 Proof\n\n```lean4\n" + formal_statement


def run_sweep(model: str, prefix_fn, output_file: Path, seeds: list[int], temperature: float | None = None):
    problems = load_data("minif2f-lean4", None, min_tier=1)
    problem = next(p for p in problems if p.id == PROBLEM_ID)
    print(f"[indent-sweep] model={model} seeds={seeds}")

    lean_server = LeanServer()
    print("[indent-sweep] Starting Lean server...")
    lean_server.start()

    local_model_path = resolve_local_snapshot(model)
    print(f"[indent-sweep] Using local model snapshot: {local_model_path}")

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
    for n in range(1, N_GROUPS + 1):
        prefix = prefix_fn(n, problem.statement)
        given_text = "\n".join(lines_through_group(n))

        for seed in seeds:
            agent.seed = seed
            print(f"\n{'='*60}\n[indent-sweep] n={n}/{N_GROUPS} blocks given, seed={seed}\n{'='*60}")

            t0 = time.time()
            proof, response = next(agent.generate_proofs(problem, k=1, prefix=prefix))
            gen_time = time.time() - t0

            lean_result = check_proof_with_heartbeats(lean_server, problem.statement, proof, MAX_HEARTBEATS)
            preserved = given_text.strip() in proof

            print(f"[indent-sweep] n={n} seed={seed}: success={lean_result.success} "
                  f"complete={lean_result.complete} preserved_given_lines={preserved} time={gen_time:.1f}s")

            results.append({
                "n_blocks_given": n,
                "n_lines_given": sum(len(g) for g in INDENT_GROUPS[:n]),
                "lines_given": lines_through_group(n),
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
                "experiment": "aime_1987_p5_prefix_indent_sweep",
                "problem_id": problem.id,
                "model": model,
                "answer_proof_full": ANSWER_PROOF,
                "n_total_lines": len(ANSWER_LINES),
                "n_total_blocks": N_GROUPS,
                "seeds": seeds,
                "results": results,
            }
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            print(f"[indent-sweep] Saved progress ({len(results)}/{N_GROUPS * len(seeds)}) to {output_file}")

    lean_server.stop()


def main():
    seeds = [int.from_bytes(os.urandom(8), "big") for _ in range(N_SEEDS)]
    print(f"[indent-sweep] {N_GROUPS} indent blocks, seeds={seeds}")

    run_sweep(
        model="Pythagoras-LM/Pythagoras-Prover-4B",
        prefix_fn=pythagoras_prefix,
        output_file=OUTPUT_DIR / "aime_1987_p5_prefix_indent_sweep_Pythagoras-Prover-4B_k1.json",
        seeds=seeds,
    )
    run_sweep(
        model="Goedel-LM/Goedel-Prover-V2-8B",
        prefix_fn=goedel_prefix,
        output_file=OUTPUT_DIR / "aime_1987_p5_prefix_indent_sweep_Goedel-Prover-V2-8B_k1.json",
        seeds=seeds,
        temperature=0.1,
    )


if __name__ == "__main__":
    main()
