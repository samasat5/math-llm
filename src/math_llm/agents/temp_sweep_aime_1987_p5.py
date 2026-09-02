"""One-off: temperature sweep for Pythagoras-Prover-4B on aime_1987_p5.

Runs k=1 (batch size 1, sequential - matches SimpleAgent's normal generation
loop) at temperature 0.6 (baseline/default), 0.8, and 0.99, capturing the raw
response, the clean NL reasoning (extract_clean_plan), the extracted Lean
proof (extract_pythagoras_proof), and Lean verification for each.
"""

import json
import time
from pathlib import Path

import math_llm.agents.simple as simple_mod
from math_llm.agents.autoformalizer import extract_clean_plan
from math_llm.data import load_data
from math_llm.lean_server import LeanServer

MODEL = "Pythagoras-LM/Pythagoras-Prover-4B"
MAX_NEW_TOKENS = 20000
PROBLEM_ID = "minif2f-lean4/aime_1987_p5"
TEMPERATURES = [0.6, 0.8, 0.99]

OUTPUT_FILE = Path("outputs/aime_1987_p5_temperature_sweep_Pythagoras-Prover-4B_k1.json")


def resolve_local_snapshot(model_name: str) -> str:
    """Point straight at the already-downloaded HF cache snapshot dir.

    transformers' tokenizer loading (mistral-regex patch) calls the Hub API
    unconditionally UNLESS it detects a local path - so even with
    HF_HUB_OFFLINE=1 (which turns that call into a hard error instead of a
    network hang) a bare repo id still crashes. Passing the local snapshot
    directory short-circuits that check entirely, avoiding both the crash
    and this environment's flaky HF proxy. The dir name keeps "pythagoras"
    in it, so is_pythagoras_model() (a plain substring check) still matches.
    """
    cache_name = "models--" + model_name.replace("/", "--")
    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
    snapshot_dirs = list(snapshots.iterdir())
    if len(snapshot_dirs) != 1:
        raise RuntimeError(f"Expected exactly one snapshot dir in {snapshots}, found {snapshot_dirs}")
    return str(snapshot_dirs[0])


def main():
    problems = load_data("minif2f-lean4", None, min_tier=1)
    problem = next(p for p in problems if p.id == PROBLEM_ID)
    print(f"[sweep] Problem: {problem.id}")

    lean_server = LeanServer()
    print("[sweep] Starting Lean server...")
    lean_server.start()

    local_model_path = resolve_local_snapshot(MODEL)
    print(f"[sweep] Using local model snapshot: {local_model_path}")

    agent = simple_mod.SimpleAgent(
        model_name=local_model_path,
        lean_server=lean_server,
        max_new_tokens=MAX_NEW_TOKENS,
        k=1,
    )
    agent.load_model()

    results = []
    for temp in TEMPERATURES:
        print(f"\n{'='*60}\n[sweep] temperature={temp}\n{'='*60}")
        simple_mod.PYTHAGORAS_TEMPERATURE = temp

        t0 = time.time()
        proof, response = next(agent.generate_proofs(problem, k=1))
        gen_time = time.time() - t0

        clean_plan = extract_clean_plan(response)

        lean_result = lean_server.check_proof(problem.statement, proof)

        results.append({
            "temperature": temp,
            "top_p": simple_mod.PYTHAGORAS_TOP_P,
            "top_k": simple_mod.PYTHAGORAS_TOP_K,
            "raw_response": response,
            "clean_nl_reasoning": clean_plan,
            "extracted_proof": proof,
            "success": lean_result.success,
            "complete": lean_result.complete,
            "error": lean_result.errors if lean_result.errors else None,
            "generation_time_seconds": gen_time,
        })

        # Save incrementally after each temperature in case a later one fails.
        output = {
            "experiment": "pythagoras_temperature_sweep",
            "problem_id": problem.id,
            "model": MODEL,
            "k": 1,
            "batch_size": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperatures_tested": TEMPERATURES,
            "results": results,
        }
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[sweep] Saved progress to {OUTPUT_FILE} ({len(results)}/{len(TEMPERATURES)} done)")

    lean_server.stop()
    print(f"\n[sweep] Done. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
