"""One-off: temperature sweep for Pythagoras-Prover-4B on a minif2f problem.

Usage: python temp_sweep_aime_1987_p5.py [problem_short_name]
(default: aime_1987_p5, e.g. run with amc12a_2017_p7 to test a different one)

Runs k=1 (batch size 1, sequential - matches SimpleAgent's normal generation
loop) for each (temperature, variant) pair in RUN_SPECS, capturing the raw
response, the clean NL reasoning (extract_clean_plan), the extracted Lean
proof (extract_pythagoras_proof, optionally post-processed - see variant
config), and Lean verification for each. Resumable: any (temperature,
variant) pair already present in OUTPUT_FILE (one per problem) is skipped,
so re-running with a longer RUN_SPECS list only generates the new ones and
merges them in - existing entries (including ones saved before the
"variant" field existed, treated as "baseline") are never overwritten.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import torch

import math_llm.agents.simple as simple_mod
from math_llm.agents.autoformalizer import extract_clean_plan
from math_llm.data import load_data
from math_llm.lean_server import LeanServer

MODEL = "Pythagoras-LM/Pythagoras-Prover-4B"
MAX_NEW_TOKENS = 20000
PROBLEM_SHORT_NAME = sys.argv[1] if len(sys.argv) > 1 else "aime_1987_p5"
PROBLEM_ID = f"minif2f-lean4/{PROBLEM_SHORT_NAME}"

# An earlier attempt asked the model in the prompt to avoid
# `norm_num at h ⊢` / interval_cases-on-compound-expressions, but that
# derailed generation into a degenerate loop repeating the `sorry`
# placeholder instead of writing tactics. The `norm_num at h ⊢` half of that
# concern is handled instead as a post-extraction fix (strip_goal_turnstile
# below). This guidance targets a different failure mode seen across the
# sweep - impossible-integer-solution branches getting a bogus asserted
# value (e.g. the h₂₆ : x = 2 ∨ x = -2 type mismatch in the 0.99 run) or a
# `linarith`/`nlinarith` call that can't close because those tactics don't
# reason about integrality - so it's phrased as a positive instruction
# ("do this instead") rather than a prohibition, to avoid the same
# derailment.
BASELINE_PROMPT_TEMPLATE = """Complete the following Lean 4 code:

```lean4
{}```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof."""

GUIDED_PROMPT_TEMPLATE = BASELINE_PROMPT_TEMPLATE + """

When a case admits no integer solution, derive False with omega from the hypothesis directly. Never assert a specific value for a variable in such a branch. Remember linarith/nlinarith do not know that variables are integers; use omega for integrality reasoning."""


def strip_goal_turnstile(proof: str) -> str:
    """Drop a trailing `⊢` from `tac at h₁ h₂ ⊢` clauses.

    Pythagoras often runs a tactic like `norm_num`/`ring_nf`/`simp` "at h ⊢"
    - simplifying the hypothesis AND the goal in one call. When that closes
    the goal outright, every subsequent tactic in the block then fails with
    "No goals to be solved" - the single most common error across every
    sweep attempt so far. Dropping the goal target ("at h" only) keeps the
    goal open for the tactics that follow instead.
    """
    return re.sub(r"[ \t]+⊢(?=\s|$)", "", proof)


VARIANTS = {
    "baseline": {"prompt_template": BASELINE_PROMPT_TEMPLATE, "strip_turnstile": False},
    "strip_turnstile_v1": {"prompt_template": BASELINE_PROMPT_TEMPLATE, "strip_turnstile": True},
    # Same config as strip_turnstile_v1 - just a different key so a fresh
    # sample can be drawn without being skipped as already-done, since the
    # first strip_turnstile_v1 draw degenerated into a #check-spam loop and
    # never actually exercised the turnstile-stripping fix.
    "strip_turnstile_v1_retry": {"prompt_template": BASELINE_PROMPT_TEMPLATE, "strip_turnstile": True},
    # Adds the omega/integrality guidance on top of the turnstile fix.
    "guided_v1": {"prompt_template": GUIDED_PROMPT_TEMPLATE, "strip_turnstile": True},
    # Retry: the saved guided_v1 result is byte-identical to strip_turnstile_v1's
    # (see the explicit-reseed fix above / commit message) - draw a genuinely
    # fresh sample now that generation is explicitly reseeded per call.
    "guided_v1_retry": {"prompt_template": GUIDED_PROMPT_TEMPLATE, "strip_turnstile": True},
    # guided_v1_retry was the first full success (complete=True) - these are
    # repeat draws (same config, fresh reseed each) to check how consistently
    # this variant actually solves the problem vs. one lucky sample.
    "guided_v1_retry2": {"prompt_template": GUIDED_PROMPT_TEMPLATE, "strip_turnstile": True},
    "guided_v1_retry3": {"prompt_template": GUIDED_PROMPT_TEMPLATE, "strip_turnstile": True},
    "guided_v1_retry4": {"prompt_template": GUIDED_PROMPT_TEMPLATE, "strip_turnstile": True},
    "guided_v1_retry5": {"prompt_template": GUIDED_PROMPT_TEMPLATE, "strip_turnstile": True},
}

# Each entry is one (temperature, variant) run. Ones already saved in
# OUTPUT_FILE are skipped automatically - see main(). Keyed by problem short
# name so a new problem doesn't inherit aime_1987_p5's whole exploration
# history - it just runs the winning config (guided_v1 @ 0.6) by default.
RUN_SPECS_BY_PROBLEM = {
    "aime_1987_p5": [
        {"temperature": 0.1, "variant": "baseline"},
        {"temperature": 0.6, "variant": "baseline"},
        {"temperature": 0.8, "variant": "baseline"},
        {"temperature": 0.99, "variant": "baseline"},
        {"temperature": 0.6, "variant": "strip_turnstile_v1"},
        # strip_turnstile_v1_retry was interrupted before it saved a result
        # (killed to prioritize testing the new omega/integrality guidance
        # below) - left out so guided_v1 goes next; re-add it later if a
        # clean (non-degenerate) turnstile-only sample is still wanted.
        {"temperature": 0.6, "variant": "guided_v1"},
        {"temperature": 0.6, "variant": "guided_v1_retry"},
        {"temperature": 0.6, "variant": "guided_v1_retry2"},
        {"temperature": 0.6, "variant": "guided_v1_retry3"},
        {"temperature": 0.6, "variant": "guided_v1_retry4"},
        {"temperature": 0.6, "variant": "guided_v1_retry5"},
        # Same guided_v1 config (guidance prompt + turnstile strip), now at
        # the other two temperatures already covered for baseline, to see
        # whether the ~40% (2/5) success rate at 0.6 holds elsewhere.
        {"temperature": 0.8, "variant": "guided_v1"},
        {"temperature": 0.99, "variant": "guided_v1"},
        {"temperature": 0.1, "variant": "guided_v1"},
    ],
}
DEFAULT_RUN_SPECS = [{"temperature": 0.6, "variant": "guided_v1"}]
RUN_SPECS = RUN_SPECS_BY_PROBLEM.get(PROBLEM_SHORT_NAME, DEFAULT_RUN_SPECS)

OUTPUT_FILE = Path(f"outputs/{PROBLEM_SHORT_NAME}_temperature_sweep_Pythagoras-Prover-4B_k1.json")


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
    results = []
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            results = json.load(f)["results"]
    for r in results:
        r.setdefault("variant", "baseline")

    done_keys = {(r["temperature"], r["variant"]) for r in results}
    pending = [s for s in RUN_SPECS if (s["temperature"], s["variant"]) not in done_keys]
    if not pending:
        print(f"[sweep] All of {RUN_SPECS} already in {OUTPUT_FILE}, nothing to do.")
        return
    print(f"[sweep] Already have: {sorted(done_keys)}. Running: {pending}")

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

    for spec in pending:
        temp, variant = spec["temperature"], spec["variant"]
        cfg = VARIANTS[variant]
        print(f"\n{'='*60}\n[sweep] temperature={temp} variant={variant}\n{'='*60}")
        simple_mod.PYTHAGORAS_TEMPERATURE = temp
        simple_mod.PYTHAGORAS_PROMPT_TEMPLATE = cfg["prompt_template"]

        # Force a fresh, high-entropy seed right before sampling. Two
        # separate processes here (strip_turnstile_v1, guided_v1 - different
        # prompts, verified-different torch/cuda process seeds) produced
        # byte-identical 29k-char generations, which independent per-process
        # entropy alone can't explain - something in the pipeline (possibly
        # a seed reset inside Pythagoras's trust_remote_code=True modeling
        # code, triggered at model load) was overriding it. Reseeding here,
        # immediately before each generate() call rather than relying on
        # process-start entropy, sidesteps that regardless of the exact
        # mechanism.
        seed = int.from_bytes(os.urandom(8), "big")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"[sweep] seed={seed}")

        t0 = time.time()
        proof, response = next(agent.generate_proofs(problem, k=1))
        gen_time = time.time() - t0

        if cfg["strip_turnstile"]:
            proof = strip_goal_turnstile(proof)

        clean_plan = extract_clean_plan(response)

        lean_result = lean_server.check_proof(problem.statement, proof)

        results.append({
            "temperature": temp,
            "variant": variant,
            "seed": seed,
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

        # Save incrementally after each run in case a later one fails.
        output = {
            "experiment": "pythagoras_temperature_sweep",
            "problem_id": problem.id,
            "model": MODEL,
            "k": 1,
            "batch_size": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "variants": {k: {"strip_turnstile": v["strip_turnstile"]} for k, v in VARIANTS.items()},
            "run_specs": RUN_SPECS,
            "results": sorted(results, key=lambda r: (r["temperature"], r["variant"])),
        }
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[sweep] Saved progress to {OUTPUT_FILE} ({len(results)}/{len(RUN_SPECS)} done)")

    lean_server.stop()
    print(f"\n[sweep] Done. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
