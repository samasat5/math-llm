"""Fill in the 10 tier0 problems missing from the interrupted Pythagoras
tier0 run (it stopped at 90/100 - status "in_progress" in the source file).
Runs each with the same config as the rest of that run (k=10,
max_new_tokens=20000), appends the results, and rewrites the file as
"complete" with the full 100.
"""

import json
import time
from pathlib import Path

from math_llm.agents.case_study.temp_sweep_aime_1987_p5 import resolve_local_snapshot
from math_llm.agents.simple import SimpleAgent
from math_llm.data import load_data
from math_llm.lean_server import LeanServer

MODEL = "Pythagoras-LM/Pythagoras-Prover-4B"
MAX_NEW_TOKENS = 20000
K = 10
TARGET_FILE = Path("outputs/minif2f-lean4_simple_Pythagoras-LM-Pythagoras-Prover-4B_k10_tier0_maxtok20000_results.json")

MISSING_PROBLEM_IDS = [
    "minif2f-lean4/induction_pprime_pdvdapowpma",
    "minif2f-lean4/mathd_algebra_107",
    "minif2f-lean4/mathd_algebra_185",
    "minif2f-lean4/mathd_algebra_192",
    "minif2f-lean4/mathd_algebra_215",
    "minif2f-lean4/mathd_algebra_31",
    "minif2f-lean4/mathd_algebra_346",
    "minif2f-lean4/mathd_numbertheory_100",
    "minif2f-lean4/mathd_numbertheory_237",
    "minif2f-lean4/mathd_numbertheory_328",
]


def main():
    with open(TARGET_FILE) as f:
        data = json.load(f)

    have_ids = {r["problem_id"] for r in data["results"]}
    pending = [pid for pid in MISSING_PROBLEM_IDS if pid not in have_ids]
    if not pending:
        print("[complete] Nothing to do - all 10 already present.")
        return
    print(f"[complete] {len(pending)} problems pending: {pending}")

    problems = load_data("minif2f-lean4", None, tier=0)
    by_id = {p.id: p for p in problems}

    lean_server = LeanServer()
    print("[complete] Starting Lean server...")
    lean_server.start()

    local_model_path = resolve_local_snapshot(MODEL)
    print(f"[complete] Using local model snapshot: {local_model_path}")
    agent = SimpleAgent(model_name=local_model_path, lean_server=lean_server, k=K, max_new_tokens=MAX_NEW_TOKENS)
    agent.load_model()

    for i, pid in enumerate(pending):
        problem = by_id[pid]
        print(f"\n{'='*60}\n[complete] [{i+1}/{len(pending)}] {pid}\n{'='*60}")
        t0 = time.time()
        result = agent.solve(problem)
        elapsed = time.time() - t0
        print(f"[complete] {pid}: success={result.success} complete={result.complete} time={elapsed:.1f}s")

        data["results"].append({
            "problem_id": pid,
            "success": result.success,
            "complete": result.complete,
            "proof": result.proof,
            "time": elapsed,
            "error": result.error,
            "attempts": result.attempts,
        })

        n_total = len(data["results"])
        n_success = sum(1 for r in data["results"] if r["success"])
        n_complete = sum(1 for r in data["results"] if r["complete"])
        data["total_problems"] = n_total
        data["successful"] = n_success
        data["complete"] = n_complete
        data["success_rate"] = n_success / n_total
        data["complete_rate"] = n_complete / n_total
        data["pass@10"] = n_complete / n_total
        data["status"] = "complete" if n_total >= data.get("total_problems_planned", n_total) else "in_progress"

        with open(TARGET_FILE, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[complete] Saved progress ({n_total}/{data.get('total_problems_planned')})")

    lean_server.stop()
    print(f"\n[complete] Done. {TARGET_FILE} now has {len(data['results'])} problems.")


if __name__ == "__main__":
    main()
