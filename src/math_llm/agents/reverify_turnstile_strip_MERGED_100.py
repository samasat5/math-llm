"""Re-verify every previously-failed (success=False) problem from the
MERGED_100 benchmark run, applying only the turnstile-strip post-extraction
fix (strip_goal_turnstile - drops a trailing `⊢` from `tac at h ⊢` clauses,
which otherwise closes the goal early and cascades into "No goals to be
solved" on every subsequent tactic).

No regeneration - this reuses the ALREADY-EXTRACTED proof text stored in
each attempt (the source results file only ever stored post-extraction
text, not raw model output - see conversation history), applies the fix
mechanically, and re-checks with the Lean server. Pure verification job, no
model/GPU needed.
"""

import json
from pathlib import Path

from math_llm.agents.temp_sweep_aime_1987_p5 import strip_goal_turnstile
from math_llm.data import load_data
from math_llm.lean_server import LeanServer

SOURCE_FILE = Path("outputs/minif2f-lean4_simple_Pythagoras-LM-Pythagoras-Prover-4B_k10_mintier1_seed0_maxtok20000_MERGED_100_results.json")
OUTPUT_FILE = Path("outputs/turnstile_strip_reverify_MERGED_100_results.json")


def main():
    with open(SOURCE_FILE) as f:
        source = json.load(f)

    failed_problems = [r for r in source["results"] if not r["success"]]
    print(f"[reverify] {len(failed_problems)} previously-failed (success=False) problems, "
          f"{sum(len(r.get('attempts') or []) for r in failed_problems)} attempts total")

    problems = load_data("minif2f-lean4", None, min_tier=1)
    by_id = {p.id: p for p in problems}

    lean_server = LeanServer()
    print("[reverify] Starting Lean server...")
    lean_server.start()

    recovered = []
    still_failed = []
    all_results = []

    for i, r in enumerate(failed_problems):
        pid = r["problem_id"]
        problem = by_id.get(pid)
        if problem is None:
            print(f"[reverify] [{i+1}/{len(failed_problems)}] {pid}: SKIPPED (not found in loaded dataset)")
            continue

        attempts = r.get("attempts") or []
        per_attempt = []
        best_complete = False
        best_success = False
        for a in attempts:
            original_proof = a.get("proof", "")
            stripped_proof = strip_goal_turnstile(original_proof)
            changed = stripped_proof != original_proof

            lean_result = lean_server.check_proof(problem.statement, stripped_proof)
            per_attempt.append({
                "source": a.get("source"),
                "turnstile_stripped": changed,
                "original_success": a.get("success"),
                "original_complete": a.get("complete"),
                "new_success": lean_result.success,
                "new_complete": lean_result.complete,
                "new_error": lean_result.errors if lean_result.errors else None,
                "proof": stripped_proof,
            })
            best_complete = best_complete or lean_result.complete
            best_success = best_success or lean_result.success

        status = "RECOVERED" if best_complete else "still fail"
        print(f"[reverify] [{i+1}/{len(failed_problems)}] {pid}: {status} "
              f"({sum(1 for a in per_attempt if a['turnstile_stripped'])}/{len(per_attempt)} attempts had a ⊢ to strip)")

        entry = {
            "problem_id": pid,
            "originally_success": r["success"],
            "originally_complete": r["complete"],
            "new_success": best_success,
            "new_complete": best_complete,
            "recovered": best_complete and not r["complete"],
            "attempts": per_attempt,
        }
        all_results.append(entry)
        (recovered if entry["recovered"] else still_failed).append(pid)

        # Save incrementally.
        with open(OUTPUT_FILE, "w") as f:
            json.dump({
                "source_file": str(SOURCE_FILE),
                "fix": "strip_goal_turnstile only, no regeneration",
                "total_reprocessed": len(all_results),
                "total_target": len(failed_problems),
                "recovered_count": len(recovered),
                "recovered_problem_ids": recovered,
                "still_failed_problem_ids": still_failed,
                "results": all_results,
            }, f, indent=2, ensure_ascii=False)

    lean_server.stop()
    print(f"\n[reverify] Done. {len(recovered)}/{len(failed_problems)} problems recovered by the turnstile-strip fix alone.")
    print(f"[reverify] Recovered: {recovered}")
    print(f"[reverify] Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
