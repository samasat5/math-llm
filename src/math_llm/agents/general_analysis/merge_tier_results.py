"""Merge a tier0 and a tier1 (mintier1/MERGED_100) results file for one
model into a single combined results file, recomputing the aggregate stats
over the combined problem set.

Usage: python merge_tier_results.py <tier1_file.json> <tier0_file.json> <output.json>
"""

import json
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        print("Usage: python merge_tier_results.py <tier1_file.json> <tier0_file.json> <output.json>")
        sys.exit(1)
    tier1_file, tier0_file, output_file = (Path(p) for p in sys.argv[1:4])

    with open(tier1_file) as f:
        tier1 = json.load(f)
    with open(tier0_file) as f:
        tier0 = json.load(f)

    ids1 = {r["problem_id"] for r in tier1["results"]}
    ids0 = {r["problem_id"] for r in tier0["results"]}
    overlap = ids1 & ids0
    if overlap:
        print(f"[merge] WARNING: {len(overlap)} overlapping problem_ids, tier1 copy kept: {sorted(overlap)}")

    results = tier1["results"] + [r for r in tier0["results"] if r["problem_id"] not in ids1]
    n_total = len(results)
    n_success = sum(1 for r in results if r["success"])
    n_complete = sum(1 for r in results if r["complete"])

    merged = {
        "dataset": tier1.get("dataset"),
        "agent": tier1.get("agent"),
        "model": tier1.get("model"),
        "k": tier1.get("k"),
        "status": "merged_tier0_tier1",
        "note": f"Merged tier0 ({tier0_file.name}) + tier1/mintier1 ({tier1_file.name})",
        "source_files": [str(tier1_file), str(tier0_file)],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_problems_planned": n_total,
        "total_problems": n_total,
        "successful": n_success,
        "complete": n_complete,
        "success_rate": n_success / n_total if n_total else 0,
        "complete_rate": n_complete / n_total if n_total else 0,
        f"pass@{tier1.get('k', 10)}": n_complete / n_total if n_total else 0,
        "total_time": tier1.get("total_time", 0) + tier0.get("total_time", 0),
        "avg_time_per_problem": (tier1.get("total_time", 0) + tier0.get("total_time", 0)) / n_total if n_total else 0,
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"[merge] tier1: {len(tier1['results'])} problems, tier0: {len(tier0['results'])} problems")
    print(f"[merge] combined: {n_total} problems, {n_success} successful ({merged['success_rate']*100:.1f}%), "
          f"{n_complete} complete ({merged['complete_rate']*100:.1f}%)")
    print(f"[merge] Saved to {output_file}")


if __name__ == "__main__":
    main()
