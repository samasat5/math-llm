"""Scan every failed (success=False) problem in the tier0 benchmark run for
hallucinations, matching the format/methodology of the earlier
minif2f-lean4_..._mintier1_seed0_..._MERGED_100_hallucinations.json file:
one entry per attempt that has at least one hallucination, each with
{name, kind, context} - kind is "invented_lemma_or_tactic" (a Mathlib-style
name that doesn't resolve) or "dangling_local_reference" (a local hypothesis
name referenced out of scope / never bound - not a Mathlib API hallucination
at all, just a scoping bug).
"""

import json
import re
from pathlib import Path

SOURCE_FILE = Path("outputs/minif2f-lean4_simple_Pythagoras-LM-Pythagoras-Prover-4B_k10_tier0_maxtok20000_results.json")
OUTPUT_FILE = Path("outputs/minif2f-lean4_simple_Pythagoras-LM-Pythagoras-Prover-4B_k10_tier0_seed0_maxtok20000_MERGED_100_hallucinations.json")

NAME_RE = re.compile(r"Unknown (?:constant|identifier) `([^`]+)`")
LEAN_KEYWORDS = {"begin", "end", "assume", "cases", "sorry", "from", "obtain", "show", "this"}


def looks_like_local_var(name: str) -> bool:
    if re.fullmatch(r"[a-zA-Z]", name):
        return True
    if re.fullmatch(r"h[₀-₉_a-zA-Z0-9ₖₙₘ]*", name):
        return True
    return False


def classify(name: str) -> str:
    if name in LEAN_KEYWORDS or looks_like_local_var(name):
        return "dangling_local_reference"
    return "invented_lemma_or_tactic"


def context_snippet(proof: str, name: str, width: int = 150) -> str:
    idx = proof.find(name)
    if idx == -1:
        return ""
    start = max(0, idx - width // 2)
    end = min(len(proof), idx + len(name) + width // 2)
    return " ".join(proof[start:end].split())


def main():
    with open(SOURCE_FILE) as f:
        data = json.load(f)

    failed = [r for r in data["results"] if not r["success"]]
    # Also scan the "looks done but isn't" set: success=True (compiles, no
    # Lean errors on the WINNING attempt) but complete=False and no literal
    # "sorry" in that winning proof - the failing OTHER attempts among their
    # k=10 samples can still carry real hallucinations worth surfacing.
    incomplete_non_sorry = [
        r for r in data["results"]
        if r["success"] and not r["complete"] and "sorry" not in r["proof"]
    ]
    targets = failed + incomplete_non_sorry
    print(f"[scan] {len(failed)} failed + {len(incomplete_non_sorry)} incomplete-non-sorry problems, "
          f"{sum(len(r.get('attempts') or []) for r in targets)} attempts total")

    entries = []
    for r in targets:
        for i, a in enumerate(r.get("attempts") or []):
            errs = a.get("error") or []
            proof = a.get("proof", "")
            hallucinations = []
            seen = set()
            for e in errs:
                for m in NAME_RE.finditer(e):
                    name = m.group(1)
                    if name in seen:
                        continue
                    seen.add(name)
                    hallucinations.append({
                        "name": name,
                        "kind": classify(name),
                        "context": context_snippet(proof, name),
                    })
            if hallucinations:
                entries.append({
                    "problem_id": r["problem_id"],
                    "attempt_index": i + 1,
                    "source": "prover",
                    "status": "FAIL",
                    "hallucinations": hallucinations,
                })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    print(f"[scan] {len(entries)} attempt(s) with a hallucination")
    for e in entries:
        for h in e["hallucinations"]:
            print(f"  {e['problem_id']} [attempt {e['attempt_index']}] {h['kind']}: {h['name']}")
    print(f"[scan] Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
