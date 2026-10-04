"""General hallucination scanner: given a benchmark results file (any model,
matching the standard {results: [{problem_id, attempts: [{proof, error,
source}, ...]}, ...]} schema), scans every attempt of every problem for
"Unknown constant `X`" / "Unknown identifier `X`" Lean errors and writes
them out in the same format as the original
minif2f-lean4_..._MERGED_100_hallucinations.json file:
one entry per attempt that has at least one hallucination, each with
{name, kind, context} - kind is "invented_lemma_or_tactic" (a Mathlib-style
name that doesn't resolve) or "dangling_local_reference" (a local hypothesis
name referenced out of scope / never bound - not a Mathlib API hallucination
at all, just a scoping bug).

Usage: python hallucination_scan.py <source_results.json> <output.json>
"""

import json
import re
import sys
from pathlib import Path

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


def scan(source_file: Path) -> list[dict]:
    with open(source_file) as f:
        data = json.load(f)

    entries = []
    for r in data["results"]:
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
                    "source": a.get("source", "prover"),
                    "status": "FAIL",
                    "hallucinations": hallucinations,
                })
    return entries


def main():
    if len(sys.argv) != 3:
        print("Usage: python hallucination_scan.py <source_results.json> <output.json>")
        sys.exit(1)
    source_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    entries = scan(source_file)

    with open(output_file, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

    invented = sum(1 for e in entries for h in e["hallucinations"] if h["kind"] == "invented_lemma_or_tactic")
    dangling = sum(1 for e in entries for h in e["hallucinations"] if h["kind"] == "dangling_local_reference")
    distinct_problems = len(set(e["problem_id"] for e in entries))

    print(f"[scan] {len(entries)} attempt(s) with a hallucination, across {distinct_problems} distinct problems")
    print(f"[scan] invented_lemma_or_tactic: {invented}  dangling_local_reference: {dangling}")
    print(f"[scan] Saved to {output_file}")


if __name__ == "__main__":
    main()
