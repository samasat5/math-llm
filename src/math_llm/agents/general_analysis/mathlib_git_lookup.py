"""For every hallucinated Mathlib-style name flagged in a hallucination-scan
output (or a raw failures file with an {"errors": [...]} list per sample,
e.g. outputs/hallucinated_identifier_failures_MERGED_100.json), look it up
against the real Mathlib git checkout at $MATHLIB_PROJECT_PATH/.lake/packages/mathlib
(default ~/.lean-bench/.lake/packages/mathlib):

  resolves_now            - ground truth: does `#check <name>` elaborate without
                            error in a live Lean/Mathlib REPL, in the exact same
                            environment ("import Mathlib\nimport Aesop\nopen
                            BigOperators Real Nat Topology", see lean_server.py's
                            DEFAULT_IMPORTS) the benchmark itself runs proofs in.
                            This is authoritative - a plain grep is NOT: an
                            earlier pass here false-positived "ln" as existing
                            because it appears as a *local* pattern-bound variable
                            name inside one unrelated tactic's implementation
                            (Mathlib/Tactic/CancelDenoms/Core.lean), not as a real
                            global identifier - grep can't tell the difference,
                            Lean's own elaborator can.
  grep_hint               - "file:line" of a whole-word match in the current
                            checkout (git grep -w HEAD), purely informational -
                            do not treat this as proof of existence, see above.
  ever_existed_in_history - if not resolves_now, "<short-hash> <date> (earliest
                            match)" for the first commit (oldest-first) whose
                            diff touched that string, or null if it never
                            appears anywhere in history.

Dangling local references (bound hypothesis names like `h1`, out of scope -
not real Mathlib API hallucinations) are skipped; only names classified as
invented_lemma_or_tactic are looked up.

Usage: python mathlib_git_lookup.py [<input.json>] [<output.json>]
Defaults: outputs/hallucinated_identifier_failures_MERGED_100.json ->
          outputs/hallucinated_identifier_failures_MERGED_100_git_lookup.json
"""

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from math_llm.agents.general_analysis.hallucination_scan import NAME_RE, classify
from math_llm.lean_server import LeanServer

DEFAULT_INPUT = Path("outputs/hallucinated_identifier_failures_MERGED_100.json")
DEFAULT_OUTPUT = Path("outputs/hallucinated_identifier_failures_MERGED_100_git_lookup.json")

MATHLIB_PATH = (
    Path(os.environ.get("MATHLIB_PROJECT_PATH", str(Path.home() / ".lean-bench")))
    / ".lake" / "packages" / "mathlib"
)


def extract_name_counts(data) -> Counter:
    """Auto-detect schema and return a Counter of invented_lemma_or_tactic names."""
    counts: Counter = Counter()

    if isinstance(data, dict) and "samples" in data:
        # Raw failures file: {"samples": [{"errors": [...]}, ...]}
        for s in data["samples"]:
            for e in s.get("errors") or []:
                for m in NAME_RE.finditer(e):
                    name = m.group(1)
                    if classify(name) == "invented_lemma_or_tactic":
                        counts[name] += 1

    elif isinstance(data, list) and data and "hallucinations" in data[0]:
        # Already-scanned hallucination_scan.py output.
        for entry in data:
            for h in entry.get("hallucinations", []):
                if h.get("kind") == "invented_lemma_or_tactic":
                    counts[h["name"]] += 1

    elif isinstance(data, dict) and "results" in data:
        # Raw benchmark results file: {"results": [{"attempts": [{"error": [...]}]}]}
        for r in data["results"]:
            for a in r.get("attempts") or []:
                for e in a.get("error") or []:
                    for m in NAME_RE.finditer(e):
                        name = m.group(1)
                        if classify(name) == "invented_lemma_or_tactic":
                            counts[name] += 1

    else:
        raise ValueError("Unrecognized input schema (expected 'samples', 'results', or a hallucination_scan.py list)")

    return counts


def grep_hint(name: str) -> str | None:
    """First 'file:line' whole-word match for `name` in the current Mathlib
    checkout, or None. Informational only - see module docstring for why this
    is not treated as authoritative (a local variable can shadow a real name)."""
    result = subprocess.run(
        ["git", "grep", "-n", "-w", "-F", "-e", name, "HEAD", "--", "*.lean"],
        cwd=MATHLIB_PATH, capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    first_line = result.stdout.splitlines()[0]
    # "HEAD:path/to/File.lean:123:content..." -> "path/to/File.lean:123"
    parts = first_line.split(":", 3)
    return f"{parts[1]}:{parts[2]}"


def ever_existed_in_history(name: str) -> str | None:
    """Earliest (oldest-first) commit whose diff touched `name`, or None if never.

    NB: `-1` cannot be combined with `--reverse` here - git applies the count
    limit during its (newest-first) walk *before* reversing the already-cut
    result, so `--reverse ... -1` silently returns nothing instead of the
    oldest match. List all matches oldest-first and take the first line.
    """
    result = subprocess.run(
        ["git", "log", "--reverse", "--format=%h %as", "-S", name, "--", "*.lean"],
        cwd=MATHLIB_PATH, capture_output=True, text=True,
    )
    lines = result.stdout.splitlines()
    if not lines:
        return None
    return f"{lines[0]} (earliest of {len(lines)} matching commit(s))"


def main():
    args = sys.argv[1:]
    input_file = Path(args[0]) if len(args) >= 1 else DEFAULT_INPUT
    output_file = Path(args[1]) if len(args) >= 2 else DEFAULT_OUTPUT

    if not MATHLIB_PATH.is_dir():
        print(f"[lookup] Mathlib checkout not found at {MATHLIB_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(input_file) as f:
        data = json.load(f)

    counts = extract_name_counts(data)
    print(f"[lookup] {len(counts)} distinct invented_lemma_or_tactic name(s) from {input_file}")

    lean_server = LeanServer()
    print("[lookup] Starting Lean server (loads Mathlib imports, ~60s-few min first time)...")
    lean_server.start()

    results = []
    try:
        for i, (name, count) in enumerate(counts.most_common()):
            resolves = lean_server.identifier_exists(name)
            hint = grep_hint(name)
            history = None if resolves else ever_existed_in_history(name)
            status = "resolves now" if resolves else ("existed once" if history else "never existed")
            flag = "" if resolves == bool(hint) else "  [grep disagreed - see grep_hint caveat]"
            print(f"[{i+1}/{len(counts)}] {name} (x{count}): {status}{flag}")
            results.append({
                "bad_name": name,
                "count": count,
                "resolves_now": resolves,
                "grep_hint": hint,
                "ever_existed_in_history": history,
            })
    finally:
        lean_server.stop()

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_exists = sum(1 for r in results if r["resolves_now"])
    n_once = sum(1 for r in results if not r["resolves_now"] and r["ever_existed_in_history"])
    n_never = sum(1 for r in results if not r["resolves_now"] and not r["ever_existed_in_history"])
    print(f"\n[lookup] resolves now: {n_exists}  existed once (renamed/removed): {n_once}  never existed: {n_never}")
    print(f"[lookup] Saved to {output_file}")


if __name__ == "__main__":
    main()
