"""Resolve stale Mathlib identifiers (from mathlib_git_lookup.py's
"existed once" bucket) to their current replacement name, using the
`alias old := new` declarations Mathlib renames go through.

Unlike ever_existed_in_history (which just finds *a* commit touching the
string), this finds the specific alias-definition line -- in current HEAD
if the alias is still there but deprecated, or anywhere in history if the
alias itself has since been deleted -- and extracts its target. The target
is then re-verified against the live Lean/Mathlib environment (not just
grepped), since an alias can itself be deprecated in favor of a third name
(A -> B -> C), and only C may actually resolve today.

Input: the *_git_lookup.json produced by mathlib_git_lookup.py.
Only entries with resolves_now == false and ever_existed_in_history are
looked up here; "never existed" entries are true fabrications and have no
alias to chase.

Usage: python build_alias_resolution_map.py [<git_lookup.json>] [<output.json>]
Defaults: outputs/hallucinated_identifier_failures_MERGED_100_git_lookup.json ->
          outputs/alias_resolution_map.json
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

from math_llm.lean_server import LeanServer

DEFAULT_INPUT = Path("outputs/hallucinated_identifier_failures_MERGED_100_git_lookup.json")
DEFAULT_OUTPUT = Path("outputs/alias_resolution_map.json")

MATHLIB_PATH = (
    Path(os.environ.get("MATHLIB_PROJECT_PATH", str(Path.home() / ".lean-bench")))
    / ".lake" / "packages" / "mathlib"
)

# Matches a +/- diff line (or a plain current-HEAD line) declaring
# `alias <name> := <target>`, optionally preceded on the same physical
# line by nothing else (the @[deprecated ...] attribute is its own line
# and is not required to match -- we only need the alias line itself).
ALIAS_LINE_RE = re.compile(
    r"^[+\- ]?\s*alias\s+([A-Za-z_][A-Za-z0-9_.']*)\s*:=\s*([A-Za-z_][A-Za-z0-9_.']*)",
    re.MULTILINE,
)


def find_alias_hops_in_current_head(name: str) -> str | None:
    """Cheapest case: the alias still exists in the current checkout
    (deprecated but not yet deleted). One git grep, no history walk."""
    result = subprocess.run(
        ["git", "grep", "-n", "-E", rf"alias\s+{re.escape(name)}\s*:="],
        cwd=MATHLIB_PATH, capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    for line in result.stdout.splitlines():
        m = ALIAS_LINE_RE.search(line.split(":", 2)[-1])
        if m and m.group(1) == name:
            return m.group(2)
    return None


def find_alias_hops_in_history(name: str) -> str | None:
    """Slower fallback: the alias itself has been deleted from HEAD.
    Walk history for any diff hunk that ever added or removed a line
    declaring `alias <name> := <target>`, and take the most recent
    such target (last deprecation wins if it was renamed more than once
    before final removal)."""
    result = subprocess.run(
        ["git", "log", "-p", "--format=COMMIT %h %as", "-G",
         rf"alias {re.escape(name)} :=", "--", "*.lean"],
        cwd=MATHLIB_PATH, capture_output=True, text=True,
    )
    if not result.stdout.strip():
        return None
    matches = [
        (n, tgt) for n, tgt in ALIAS_LINE_RE.findall(result.stdout) if n == name
    ]
    if not matches:
        return None
    return matches[-1][1]  # last in log --reverse-free (newest-first) order


def resolve_chain(name: str, lean_server: LeanServer, max_hops: int = 5) -> dict:
    """Follow alias -> alias -> ... until a name resolves live in Lean,
    a cycle/dead end is hit, or max_hops is exceeded."""
    chain = [name]
    seen = {name}
    current = name
    for _ in range(max_hops):
        if lean_server.identifier_exists(current):
            return {"final_target": current, "chain": chain, "resolves_now": True}
        nxt = find_alias_hops_in_current_head(current) or find_alias_hops_in_history(current)
        if nxt is None or nxt in seen:
            break
        chain.append(nxt)
        seen.add(nxt)
        current = nxt
    return {"final_target": None, "chain": chain, "resolves_now": False}


def main():
    args = sys.argv[1:]
    input_file = Path(args[0]) if len(args) >= 1 else DEFAULT_INPUT
    output_file = Path(args[1]) if len(args) >= 2 else DEFAULT_OUTPUT

    if not MATHLIB_PATH.is_dir():
        print(f"[alias-map] Mathlib checkout not found at {MATHLIB_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(input_file) as f:
        entries = json.load(f)

    targets = [
        e for e in entries
        if not e["resolves_now"] and e.get("ever_existed_in_history")
    ]
    print(f"[alias-map] {len(targets)} stale ('existed once') name(s) to chase from {input_file}")

    lean_server = LeanServer()
    print("[alias-map] Starting Lean server...")
    lean_server.start()

    results = []
    try:
        for i, e in enumerate(targets):
            name = e["bad_name"]
            res = resolve_chain(name, lean_server)
            status = f"-> {res['final_target']}" if res["resolves_now"] else "NO ALIAS FOUND"
            print(f"[{i+1}/{len(targets)}] {name} {status}  (chain: {' -> '.join(res['chain'])})")
            results.append({"bad_name": name, "count": e["count"], **res})
    finally:
        lean_server.stop()

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n_found = sum(1 for r in results if r["resolves_now"])
    print(f"\n[alias-map] resolved via alias chain: {n_found}/{len(results)}")
    print(f"[alias-map] Saved to {output_file}")


if __name__ == "__main__":
    main()