#!/usr/bin/env python3
"""
Categorize failure modes in a benchmark results file and plot a histogram.

Usage:
    poetry run python scripts/categorize_errors.py <results.json> [--output-dir DIR]

Looks at every attempt across every problem in the results file, buckets
each non-complete attempt into a failure category (Lean error type, or the
"wrote sorry instead of finishing" case), and writes:
    - <output-dir>/<results-stem>_error_histogram.png
    - <output-dir>/<results-stem>_error_categories.json (counts + examples)
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Ordered (category, matcher) rules - first match wins. Matching is done
# against the attempt's error messages joined together (case-insensitive),
# except the "sorry" rule which looks at the proof text of attempts that
# compiled cleanly but never actually finished.
CATEGORIES = [
    ("sorry_placeholder", None),  # handled specially, see classify()
    ("no_goals_to_solve", re.compile(r"no goals to be solved", re.I)),
    ("timeout", re.compile(r"timeout|maximum number of heartbeats", re.I)),
    ("unknown_premise", None),          # handled specially, see classify()
    ("unknown_local_hypothesis", None), # handled specially, see classify()
    ("unknown_tactic", re.compile(r"unknown tactic", re.I)),
    ("type_mismatch", re.compile(r"type mismatch", re.I)),
    ("function_expected", re.compile(r"function expected", re.I)),
    ("failed_to_synthesize", re.compile(r"failed to synthesize", re.I)),
    ("recursion_depth", re.compile(r"maximum recursion depth", re.I)),
    ("invalid_scope_or_redecl", re.compile(
        r"invalid `end`|no current scope|invalid 'include'|has already been declared", re.I)),
    ("unsolved_goals", re.compile(r"unsolved goals", re.I)),
    ("tactic_failed", re.compile(
        r"\bfailed\b|could not prove|made no progress|failed to prove", re.I)),
    ("syntax_error", re.compile(
        r"unexpected token|unexpected identifier|unexpected end of input|expected ", re.I)),
]


# "Unknown identifier `X`" / "Unknown constant `X`" errors are a mix of two
# unrelated bugs: X can be a genuine (hallucinated or outdated) Mathlib
# lemma/def name like `add_left_neg` or `Nat.odd_iff_not_even`, or it can be
# one of Lean's own auto-generated hypothesis names (h, h1, h₁₅, ...) that
# the model referenced out of the scope it was introduced in - a proof
# structure/scoping bug, not a knowledge hallucination. Distinguish by
# shape: hypothesis names are always "h" plus digits/unicode subscript
# digits, lemma names always contain a "_" or a namespace "."
_UNKNOWN_NAME_RE = re.compile(r"unknown (?:identifier|constant) `([^`]+)`", re.I)
_LOCAL_HYPOTHESIS_RE = re.compile(r"^h[\d₀-₉]*$")


def classify_unknown_name(name: str) -> str:
    if _LOCAL_HYPOTHESIS_RE.match(name):
        return "unknown_local_hypothesis"
    return "unknown_premise"


def classify(attempt: dict) -> str:
    """Return a category name for one attempt (already known non-complete)."""
    if attempt.get("success") and not attempt.get("error"):
        # Compiled without any Lean error but still marked incomplete. Two
        # distinct cases share this shape, so don't collapse them into one
        # bucket (or worse, an unlabeled "other"):
        #   - the proof still contains a literal `sorry` placeholder - the
        #     model gave up mid-proof and asked Lean to skip a subgoal.
        #   - the proof has no `sorry` at all, yet Lean still reports the
        #     theorem as not fully closed (e.g. a leftover goal Lean's
        #     `errors` field didn't surface as a hard error). Seen in prior
        #     Pythagoras runs (the "incomplete_non_sorry" case) - worth
        #     surfacing on its own since it points at a genuinely different
        #     problem than "the model punted with sorry".
        if "sorry" in (attempt.get("proof") or ""):
            return "sorry_placeholder"
        return "compiled_but_incomplete_no_sorry"

    errors = attempt.get("error")
    if not errors:
        return "other"
    text = " | ".join(errors) if isinstance(errors, list) else str(errors)

    m = _UNKNOWN_NAME_RE.search(text)
    if m:
        return classify_unknown_name(m.group(1))

    for name, pattern in CATEGORIES:
        if pattern is None:
            continue
        if pattern.search(text):
            return name
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_file", type=Path, help="Path to a *_results.json benchmark file")
    ap.add_argument("--output-dir", type=Path, default=None,
                     help="Where to write the histogram/json (default: same dir as results_file)")
    ap.add_argument("--top-n", type=int, default=None,
                     help="Only plot the top N categories (default: all)")
    args = ap.parse_args()

    data = json.loads(args.results_file.read_text())
    out_dir = args.output_dir or args.results_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.results_file.stem

    counts = Counter()
    examples = defaultdict(list)
    n_complete_attempts = 0
    n_total_attempts = 0

    for problem in data.get("results", []):
        for attempt in (problem.get("attempts") or []):
            n_total_attempts += 1
            if attempt.get("complete"):
                n_complete_attempts += 1
                continue
            category = classify(attempt)
            counts[category] += 1
            if len(examples[category]) < 3:
                errors = attempt.get("error")
                snippet = (" | ".join(errors) if isinstance(errors, list) else str(errors)) if errors else ""
                examples[category].append({
                    "problem_id": problem.get("problem_id"),
                    "error": snippet[:300],
                })

    n_failed_attempts = n_total_attempts - n_complete_attempts
    k = data.get("k")
    pass_at_k = data.get(f"pass@{k}")
    print(f"Results file: {args.results_file}")
    print(f"Model: {data.get('model')}")
    print(f"Problems: {data.get('total_problems')}  Complete: {data.get('complete')}  "
          f"pass@{k}: {pass_at_k}")
    print(f"Total attempts: {n_total_attempts}  Complete attempts: {n_complete_attempts}  "
          f"Non-complete attempts: {n_failed_attempts}")
    print()
    print(f"{'category':<28} {'count':>8} {'% of non-complete':>18}")
    for name, cnt in counts.most_common():
        pct = 100 * cnt / n_failed_attempts if n_failed_attempts else 0
        print(f"{name:<28} {cnt:>8} {pct:>17.1f}%")

    # --- write categorized json (counts + example errors per category) ---
    categories_path = out_dir / f"{stem}_error_categories.json"
    categories_path.write_text(json.dumps({
        "results_file": str(args.results_file),
        "n_total_attempts": n_total_attempts,
        "n_complete_attempts": n_complete_attempts,
        "n_noncomplete_attempts": n_failed_attempts,
        "counts": dict(counts),
        "examples": examples,
    }, indent=2))
    print(f"\nWrote category breakdown -> {categories_path}")

    # --- histogram ---
    items = counts.most_common(args.top_n)
    labels = [name for name, _ in items]
    values = [cnt for _, cnt in items]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(labels))))
    bars = ax.barh(labels, values, color="#4C72B0")
    ax.invert_yaxis()  # largest category on top
    ax.set_xlabel("Number of attempts")
    ax.set_title(f"Failure-mode breakdown: {data.get('model')}\n"
                 f"({n_failed_attempts} non-complete attempts across {data.get('total_problems')} problems)")
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(value), va="center", fontsize=9)
    fig.tight_layout()

    png_path = out_dir / f"{stem}_error_histogram.png"
    fig.savefig(png_path, dpi=150)
    print(f"Wrote histogram -> {png_path}")


if __name__ == "__main__":
    main()
