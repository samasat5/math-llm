"""Error histograms for the turnstile-strip re-verification results
(outputs/turnstile_strip_reverify_MERGED_100_results.json), excluding
aime_1987_p5 (recovered by the fix, so not a "failure" anymore).

Three views:
  1. Per-sample: one vote per attempt, using its first/root error. The
     representative unit, since a single degenerate attempt can otherwise
     emit thousands of cascading duplicate error messages and swamp a raw
     per-instance count (one attempt had 7,888).
  2. Per-problem (dominant): one vote per problem, whichever error category
     is most common across that problem's own samples.
  3. Per-problem (any-occurrence): a problem counts toward every category
     any of its samples hit - shows breadth of failure modes per problem.
"""

import json
import re
from collections import Counter

RESULTS_FILE = "outputs/turnstile_strip_reverify_MERGED_100_results.json"
EXCLUDE_PROBLEM_IDS = {"minif2f-lean4/aime_1987_p5"}


def categorize(err: str) -> str:
    e = err.strip()
    first_line = e.split("\n")[0]
    if "No goals to be solved" in e:
        return "No goals to be solved"
    if re.search(r"\bnlinarith\b.*failed", e, re.I):
        return "nlinarith failed"
    if re.search(r"\blinarith\b.*failed", e, re.I):
        return "linarith failed"
    if "omega could not prove" in e or "omega failed" in e.lower():
        return "omega failed"
    if "unsolved goals" in e:
        return "unsolved goals (leftover goal)"
    if "Type mismatch" in e:
        return "Type mismatch"
    if re.search(r"Unknown (constant|identifier)", e):
        return "Unknown constant/identifier (hallucinated name)"
    if "unexpected token" in e or "unexpected end of input" in e or "Invalid `end`" in e:
        return "Parse/structure error (broken syntax)"
    if "already been declared" in e:
        return "Duplicate declaration (extraction bug)"
    if "maximum recursion depth" in e or "maxRecDepth" in e:
        return "Max recursion depth"
    if "heartbeats" in e.lower():
        return "Timeout (maxHeartbeats)"
    if "Tactic `simp`" in e or "simp made no progress" in e or "simp_all made no progress" in e:
        return "simp failed"
    if "Tactic `rw`" in e or "Did not find an occurrence" in e:
        return "rw/rewrite failed"
    if "Tactic `split`" in e:
        return "split failed (if/match)"
    if re.search(r"Tactic `apply`", e):
        return "apply failed (unification)"
    return f"OTHER: {first_line[:70]}"


def print_histogram(counter: Counter, denom: int, title: str, top_n: int = 20) -> None:
    print(f"\n=== {title} ===")
    if not counter:
        print("  (no data)")
        return
    maxc = max(counter.values())
    for cat, cnt in counter.most_common(top_n):
        bar = "#" * int(30 * cnt / maxc)
        print(f"{cnt:4d} ({100*cnt/denom:4.1f}%)  {bar:<30s} {cat}")


def main():
    with open(RESULTS_FILE) as f:
        data = json.load(f)

    per_sample = Counter()
    per_problem_dominant = Counter()
    per_problem_any = Counter()
    n_problems = 0
    n_samples = 0

    for entry in data["results"]:
        if entry["problem_id"] in EXCLUDE_PROBLEM_IDS:
            continue

        cats_this_problem = []
        for a in entry["attempts"]:
            errs = a.get("new_error") or []
            if not errs:
                continue
            n_samples += 1
            cat = categorize(errs[0])
            per_sample[cat] += 1
            cats_this_problem.append(cat)

        if not cats_this_problem:
            continue
        n_problems += 1
        dominant = Counter(cats_this_problem).most_common(1)[0][0]
        per_problem_dominant[dominant] += 1
        for cat in set(cats_this_problem):
            per_problem_any[cat] += 1

    print(f"Problems included: {n_problems}  |  Samples (attempts) with >=1 error: {n_samples}")
    print_histogram(per_sample, n_samples, "Per-SAMPLE histogram (first/root error, one vote per attempt)")
    print_histogram(per_problem_dominant, n_problems, "Per-PROBLEM histogram (dominant error, one vote per problem)")
    print_histogram(per_problem_any, n_problems, "Per-PROBLEM histogram (any occurrence, a problem can count in multiple categories)")


if __name__ == "__main__":
    main()
