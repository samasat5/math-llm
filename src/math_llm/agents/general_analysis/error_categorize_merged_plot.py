#!/usr/bin/env python3
"""
Merge two error_categorize.py category-breakdown JSON files into one
side-by-side grouped bar chart (Pythagoras vs Goedel failure-mode shares).

Usage:
    poetry run python -m math_llm.agents.general_analysis.error_categorize_merged_plot \
        <pythagoras_error_categories.json> <goedel_error_categories.json> \
        [--output PATH]

Shares (% of that model's own non-complete attempts) are compared rather than
raw counts, since the two models have different total non-complete attempt
counts (1056 vs 883) and raw counts would misrepresent relative prevalence.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Same blue as the single-model error_categorize.py histograms ("#4C72B0"),
# plus its seaborn-muted companion orange, so the merged chart matches the
# look of the plots it's combining.
COLOR_A = "#4C72B0"
COLOR_B = "#DD8452"


def model_label(results_file: str) -> str:
    name = Path(results_file).stem
    m = re.search(r"simple_([\w-]+?)_k\d", name)
    if not m:
        return name
    # Strip the HF org prefix ("Pythagoras-LM-", "Goedel-LM-") - keep just
    # the model name itself for a compact legend entry.
    return re.sub(r"^[\w]+-LM-", "", m.group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("categories_json", nargs=2, type=Path,
                     help="Two *_error_categories.json files (from error_categorize.py)")
    ap.add_argument("--output", type=Path, default=None,
                     help="Output PNG path (default: outputs/merged_error_histogram.png)")
    args = ap.parse_args()

    data = [json.loads(p.read_text()) for p in args.categories_json]
    labels_full = [model_label(d["results_file"]) for d in data]

    shares = []
    for d in data:
        total = d["n_noncomplete_attempts"]
        shares.append({k: 100 * v / total for k, v in d["counts"].items()})

    all_categories = set(shares[0]) | set(shares[1])
    # Order by combined share, descending - largest shared pain point on top.
    order = sorted(all_categories, key=lambda c: -(shares[0].get(c, 0) + shares[1].get(c, 0)))

    vals_a = [shares[0].get(c, 0) for c in order]
    vals_b = [shares[1].get(c, 0) for c in order]

    y = list(range(len(order)))
    height = 0.36

    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(order))))
    bars_a = ax.barh([i + height / 2 + 0.02 for i in y], vals_a, height=height,
                      color=COLOR_A, label=labels_full[0])
    bars_b = ax.barh([i - height / 2 - 0.02 for i in y], vals_b, height=height,
                      color=COLOR_B, label=labels_full[1])

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("% of that model's non-complete attempts")
    ax.set_title("Failure-mode breakdown: Pythagoras-Prover-4B vs Goedel-Prover-V2-8B\n"
                  f"({data[0]['n_noncomplete_attempts']} vs {data[1]['n_noncomplete_attempts']} "
                  "non-complete attempts, minif2f-lean4 ALL_TIERS k10)")

    for bars, vals in ((bars_a, vals_a), (bars_b, vals_b)):
        for bar, v in zip(bars, vals):
            if v <= 0:
                continue
            ax.text(bar.get_width() + max(vals_a + vals_b) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f"{v:.1f}%", va="center", fontsize=9)

    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = args.output or Path("outputs/merged_error_histogram.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    print(f"Wrote merged histogram -> {out_path}")


if __name__ == "__main__":
    main()
