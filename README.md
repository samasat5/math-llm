# math-llm: dissecting Lean provers


One-shot and few-shots inference theorem generation by Pythagoras prover 4B and Godel prover 8B.


## Setup

```bash
# Prerequisites: Python 3.10+, Poetry, Elan (Lean version manager)
make install
make lean-server   # Lean + Mathlib + REPL (~2GB, 10-20 min)
```

## Running a benchmark

```bash
python -m math_llm <dataset> simple [--model M] [--k K] [--tier T | --min-tier T] \
    [--seed S] [--samples N] [--max-tokens N] [--batch-size B]
```

The report runs used `--k 10 --max-tokens 20000`, with `--tier 0` (Tier 0) or
`--min-tier 1 --seed 0 --samples 100` (Tier 1). See `scripts/run_tier0_simple.sh`.

## Where each result comes from

| Report | Code | Data (`outputs/`) |
|---|---|---|
| Table 4.1, pass@10 | `src/math_llm/cli.py`, `agents/simple.py` | Pythagoras: `*Pythagoras*_k10_tier0_*_results.json` + `remaining/` (Tier 0), `*Pythagoras*_MERGED_100_results.json` (Tier 1). Goedel: `*Goedel*_k10_tier0_*_results.json`, `*Goedel*_MERGED_100_results.json` |
| Tier 1 merges | `scripts/merge_mintier1_100.py` (Goedel) | partial runs `*_mintier1_seed0_*_results.json`, `*_partial31_backup.json`, `*Pythagoras*_mintier1_seed0_*_autoformalizer_results.json` |
| Fig. 4.1, failure taxonomy | `scripts/categorize_errors.py` | `*_error_categories.json`, `*_error_histogram.png`, `*_MERGED_100_failures_*.json`, `*_incomplete_non_sorry.json` |
| §2.4, Table B.1, Fig. B.1 (unknown identifiers) | `LeanServer.identifier_exists` | `failure_taxonomy_and_hallucination_analysis.json`, `hallucinated_names_lean_verification.json`, `*_MERGED_100_hallucinations.json` |
| Fig. B.2 (`sorry` proofs) | | `*_MERGED_100_sorry_cheats.json` |
| App. B.4.1, hints on `aime_1987_p5` | `agents/case_study/` (`temp_sweep`, `prefix_line_sweep`, `prefix_conditioned`) | `aime_1987_p5_temperature_sweep_*.json` (prompt guidance), `aime_1987_p5_prefix_line_sweep_*` |
| App. B.4.2–4.3, `mathd_numbertheory_188` | `scripts/*_challenge_mathd_numbertheory_188.py` | `mathd_numbertheory_188_*_challenge_*.json` |
| App. B.1, RL-only training | `src/math_llm/training/` (GRPO) | |

Run logs (git-ignored) are in `logs/`.
