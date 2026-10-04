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
