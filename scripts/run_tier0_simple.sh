#!/bin/bash
#
# Run the simple agent with 10 attempts (pass@10) over the 100 easiest
# tier-0 miniF2F problems (MATH-sourced, non-competition: mathd_*/algebra_*/
# numbertheory_*/induction_*).
#
# Usage:
#   ./scripts/run_tier0_simple.sh [extra python -m math_llm args...]
#
# Env overrides:
#   MODEL       - model name (default: Pythagoras-LM/Pythagoras-Prover-4B)
#   MAX_TOKENS  - max new tokens per sample (default: 20000)

set -e

cd "$(dirname "$0")/.."

MODEL="${MODEL:-Pythagoras-LM/Pythagoras-Prover-4B}"
MAX_TOKENS="${MAX_TOKENS:-20000}"

poetry run python -m math_llm minif2f-lean4 simple \
  --tier 0 \
  --samples 100 \
  --k 10 \
  --model "$MODEL" \
  --max-tokens "$MAX_TOKENS" \
  "$@"
