#!/bin/bash
#SBATCH --account=priority-farazdadgostari
#SBATCH --job-name=mathllm_goedel_sanity
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=mathllm_goedel_sanity_%j.out
#SBATCH --error=mathllm_goedel_sanity_%j.err

set -e

module purge
module load Python/3.12.3-GCCcore-13.3.0

cd "$HOME/math-llm"
source venv/bin/activate

export PATH="$HOME/.elan/bin:$PATH"
export MATHLIB_PROJECT_PATH="$HOME/.lean-bench"

echo "NODE: $(hostname)"
nvidia-smi

MODEL="${MODEL:-Goedel-LM/Goedel-Prover-V2-8B}"
MAX_TOKENS="${MAX_TOKENS:-20000}"

# Tier-0 (easiest) sanity check to confirm the fixed extract_proof()
# is pulling real tactics out of Goedel's output, not the restated theorem.
time python -m math_llm minif2f-lean4 simple \
    --tier 0 \
    --samples 20 \
    --k 5 \
    --model "$MODEL" \
    --max-tokens "$MAX_TOKENS" \
    --output outputs_sanity_goedel \
    "$@"
