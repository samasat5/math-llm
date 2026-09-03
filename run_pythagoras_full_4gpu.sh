#!/bin/bash
#SBATCH --account=priority-farazdadgostari
#SBATCH --job-name=mathllm_full4
#SBATCH --partition=gpupriority
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=96:00:00
#SBATCH --output=mathllm_full4_%j.out
#SBATCH --error=mathllm_full4_%j.err

module purge
module load Python/3.12.3-GCCcore-13.3.0

cd "$HOME/math-llm"
source venv/bin/activate

export PATH="$HOME/.elan/bin:$PATH"
export MATHLIB_PROJECT_PATH="$HOME/.lean-bench"

echo "NODE: $(hostname)"
nvidia-smi

MODEL="${MODEL:-Pythagoras-LM/Pythagoras-Prover-4B}"
MAX_TOKENS="${MAX_TOKENS:-20000}"

# Full minif2f-lean4 (488 problems, no tier filter), split across 4 GPUs:
# worker N takes problems [N*122 : N*122+122).
# CLI saves the full results JSON incrementally after every problem
# (see save_summary("in_progress") in src/math_llm/cli.py), so each
# worker's outputs_gpuN/*.json is always up to date even if a worker dies.
declare -a PIDS

for i in 0 1 2 3; do
    OFFSET=$((i * 122))
    LOG="worker_gpu${i}_${SLURM_JOB_ID}.log"
    python -m math_llm minif2f-lean4 simple \
        --offset "$OFFSET" --samples 122 \
        --k 10 \
        --model "$MODEL" \
        --max-tokens "$MAX_TOKENS" \
        --gpu "$i" \
        --output "outputs_gpu${i}" \
        > "$LOG" 2>&1 &
    PIDS[$i]=$!
    echo "Worker $i (problems ${OFFSET}-$((OFFSET+121))) PID: ${PIDS[$i]}, log: $LOG"
done

STATUS=0
for i in 0 1 2 3; do
    wait "${PIDS[$i]}"
    WSTATUS=$?
    echo "Worker $i exited with status $WSTATUS"
    if [ "$WSTATUS" -ne 0 ]; then STATUS=1; fi
done

exit $STATUS
