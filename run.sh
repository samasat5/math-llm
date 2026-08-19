#!/bin/bash
#SBATCH --partition=hard
#SBATCH --gres=gpu:A6000:1
#SBATCH --job-name=mathllm
#SBATCH --time=12:00:00
#SBATCH --output=out_%j.log

cd ~/mathllm/math-llm
conda activate lab
python your_eval_script.py
