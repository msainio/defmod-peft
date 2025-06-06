#!/bin/bash

#SBATCH --account=project_2007780
#SBATCH --job-name=generate-opt-cha-debug
#SBATCH --output=./stdio/%j-%x.out
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=15
#SBATCH --mail-type=ALL

module purge
module load pytorch/2.4
export HF_HOME=".cache/huggingface"

srun python3 src/generate.py \
    --data_config config/datasets/cha_mini.json \
    --task_config config/debug/generate_opt_debug.json \
    --peft_model models/24205379-train-opt-cha
