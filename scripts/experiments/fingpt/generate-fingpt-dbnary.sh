#!/bin/bash

#SBATCH --account=project_2007780
#SBATCH --job-name=generate-fingpt-dbnary
#SBATCH --output=./stdio/%j-%x.out
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=15
#SBATCH --mail-type=ALL

module purge
module load pytorch/2.3
export HF_HOME=".cache/huggingface"

srun python3 src/generate.py \
    --data_config config/data/dbnary.json \
    --task_config config/experiments/generate_fingpt.json \
    --peft_model models/24423482-train-fingpt-dbnary
