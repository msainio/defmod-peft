#!/bin/bash

#SBATCH --account=project_2007780
#SBATCH --job-name=generate-opt-cha
#SBATCH --output=./io/%j-%x.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-4
#SBATCH --mail-type=ALL

module purge
module load pytorch/2.4
export HF_HOME=".cache/huggingface"

srun python3 src/generate.py \
    --data_config config/data/cha.json \
    --task_config config/experiments/generate_opt.json \
    --peft_model models/24205379-train-opt-cha
