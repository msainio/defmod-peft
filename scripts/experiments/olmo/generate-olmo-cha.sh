#!/bin/bash

#SBATCH --account=project_2007780
#SBATCH --job-name=generate-olmo-cha
#SBATCH --output=./stdio/%j-%x.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-18
#SBATCH --mail-type=ALL

module purge
module load pytorch/2.3
export HF_HOME=".cache/huggingface"

srun python3 src/generate.py \
    --data_config config/datasets/cha.json \
    --task_config config/experiments/generate_olmo.json \
    --peft_model models/24457112-train-olmo-cha
