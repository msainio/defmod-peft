#!/bin/bash

#SBATCH --account=project_2007780
#SBATCH --job-name=train-fingpt-dbnary-debug
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

srun python3 src/train.py \
    --data_config config/datasets/dbnary.json \
    --task_config config/experiments/train_fingpt.json
