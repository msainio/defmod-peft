#!/bin/bash

#SBATCH --account=project_2007780
#SBATCH --job-name=train-fingpt-dbnary
#SBATCH --output=./stdio/%j-%x.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-4
#SBATCH --mail-type=ALL

module purge
module load pytorch/2.3
export HF_HOME=".cache/huggingface"

srun python3 src/train.py \
    --data_config config/data/dbnary.json
    --task_config config/experiments/train_fingpt.json
