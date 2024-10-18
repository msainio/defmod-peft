#!/bin/bash

#SBATCH --account=project_2007780
#SBATCH --job-name=train-opt-mini
#SBATCH --output=./io/%j-%x.out
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=15
#SBATCH --mail-type=ALL

module purge
module load pytorch/2.4
export HF_HOME=".cache/huggingface"

srun python3 src/train.py --config config/train_opt_mini.json
