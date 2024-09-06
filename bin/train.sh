#!/bin/bash

#SBATCH --account=project_2007780
#SBATCH --job-name=train
#SBATCH --output=./io/%j_%x.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-16
#SBATCH --mail-type=ALL

module purge
module load pytorch

srun python3 src/train.py --config config/train.json
