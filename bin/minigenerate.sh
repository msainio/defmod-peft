#!/bin/bash

#SBATCH --account=project_2007780
#SBATCH --job-name=minigenerate
#SBATCH --output=./io/%j_%x.out
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=15
#SBATCH --mail-type=ALL

module purge
module load pytorch

srun python3 src/generate.py --config config/minigenerate.json
