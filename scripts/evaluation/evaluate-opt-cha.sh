#!/bin/bash

#SBATCH --account=project_2007780
#SBATCH --job-name=evaluate-opt-cha
#SBATCH --output=./stdio/%j-%x.out
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=0-2
#SBATCH --mail-type=ALL

module purge
module load pytorch/2.3
source .venv/bin/activate

srun python3 src/run_eval.py \
    --predictions preds/24611797-generate-opt-cha.csv \
    --data_config config/data/cha.json
