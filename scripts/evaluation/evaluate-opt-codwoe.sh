#!/bin/bash

#SBATCH --account=project_2007780
#SBATCH --job-name=evaluate-opt-codwoe
#SBATCH --output=./stdio/%j-%x.out
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=20G
#SBATCH --time=0-1
#SBATCH --mail-type=ALL

module purge
module load pytorch/2.3
source .venv/bin/activate

srun python3 src/run_eval.py \
    --predictions preds/24611498-generate-opt-codwoe.csv \
    --data_config config/datasets/codwoe.json
