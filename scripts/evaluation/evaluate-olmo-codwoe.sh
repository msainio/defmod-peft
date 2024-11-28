#!/bin/bash

#SBATCH --account=project_2007780
#SBATCH --job-name=evaluate-olmo-codwoe
#SBATCH --output=./io/%j-%x.out
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=15
#SBATCH --mail-type=ALL

module purge
module load pytorch/2.3
source .venv/bin/activate
    
srun python3 src/run_eval.py \
    --predictions preds/24422784-generate-olmo-codwoe.csv \
    --data_config config/data/codwoe.json
