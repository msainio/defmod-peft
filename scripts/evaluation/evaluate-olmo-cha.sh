#!/bin/bash

#SBATCH --account=project_2007780
#SBATCH --job-name=evaluate-olmo-cha
#SBATCH --output=./stdio/%j-%x.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=0-2
#SBATCH --mail-type=ALL

module purge
module load pytorch/2.3
source .venv/bin/activate

srun python3 src/run_eval.py \
    --predictions preds/24529353-generate-olmo-cha.csv \
    --data_config config/data/cha.json
