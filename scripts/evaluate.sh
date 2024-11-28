#!/bin/bash

#SBATCH --account=project_2007780
#SBATCH --job-name=evaluate
#SBATCH --output=./io/%j-%x.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=0-1
#SBATCH --mail-type=ALL

module purge
module load pytorch/2.4

preds=(
    "preds/24210970-generate-opt-cha.csv config/data/cha.json" \
    "preds/24422784-generate-olmo-codwoe.csv config/data/codwoe.json" \
    "preds/24529353-generate-olmo-cha.csv config/data/cha.json" \
    "preds/24529354-generate-opt-codwoe.csv config/data/codwoe.json" \
    "preds/24529418-generate-fingpt-dbnary.csv config/data/dbnary.json"
    )

for p in "${preds[@]}"
do
    set -- $p
    srun python3 src/evaluate.py --predictions $1 --data_config $2
done
