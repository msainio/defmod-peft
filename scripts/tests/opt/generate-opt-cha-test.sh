#!/bin/bash

#SBATCH --account=project_2007780
#SBATCH --job-name=generate-opt-cha-test
#SBATCH --output=./stdio/%j-%x.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=60
#SBATCH --mail-type=ALL

module purge
module load pytorch/2.4
export HF_HOME=".cache/huggingface"

srun python3 src/generate.py \
    --data_config config/data/cha_mini.json \
    --task_config config/tests/generate_opt_test.json
