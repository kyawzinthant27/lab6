#!/bin/bash
#SBATCH --job-name=lab6
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=lab6-%j.out

set -e

cd "$SLURM_SUBMIT_DIR"
source "$HOME/deep_learning/env_dl/bin/activate"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'

srun python -u lab6.py
