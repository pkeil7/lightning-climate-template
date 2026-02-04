#!/bin/bash
#SBATCH --account=
#SBATCH --job-name=
#SBATCH --partition=
#SBATCH --gres=
#SBATCH --nodes=
#SBATCH --mem=
#SBATCH --time=
#SBATCH --output=logs/slurm_job.o%j

mkdir -p ../logs

# Activate virtual environment
source .venv/bin/activate

# Run the training script
python -u train.py --data_dir /path/to/your/data --in_channels 10 --out_channels 1
