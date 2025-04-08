#!/bin/bash
#SBATCH --job-name=create_small_dataset
#SBATCH --output=create_small_dataset_%j.out
#SBATCH --error=create_small_dataset_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=general

# Activate your conda environment if needed
# source activate your_env_name

# Run the script
python3 create_small_dataset.py 