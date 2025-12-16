#!/bin/bash

#SBATCH --account=bebv-delta-gpu
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --job-name=value_distribution_figure
#SBATCH --error=figures/value_distribution/error.txt
#SBATCH --output=figures/value_distribution/output.txt