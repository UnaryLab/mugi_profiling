#!/bin/bash

#SBATCH --account=bebv-delta-gpu
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --partition=gpuA100x4,gpuA40x4,gpuA100x8,gpuH200x8
#SBATCH --gres=gpu:2
#SBATCH --mem=64g
#SBATCH --job-name=value_distribution_figure
#SBATCH --error=output/figures/value_distribution/error.txt
#SBATCH --output=output/figures/value_distribution/output.txt

# module load python
# module load anaconda3_gpu
# module load cuda

# Initialize conda properly for bash script
source $(conda info --base)/etc/profile.d/conda.sh

conda deactivate
conda activate mugi_profiling

cd ~/mugi_profiling

python figures/code/distribution_figure.py
