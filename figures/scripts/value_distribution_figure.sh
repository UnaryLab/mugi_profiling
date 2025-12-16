#!/bin/bash

#SBATCH --account=bebv-delta-gpu
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --job-name=value_distribution_figure
#SBATCH --error=output/figures/value_distribution/error.txt
#SBATCH --output=output/figures/value_distribution/output.txt

module load python
module load anaconda3_gpu
module load cuda

# Initialize conda properly for bash script
source $(conda info --base)/etc/profile.d/conda.sh

conda deactivate
conda activate mugi_profiling

cd ~/mugi_profiling

python distribution_figure.py
