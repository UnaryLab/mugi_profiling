#!/bin/bash

#SBATCH --account=bebv-delta-gpu
#SBATCH --time=00:02:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8g
#SBATCH --job-name=heatmap
#SBATCH --error=output/figures/heatmap/error.txt
#SBATCH --output=output/figures/heatmap/output.txt

# module load python
# module load anaconda3_gpu
# module load cuda

# Initialize conda properly for bash script
source $(conda info --base)/etc/profile.d/conda.sh

conda deactivate
conda activate mugi_profiling

python figures/code/perplexity_figure.py