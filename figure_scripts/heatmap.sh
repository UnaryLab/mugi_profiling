#!/bin/bash

#SBATCH --account=bebv-delta-gpu
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --job-name=heatmap
#SBATCH --error=output/heatmap.txt
#SBATCH --output=output/heatmap.txt

module load python
module load anaconda3_gpu
module load cuda

# Initialize conda properly for bash script
source $(conda info --base)/etc/profile.d/conda.sh

conda deactivate
conda activate mugi_profiling

cd ~/mugi_profiling

python perplexity_figure.py
