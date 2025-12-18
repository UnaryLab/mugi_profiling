#!/bin/bash

#SBATCH --account=bebv-delta-gpu
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=16
#SBATCH --partition=gpuA100x4,gpuA40x4,gpuA100x8,gpuH200x8
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH --job-name=nonlinear_error
#SBATCH --error=output/figures/nonlinear_error/error.txt
#SBATCH --output=output/figures/nonlinear_error/output.txt

# module load python
# module load anaconda3_gpu
# module load cuda

# Initialize conda properly for bash script
source $(conda info --base)/etc/profile.d/conda.sh

conda deactivate
conda activate mugi_profiling

cd ~/mugi_profiling

python figures/code/nonlinear_data.py
python figures/code/nonlinear_error.py
