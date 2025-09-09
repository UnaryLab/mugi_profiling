#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=whisper_tiny
#SBATCH --constraint=gpu32
#SBATCH --job-name=llama_2_7b_profiling
#SBATCH --error=output/run/llama_2/llama_2_7b/error.txt
#SBATCH --output=output/run/llama_2/llama_2_7b/output.txt

# module load anaconda
# module load cuda
# module load openblas

# Initialize conda properly for bash script
# source $(conda info --base)/etc/profile.d/conda.sh
# eval "$(conda shell.bash hook)"

# conda activate mugi_profiling

cd ~/mugi_profiling

# Configuration files to process
model_config="config/model_config/llama/llama_2_7b.yaml"
nonlinear_config="config/nonlinear_config/torch/torch.yaml"
parameter_config="config/parameter_config/parameter_config.yaml"
hf_token="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"

huggingface-cli login --token "$hf_token"

# Loop through each configuration
echo ""
echo "Running experiment with configuration: $model_config"
echo "----------------------------------------"

# Run the transformer script with the current config
export PYTHONPATH=~/mugi_profiling:$PYTHONPATH

NUM_GPUS=$(nvidia-smi -L | wc -l)

python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    src/model_script.py \
    --model_config "$model_config" \
    --nonlinear_config "$nonlinear_config" \
    --parameter_config "$parameter_config"

# python src/model_script.py --model_config "$model_config" \
#                             --nonlinear_config "$nonlinear_config" \
#                             --parameter_config "$parameter_config"

# Capture the exit code
exit_code=$?

# Check if the script ran successfully
if [ $exit_code -eq 0 ]; then
    echo "✓ Successfully completed experiment with $model_config"
else
    echo "✗ Error occurred while running experiment with $model_config (exit code: $exit_code)"
fi

echo "----------------------------------------"