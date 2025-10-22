#!/bin/bash

#SBATCH --account=bebv-delta-gpu
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:2
#SBATCH --mem=64g
#SBATCH --job-name=llama_2_70b_profiling
#SBATCH --error=error/llama_2/llama_2_70b_error.txt
#SBATCH --output=output/llama_2/llama_2_70b_output.txt

module load cuda
module load openblas
module load anaconda

# Initialize conda properly for bash script
source $(conda info --base)/etc/profile.d/conda.sh

conda deactivate
conda activate mugi_profiling

cd ~/mugi_profiling

# Configuration files to process
model_configs=("config/model_config/llama/llama_2_70b.yaml")
model_config="config/model_config/llama/llama_2_70b.yaml"
nonlinear_config="config/nonlinear_config/vlp/vlp_softmax_layers_llama_2_70b.yaml"
parameter_config="config/parameter_config/parameter_config.yaml"

hf_token="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"

huggingface-cli login --token "$hf_token"

# Loop through each configuration
echo ""
echo "Running experiment with configuration: $model_config"
echo "----------------------------------------"

# Run the transformer script with the current config
export PYTHONPATH=~/mugi_profiling:$PYTHONPATH

# NUM_GPUS=$(nvidia-smi -L | wc -l)

# python -m torch.distributed.run \
#     --nproc_per_node=$NUM_GPUS \
#     src/model_script.py \
#     --model_config "$model_config" \
#     --nonlinear_config "$nonlinear_config" \
#     --parameter_config "$parameter_config"

python src/model_script.py --model_config "$model_config" \
                            --nonlinear_config "$nonlinear_config" \
                            --parameter_config "$parameter_config"

# Capture the exit code
exit_code=$?

# Check if the script ran successfully
if [ $exit_code -eq 0 ]; then
    echo "✓ Successfully completed experiment with $model_config"
else
    echo "✗ Error occurred while running experiment with $model_config (exit code: $exit_code)"
fi

echo "----------------------------------------"