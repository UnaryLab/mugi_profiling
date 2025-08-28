#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --job-name=llama_2_7b_profiling
#SBATCH --error=output/run/llama_2/llama_2_7b/error.txt
#SBATCH --output=output/run/llama_2/llama_2_7b/output.txt
#SBATCH --constraint=gpu32|gpu80

module load anaconda
module load cuda

# Initialize conda properly for bash script
# source $(conda info --base)/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"

conda activate mugi_profiling

cd ~/mugi_profiling

# Configuration files to process
model_configs=("config/model_config/whisper/whisper_tiny.yaml")
nonlinear_config="config/nonlinear_config/nonlinear_config.yaml"
parameter_config="config/parameter_config/parameter_config.yaml"
hf_token="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"

huggingface-cli login --token "$hf_token"

# Loop through each configuration
echo ""
echo "Running experiment with configuration: $model_config"
echo "----------------------------------------"

# Run the transformer script with the current config
export PYTHONPATH=~/mugi_profiling:$PYTHONPATH

which pip

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
    echo "Check whisper_detailed_log.txt and whisper_error.txt for details"
    echo "Continuing with next configuration..."
fi

echo "----------------------------------------"