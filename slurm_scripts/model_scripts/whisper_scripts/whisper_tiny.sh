#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=whisper_tiny
#SBATCH --error=output/run/whisper/whisper_tiny/error.txt
#SBATCH --output=output/run/whisper/whisper_tiny/output.txt

# module load anaconda
module load cuda
module load openblas

# Initialize conda properly for bash script
# source $(conda info --base)/etc/profile.d/conda.sh
# eval "$(conda shell.bash hook)"

conda activate mugi_profiling

cd ~/mugi_profiling

# Configuration files to process
model_config="config/model_config/whisper/whisper_tiny.yaml"
nonlinear_config="config/nonlinear_config/vlp/vlp_softmax.yaml"
parameter_config="config/parameter_config/parameter_config.yaml"
hf_token="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"

huggingface-cli login --token "$hf_token"

# Loop through each configuration
echo ""
echo "Running experiment with configuration: $model_config"
echo "----------------------------------------"

# Run the transformer script with the current config
export PYTHONPATH=~/mugi_profiling:$PYTHONPATH

# deepspeed --num_gpus=2 src/model_script.py \
#           --model_config "$model_config" \
#           --nonlinear_config "$nonlinear_config" \
#           --parameter_config "$parameter_config"

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