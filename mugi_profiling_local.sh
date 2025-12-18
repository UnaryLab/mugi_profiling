#!/usr/bin/env bash
export TOKEN=""

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install triton==3.3.0
pip install transformers==4.54.1
pip install tokenizers==0.21.4
pip install accelerate==1.9.0
pip install datasets==3.6.0

PROFILE_DIR="profile"
PPL_DIR="csv"
ERROR_DIR="output"
[ -d "$PROFILE_DIR" ] && rm -rf "$PROFILE_DIR"
[ -d "$PPL_DIR" ] && rm -rf "$PPL_DIR"
[ -d "$ERROR_DIR" ] && rm -rf "$ERROR_DIR"

# value distribution profiling
bash value_distribution_scripts/vivit.sh
bash value_distribution_scripts/swin.sh
bash value_distribution_scripts/llama.sh
bash value_distribution_scripts/whisper.sh
bash figures/scripts/value_distribution_figure.sh

# perplexity profiling
bash ppl_distribution_scripts/swin/swin_tiny.sh
bash ppl_distribution_scripts/swin/swin_large.sh
bash ppl_distribution_scripts/whisper/whisper_tiny.sh
bash ppl_distribution_scripts/whisper/whisper_large.sh
bash ppl_distribution_scripts/vivit/vivit.sh
bash ppl_distribution_scripts/llama/llama_2_7b.sh
bash ppl_distribution_scripts/llama/llama_2_13b.sh

bash end_to_end_scripts/swin.sh
bash end_to_end_scripts/whisper.sh
bash end_to_end_scripts/vivit.sh
bash end_to_end_scripts/llama.sh
bash figures/scripts/heatmap.sh

# theoretical nonlinear error
bash figures/scripts/nonlinear_error.sh