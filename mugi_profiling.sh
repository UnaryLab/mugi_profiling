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
llama_job=$(sbatch --parsable value_distribution_scripts/llama.sh)
swin_job=$(sbatch --parsable value_distribution_scripts/swin.sh)
vivit_job=$(sbatch --parsable value_distribution_scripts/vivit.sh)
whisper_job=$(sbatch --parsable value_distribution_scripts/whisper.sh)
sbatch --dependency=afterok:$llama_job:$swin_job:$vivit_job:$whisper_job figures/scripts/value_distribution_figure.sh

# perplexity profiling
st_0=$(sbatch --parsable ppl_distribution_scripts/swin/swin_tiny_0.sh)
st_1=$(sbatch --parsable ppl_distribution_scripts/swin/swin_tiny_1.sh)
sl_0=$(sbatch --parsable ppl_distribution_scripts/swin/swin_large_0.sh)
sl_1=$(sbatch --parsable ppl_distribution_scripts/swin/swin_large_1.sh)
wt_0=$(sbatch --parsable ppl_distribution_scripts/whisper/whisper_tiny_0.sh)
wt_1=$(sbatch --parsable ppl_distribution_scripts/whisper/whisper_tiny_1.sh)
wl_0=$(sbatch --parsable ppl_distribution_scripts/whisper/whisper_large_0.sh)
wl_1=$(sbatch --parsable ppl_distribution_scripts/whisper/whisper_large_1.sh)
v_0=$(sbatch --parsable ppl_distribution_scripts/vivit/vivit_0.sh)
v_1=$(sbatch --parsable ppl_distribution_scripts/vivit/vivit_1.sh)
l7_0=$(sbatch --parsable ppl_distribution_scripts/llama/llama_2_7b_0.sh)
l7_1=$(sbatch --parsable ppl_distribution_scripts/llama/llama_2_7b_1.sh)
l13_0=$(sbatch --parsable ppl_distribution_scripts/llama/llama_2_13b_0.sh)
l13_1=$(sbatch --parsable ppl_distribution_scripts/llama/llama_2_13b_1.sh)

swin_e2e=$(sbatch --parsable --dependency=afterok:$st_0:$st_1:$sl_0:$sl_1 end_to_end_scripts/swin.sh)
whisper_e2e=$(sbatch --parsable --dependency=afterok:$wt_0:$wt_1:$wl_0:$wl_1 end_to_end_scripts/whisper.sh)
vivit_e2e=$(sbatch --parsable --dependency=afterok:$v_0:$v_1 end_to_end_scripts/vivit.sh)
llama_e2e=$(sbatch --parsable --dependency=afterok:$l7_0:$l7_1:$l13_0:$l13_1 end_to_end_scripts/llama.sh)

sbatch --dependency=afterok:$swin_e2e:$whisper_e2e:$vivit_e2e:$llama_e2e figures/scripts/heatmap.sh

sbatch figures/scripts/nonlinear_error.sh