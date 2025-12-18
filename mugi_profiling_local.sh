#!/usr/bin/env bash
export TOKEN=""

# value distribution profiling
bash value_distribution_scripts/vivit.sh
bash value_distribution_scripts/swin.sh
bash value_distribution_scripts/llama.sh
bash value_distribution_scripts/whisper.sh

# perplexity profiling
bash ppl_distribution_scripts/swin/swin_tiny.sh
bash ppl_distribution_scripts/swin/swin_large.sh
bash ppl_distribution_scripts/whisper/whisper_tiny.sh
bash ppl_distribution_scripts/whisper/whisper_large.sh
bash ppl_distribution_scripts/vivit/vivit.sh
bash ppl_distribution_scripts/llama/llama_2_7b.sh
bash ppl_distribution_scripts/llama/llama_2_13b.sh

# theoretical nonlinear error
bash figures/scripts/nonlinear_error.sh