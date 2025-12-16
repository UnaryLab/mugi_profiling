
# value distribution profiling
# sbatch value_distribution_scripts/swin.sh
# sbatch value_distribution_scripts/llama.sh
# sbatch value_distribution_scripts/whisper.sh
sbatch value_distribution_scripts/vivit.sh

# perplexity profiling
sbatch ppl_distribution_scripts/swim/swin_tiny.sh
sbatch ppl_distribution_scripts/swim/swin_large.sh
#sbatch ppl_distribution_scripts/whisper/whisper_tiny.sh
#sbatch ppl_distribution_scripts/whisper/whisper_large.sh
sbatch ppl_distribution_scripts/vivit/vivit.sh
#sbatch ppl_distribution_scripts/llama/llama_2_7b.sh
sbatch ppl_distribution_scripts/llama/llama_2_13b.sh

# theoretical nonlinear error