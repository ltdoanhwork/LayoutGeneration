#!/bin/bash
# Quick script to print summary with conda environment

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sam

# Run the summary script
python -m eval.print_summary --summary "$@"
