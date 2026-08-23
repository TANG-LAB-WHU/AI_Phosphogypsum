#!/bin/bash
# ====================================================================
#  DeePMD-kit Production Training Script on WHU-HPC
# ====================================================================

set -e

echo "===================================================================="
echo " Starting DeePMD-kit Production Model Training (1,000,000 steps)..."
echo "===================================================================="

# Activate Conda Environment AI_phosphogypsum_env
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    source ~/.bashrc 2>/dev/null || true
fi

conda activate AI_phosphogypsum_env 2>/dev/null || echo "[Info] Running in active Python environment"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "[Info] GPU device(s) detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Run DeePMD training
dp train input.json

echo "===================================================================="
echo " Training finished! Check lcurve.out and checkpoint files."
echo " Proceed to 02.model_evaluation and 03.freeze_and_compress."
echo "===================================================================="
