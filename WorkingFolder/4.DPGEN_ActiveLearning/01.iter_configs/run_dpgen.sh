#!/bin/bash
# ====================================================================
#  DP-GEN Concurrent Active Learning Execution Script on WHU-HPC
# ====================================================================

set -e

echo "===================================================================="
echo " Starting DP-GEN Active Learning Exploration & Labeling Pipeline"
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

# Verify tools
if ! command -v dpgen &> /dev/null; then
    echo "[Error] dpgen command not found in AI_phosphogypsum_env. Please install dpgen."
    exit 1
fi

echo "[Step 1] Initializing seeds if not present..."
if [ ! -d "../00.init_seeds/init_3.1.1CSO-2H2O" ]; then
    python3 ../00.init_seeds/prepare_init_data_from_cp2k.py
fi

echo "[Step 2] Running DP-GEN active learning iteration loop..."
dpgen run param.json machine.json

echo "===================================================================="
echo " DP-GEN active learning loop finished!"
echo " Proceed to 5.DeePMD_MLIP for production potential training."
echo "===================================================================="
