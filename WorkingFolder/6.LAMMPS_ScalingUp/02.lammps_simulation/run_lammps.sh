#!/bin/bash
# ====================================================================
#  LAMMPS Large-Scale Hydrothermal Simulation Script on WHU-HPC
# ====================================================================

set -e

echo "===================================================================="
echo " Starting LAMMPS Large-Scale Hydrothermal Simulation (180 °C, 50 bar)"
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

# Check for potential model
if [ ! -f "frozen_model_compressed.pb" ] && [ ! -f "frozen_model.pb" ]; then
    echo "[Info] Model not found in current directory. Checking Stage 5..."
    if [ -f "../../5.DeePMD_MLIP/03.freeze_and_compress/frozen_model_compressed.pb" ]; then
        cp "../../5.DeePMD_MLIP/03.freeze_and_compress/frozen_model_compressed.pb" ./
    elif [ -f "../../5.DeePMD_MLIP/03.freeze_and_compress/frozen_model.pb" ]; then
        cp "../../5.DeePMD_MLIP/03.freeze_and_compress/frozen_model.pb" ./
    fi
fi

# Check for data file
if [ ! -f "data.phosphogypsum_hydrothermal" ]; then
    echo "[Info] Generating data.phosphogypsum_hydrothermal..."
    python3 ../01.system_setup/build_large_scale_hydrothermal_box.py
fi

echo "[Step 1] Running GPU-accelerated LAMMPS with Deep Potential..."
lmp -in in.phosphogypsum_hydrothermal.lammps -log lammps_hydrothermal.log

echo "===================================================================="
echo " LAMMPS simulation finished! Trajectory saved to hydrothermal_prod.lammpstrj"
echo " Proceed to 03.trajectory_analysis for reaction kinetics analysis."
echo "===================================================================="
