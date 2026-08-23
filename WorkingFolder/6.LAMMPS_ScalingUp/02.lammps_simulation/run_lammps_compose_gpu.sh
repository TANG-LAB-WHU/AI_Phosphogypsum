#!/usr/bin/env bash
# ==============================================================================
#  STAGE 6: BATCH PRODUCTION MOLECULAR DYNAMICS RUNNER (DOCKER COMPOSE GPU)
# ==============================================================================
# Executes large-scale (10^5 atoms) hydrothermal reactive MD on RTX 5090 (32GB).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

echo "================================================================================"
echo "  SUBMITTING PRODUCTION HYDROTHERMAL MD ON RTX 5090 (DOCKER COMPOSE GPU)"
echo "================================================================================"

# Copy compressed model if needed
COMP_MODEL="${ROOT_DIR}/5.DeePMD_MLIP/03.freeze_and_compress/frozen_model_compressed.pb"
if [ -f "${COMP_MODEL}" ]; then
    cp "${COMP_MODEL}" "${SCRIPT_DIR}/frozen_model_compressed.pb"
fi

# Generate structure if missing
if [ ! -f "${SCRIPT_DIR}/data.phosphogypsum_hydrothermal" ]; then
    echo "[Info] Generating data.phosphogypsum_hydrothermal box..."
    docker compose run --rm -w /work/WorkingFolder/6.LAMMPS_ScalingUp/01.system_setup deepmd python3 build_large_scale_hydrothermal_box.py
    cp "${ROOT_DIR}/6.LAMMPS_ScalingUp/01.system_setup/data.phosphogypsum_hydrothermal" "${SCRIPT_DIR}/"
fi

echo -e "\n[STARTING] LAMMPS Large-Scale Hydrothermal Simulation..."
docker compose run --rm \
    -w /work/WorkingFolder/6.LAMMPS_ScalingUp/02.lammps_simulation \
    lammps \
    lmp -in in.phosphogypsum_hydrothermal.lammps -log lammps_hydrothermal.log

echo -e "\n================================================================================"
echo "  PRODUCTION MD COMPLETE! READY FOR REACTION KINETICS & PNC CLUSTERING ANALYSIS."
echo "================================================================================"
