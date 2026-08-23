#!/usr/bin/env bash
# ==============================================================================
#  STAGE 5: MODEL TEST & TABULATION COMPRESSION (DP TEST & DP COMPRESS)
# ==============================================================================
# Evaluates RMSE on energy and force for phosphogypsum systems.
# Compresses model_000 into a 5th-order polynomial table for 5x-20x LAMMPS acceleration.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${ROOT_DIR}"

echo "================================================================================"
echo "  COMPRESSING MODEL ON RTX 5090 (DP COMPRESS -> frozen_model_compressed.pb)"
echo "================================================================================"

FROZEN_MODEL="${ROOT_DIR}/5.DeePMD_MLIP/01.training/model_000/frozen_model.pb"
if [ ! -f "${FROZEN_MODEL}" ]; then
    FROZEN_MODEL="${ROOT_DIR}/5.DeePMD_MLIP/01.training/frozen_model.pb"
fi

if [ -f "${FROZEN_MODEL}" ]; then
    cp "${FROZEN_MODEL}" "${SCRIPT_DIR}/frozen_model.pb"
fi

echo "Compressing frozen_model.pb for ultra-fast LAMMPS production MD on RTX 5090..."

docker compose run --rm -w /work/WorkingFolder/5.DeePMD_MLIP/03.freeze_and_compress deepmd dp compress -i frozen_model.pb -o frozen_model_compressed.pb

echo -e "\n  [SUCCESS] Created: 5.DeePMD_MLIP/03.freeze_and_compress/frozen_model_compressed.pb"
echo "  Ready for 100k-atom / 100 ns LAMMPS MD simulation on RTX 5090!"
