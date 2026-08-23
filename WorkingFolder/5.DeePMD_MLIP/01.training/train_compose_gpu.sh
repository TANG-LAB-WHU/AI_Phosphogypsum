#!/usr/bin/env bash
# ==============================================================================
#  STAGE 5: DOCKER COMPOSE GPU BATCH TRAINING SCRIPT (RTX 5090 WSL2)
# ==============================================================================
# Trains 4 independent ensemble models (model_000 ~ model_003) on RTX 5090 (32GB)
# using Docker Compose GPU service 'deepmd' (or native conda if configured).
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${SCRIPT_DIR}"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "================================================================================"
echo "  LAUNCHING DEEPMD-KIT ENSEMBLE TRAINING ON RTX 5090 (DOCKER COMPOSE GPU)"
echo "================================================================================"
echo "Working Directory: ${ROOT_DIR}"
echo "Training Config:   ${WORK_DIR}/input.json"

cd "${ROOT_DIR}"

# Train 4 ensemble models with different random seeds
for SEED_ID in 000 001 002 003; do
    MODEL_DIR="${WORK_DIR}/model_${SEED_ID}"
    mkdir -p "${MODEL_DIR}"
    
    # Generate seed-specific input.json
    RAND_SEED=$((42 + 1000 * 10#${SEED_ID} + RANDOM % 100))
    python3 -c "
import json
with open('${WORK_DIR}/input.json') as f:
    d = json.load(f)
d['model']['descriptor']['seed'] = ${RAND_SEED}
d['model']['fitting_net']['seed'] = ${RAND_SEED} + 1
d['training']['seed'] = ${RAND_SEED} + 2
with open('${MODEL_DIR}/input.json', 'w') as f:
    json.dump(d, f, indent=2)
"
    
    echo -e "\n--------------------------------------------------------------------------------"
    echo "  [STARTING] Training Ensemble Model ${SEED_ID} (Random Seed: ${RAND_SEED})"
    echo "  Target Output: 5.DeePMD_MLIP/01.training/model_${SEED_ID}/"
    echo "--------------------------------------------------------------------------------"
    
    docker compose run --rm -w /work/WorkingFolder/5.DeePMD_MLIP/01.training/model_${SEED_ID} deepmd dp train input.json
    
    echo -e "\n  [FREEZING] Model ${SEED_ID} -> frozen_model.pb"
    docker compose run --rm -w /work/WorkingFolder/5.DeePMD_MLIP/01.training/model_${SEED_ID} deepmd dp freeze -o frozen_model.pb
done

echo -e "\n================================================================================"
echo "  ★ ALL 4 ENSEMBLE MODELS SUCCESSFULLY TRAINED AND FROZEN ON RTX 5090!"
echo "================================================================================"
