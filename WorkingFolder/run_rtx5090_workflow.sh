#!/usr/bin/env bash
# ==============================================================================
#  MASTER PIPELINE RUNNER: STAGES 4 -> 5 -> 6 ON RTX 5090 (WSL2 UBUNTU)
# ==============================================================================
# Reference Architecture: Step3_dpgen-deepmdkit (Tang Lab WHU)
# Hardware Target: NVIDIA GeForce RTX 5090 (32GB GDDR7, Blackwell Tensor Cores)
#
# Usage:
#   ./run_rtx5090_workflow.sh --step all      # Execute full Stage 4 -> 5 -> 6 pipeline
#   ./run_rtx5090_workflow.sh --step 4        # Run Stage 4 DP-GEN active learning loop
#   ./run_rtx5090_workflow.sh --step 5        # Run Stage 5 DeePMD ensemble training & compression
#   ./run_rtx5090_workflow.sh --step 6        # Run Stage 6 LAMMPS 10^5-atom reactive dynamics
# ==============================================================================

set -e

# ANSI Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

STEP_TARGET="all"
MODE="docker"  # 'docker' or 'conda'

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --step) STEP_TARGET="$2"; shift ;;
        --mode) MODE="$2"; shift ;;
        -h|--help)
            echo "Usage: ./run_rtx5090_workflow.sh [--step 4|5|6|all] [--mode docker|conda]"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

log_section() {
    echo -e "\n${BLUE}================================================================================${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

START_TIME=$(date +%s)

log_section "AI_PHOSPHOGYPSUM RTX 5090 ACCELERATED PIPELINE"
echo -e "${CYAN}  Working Directory:${NC} ${SCRIPT_DIR}"
echo -e "${CYAN}  Execution Target:${NC}  Stage ${STEP_TARGET}"
echo -e "${CYAN}  Execution Mode:${NC}    ${MODE}"
echo -e "${CYAN}  Hardware:${NC}          NVIDIA GeForce RTX 5090 (32GB VRAM)"

# ==============================================================================
# STAGE 4: DP-GEN CONCURRENT ACTIVE LEARNING & GPU LABELING
# ==============================================================================
if [[ "${STEP_TARGET}" == "4" || "${STEP_TARGET}" == "all" ]]; then
    log_section "STAGE 4: DP-GEN CONCURRENT ACTIVE LEARNING ON RTX 5090"
    
    cd "${SCRIPT_DIR}/4.DPGEN_ActiveLearning/01.iter_configs"
    if [[ "${MODE}" == "docker" ]]; then
        log_info "Running DP-GEN via Docker Compose GPU..."
        dpgen run param.json machine_docker_gpu.json
    else
        log_info "Running DP-GEN via native Conda AI_phosphogypsum_env..."
        ./run_dpgen.sh
    fi
    cd "${SCRIPT_DIR}"
    log_success "Stage 4 Complete: Active learning iterative exploration & labeling finished."
fi

# ==============================================================================
# STAGE 5: PRODUCTION DEEPMD-KIT ENSEMBLE TRAINING & TABULATION COMPRESSION
# ==============================================================================
if [[ "${STEP_TARGET}" == "5" || "${STEP_TARGET}" == "all" ]]; then
    log_section "STAGE 5: DEEPMD-KIT PRODUCTION TRAINING & MODEL COMPRESSION"
    
    cd "${SCRIPT_DIR}/5.DeePMD_MLIP"
    if [[ "${MODE}" == "docker" ]]; then
        log_info "Training 4 ensemble models via Docker Compose GPU on RTX 5090..."
        ./01.training/train_compose_gpu.sh
        
        log_info "Compressing model into tabulated grid for 10x inference acceleration..."
        ./03.freeze_and_compress/test_and_compress.sh
    else
        log_info "Training production model via native Conda AI_phosphogypsum_env..."
        cd 01.training && ./run_dp_train.sh && cd ..
        cd 03.freeze_and_compress && python3 freeze_and_compress.py && cd ..
    fi
    cd "${SCRIPT_DIR}"
    log_success "Stage 5 Complete: Production Deep Potential frozen and compressed (frozen_model_compressed.pb)."
fi

# ==============================================================================
# STAGE 6: LARGE-SCALE HYDROTHERMAL MD & REACTION KINETICS
# ==============================================================================
if [[ "${STEP_TARGET}" == "6" || "${STEP_TARGET}" == "all" ]]; then
    log_section "STAGE 6: LARGE-SCALE (10^5 ATOMS) HYDROTHERMAL REACTIVE DYNAMICS"
    
    cd "${SCRIPT_DIR}/6.LAMMPS_ScalingUp"
    if [[ "${MODE}" == "docker" ]]; then
        log_info "Running 100k-atom reactive MD via Docker Compose GPU on RTX 5090..."
        ./02.lammps_simulation/run_lammps_compose_gpu.sh
    else
        log_info "Running LAMMPS via native Conda AI_phosphogypsum_env..."
        cd 02.lammps_simulation && ./run_lammps.sh && cd ..
    fi
    
    log_info "Analyzing dissolution kinetics, ion diffusion coefficients and PNC clustering..."
    cd 03.trajectory_analysis && python3 analyze_reaction_kinetics.py && cd ..
    
    cd "${SCRIPT_DIR}"
    log_success "Stage 6 Complete: Large-scale reactive dynamics and reaction kinetics analysis complete."
fi

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))

log_section "★ WORKFLOW EXECUTION SUMMARY"
echo -e "${GREEN}  All requested stages finished successfully!${NC}"
echo -e "${CYAN}  Total Elapsed Time: ${TOTAL_ELAPSED} seconds (${TOTAL_ELAPSED}s)${NC}"
echo -e "${BLUE}================================================================================${NC}\n"
