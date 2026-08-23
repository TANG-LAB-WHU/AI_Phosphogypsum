#!/bin/bash
#SBATCH --job-name=aimd_3.1.12w
#SBATCH --partition=9a14a
#SBATCH --account=tangsiqi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.error

# User config
INPUT_FILE="aimd.inp"
OUTPUT_FILE="aimd.out"

# Activate CP2K v2026.2 environment on WHU-HPC
source /scratch/tangsiqi/CP2K_pkg/cp2k-v202602/install/bin/cp2k_env.sh

# Relocate to submit directory
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Set stack & memory limits
ulimit -s unlimited
export OMP_STACKSIZE=128M

# Pure MPI parallelism on 192 hardware threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export MKL_NUM_THREADS=1

# OpenMPI 5.0 + UCX 1.19 settings
export OMPI_MCA_pml=ucx
export UCX_TLS=^mm
export PRTE_MCA_hwloc_default_binding_policy=none
export FI_PROVIDER='^psm2,psm'
export GFORTRAN_ERROR_BACKTRACE=1

# Run CP2K AIMD job
mpirun -np ${SLURM_NTASKS} --bind-to hwthread cp2k.psmp -i "${INPUT_FILE}" -o "${OUTPUT_FILE}"
