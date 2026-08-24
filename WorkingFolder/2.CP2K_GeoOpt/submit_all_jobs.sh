#!/bin/bash
# ==============================================================================
# Master Batch Submission Script for WHU-HPC Slurm Cluster
# Submits all 10 CP2K Geometry Optimization Jobs (Partition: 9a14a)
# ==============================================================================

SUBDIRS=(
    "2.1.1CSO-2H2O"
    "2.1.2CSO-0.583H2O"
    "2.1.2CSO-0.5H2O"
    "2.1.2CSO-0.625H2O"
    "2.1.3CSO"
    "2.2.1CSO-2H2O+NH4"
    "2.2.2CSO-0.583H2O+NH4"
    "2.2.2CSO-0.5H2O+NH4"
    "2.2.2CSO-0.625H2O+NH4"
    "2.2.3CSO+NH4"
)

echo "========================================================================"
echo " Submitting 10 CP2K GEO_OPT Jobs to WHU-HPC (Partition: 9a14a)..."
echo "========================================================================"

for dir in "${SUBDIRS[@]}"; do
    if [ -d "$dir" ]; then
        cd "$dir"
        if compgen -G "*FINAL*" > /dev/null; then
            echo "[$dir] SKIPPING: Already completed and converged (FINAL file exists)."
        elif [ -f "run_cp2k.slurm" ]; then
            echo -n "[$dir] Submitting job ... "
            JOB_ID=$(sbatch run_cp2k.slurm | awk '{print $NF}')
            echo "Submitted! Job ID: $JOB_ID"
        else
            echo "[$dir] ERROR: run_cp2k.slurm not found."
        fi
        cd ..
    else
        echo "WARNING: Directory $dir does not exist."
    fi
done

echo "========================================================================"
echo "All jobs submitted! Use 'squeue -u $USER' to monitor status."
echo "========================================================================"
