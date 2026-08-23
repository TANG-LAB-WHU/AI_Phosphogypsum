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
        echo -n "Submitting job in $dir ... "
        cd "$dir"
        if [ -f "run_cp2k.slurm" ]; then
            JOB_ID=$(sbatch run_cp2k.slurm | awk '{print $NF}')
            echo "Submitted! Job ID: $JOB_ID"
        else
            echo "ERROR: run_cp2k.slurm not found in $dir"
        fi
        cd ..
    else
        echo "WARNING: Directory $dir does not exist."
    fi
done

echo "========================================================================"
echo "All jobs submitted! Use 'squeue -u $USER' to monitor status."
echo "========================================================================"
