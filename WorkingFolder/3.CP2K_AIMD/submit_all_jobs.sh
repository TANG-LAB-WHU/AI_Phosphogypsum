#!/bin/bash
# ==============================================================================
# Master Batch Submission Script for WHU-HPC Slurm Cluster
# Submits all 10 CP2K AIMD Simulation Jobs (180 ¡ãC, Partition: 9a14a)
# ==============================================================================

SUBDIRS=(
    "3.1.1CSO-2H2O"
    "3.1.2CSO-0.583H2O"
    "3.1.2CSO-0.5H2O"
    "3.1.2CSO-0.625H2O"
    "3.1.3CSO"
    "3.2.1CSO-2H2O+NH4"
    "3.2.2CSO-0.583H2O+NH4"
    "3.2.2CSO-0.5H2O+NH4"
    "3.2.2CSO-0.625H2O+NH4"
    "3.2.3CSO+NH4"
)

echo "========================================================================"
echo " Submitting 10 CP2K AIMD Jobs to WHU-HPC (180 ¡ãC, Partition: 9a14a)..."
echo "========================================================================"

for dir in "${SUBDIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -n "Submitting AIMD job in $dir ... "
        cd "$dir"
        if [ -f "run_cp2k.slurm" ]; then
            # Clean old failed logs if aborted previously
            if [ -f "aimd.out" ] && grep -q "ABORT" aimd.out 2>/dev/null; then
                rm -f aimd.out slurm-*.out slurm-*.error
            fi
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
