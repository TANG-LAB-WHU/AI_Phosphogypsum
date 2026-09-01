#!/usr/bin/env python3
"""
sync_and_setup_aimd_hpc.py
==========================
Automated setup script for CP2K AIMD simulation jobs on WHU-HPC.
Strictly unifies 2D Slab physics (PERIODIC XY + PSOLVER MT) across Stage 2, 3, and 4:
1. &POISSON and &CELL with PERIODIC XY and PSOLVER MT (Martyna-Tuckerman solver) to prevent spurious dipole interactions across vacuum.
2. Full observables output at &EACH MD 1 (&TRAJECTORY, &VELOCITIES, &FORCES, &ENERGY, &CELL) for DeePMD-kit training.
3. Strict alignment with 180 °C (453.15 K) hydrothermal conditions with CSVR thermostat.
4. Generates WHU-HPC Slurm submission scripts (run_cp2k.slurm) for partition 9a14a (192 MPI ranks).
5. Generates Master batch submission script (submit_all_jobs.sh).
"""

import os
import glob
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
GEOOPT_DIR = os.path.join(REPO_ROOT, "WorkingFolder", "2.CP2K_GeoOpt")
MODELCONFIG_DIR = os.path.join(REPO_ROOT, "WorkingFolder", "1.ModelConfig")

SYSTEMS = [
    ("2.1.1CSO-2H2O", "3.1.1CSO-2H2O", False, "Gypsum (CaSO4-2H2O) Slab Hydrothermal AIMD"),
    ("2.1.2CSO-0.583H2O", "3.1.2CSO-0.583H2O", False, "Hemihydrate-like Intermediate (CaSO4-0.583H2O) Slab AIMD"),
    ("2.1.2CSO-0.5H2O", "3.1.2CSO-0.5H2O", False, "Hemihydrate (CaSO4-0.5H2O) Slab AIMD"),
    ("2.1.2CSO-0.625H2O", "3.1.2CSO-0.625H2O", False, "Intermediate Hydrate (CaSO4-0.625H2O) Slab AIMD"),
    ("2.1.3CSO", "3.1.3CSO", False, "Anhydrite (CaSO4) Slab AIMD"),
    ("2.2.1CSO-2H2O+NH4", "3.2.1CSO-2H2O+NH4", True, "Dihydrate + NH4/PO4/H2O Confined Depolymerization AIMD"),
    ("2.2.2CSO-0.583H2O+NH4", "3.2.2CSO-0.583H2O+NH4", True, "Intermediate 0.583H2O + NH4/PO4/H2O Confined Depolymerization AIMD"),
    ("2.2.2CSO-0.5H2O+NH4", "3.2.2CSO-0.5H2O+NH4", True, "Hemihydrate 0.5H2O + NH4/PO4/H2O Confined Depolymerization AIMD"),
    ("2.2.2CSO-0.625H2O+NH4", "3.2.2CSO-0.625H2O+NH4", True, "Intermediate 0.625H2O Strong Confinement + NH4/PO4/H2O AIMD"),
    ("2.2.3CSO+NH4", "3.2.3CSO+NH4", True, "Anhydrite + NH4/PO4/H2O Confined Depolymerization AIMD"),
]

# CP2K KIND blocks
KIND_DEFINITIONS = {
    "Ca": """    &KIND Ca
      BASIS_SET ORB DZVP-MOLOPT-SCAN-GTH-q10
      POTENTIAL GTH-SCAN-q10
    &END KIND""",
    "O": """    &KIND O
      BASIS_SET ORB DZVP-MOLOPT-SCAN-GTH-q6
      POTENTIAL GTH-SCAN-q6
    &END KIND""",
    "S": """    &KIND S
      BASIS_SET ORB DZVP-MOLOPT-SCAN-GTH-q6
      POTENTIAL GTH-SCAN-q6
    &END KIND""",
    "H": """    &KIND H
      BASIS_SET ORB TZV2P-MOLOPT-SCAN-GTH-q1
      POTENTIAL GTH-SCAN-q1
    &END KIND""",
    "N": """    &KIND N
      BASIS_SET ORB DZVP-MOLOPT-SCAN-GTH-q5
      POTENTIAL GTH-SCAN-q5
    &END KIND""",
    "P": """    &KIND P
      BASIS_SET ORB DZVP-MOLOPT-SCAN-GTH-q5
      POTENTIAL GTH-SCAN-q5
    &END KIND""",
}

INP_TEMPLATE = """@SET PROJECT_NAME {PROJECT_NAME}
@SET EPS_SCF 1.0E-6
@SET CUT_OFF 420
@SET MAX_SCF 200
@SET TEMPERATURE 453.15  # 180 °C Hydrothermal condition (453.15 K)
@SET TIMESTEP 0.5
@SET STEPS 30000

&GLOBAL
  PROJECT_NAME  ${{PROJECT_NAME}}
  RUN_TYPE      MD
  PRINT_LEVEL   MEDIUM
  EXTENDED_FFT_LENGTHS  TRUE
&END GLOBAL

&FORCE_EVAL
  METHOD           QUICKSTEP
  &DFT
    BASIS_SET_FILE_NAME  BASIS_MOLOPT_UZH
    POTENTIAL_FILE_NAME  POTENTIAL_UZH
    CHARGE        0
    MULTIPLICITY  1

    &MGRID
      CUTOFF       ${{CUT_OFF}}
      REL_CUTOFF   60
      NGRIDS       4
    &END MGRID

    &QS
      METHOD           GPW
      EPS_DEFAULT      1.0E-12
      EXTRAPOLATION    ASPC
      EXTRAPOLATION_ORDER 4
    &END QS

    &POISSON
      PERIODIC    XY
      PSOLVER     MT
      &MT
        REL_CUTOFF 2.0
      &END MT
    &END POISSON

    &SCF
      MAX_SCF      ${{MAX_SCF}}
      EPS_SCF      ${{EPS_SCF}}
      SCF_GUESS    ATOMIC
      IGNORE_CONVERGENCE_FAILURE
      &OT
        MINIMIZER           DIIS
        LINESEARCH          3PNT
        PRECONDITIONER      FULL_SINGLE_INVERSE
        ENERGY_GAP          0.08
      &END OT
      &OUTER_SCF
        MAX_SCF     10
        EPS_SCF     ${{EPS_SCF}}
      &END OUTER_SCF
      &PRINT
        &RESTART
          BACKUP_COPIES 1
          &EACH
            MD 100
          &END EACH
        &END RESTART
      &END PRINT
    &END SCF

    &XC
      &XC_FUNCTIONAL
        &MGGA_X_R2SCAN
        &END MGGA_X_R2SCAN
        &MGGA_C_R2SCAN
        &END MGGA_C_R2SCAN
      &END XC_FUNCTIONAL
      &VDW_POTENTIAL
        POTENTIAL_TYPE PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE                  DFTD4
          PARAMETER_FILE_NAME   dftd4.dat
          REFERENCE_FUNCTIONAL  r2SCAN
          D4_CUTOFF             [angstrom] 25.0
          R_CUTOFF              [angstrom] 15.0
        &END PAIR_POTENTIAL
      &END VDW_POTENTIAL
    &END XC
  &END DFT

  &SUBSYS
    &TOPOLOGY
      COORD_FILE_NAME    {XYZ_NAME}
      COORD_FILE_FORMAT  XYZ
    &END TOPOLOGY

    &CELL
      A   {A1} {A2} {A3}
      B   {B1} {B2} {B3}
      C   {C1} {C2} {C3}
      PERIODIC  XY
    &END CELL

{KIND_BLOCK}
  &END SUBSYS
&END FORCE_EVAL

&MOTION
  &MD
    ENSEMBLE     NVT
    STEPS        ${{STEPS}}
    TIMESTEP     ${{TIMESTEP}}
    TEMPERATURE  ${{TEMPERATURE}}
    &THERMOSTAT
      TYPE  CSVR
      &CSVR
        TIMECON 20.0
      &END CSVR
    &END THERMOSTAT
    &PRINT
      &ENERGY
        &EACH
          MD 1
        &END EACH
      &END ENERGY
    &END PRINT
  &END MD
  &PRINT
    &TRAJECTORY
      FORMAT XYZ
      &EACH
        MD 1
      &END EACH
    &END TRAJECTORY
    &VELOCITIES
      FORMAT XYZ
      &EACH
        MD 1
      &END EACH
    &END VELOCITIES
    &FORCES
      FORMAT XYZ
      &EACH
        MD 1
      &END EACH
    &END FORCES
    &CELL
      &EACH
        MD 1
      &END EACH
    &END CELL
    &RESTART
      BACKUP_COPIES 1
      &EACH
        MD 100
      &END EACH
    &END RESTART
    &RESTART_HISTORY OFF
    &END RESTART_HISTORY
  &END PRINT
&END MOTION
"""

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={JOB_NAME}
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
cd "${{SLURM_SUBMIT_DIR:-$PWD}}"

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
mpirun -np ${{SLURM_NTASKS}} --bind-to hwthread cp2k.psmp -i "${{INPUT_FILE}}" -o "${{OUTPUT_FILE}}"
"""

SUBMIT_ALL_TEMPLATE = """#!/bin/bash
# ==============================================================================
# Master Batch Submission Script for WHU-HPC Slurm Cluster
# Submits all 10 CP2K AIMD Simulation Jobs (180 °C, Partition: 9a14a)
# ==============================================================================

SUBDIRS=(
{SUBDIR_LIST}
)

echo "========================================================================"
echo " Submitting 10 CP2K AIMD Jobs to WHU-HPC (180 °C, Partition: 9a14a)..."
echo "========================================================================"

for dir in "${{SUBDIRS[@]}}"; do
    if [ -d "$dir" ]; then
        echo -n "Submitting AIMD job in $dir ... "
        cd "$dir"
        if [ -f "run_cp2k.slurm" ]; then
            # Clean old failed logs if aborted previously
            if [ -f "aimd.out" ] && grep -q "ABORT" aimd.out 2>/dev/null; then
                rm -f aimd.out slurm-*.out slurm-*.error
            fi
            JOB_ID=$(sbatch run_cp2k.slurm | awk '{{print $NF}}')
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
"""

def main():
    print("=" * 70)
    print("UNIFYING 3.CP2K_AIMD WITH 2D SLAB PERIODICITY (PERIODIC XY + PSOLVER MT)")
    print("=" * 70)
    
    subdir_names = []
    
    for src_geo, dst_aimd, is_nh4, desc in SYSTEMS:
        geo_dir = os.path.join(GEOOPT_DIR, src_geo)
        dst_dir = os.path.join(BASE_DIR, dst_aimd)
        os.makedirs(dst_dir, exist_ok=True)
        subdir_names.append(f'    "{dst_aimd}"')
        
        print(f"\nProcessing: {src_geo} -> {dst_aimd}")
        
        # 1. Clean up old relics
        for f in os.listdir(dst_dir):
            if f not in [f"{dst_aimd}.xyz", "aimd.inp", "run_cp2k.slurm"]:
                p = os.path.join(dst_dir, f)
                if os.path.isfile(p):
                    os.remove(p)
                    
        # 2. Extract authentic optimized structure and lattice
        pos_files = glob.glob(os.path.join(geo_dir, "*-pos-1.xyz"))
        if not pos_files:
            raise FileNotFoundError(f"Could not locate trajectory file *-pos-1.xyz in {geo_dir}")
        pos_file = pos_files[0]
        
        with open(pos_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            
        atom_count = int(lines[0].strip())
        frame_len = atom_count + 2
        last_frame_lines = lines[-frame_len:]
        coord_raw_lines = last_frame_lines[2:]
        print(f"   [Source] Extracted final step from {os.path.basename(pos_file)} ({atom_count} atoms)")
        
        # Extract lattice from geo_opt.inp
        inp_path = os.path.join(geo_dir, "geo_opt.inp")
        with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
            
        m_a = re.search(r'^\s*A\s+([0-9\.\+\-E]+)\s+([0-9\.\+\-E]+)\s+([0-9\.\+\-E]+)', txt, re.M)
        m_b = re.search(r'^\s*B\s+([0-9\.\+\-E]+)\s+([0-9\.\+\-E]+)\s+([0-9\.\+\-E]+)', txt, re.M)
        m_c = re.search(r'^\s*C\s+([0-9\.\+\-E]+)\s+([0-9\.\+\-E]+)\s+([0-9\.\+\-E]+)', txt, re.M)
        
        if not (m_a and m_b and m_c):
            raise ValueError(f"Could not extract cell vectors from {inp_path}")
            
        a1, a2, a3 = m_a.group(1), m_a.group(2), m_a.group(3)
        b1, b2, b3 = m_b.group(1), m_b.group(2), m_b.group(3)
        c1, c2, c3 = m_c.group(1), m_c.group(2), m_c.group(3)
        lattice_str = f"{float(a1):.10f} {float(a2):.10f} {float(a3):.10f} {float(b1):.10f} {float(b2):.10f} {float(b3):.10f} {float(c1):.10f} {float(c2):.10f} {float(c3):.10f}"
        
        # Format coordinate lines
        standard_header = [
            f"{atom_count}\n",
            f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3 pbc="T T F"\n'
        ]
        formatted_coords = []
        for l in coord_raw_lines:
            parts = l.split()
            if len(parts) >= 4:
                sym = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                formatted_coords.append(f"{sym:<4} {x:18.10f} {y:18.10f} {z:18.10f}\n")
                
        if len(formatted_coords) != atom_count:
            raise ValueError(f"Formatted coords count {len(formatted_coords)} does not match atom count {atom_count}")
            
        out_lines = standard_header + formatted_coords
        
        # Write structure file named <dst_aimd>.xyz
        xyz_name = f"{dst_aimd}.xyz"
        out_xyz_path = os.path.join(dst_dir, xyz_name)
        with open(out_xyz_path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)
        print(f"   [Structure] Wrote {atom_count} atoms -> {xyz_name}")
        
        # Count elements & build KIND block
        elements = [l.split()[0] for l in formatted_coords]
        elem_counts = Counter(elements)
        print(f"   [Composition] {dict(elem_counts)}")
        kind_blocks = [KIND_DEFINITIONS[elem] for elem in ["Ca", "O", "S", "H", "N", "P"] if elem in elem_counts]
        kind_block_str = "\n".join(kind_blocks)
        
        # 3. Write aimd.inp with UNIFIED 2D PERIODIC XY + PSOLVER MT
        project_name = f"gypsum_slab_{'nh4_' if is_nh4 else ''}aimd"
        inp_content = INP_TEMPLATE.format(
            PROJECT_NAME=project_name,
            A1=a1, A2=a2, A3=a3,
            B1=b1, B2=b2, B3=b3,
            C1=c1, C2=c2, C3=c3,
            XYZ_NAME=xyz_name,
            KIND_BLOCK=kind_block_str,
        )
        with open(os.path.join(dst_dir, "aimd.inp"), "w") as f:
            f.write(inp_content)
        print(f"   [Input] Created aimd.inp (UNIFIED PERIODIC XY + PSOLVER MT)")
        
        # 4. Write run_cp2k.slurm for WHU-HPC
        job_name = f"aimd_{dst_aimd.replace('CSO-', '').replace('CSO+', '').replace('H2O', 'w').replace('+', '_')}"
        slurm_content = SLURM_TEMPLATE.format(
            JOB_NAME=job_name,
        )
        with open(os.path.join(dst_dir, "run_cp2k.slurm"), "w", newline="\n") as f:
            f.write(slurm_content)
        print(f"   [WHU-HPC] Created run_cp2k.slurm (job-name: {job_name})")
        
    # 5. Master submit_all_jobs.sh in 3.CP2K_AIMD
    master_submit_path = os.path.join(BASE_DIR, "submit_all_jobs.sh")
    with open(master_submit_path, "w", newline="\n") as f:
        f.write(SUBMIT_ALL_TEMPLATE.format(SUBDIR_LIST="\n".join(subdir_names)))
    os.chmod(master_submit_path, 0o755)
    print(f"\n[Master] Successfully created WHU-HPC master AIMD submission script: {master_submit_path}")
    
    print("\n" + "=" * 70)
    print("ALL 10 SYSTEMS IN 3.CP2K_AIMD UNIFIED TO 2D PERIODIC XY + PSOLVER MT!")
    print("=" * 70)

if __name__ == "__main__":
    main()
