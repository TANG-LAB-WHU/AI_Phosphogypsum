#!/usr/bin/env python3
"""
sync_and_setup_geoopt_hpc.py
============================
Automated setup script for CP2K DFT Geometry Optimization jobs on WHU-HPC.
1. Directly inherits pristine initial structures from WorkingFolder/1.ModelConfig.
2. Generates r2SCAN + DFTD4 input files (geo_opt.inp) with 2D Martyna-Tuckerman Poisson solver.
3. Generates WHU-HPC Slurm submission scripts (run_cp2k.slurm) for partition 9a14a (192 MPI ranks).
4. Generates Master batch submission script (submit_all_jobs.sh).
"""

import os
import glob
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
MODELCONFIG_DIR = os.path.join(REPO_ROOT, "WorkingFolder", "1.ModelConfig")

SYSTEMS = [
    ("1.1.1CSO-2H2O", "2.1.1CSO-2H2O", False, "Gypsum (CaSO4-2H2O) Slab GeoOpt"),
    ("1.1.2CSO-0.583H2O", "2.1.2CSO-0.583H2O", False, "Hemihydrate-like Intermediate (CaSO4-0.583H2O) Slab GeoOpt"),
    ("1.1.2CSO-0.5H2O", "2.1.2CSO-0.5H2O", False, "Hemihydrate (CaSO4-0.5H2O) Slab GeoOpt"),
    ("1.1.2CSO-0.625H2O", "2.1.2CSO-0.625H2O", False, "Intermediate Hydrate (CaSO4-0.625H2O) Slab GeoOpt"),
    ("1.1.3CSO", "2.1.3CSO", False, "Anhydrite (CaSO4) Slab GeoOpt"),
    ("1.2.1CSO-2H2O+NH4", "2.2.1CSO-2H2O+NH4", True, "Dihydrate + NH4/PO4/H2O Confined Adsorption GeoOpt"),
    ("1.2.2CSO-0.583H2O+NH4", "2.2.2CSO-0.583H2O+NH4", True, "Intermediate 0.583H2O + NH4/PO4/H2O Confined Adsorption GeoOpt"),
    ("1.2.2CSO-0.5H2O+NH4", "2.2.2CSO-0.5H2O+NH4", True, "Hemihydrate 0.5H2O + NH4/PO4/H2O Confined Adsorption GeoOpt"),
    ("1.2.2CSO-0.625H2O+NH4", "2.2.2CSO-0.625H2O+NH4", True, "Intermediate 0.625H2O Strong Confinement + NH4/PO4/H2O GeoOpt"),
    ("1.2.3CSO+NH4", "2.2.3CSO+NH4", True, "Anhydrite + NH4/PO4/H2O Confined Adsorption GeoOpt"),
]

# CP2K KIND blocks
KIND_DEFINITIONS = {
    "Ca": """    &KIND Ca
      BASIS_SET DZVP-MOLOPT-SCAN-GTH-q10
      POTENTIAL GTH-SCAN-q10
    &END KIND""",
    "O": """    &KIND O
      BASIS_SET DZVP-MOLOPT-SCAN-GTH-q6
      POTENTIAL GTH-SCAN-q6
    &END KIND""",
    "S": """    &KIND S
      BASIS_SET DZVP-MOLOPT-SCAN-GTH-q6
      POTENTIAL GTH-SCAN-q6
    &END KIND""",
    "H": """    &KIND H
      BASIS_SET TZV2P-MOLOPT-SCAN-GTH-q1
      POTENTIAL GTH-SCAN-q1
    &END KIND""",
    "N": """    &KIND N
      BASIS_SET DZVP-MOLOPT-SCAN-GTH-q5
      POTENTIAL GTH-SCAN-q5
    &END KIND""",
    "P": """    &KIND P
      BASIS_SET DZVP-MOLOPT-SCAN-GTH-q5
      POTENTIAL GTH-SCAN-q5
    &END KIND""",
}

INP_TEMPLATE = """@SET PROJECT_NAME {PROJECT_NAME}
@SET EPS_SCF 1.5E-6
@SET CUT_OFF 420
@SET MAX_SCF 200

&GLOBAL
  PROJECT_NAME  ${{PROJECT_NAME}}
  RUN_TYPE      GEO_OPT
  PRINT_LEVEL   MEDIUM
&END GLOBAL

&FORCE_EVAL
  METHOD           QUICKSTEP
  &DFT
    BASIS_SET_FILE_NAME  BASIS_MOLOPT_UZH
    POTENTIAL_FILE_NAME  POTENTIAL_UZH
    CHARGE        0
    MULTIPLICITY  1

    &MGRID
      CUTOFF      ${{CUT_OFF}}
      REL_CUTOFF  60
      NGRIDS      5
    &END MGRID

    &QS
      METHOD           GPW
      EPS_DEFAULT      1.0E-12
      EXTRAPOLATION    ASPC
      EXTRAPOLATION_ORDER 3
    &END QS

    &POISSON
      PERIODIC    XY
      PSOLVER     MT
      &MT
        REL_CUTOFF 2.0
      &END MT
    &END POISSON

    &SCF
      SCF_GUESS    ATOMIC
      EPS_SCF      ${{EPS_SCF}}
      MAX_SCF      ${{MAX_SCF}}
      &OT
        MINIMIZER           DIIS
        PRECONDITIONER      FULL_SINGLE_INVERSE
        ENERGY_GAP          0.001
      &END OT
      &OUTER_SCF
        MAX_SCF    10
        EPS_SCF    ${{EPS_SCF}}
      &END OUTER_SCF
      &PRINT
        &RESTART
          BACKUP_COPIES  1
          &EACH
            GEO_OPT 1
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
        POTENTIAL_TYPE  PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE                  DFTD4
          PARAMETER_FILE_NAME   dftd4.dat
          REFERENCE_FUNCTIONAL  r2SCAN
          R_CUTOFF              15.0
        &END PAIR_POTENTIAL
      &END VDW_POTENTIAL
    &END XC
  &END DFT

  &SUBSYS
    &CELL
      A   {A1} {A2} {A3}
      B   {B1} {B2} {B3}
      C   {C1} {C2} {C3}
      PERIODIC  XY
    &END CELL
    &TOPOLOGY
      COORD_FILE_FORMAT  XYZ
      COORD_FILE_NAME    {XYZ_NAME}
    &END TOPOLOGY

{KIND_BLOCK}
  &END SUBSYS
&END FORCE_EVAL

&MOTION
  &GEO_OPT
    TYPE        MINIMIZATION
    OPTIMIZER   BFGS
    MAX_ITER    500
    MAX_DR      3.0E-3
    MAX_FORCE   4.5E-4
    RMS_DR      1.5E-3
    RMS_FORCE   3.0E-4
    &BFGS
      TRUST_RADIUS 0.2
    &END BFGS
  &END GEO_OPT
  &PRINT
    &TRAJECTORY
      FORMAT XYZ
      &EACH
        GEO_OPT 1
      &END EACH
    &END TRAJECTORY
    &RESTART
      BACKUP_COPIES 1
      &EACH
        GEO_OPT 1
      &END EACH
    &END RESTART
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
INPUT_FILE="geo_opt.inp"
OUTPUT_FILE="geo_opt.out"

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

# Run CP2K job
mpirun -np ${{SLURM_NTASKS}} --bind-to hwthread cp2k.psmp -i "${{INPUT_FILE}}" -o "${{OUTPUT_FILE}}"
"""

SUBMIT_ALL_TEMPLATE = """#!/bin/bash
# ==============================================================================
# Master Batch Submission Script for WHU-HPC Slurm Cluster
# Submits all 10 CP2K Geometry Optimization Jobs (Partition: 9a14a)
# ==============================================================================

SUBDIRS=(
{SUBDIR_LIST}
)

echo "========================================================================"
echo " Submitting 10 CP2K GEO_OPT Jobs to WHU-HPC (Partition: 9a14a)..."
echo "========================================================================"

for dir in "${{SUBDIRS[@]}}"; do
    if [ -d "$dir" ]; then
        echo -n "Submitting job in $dir ... "
        cd "$dir"
        if [ -f "run_cp2k.slurm" ]; then
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
    print("SYNCING 1.ModelConfig TO 2.CP2K_GeoOpt (CLEAN 3-FILE HPC SETUP)")
    print("=" * 70)
    
    subdir_names = []
    
    for src_sub, dst_sub, is_nh4, desc in SYSTEMS:
        src_dir = os.path.join(MODELCONFIG_DIR, src_sub)
        dst_dir = os.path.join(BASE_DIR, dst_sub)
        os.makedirs(dst_dir, exist_ok=True)
        subdir_names.append(f'    "{dst_sub}"')
        
        print(f"\nProcessing: {src_sub} -> {dst_sub}")
        
        # 1. Clean up old relics
        for f in os.listdir(dst_dir):
            if f not in [f"{dst_sub}.xyz", "geo_opt.inp", "run_cp2k.slurm"]:
                p = os.path.join(dst_dir, f)
                if os.path.isfile(p):
                    os.remove(p)
                    
        # 2. Extract structure directly from 1.ModelConfig
        xyzs = glob.glob(os.path.join(src_dir, "*.xyz"))
        if not xyzs:
            raise FileNotFoundError(f"Cannot find source structure in {src_dir}")
        source_xyz = xyzs[0]
        
        with open(source_xyz, "r") as f:
            base_lines = f.readlines()
            
        atom_count = int(base_lines[0].strip())
        comment = base_lines[1].strip()
        
        # Extract lattice
        m = re.search(r'Lattice="([^"]+)"', comment)
        if not m:
            raise ValueError(f"Could not find Lattice in {source_xyz}")
        lattice_str = m.group(1)
        lat_vals = lattice_str.split()
        a1, a2, a3 = lat_vals[0], lat_vals[1], lat_vals[2]
        b1, b2, b3 = lat_vals[3], lat_vals[4], lat_vals[5]
        c1, c2, c3 = lat_vals[6], lat_vals[7], lat_vals[8]
        
        # Format coordinate lines
        coords_lines = base_lines[2:2+atom_count]
        standard_header = [
            f"{atom_count}\n",
            f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3 pbc="T T F"\n'
        ]
        formatted_coords = []
        for l in coords_lines:
            parts = l.split()
            if len(parts) >= 4:
                sym = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                formatted_coords.append(f"{sym:<4} {x:18.10f} {y:18.10f} {z:18.10f}\n")
                
        out_lines = standard_header + formatted_coords
        
        # Write structure file named <dst_sub>.xyz
        xyz_name = f"{dst_sub}.xyz"
        out_xyz_path = os.path.join(dst_dir, xyz_name)
        with open(out_xyz_path, "w") as f:
            f.writelines(out_lines)
        print(f"   [Structure] Directly inherited {atom_count} atoms from 1.ModelConfig -> {xyz_name}")
        
        # Count elements & build KIND block
        elements = [l.split()[0] for l in formatted_coords]
        elem_counts = Counter(elements)
        print(f"   [Composition] {dict(elem_counts)}")
        kind_blocks = [KIND_DEFINITIONS[elem] for elem in ["Ca", "O", "S", "H", "N", "P"] if elem in elem_counts]
        kind_block_str = "\n".join(kind_blocks)
        
        # 3. Write geo_opt.inp
        project_name = f"gypsum_slab_{'nh4_' if is_nh4 else ''}geoopt"
        inp_content = INP_TEMPLATE.format(
            PROJECT_NAME=project_name,
            A1=a1, A2=a2, A3=a3,
            B1=b1, B2=b2, B3=b3,
            C1=c1, C2=c2, C3=c3,
            XYZ_NAME=xyz_name,
            KIND_BLOCK=kind_block_str,
        )
        with open(os.path.join(dst_dir, "geo_opt.inp"), "w") as f:
            f.write(inp_content)
        print(f"   [Input] Created geo_opt.inp (Lattice A=({a1},{a2},{a3}), B=({b1},{b2},{b3}), C=({c1},{c2},{c3}))")
        
        # 4. Write run_cp2k.slurm for WHU-HPC
        job_name = f"geo_{dst_sub.replace('CSO-', '').replace('CSO+', '').replace('H2O', 'w').replace('+', '_')}"
        slurm_content = SLURM_TEMPLATE.format(
            JOB_NAME=job_name,
        )
        with open(os.path.join(dst_dir, "run_cp2k.slurm"), "w", newline="\n") as f:
            f.write(slurm_content)
        print(f"   [WHU-HPC] Created run_cp2k.slurm (job-name: {job_name})")
        
    # 5. Master submit_all_jobs.sh in 2.CP2K_GeoOpt
    master_submit_path = os.path.join(BASE_DIR, "submit_all_jobs.sh")
    with open(master_submit_path, "w", newline="\n") as f:
        f.write(SUBMIT_ALL_TEMPLATE.format(SUBDIR_LIST="\n".join(subdir_names)))
    os.chmod(master_submit_path, 0o755)
    print(f"\n[Master] Successfully created WHU-HPC master submission script: {master_submit_path}")
    
    print("\n" + "=" * 70)
    print("ALL 10 SYSTEMS IN 2.CP2K_GeoOpt CLEANED & STANDARDIZED (3 FILES PER SYSTEM)!")
    print("=" * 70)

if __name__ == "__main__":
    main()
