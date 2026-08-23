#!/usr/bin/env python3
"""
align_and_generate_aimd.py
==========================
Automated alignment and setup script for CP2K AIMD simulations.
Aligns structures from WorkingFolder/2.CP2K_GeoOpt to WorkingFolder/3.CP2K_AIMD.
"""

import os
import shutil
import re
import glob
from collections import Counter
from ase.io import read, write
from ase.atoms import Atoms
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
GEOOPT_DIR = os.path.join(REPO_ROOT, "WorkingFolder", "2.CP2K_GeoOpt")
MODELCONFIG_DIR = os.path.join(REPO_ROOT, "WorkingFolder", "1.ModelConfig")

# System mapping: (2b source dir, 4 target dir, is_nh4_system, project_desc)
SYSTEM_SPECS = [
    ("2.1.1CSO-2H2O", "3.1.1CSO-2H2O", False, "Gypsum (CaSO4-2H2O) Slab AIMD"),
    ("2.1.2CSO-0.583H2O", "3.1.2CSO-0.583H2O", False, "Hemihydrate-like Intermediate (CaSO4-0.583H2O) Slab AIMD"),
    ("2.1.2CSO-0.5H2O", "3.1.2CSO-0.5H2O", False, "Hemihydrate (CaSO4-0.5H2O) Slab AIMD"),
    ("2.1.2CSO-0.625H2O", "3.1.2CSO-0.625H2O", False, "Intermediate Hydrate (CaSO4-0.625H2O) Slab AIMD"),
    ("2.1.3CSO", "3.1.3CSO", False, "Anhydrite (CaSO4) Slab AIMD"),
    ("2.2.1CSO-2H2O+NH4", "3.2.1CSO-2H2O+NH4", True, "Dihydrate + NH4/PO4/H2O Confined Depolymerization AIMD"),
    ("2.2.2CSO-0.583H2O+NH4", "3.2.2CSO-0.583H2O+NH4", True, "Intermediate 0.583H2O + NH4/PO4/H2O Confined Depolymerization AIMD"),
    ("2.2.2CSO-0.5H2O+NH4", "3.2.2CSO-0.5H2O+NH4", True, "Hemihydrate 0.5H2O + NH4/PO4/H2O Confined Depolymerization AIMD"),
    ("2.2.2CSO-0.625H2O+NH4", "3.2.2CSO-0.625H2O+NH4", True, "Intermediate 0.625H2O + NH4/PO4/H2O Confined Depolymerization AIMD"),
    ("2.2.3CSO+NH4", "3.2.3CSO+NH4", True, "Anhydrite + NH4/PO4/H2O Confined Depolymerization AIMD"),
]

# CP2K KIND definitions
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
@SET EPS_SCF 1.0E-6
@SET CUT_OFF 420
@SET MAX_SCF 200
@SET TEMPERATURE 453.15 ! 180 °C Hydrothermal condition
@SET TIMESTEP 0.5
@SET STEPS 20000

&GLOBAL
  PROJECT_NAME  ${{PROJECT_NAME}}
  RUN_TYPE      MD
  PRINT_LEVEL   MEDIUM
&END GLOBAL

&FORCE_EVAL
  METHOD           QUICKSTEP
  &DFT
    BASIS_SET_FILE_NAME  BASIS_MOLOPT_UZH
    POTENTIAL_FILE_NAME  POTENTIAL_UZH
    CHARGE        0
    MULTIPLICITY  1
    UKS
    &MGRID
      CUTOFF       ${{CUT_OFF}}
      REL_CUTOFF   60
      NGRIDS       5
    &END MGRID
    &QS
      METHOD       GPW     
      EPS_DEFAULT  1.0E-12
      EPS_PGF_ORB  1.0E-14
      EXTRAPOLATION ASPC
      EXTRAPOLATION_ORDER 3
    &END QS
    &POISSON
      PERIODIC XY
      PSOLVER  MT
    &END POISSON
    &SCF
      MAX_SCF      ${{MAX_SCF}}
      EPS_SCF      ${{EPS_SCF}}
      SCF_GUESS    ATOMIC
      IGNORE_CONVERGENCE_FAILURE
      &OT
        MINIMIZER       DIIS
        PRECONDITIONER  FULL_SINGLE_INVERSE
        ENERGY_GAP      0.001
        STEPSIZE        0.15
        PRECOND_SOLVER  INVERSE_UPDATE
      &END OT
      &OUTER_SCF
        MAX_SCF     10
        EPS_SCF     ${{EPS_SCF}}
      &END OUTER_SCF
      &PRINT
        &RESTART
          BACKUP_COPIES 1
          &EACH
            MD 10
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
          TYPE           DFTD4
          REFERENCE_FUNCTIONAL r2SCAN
          R_CUTOFF       15.0
          D4_CUTOFF      30.0
        &END PAIR_POTENTIAL
      &END VDW_POTENTIAL
      &XC_GRID
        XC_DERIV    SPLINE3_SMOOTH
        XC_SMOOTH_RHO  SPLINE3
      &END XC_GRID
    &END XC
    &PRINT
      &E_DENSITY_CUBE OFF
      &END E_DENSITY_CUBE
      &MO_CUBES OFF
      &END MO_CUBES
      &MULLIKEN ON
        &EACH
          MD 20
        &END EACH
      &END MULLIKEN
      &HIRSHFELD ON
        &EACH
          MD 20
        &END EACH
      &END HIRSHFELD
      &MOMENTS ON
        &EACH
          MD 10
        &END EACH
      &END MOMENTS
    &END PRINT
  &END DFT
  
  &SUBSYS
    &CELL
      A   {A1} {A2} {A3}
      B   {B1} {B2} {B3}
      C   {C1} {C2} {C3}
      PERIODIC  XY
    &END CELL
    &TOPOLOGY
      COORD_FILE_NAME   geoopt_optimized_structure_extxyz_wrap.xyz
      COORD_FILE_FORMAT XYZ
    &END TOPOLOGY
{KIND_BLOCK}
  &END SUBSYS
&END FORCE_EVAL

&MOTION
  &MD
    ENSEMBLE NVT
    STEPS ${{STEPS}}
    TIMESTEP ${{TIMESTEP}}
    TEMPERATURE ${{TEMPERATURE}}
    &THERMOSTAT
       REGION GLOBAL
       TYPE CSVR
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
        MD 10
      &END EACH
    &END RESTART
    &RESTART_HISTORY OFF
    &END RESTART_HISTORY
  &END PRINT
&END MOTION
"""

DOCKER_COMPOSE_TEMPLATE = """x-gpu: &gpu
  deploy:
    resources:
      reservations:
        devices:
          - capabilities: [ gpu ]

services:
  cp2k_aimd:
    image: mycp2k-rtx5080:master_mpich_native_cuda_A100_psmp
    #image: mycp2k-rtx5090:master_mpich_native_cuda_A100_psmp
    working_dir: /mnt/cp2k
    volumes:
      - .:/mnt/cp2k # Mount current directory for input/output
    environment:
      - OMP_NUM_THREADS=16
    command: mpirun -np 1 cp2k.psmp -i aimd.inp -o aimd.log
    restart: "no"
    <<: *gpu
"""

BAT_TEMPLATE = """@echo off
REM ====================================================================
REM  Terminal Logging Wrapper
REM ====================================================================
if "%1"=="--no-log-wrapper" (
    shift
    goto :main
)

echo [Info] Starting simulation with terminal logging to aimd_terminal.log...
%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -NoProfile -Command "& {{ .\\%~nx0 --no-log-wrapper | Tee-Object -FilePath 'aimd_terminal.log' }}"
exit /b

:main
REM ====================================================================
REM  Windows batch script to launch a GPU-accelerated CP2K AIMD
REM  simulation using Docker
REM ====================================================================

REM --------------------------------------------------------------------
REM 0. Display startup information
REM --------------------------------------------------------------------
echo ====================================================================
echo  CP2K AIMD Simulation Runner
echo  Project: {PROJECT_DESC}
echo ====================================================================
echo.

REM --------------------------------------------------------------------
REM 1. Check required files exist
REM --------------------------------------------------------------------
echo [Step 1.0] Checking required files...

if not exist "aimd.inp" (
    echo ERROR: aimd.inp not found. Aborting.
    pause
    exit /b 1
)

if not exist "geoopt_optimized_structure_extxyz_wrap.xyz" (
    echo ERROR: Structure file geoopt_optimized_structure_extxyz_wrap.xyz not found. Aborting.
    pause
    exit /b 1
)

if not exist "docker-compose-aimd.yml" (
    echo ERROR: docker-compose-aimd.yml not found. Aborting.
    pause
    exit /b 1
)

echo    - aimd.inp: OK
echo    - geoopt_optimized_structure_extxyz_wrap.xyz: OK
echo    - docker-compose-aimd.yml: OK
echo.

REM --------------------------------------------------------------------
REM 1.1 Extract last frame from trajectory and update coordinate file if available
REM --------------------------------------------------------------------
set TRAJ_FILE={TRAJ_FILE}
set STRUCT_FILE=geoopt_optimized_structure_extxyz_wrap.xyz

if exist "%TRAJ_FILE%" (
    echo [Step 1.1] Extracting last frame from %TRAJ_FILE%...
    %SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -NoProfile -Command "$traj = Get-Content '%TRAJ_FILE%'; $atomCount = [int]$traj[0].Trim(); $totalLines = $traj.Count; $lastFrameStart = $totalLines - ($atomCount + 2); if ($lastFrameStart -ge 0) {{ $coords = $traj[$($lastFrameStart+2)..($totalLines-1)]; $header = Get-Content '%STRUCT_FILE%' -TotalCount 2; $newContent = $header + $coords; $utf8NoBom = New-Object System.Text.UTF8Encoding($false); [IO.File]::WriteAllLines((Join-Path $PWD '%STRUCT_FILE%'), [string[]]$newContent, $utf8NoBom); Write-Host '   Structure coordinates updated from last trajectory frame.' -ForegroundColor Green }} else {{ Write-Host '   ERROR: Could not find complete last frame in trajectory.' -ForegroundColor Red }}"
) else (
    echo [Step 1.1] Trajectory file %TRAJ_FILE% not present. Using pre-aligned geoopt_optimized_structure_extxyz_wrap.xyz.
)

echo.

REM --------------------------------------------------------------------
REM 1.2 Update cell vectors in aimd.inp from extxyz Lattice
REM --------------------------------------------------------------------
echo [Step 1.2] Syncing cell vectors from optimized structure...

%SystemRoot%\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -NoProfile -Command "$xyz = Get-Content 'geoopt_optimized_structure_extxyz_wrap.xyz' | Select-Object -Index 1; $match = [regex]::Match($xyz, 'Lattice=[\\x22]([^\\x22]+)[\\x22]'); if ($match.Success) {{ $vals = $match.Groups[1].Value -split ' '; $A = '      A   ' + $vals[0] + ' ' + $vals[1] + ' ' + $vals[2]; $B = '      B   ' + $vals[3] + ' ' + $vals[4] + ' ' + $vals[5]; $C = '      C   ' + $vals[6] + ' ' + $vals[7] + ' ' + $vals[8]; $inp = Get-Content 'aimd.inp'; $newInp = @(); foreach ($line in $inp) {{ if ($line -match '^\\s*A\\s+') {{ $newInp += $A }} elseif ($line -match '^\\s*B\\s+') {{ $newInp += $B }} elseif ($line -match '^\\s*C\\s+') {{ $newInp += $C }} else {{ $newInp += $line }} }}; $utf8NoBom = New-Object System.Text.UTF8Encoding($false); [IO.File]::WriteAllLines((Join-Path $PWD 'aimd.inp'), [string[]]$newInp, $utf8NoBom); Write-Host '   Cell vectors updated:' -ForegroundColor Green; Write-Host ('   A: ' + $vals[0] + ' ' + $vals[1] + ' ' + $vals[2]); Write-Host ('   B: ' + $vals[3] + ' ' + $vals[4] + ' ' + $vals[5]); Write-Host ('   C: ' + $vals[6] + ' ' + $vals[7] + ' ' + $vals[8]) }} else {{ Write-Host '   WARNING: Could not parse Lattice from extxyz file' -ForegroundColor Yellow }}"

echo.

REM --------------------------------------------------------------------
REM 2. Clean up any previous Docker containers
REM --------------------------------------------------------------------
echo [Step 2] Cleaning up previous Docker containers...
docker compose -f docker-compose-aimd.yml down --remove-orphans 2>nul
echo.

REM --------------------------------------------------------------------
REM 3. Launch CP2K in Docker with GPU support
REM --------------------------------------------------------------------
echo [Step 3] Launching CP2K AIMD simulation...
echo    Image: mycp2k-rtx5080:master_mpich_native_cuda_A100_psmp
echo    Input: aimd.inp
echo    Output: aimd.log
echo.
echo Starting CP2K container...
echo ====================================================================
echo.

docker compose -f docker-compose-aimd.yml up --abort-on-container-exit

REM --------------------------------------------------------------------
REM 4. Post-run information
REM --------------------------------------------------------------------
echo.
echo ====================================================================
echo  CP2K job completed.
echo  Check aimd.log for output details.
echo ====================================================================

pause
"""

SH_TEMPLATE = """#!/usr/bin/env bash
set -e

echo "===================================================================="
echo " CP2K AIMD Simulation Runner (Linux/Docker/Server)"
echo " Project: {PROJECT_DESC}"
echo "===================================================================="

# Check required files
for f in aimd.inp geoopt_optimized_structure_extxyz_wrap.xyz docker-compose-aimd.yml; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file $f not found!" >&2
        exit 1
    fi
done

echo "[Step 1] Cleaning up old containers..."
docker compose -f docker-compose-aimd.yml down --remove-orphans 2>/dev/null || true

echo "[Step 2] Launching CP2K AIMD simulation container..."
docker compose -f docker-compose-aimd.yml up --abort-on-container-exit

echo "===================================================================="
echo " CP2K job completed. Check aimd.log for details."
echo "===================================================================="
"""

def extract_lattice_and_coords(src_geo_dir, sys_name):
    """
    Extracts the best structure (with latest trajectory frame if available)
    and authentic extxyz Lattice parameters from 2.CP2K_GeoOpt or 1.ModelConfig.
    """
    source_file = None
    if os.path.exists(src_geo_dir):
        opt_xyz = os.path.join(src_geo_dir, "optimized_structure_extxyz_wrap.xyz")
        if os.path.exists(opt_xyz):
            source_file = opt_xyz
            
    if not source_file:
        # Check in 1.ModelConfig
        candidates = [
            "1." + sys_name[2:],
            "1." + sys_name[2:].replace("+NH4_improved", "_improved+NH4"),
            "1." + sys_name[2:].replace("_improved+NH4", "+NH4_improved"),
            "1." + sys_name[2:].replace("_3", "+NH4_3"),
            "1." + sys_name[2:].replace("+NH4_improved_3", "_improved+NH4_3"),
            "1." + sys_name[2:].replace("_improved_3", "_improved+NH4_3"),
        ]
        for cand in candidates:
            md = os.path.join(MODELCONFIG_DIR, cand)
            if os.path.exists(md):
                xyzs = glob.glob(os.path.join(md, "*.xyz"))
                if xyzs:
                    source_file = xyzs[0]
                    break
    if not source_file:
        raise FileNotFoundError(f"Cannot find source structure for {sys_name}")
        
    with open(source_file, "r") as f:
        base_lines = f.readlines()
        
    atom_count = int(base_lines[0].strip())
    comment = base_lines[1].strip()
    
    # Extract lattice
    m = re.search(r'Lattice="([^"]+)"', comment)
    if not m:
        raise ValueError(f"Could not find Lattice in {source_file}")
    lattice_str = m.group(1)
    
    # Check for trajectory in 2.CP2K_GeoOpt
    trajs = glob.glob(os.path.join(src_geo_dir, "*-pos-1.xyz")) if os.path.exists(src_geo_dir) else []
    traj_path = None
    coords_lines = base_lines[2:2+atom_count]
    
    if trajs:
        traj_path = trajs[0]
        try:
            with open(traj_path, "r") as f:
                traj_lines = f.readlines()
            frame_len = atom_count + 2
            if len(traj_lines) >= frame_len:
                coords_lines = traj_lines[-atom_count:]
        except Exception as e:
            print(f"Warning parsing trajectory in {src_geo_dir}: {e}")

    # Standard clean extxyz format (species:S:1:pos:R:3)
    standard_header = [
        f"{atom_count}\n",
        f'Lattice="{lattice_str}" Properties=species:S:1:pos:R:3 pbc="T T F"\n'
    ]
    
    # Format coordinate lines: Symbol X Y Z
    formatted_coords = []
    for l in coords_lines:
        parts = l.split()
        if len(parts) >= 4:
            sym = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            formatted_coords.append(f"{sym:<4} {x:18.10f} {y:18.10f} {z:18.10f}\n")
            
    out_lines = standard_header + formatted_coords
    return out_lines, traj_path, lattice_str


def main():
    print(f"Starting alignment from 2b.GeoOpt to 3.CP2K_AIMD...")
    
    for src_name, dst_name, is_nh4, proj_desc in SYSTEM_SPECS:
        src_geo_dir = os.path.join(GEOOPT_DIR, src_name)
        
        dst_dir = os.path.join(BASE_DIR, dst_name)
        os.makedirs(dst_dir, exist_ok=True)
        
        print(f"\nProcessing: {src_name} -> {dst_name}")
        
        # 1. Structure file
        xyz_lines, traj_path, lattice_str = extract_lattice_and_coords(src_geo_dir, src_name)
        out_xyz_path = os.path.join(dst_dir, "geoopt_optimized_structure_extxyz_wrap.xyz")
        with open(out_xyz_path, "w") as f:
            f.writelines(xyz_lines)
        print(f"   [Structure] Wrote {len(xyz_lines)-2} atoms to geoopt_optimized_structure_extxyz_wrap.xyz")
        
        # Copy trajectory if present
        traj_filename = "gypsum_slab_nh4_geoopt-pos-1.xyz" if is_nh4 else "gypsum_slab_geoopt-pos-1.xyz"
        if traj_path and os.path.exists(traj_path):
            shutil.copy2(traj_path, os.path.join(dst_dir, traj_filename))
            print(f"   [Trajectory] Copied {os.path.basename(traj_path)}")
            
        # Parse Lattice components
        lat_vals = lattice_str.split()
        a1, a2, a3 = lat_vals[0], lat_vals[1], lat_vals[2]
        b1, b2, b3 = lat_vals[3], lat_vals[4], lat_vals[5]
        c1, c2, c3 = lat_vals[6], lat_vals[7], lat_vals[8]
        
        # Count elements
        elements = [l.split()[0] for l in xyz_lines[2:]]
        elem_counts = Counter(elements)
        print(f"   [Composition] {dict(elem_counts)}")
        
        # Build KIND block
        kind_blocks = []
        for elem in ["Ca", "O", "S", "H", "N", "P"]:
            if elem in elem_counts:
                kind_blocks.append(KIND_DEFINITIONS[elem])
        kind_block_str = "\n".join(kind_blocks)
        
        # 2. Write aimd.inp
        project_name = f"gypsum_slab_{'nh4_' if is_nh4 else ''}aimd"
        inp_content = INP_TEMPLATE.format(
            PROJECT_NAME=project_name,
            A1=a1, A2=a2, A3=a3,
            B1=b1, B2=b2, B3=b3,
            C1=c1, C2=c2, C3=c3,
            KIND_BLOCK=kind_block_str,
        )
        with open(os.path.join(dst_dir, "aimd.inp"), "w") as f:
            f.write(inp_content)
        print(f"   [Input] Created aimd.inp with lattice: A=({a1},{a2},{a3}), B=({b1},{b2},{b3}), C=({c1},{c2},{c3})")
        
        # 3. Write docker-compose-aimd.yml
        with open(os.path.join(dst_dir, "docker-compose-aimd.yml"), "w") as f:
            f.write(DOCKER_COMPOSE_TEMPLATE)
        print(f"   [Docker] Created docker-compose-aimd.yml")
        
        # 4. Write run_cp2k_aimd.bat
        bat_content = BAT_TEMPLATE.format(
            PROJECT_DESC=proj_desc,
            TRAJ_FILE=traj_filename,
        )
        with open(os.path.join(dst_dir, "run_cp2k_aimd.bat"), "w", newline="\r\n") as f:
            f.write(bat_content)
        print(f"   [Script] Created run_cp2k_aimd.bat")
        
        # 5. Write run_cp2k_aimd.sh
        sh_content = SH_TEMPLATE.format(
            PROJECT_DESC=proj_desc,
        )
        sh_path = os.path.join(dst_dir, "run_cp2k_aimd.sh")
        with open(sh_path, "w", newline="\n") as f:
            f.write(sh_content)
        os.chmod(sh_path, 0o755)
        print(f"   [Script] Created run_cp2k_aimd.sh")
        
    print("\nAll 14 systems aligned and generated successfully!")

if __name__ == "__main__":
    main()
