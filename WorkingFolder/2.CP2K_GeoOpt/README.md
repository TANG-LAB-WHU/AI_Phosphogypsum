# 2. CP2K DFT Geometry Optimization Center

This directory manages **first-principles Density Functional Theory (DFT) geometry optimization** tasks for all 10 standard phosphogypsum slab systems.

All initial atomic structures in this directory are **directly inherited from `WorkingFolder/1.ModelConfig`**, ensuring 100% fidelity without any unphysical pre-conditioning or MACE relics.

---

## System Overview (All 10 Systems)

| Subdirectory | Chemical Formula | Total Atoms | Lattice Dimensions ($a \times b \times c$, Å) | Physical & Chemical Description |
| :--- | :--- | :--- | :--- | :--- |
| **`2.1.1CSO-2H2O`** | $H_{64}Ca_{16}O_{96}S_{16}$ | 192 | $11.36 \times 12.57 \times 29.53$ | Pure gypsum dihydrate slab |
| **`2.1.2CSO-0.583H2O`** | $H_{64}Ca_{48}O_{224}S_{48}$ | 384 | $25.50 \times 23.97 \times 21.90$ | Pure intermediate / hemihydrate-like slab |
| **`2.1.2CSO-0.5H2O`** | $H_{24}Ca_{24}O_{108}S_{24}$ | 180 | $25.51 \times 13.85 \times 20.43$ | Pure bassanite (hemihydrate) slab |
| **`2.1.2CSO-0.625H2O`** | $H_{48}Ca_{48}O_{216}S_{48}$ | 360 | $24.07 \times 35.04 \times 21.73$ | Pure intermediate hydrate slab ($\gamma = 133.66^\circ$) |
| **`2.1.3CSO`** | $Ca_{16}O_{64}S_{16}$ | 96 | $12.77 \times 14.11 \times 20.91$ | Pure anhydrite (anhydrous) slab |
| **`2.2.1CSO-2H2O+NH4`** | $H_{131}Ca_{16}N_{7}O_{124}P_{2}S_{16}$ | 296 | $11.36 \times 12.57 \times 60.05$ | Dihydrate confined adsorption slab |
| **`2.2.2CSO-0.583H2O+NH4`** | $H_{131}Ca_{48}N_{7}O_{252}P_{2}S_{48}$ | 488 | $25.50 \times 23.97 \times 43.86$ | Intermediate 0.583H2O confined adsorption slab |
| **`2.2.2CSO-0.5H2O+NH4`** | $H_{91}Ca_{24}N_{7}O_{136}P_{2}S_{24}$ | 284 | $25.51 \times 13.85 \times 42.03$ | Hemihydrate 0.5H2O confined adsorption slab |
| **`2.2.2CSO-0.625H2O+NH4`** | $H_{115}Ca_{48}N_{7}O_{244}P_{2}S_{48}$ | 464 | $24.07 \times 35.04 \times 36.83$ | Sub-nanometer strong confinement adsorption slab |
| **`2.2.3CSO+NH4`** | $H_{67}Ca_{16}N_{7}O_{92}P_{2}S_{16}$ | 200 | $12.77 \times 14.11 \times 46.17$ | Anhydrite confined adsorption slab |

---

## Technical Specifications (CP2K Quickstep GPW)

- **Exchange-Correlation Functional**: `r2SCAN` Meta-GGA (`&MGGA_X_R2SCAN` + `&MGGA_C_R2SCAN`).
- **Dispersion Correction**: Grimme `DFTD4` pair potential (`TYPE DFTD4`, `REFERENCE_FUNCTIONAL r2SCAN`, cutoff 15.0 Å).
- **Basis Sets & Pseudopotentials**: `MOLOPT_UZH` basis sets (`DZVP-MOLOPT-SCAN-GTH` / `TZV2P-MOLOPT-SCAN-GTH-q1` for H) and `GTH-SCAN` pseudopotentials.
- **Electrostatic Boundary**: `&CELL PERIODIC XY` + `&POISSON PERIODIC XY` with Martyna-Tuckerman (`MT`) solver.
- **Convergence Parameters**: `CUTOFF 420` Ry, `REL_CUTOFF 60` Ry, `EPS_SCF 1.5E-6`, `MAX_SCF 200`, OT DIIS `FULL_SINGLE_INVERSE`.
- **Optimizer**: `BFGS` with `MAX_ITER 500`, `MAX_FORCE 4.5E-4` a.u., `RMS_FORCE 3.0E-4` a.u.

---

## Wuhan University (WHU) HPC Submission Guide

### 1. Master Batch Submission (All 10 Systems)

On the WHU HPC login node, navigate to `WorkingFolder/2.CP2K_GeoOpt` and run:

```bash
./submit_all_jobs.sh
```

This script will automatically iterate through all 10 subdirectories and submit jobs to the Slurm queue.

### 2. Single System Submission

```bash
cd 2.2.2CSO-0.625H2O+NH4
sbatch run_cp2k.slurm
```

### 3. Slurm Configuration Details
- **Partition**: `9a14a`
- **Account**: `tangsiqi`
- **MPI Parallelism**: 1 node, 192 tasks per node (`--ntasks-per-node=192`)
- **Environment**: `source /scratch/tangsiqi/CP2K_pkg/cp2k-v202602/install/bin/cp2k_env.sh`
- **Monitoring**: `squeue -u tangsiqi` or `tail -f geo_opt.out`

---

## Local / Workstation Testing Guide

### 1. Windows Batch Execution
```cmd
cd 2.1.1CSO-2H2O
.\run_cp2k_geoopt.bat
```

### 2. Linux / Docker Compose Execution
```bash
cd 2.1.1CSO-2H2O
./run_cp2k_geoopt.sh
```
