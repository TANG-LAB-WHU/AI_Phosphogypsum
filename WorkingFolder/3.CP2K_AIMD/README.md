# 3. CP2K AIMD Hydrothermal Dynamics Simulation Center

This directory manages **first-principles Born-Oppenheimer Ab-Initio Molecular Dynamics (AIMD)** simulations for investigating the hydrothermal dissolution and reaction kinetics of phosphogypsum slabs.

All initial atomic structures are directly inherited from Stage 2 (`WorkingFolder/2.CP2K_GeoOpt`) / Stage 1 (`WorkingFolder/1.ModelConfig`). The simulation configuration strictly follows the Wuhan University (WHU) HPC reference architecture (`R2_AIMD`), producing comprehensive trajectory, force, energy, and analytical stress tensor datasets tailored for DeePMD-kit and DP-GEN training.

---

## System Overview (All 10 Systems)

| Directory | Chemical Formula | Total Atoms | Lattice Dimensions ($a \times b \times c$, Å) | Hydrothermal Chemical Context |
| :--- | :--- | :--- | :--- | :--- |
| **`3.1.1CSO-2H2O`** | $H_{64}Ca_{16}O_{96}S_{16}$ | 192 | $11.36 \times 12.57 \times 29.53$ | Gypsum dihydrate slab AIMD |
| **`3.1.2CSO-0.583H2O`** | $H_{64}Ca_{48}O_{224}S_{48}$ | 384 | $25.50 \times 23.97 \times 21.90$ | Hemihydrate-like intermediate slab AIMD |
| **`3.1.2CSO-0.5H2O`** | $H_{24}Ca_{24}O_{108}S_{24}$ | 180 | $25.51 \times 13.85 \times 20.43$ | Bassanite (hemihydrate) slab AIMD |
| **`3.1.2CSO-0.625H2O`** | $H_{48}Ca_{48}O_{216}S_{48}$ | 360 | $24.07 \times 35.04 \times 21.73$ | Intermediate hydrate slab ($\gamma = 133.66^\circ$) AIMD |
| **`3.1.3CSO`** | $Ca_{16}O_{64}S_{16}$ | 96 | $12.77 \times 14.11 \times 20.91$ | Anhydrite (anhydrous) slab AIMD |
| **`3.2.1CSO-2H2O+NH4`** | $H_{131}Ca_{16}N_{7}O_{124}P_{2}S_{16}$ | 296 | $11.36 \times 12.57 \times 60.05$ | Dihydrate confined depolymerization AIMD |
| **`3.2.2CSO-0.583H2O+NH4`** | $H_{131}Ca_{48}N_{7}O_{252}P_{2}S_{48}$ | 488 | $25.50 \times 23.97 \times 43.86$ | Intermediate 0.583H2O confined reaction AIMD |
| **`3.2.2CSO-0.5H2O+NH4`** | $H_{91}Ca_{24}N_{7}O_{136}P_{2}S_{24}$ | 284 | $25.51 \times 13.85 \times 42.03$ | Hemihydrate 0.5H2O confined reaction AIMD |
| **`3.2.2CSO-0.625H2O+NH4`** | $H_{115}Ca_{48}N_{7}O_{244}P_{2}S_{48}$ | 464 | $24.07 \times 35.04 \times 36.83$ | Sub-nanometer strong confinement AIMD |
| **`3.2.3CSO+NH4`** | $H_{67}Ca_{16}N_{7}O_{92}P_{2}S_{16}$ | 200 | $12.77 \times 14.11 \times 46.17$ | Anhydrite confined reaction AIMD |

---

## Technical Specifications (Aligned with WHU R2_AIMD)

1. **Analytical Stress Tensor & Full Virial Support**:
   - `&FORCE_EVAL &STRESS_TENSOR ANALYTICAL`
   - `&POISSON &PERIODIC XYZ &POISSON_SOLVER PERIODIC`
   - `&CELL &PERIODIC XYZ`
   - Generates step-by-step `*-1.stress` files for training pressure-dependent virials in DeePMD-kit.
2. **First-Principles Density Functional**:
   - `r2SCAN` Meta-GGA (`&MGGA_X_R2SCAN` + `&MGGA_C_R2SCAN`) with `DFTD4` dispersion.
   - `MOLOPT_UZH` basis sets + `GTH-SCAN` pseudopotentials.
3. **Hydrothermal Reaction Dynamics**:
   - Temperature: $T = 453.15\text{ K}$ ($180^\circ\text{C}$ strictly aligned with experiments).
   - Integration Timestep: $\Delta t = 0.5\text{ fs}$.
   - Thermostat: `CSVR` canonical thermostat ($\tau = 20.0\text{ fs}$).
4. **DeePMD Full Observables Logging (`&EACH MD 1`)**:
   - `*-pos-1.xyz` (Coordinates)
   - `*-vel-1.xyz` (Velocities)
   - `*-frc-1.xyz` (Forces)
   - `*-1.ener` (Energies)
   - `*-1.stress` (Stress tensors)
   - `*-1.cell` (Cell vectors)

---

## Wuhan University (WHU) HPC Submission Guide

### 1. Master Batch Submission (All 10 Systems)

```bash
cd WorkingFolder/3.CP2K_AIMD
./submit_all_jobs.sh
```

### 2. Single System Submission

```bash
cd 3.2.2CSO-0.625H2O+NH4
sbatch run_cp2k.slurm
```

### 3. Slurm Configuration
- **Partition**: `9a14a`
- **Account**: `tangsiqi`
- **MPI Parallelism**: 1 node, 192 tasks per node (`--ntasks-per-node=192`)
- **Environment**: `source /scratch/tangsiqi/CP2K_pkg/cp2k-v202602/install/bin/cp2k_env.sh`
- **Monitoring**: `squeue -u tangsiqi`
