# 2. Geometry Optimization Task

This directory contains CP2K geometry optimization tasks for Calcium Sulfate (gypsum) slab models under various conditions. The simulations are divided into two main series: pure systems and systems with Ammonium (NH4) adsorption.

## Directory Structure

### 2.1 Pure System Series (CSO)

Calculations for pure Calcium Sulfate slabs with varying water content.

- **2.1.1CSO-2H2O**: Dihydrate (Gypsum) slab.
- **2.1.2CSO-0.5H2O**: Hemihydrate slab (0.5 H2O).
- **2.1.2CSO-0.583H2O**: Soluble Anhydrite-like intermediate (0.583 H2O).
- **2.1.2CSO-0.625H2O**: Intermediate hydration state (0.625 H2O).
- **2.1.3CSO**: Anhydrite slab (0 H2O).

### 2.2 System with Ammonium Series (CSO + NH4)

Calculations for Calcium Sulfate slabs interacting with NH3/NH4/HPO4 species.

- **2.2.1CSO-2H2O+NH4**: Dihydrate with adsorbed NH4 species.
- **2.2.2CSO-0.5H2O+NH4**: Hemihydrate with adsorbed NH4 species.
- **2.2.2CSO-0.583H2O+NH4**: Intermediate with adsorbed NH4 species.
- **2.2.2CSO-0.625H2O+NH4**: Intermediate with adsorbed NH4 species.
- **2.2.3CSO+NH4**: Anhydrite with adsorbed NH4 species.

## Simulation Details

Each subdirectory contains the necessary files to run a geometry optimization using CP2K:

1. **`geo_opt.inp`**: The main CP2K input file. It specifies:
   * **Run Type**: `GEO_OPT` (Geometry Optimization)
   * **Method**: Quickstep (GPW) using `revPBE` functional with `DFTD4` dispersion correction.
   * **Cell Parameters**: Specifically adapted to match the XYZ structure of each system.
   * **Basis Sets/Potentials**: `MOLOPT` basis sets and `GTH` pseudopotentials.
2. **`*.xyz`**: The initial atomic structure file.
3. **`docker-compose-cp2k.yml`**: Docker Compose configuration for running CP2K with GPU support (`mycp2k-rtx5080` or similar image).
4. **`run_cp2k_geoopt.bat`**: A Windows batch script to automate the execution of the simulation via Docker.

## Usage

To run a simulation for a specific system:

1. Navigate to the target directory (e.g., `cd 2.1.2CSO-0.5H2O`).
2. Execute the batch script:

   ```cmd
   .\run_cp2k_geoopt.bat
   ```

   This script will:* Check for required files.* Clean up old containers.* Launch CP2K using Docker.* Output logs to `geo_opt.log`.
3. Prior to initialize a specific system, a Python virtual environment shoudl built via conda in a powershell terminal as below. After that, activate this environment and install required dependencies over it.

   a) conda create -n AI_Phosphogypsum python=3.10  -y

   b) conda activate AI_Phosphogypsum

   c) pip install ase matplotlib numpy
4. Use cd in the powershell terminal to set the current folder to that holding the batch script run_cp2k_geoopt.bat, and then type ".\run_cp2k_geoopt.bat" to start the calculation task.
5. When encountering error or completion signals in the terminal, use "docker compose -f docker-compose-cp2k.yml down --remove-orphans" to remove the previous container cached in Docker Desktop.

## Notes

- **Cell Consistency**: The `&CELL` section in each `geo_opt.inp` has been carefully verified to match the `Lattice` parameters from its corresponding XYZ file.
- **Resources**: Scripts are configured to use GPU resources and 16 OMP threads by default.
