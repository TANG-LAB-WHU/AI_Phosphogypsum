# 2a. Geometry Optimization Task (MACE-MP)

This directory contains **MACE-MP** (Machine Learning Interatomic Potential) geometry optimization tasks for Calcium Sulfate (gypsum) slab models under various conditions. This is a faster alternative to DFT-based CP2K calculations, using pre-trained universal machine learning potentials.

## Directory Structure

### 2.1 Pure System Series (CSO)

Calculations for pure Calcium Sulfate slabs with varying water content.

| Directory | Description | Input Structure |
|-----------|-------------|-----------------|
| **2.1.1CSO-2H2O** | Dihydrate (Gypsum) slab | `conventional_cell_slab_020_L1_2x2.xyz` |
| **2.1.2CSO-0.5H2O** | Hemihydrate slab (0.5 H2O) | `conventional_cell_slab_020_L1_2x2.xyz` |
| **2.1.2CSO-0.583H2O** | Soluble Anhydrite-like intermediate (0.583 H2O) | `conventional_cell_slab_020_L1_2x2.xyz` |
| **2.1.2CSO-0.625H2O** | Intermediate hydration state (0.625 H2O) | `conventional_cell_slab_020_L1_2x2.xyz` |
| **2.1.3CSO** | Anhydrite slab (0 H2O) | `conventional_cell_slab_020_L1_2x2.xyz` |

### 2.2 System with Ammonium Series (CSO + NH4)

Calculations for Calcium Sulfate slabs interacting with NH3/NH4/HPO4 species.

| Directory | Description | Input Structure |
|-----------|-------------|-----------------|
| **2.2.1CSO-2H2O+NH4** | Dihydrate with adsorbed NH4 species | `conventional_cell_slab_020_L1_2x2_packed_w20_nh3_3_nh4_4_hpo4_2.xyz` |
| **2.2.2CSO-0.5H2O+NH4** | Hemihydrate with adsorbed NH4 species | `conventional_cell_slab_020_L1_2x2_packed_w20_nh3_3_nh4_4_hpo4_2.xyz` |
| **2.2.2CSO-0.583H2O+NH4** | Intermediate with adsorbed NH4 species | `conventional_cell_slab_020_L1_2x2_packed_w20_nh3_3_nh4_4_hpo4_2.xyz` |
| **2.2.2CSO-0.625H2O+NH4** | Intermediate with adsorbed NH4 species | `conventional_cell_slab_020_L1_2x2_packed_w20_nh3_3_nh4_4_hpo4_2.xyz` |
| **2.2.3CSO+NH4** | Anhydrite with adsorbed NH4 species | `conventional_cell_slab_020_L1_2x2_packed_w20_nh3_3_nh4_4_hpo4_2.xyz` |

## MACE-MP Models

Three pre-trained MACE-MP models are available:

| Model | Description |
|-------|-------------|
| `medium-mpa-0` | Latest recommended model (PBE+U, Materials Project) |
| `mace-matpes-pbe-0` | Pure PBE without +U correction (MATPES dataset) |
| `mace-matpes-r2scan-0` | r2SCAN functional (default, better for some materials) |

## File Structure

Each subdirectory contains:

| File | Description |
|------|-------------|
| `run_mace_GeoOpt.py` | Main Python script for MACE geometry optimization |
| `*.xyz` | Initial atomic structure file (extended XYZ format with lattice) |

### Output Files (generated after running)

| File | Description |
|------|-------------|
| `*_optimized.xyz` | Optimized structure |
| `optimized_structure_extxyz_wrap.xyz` | Extended XYZ with energy/forces (wrapped) |
| `optimized_structure_extxyz_unwrap.xyz` | Extended XYZ with energy/forces (unwrapped) |
| `optimization.traj` | ASE trajectory file |
| `optimization_trajectory.xyz` | Trajectory in XYZ format (for VMD) |
| `optimization.log` | Optimization log |
| `optimization_report.txt` | Summary report |

## Environment Setup

### Prerequisites

1. Create and activate a conda environment with MACE:

   ```bash
   conda create -n mace_env python=3.10 -y
   conda activate mace_env
   ```

2. Install required packages:

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu129
   pip install mace-torch ase matplotlib numpy
   ```

## Usage

### Basic Usage

Navigate to the target directory and run:

```bash
cd 2.1.2CSO-0.583H2O
python run_mace_GeoOpt.py
```

### Command Line Options

```bash
python run_mace_GeoOpt.py [OPTIONS]

Options:
  -m, --model         MACE model (default: mace-matpes-r2scan-0)
  -i, --input         Input XYZ file (default: conventional_cell_slab_020_L1_2x2.xyz)
  -o, --output        Output XYZ file (auto-generated if not specified)
  --fmax              Force convergence criterion in eV/Å (default: 0.00001)
  --max-steps         Maximum optimization steps (default: 5000)
  --optimizer         Optimizer: BFGS, LBFGS, or FIRE (default: BFGS)
  --no-dispersion     Disable D3 dispersion correction
  --damping           D3 damping function: bj, zero, zerom, bjm (default: bj)
  --list-models       List available models and exit
```

### Examples

```bash
# Use default r2SCAN model
python run_mace_GeoOpt.py

# Use PBE model with tighter convergence
python run_mace_GeoOpt.py --model mace-matpes-pbe-0 --fmax 0.001

# Use FIRE optimizer for difficult cases
python run_mace_GeoOpt.py --optimizer FIRE --max-steps 10000

# List available models
python run_mace_GeoOpt.py --list-models
```

## Optimization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fmax` | 0.00001 eV/Å | Force convergence criterion |
| `max_steps` | 5000 | Maximum optimization steps |
| `optimizer` | BFGS | Optimization algorithm |
| `dispersion` | Enabled (D3-BJ) | DFT-D3 dispersion correction |
| `fix_bottom_layers` | False | Fix atoms at bottom of slab |

## Notes

- **GPU Acceleration**: The script automatically detects and uses CUDA GPU if available. CPU mode is significantly slower.
- **Dispersion Correction**: D3 dispersion correction with Becke-Johnson damping is enabled by default for accurate description of van der Waals interactions.
- **Precision**: Uses `float64` precision for geometry optimization accuracy.
- **Trajectory**: Optimization trajectory is saved for visualization and analysis in VMD or other molecular viewers.
