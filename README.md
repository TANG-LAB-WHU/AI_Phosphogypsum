# AI_Phosphogypsum

AI-empowering upcycling of world-issue phosphogypsum (PG) through computational chemistry and machine learning interatomic potentials.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains computational workflows for studying the dehydration mechanisms and surface chemistry of phosphogypsum (calcium sulfate hydrates, CaSO₄·nH₂O) using:

- **Machine Learning Interatomic Potentials (MACE-MP)**: Fast and accurate geometry optimization and molecular dynamics
- **Density Functional Theory (CP2K r²SCAN)**: High-accuracy reference calculations for validation
- **AIMD Simulations**: Ab-initio molecular dynamics for reaction mechanism investigation

The project aims to understand fundamental processes involved in PG upcycling, including:
- Dehydration pathways from gypsum (CaSO₄·2H₂O) to anhydrite (CaSO₄)
- Surface interactions with NH₃/NH₄⁺/HPO₄²⁻ species
- Hydrogen bonding dynamics during phase transitions

## Repository Structure

```
AI_Phosphogypsum/
├── WorkingFolder/                      # Active computational workflows
│   ├── 1.ModelConfig/                  # Structure preparation & model building
│   ├── 2a.GeoOpt_Task_byMACE/          # MACE-MP geometry optimization
│   ├── 2b.GeoOpt_Task_byCP2K_r2SCAN/   # CP2K DFT geometry optimization
│   └── 3.MLIP_AIMD/                    # MLIP-based AIMD simulations
├── References/                         # Reference materials
│   ├── MACE_installation/              # MACE + cuEquivariance setup guide
│   ├── Structural_Data/                # Crystal structure database (ICSD)
│   ├── Papers_Parsed/                  # Parsed literature references
│   └── Published_Articles/             # Related publications
├── LICENSE
└── README.md
```

## Workflow Modules

### 1. Model Configuration (`WorkingFolder/1.ModelConfig/`)

Tools for generating slab models from crystallographic data:

| Script | Function |
|--------|----------|
| `create_slab_from_cif.py` | Create surface slabs from CIF files with specified Miller indices |
| `NH4_packing_CSOslab.py` | Pack reactant molecules (H₂O, NH₃, NH₄⁺, HPO₄²⁻) into vacuum region |
| `generate_dehydration_series.py` | Generate CaSO₄·nH₂O structures with varying water content (n = 2 → 0) |

**Supported Systems:**
- Pure CSO slabs: CaSO₄·2H₂O, CaSO₄·0.625H₂O, CaSO₄·0.583H₂O, CaSO₄·0.5H₂O, CaSO₄
- CSO + Reactants: All above with NH₃/NH₄⁺/HPO₄²⁻ adsorption

### 2a. MACE-MP Geometry Optimization (`WorkingFolder/2a.GeoOpt_Task_byMACE/`)

Fast geometry optimization using pre-trained universal machine learning potentials.

**Features:**
- GPU-accelerated calculations with CUDA support
- DFT-D3 dispersion correction (Becke-Johnson damping)
- Multiple pre-trained models: `mace-matpes-r2scan-0`, `mace-matpes-pbe-0`, `medium-mpa-0`
- Typical performance: ~90 ms/step on RTX 5080

**Quick Start:**
```bash
conda activate mace_env
cd WorkingFolder/2a.GeoOpt_Task_byMACE/2.1.1CSO-2H2O
python run_mace_GeoOpt.py --model mace-matpes-r2scan-0 --fmax 0.00001
```

### 2b. CP2K DFT Geometry Optimization (`WorkingFolder/2b.GeoOpt_Task_byCP2K_r2SCAN/`)

High-accuracy DFT calculations using the r²SCAN functional with D4 dispersion correction.

**Features:**
- revPBE + DFT-D4 dispersion correction
- MOLOPT basis sets + GTH pseudopotentials
- Docker-based execution with GPU support

**Quick Start:**
```bash
cd WorkingFolder/2b.GeoOpt_Task_byCP2K_r2SCAN/2.1.1CSO-2H2O
.\run_cp2k_geoopt.bat
```

### 3. MLIP-AIMD Simulations (`WorkingFolder/3.MLIP_AIMD/`)

Machine-learning potential driven ab-initio molecular dynamics for investigating reaction mechanisms and dynamics.

## Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.x support
- Conda or venv for environment management

### MACE Environment Setup

```bash
# Create conda environment
conda create -n mace_env python=3.10 -y
conda activate mace_env

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu129

# Install MACE and dependencies
pip install mace-torch ase matplotlib numpy

# (Optional) Install cuEquivariance for GPU acceleration
pip install cuequivariance==0.8.1 cuequivariance-torch==0.8.1
pip install cuequivariance-ops-cu12==0.8.1 cuequivariance-ops-torch-cu12==0.8.1
```

For detailed installation instructions including Blackwell architecture GPU (RTX 5080/5090) support, see [`References/MACE_installation/MACE_installation.md`](References/MACE_installation/MACE_installation.md).

### CP2K Environment Setup

```bash
# Create conda environment
conda create -n AI_Phosphogypsum python=3.10 -y
conda activate AI_Phosphogypsum
pip install ase matplotlib numpy
```

CP2K runs via Docker. Ensure Docker Desktop is installed and GPU passthrough is configured.

## Usage Examples

### Generate a Gypsum Slab

```bash
cd WorkingFolder/1.ModelConfig
python create_slab_from_cif.py \
    --cif-input 1.1.1CSO-2H2O/conventional_cell.cif \
    --miller 0 2 0 \
    --layers 1 \
    --vacuum 15.0 \
    --supercell 2 2
```

### Pack Reactants onto Slab

```bash
python NH4_packing_CSOslab.py \
    --slab-input 1.1.1CSO-2H2O/conventional_cell_slab_020_L1_2x2.xyz \
    --output-dir 1.2.1CSO-2H2O+NH4 \
    --n-water 20 --n-nh3 3 --n-nh4 4 --n-hpo4 2 \
    --auto-vacuum --min-distance 2.5
```

### Run MACE Geometry Optimization

```bash
cd WorkingFolder/2a.GeoOpt_Task_byMACE/2.1.1CSO-2H2O
python run_mace_GeoOpt.py \
    --model mace-matpes-r2scan-0 \
    --fmax 0.00001 \
    --optimizer BFGS
```

## System Overview

| System ID | Composition | Description |
|-----------|-------------|-------------|
| 2.1.1 | CaSO₄·2H₂O | Gypsum (dihydrate) |
| 2.1.2 | CaSO₄·0.5-0.625H₂O | Bassanite (hemihydrate) variants |
| 2.1.3 | CaSO₄ | Anhydrite |
| 2.2.x | CSO·nH₂O + NH₄/NH₃/HPO₄ | Reactant-adsorbed systems |

## References

- MACE: [ACEsuit/mace](https://github.com/ACEsuit/mace)
- CP2K: [CP2K Open Source Molecular Dynamics](https://www.cp2k.org/)
- ASE: [Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
