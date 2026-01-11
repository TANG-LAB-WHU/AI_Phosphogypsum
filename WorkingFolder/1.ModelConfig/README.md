# 1.ModelConfig

This directory contains initial configurations and tools for Phosphogypsum (PG) quantum mechanical simulations. It includes scripts for creating slab models from CIF files, packing them with reactant molecules (NH4+, HPO4^2-, NH3, H2O), and generating dehydration series.

## Directory Structure

The directory is organized into main categories of models and tools:

*   **1.1.x Pure CSO-nH2O Slabs**: Contains Calcium Sulfate (CSO) slab models with varying degrees of hydration (Di-hydrate, Hemi-hydrate, Anhydrite). These are generated directly from crystal structure files (CIF).
*   **1.2.x CSO Slabs + Reactants**: Contains the CSO slabs from 1.1.x packed with ammonium and phosphate species to simulate the reaction environment.
*   **gradient_CSO_nH2O**: Contains tools and output for generating dehydration series of CaSO4·nH2O structures (n = 2 → 0) for AIMD simulations.

## Scripts

### 1. create_slab_from_cif.py

A tool to create slab models from bulk CIF files with specified Miller indices, vacuum thickness, and supercell dimensions. It handles orthogonality checks and effectively manages file formats.

**Key Arguments:**
*   `--cif-input`: Path to the input CIF file.
*   `--miller H K L`: Miller indices for the surface plane.
*   `--layers N`: Number of unit cell layers in the slab.
*   `--vacuum N`: Vacuum thickness in Angstroms.
*   `--supercell A B`: In-plane supercell dimensions (e.g., `2 2`).
*   `--require-ortho`: Enforce orthogonality of the simulation cell.

### 2. NH4_packing_CSOslab.py

A tool to pack reactant molecules (H2O, NH3, NH4+, HPO4^2-) into the vacuum region of the generated slab models. It uses random rotation and collision detection to place molecules.

**Key Arguments:**
*   `--slab-input`: Path to the input slab geometry file (e.g., .xyz).
*   `--n-water`, `--n-nh3`, `--n-nh4`, `--n-hpo4`: Counts for each species.
*   `--min-distance`: Minimum atomic distance to avoid clashes.
*   `--auto-vacuum`: Automatically adjust vacuum height to accommodate molecules.

### 3. gradient_CSO_nH2O/generate_dehydration_series.py

A module to generate CaSO4·nH2O structures with varying water content (n = 2 → 0). It supports different water removal strategies to simulate physical processes.

**Strategies:**
*   `random`: Uniform random selection.
*   `layer_ordered`: Preferential removal from (010) inter-layer surfaces (simulating cracks).
*   `hbond_weakest`: Removes water molecules with weakest hydrogen bonding.

**Key Arguments:**
*   `--cif-input`: Path to input CIF file.
*   `--n-values`: Explicit list of n values (e.g., `2.0 1.5 1.0`).
*   `--n-step`: Step size for chemical formula n reduction.
*   `--supercell`: Supercell dimensions (default `2 1 2`).

## Usage Workflows

Below are standard workflows used to generate the models in this directory.

### 1. Create Pure CSO-nH2O Slab Models

**Calcium Sulfate Dihydrate (Gypsum)**
```bash
python create_slab_from_cif.py --cif-input 1.1.1CSO-2H2O/conventional_cell.cif --miller 0 2 0 --layers 1 --vacuum 15.0 --supercell 2 2
```

**Calcium Sulfate Intermediates (Hemi-hydrate)**
```bash
python create_slab_from_cif.py --cif-input 1.1.2CSO-0.5H2O/conventional_cell.cif --miller 0 2 0 --layers 1 --vacuum 15.0 --supercell 2 2 --require-ortho
```

**Calcium Sulfate Anhydrate**
```bash
python create_slab_from_cif.py --cif-input 1.1.3CSO/conventional_cell.cif --miller 0 2 0 --layers 1 --vacuum 15.0 --supercell 2 2 --require-ortho
```

### 2. Pack Reactants (NH4+) onto Slab Models

**Packing onto CSO-2H2O**
```bash
python NH4_packing_CSOslab.py \
    --slab-input 1.1.1CSO-2H2O/conventional_cell_slab_020_L1_2x2.xyz \
    --output-dir 1.2.1CSO-2H2O+NH4 \
    --n-water 20 \
    --n-nh3 3 \
    --n-nh4 4 \
    --n-hpo4 2 \
    --auto-vacuum \
    --min-distance 2.5 \
    --z-reoffset 2 \
    --wrap \
    --no-boundary-cross \
    --pbc-xyz \
    --target-vacuum 18 \
    --verbose
```

### 3. Generate Dehydration Series

**Generate with default settings (n = 2.0, 1.5, 1.0, 0.5, 0.0)**
```bash
cd gradient_CSO_nH2O
python generate_dehydration_series.py --cif-input ../../References/Structural_Data/PG_ICSD/33.cif
```

**Generate with specific n values**
```bash
python generate_dehydration_series.py --cif-input ../../References/Structural_Data/PG_ICSD/33.cif --n-values 2.0 1.75 1.25 0.75 0.0
```