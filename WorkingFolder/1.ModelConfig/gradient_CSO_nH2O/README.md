# Gypsum Dehydration Series Generator

This module generates CaSO₄·nH₂O structures with varying water content (n = 2 → 0) for AIMD simulations.

## Quick Start

```powershell
# Generate with default settings (n = 2.0, 1.5, 1.0, 0.5, 0.0)
python generate_dehydration_series.py --cif-input ../../References/Structural_Data/PG_ICSD/33.cif

# Generate with custom step size
python generate_dehydration_series.py --cif-input ../../References/Structural_Data/PG_ICSD/33.cif --n-step 0.25

# Generate specific n values
python generate_dehydration_series.py --cif-input ../../References/Structural_Data/PG_ICSD/33.cif --n-values 2.0 1.75 1.25 0.75 0.0
```

## Water Removal Strategies

| Strategy | Description | Physical Basis |
|----------|-------------|----------------|
| `random` | Uniform random selection | Statistical sampling |
| `layer_ordered` | Remove from (010) surfaces first | Dehydration cracks parallel to {010} cleavage |
| `hbond_weakest` | Remove weakly H-bonded water first | Thermodynamic preference |

## CP2K Considerations

- **Default supercell (2×1×2)**: ~150-200 atoms, suitable for DFT-MD
- **Larger supercell (3×1×3)**: ~350-450 atoms, finer n granularity but higher cost

## Output Structure

```
dehydration_series/
├── random/
│   ├── CaSO4_n2p00.xyz
│   ├── CaSO4_n1p50.xyz
│   └── ...
├── layer_ordered/
│   └── ...
├── hbond_weakest/
│   └── ...
├── CaSO4_n2p00_reference.xyz
└── generation_log.txt
```

## References

The `layer_ordered` strategy is based on crystallographic studies of gypsum dehydration:

1. **Stawski, T. M., et al.** (2020). *The dehydration of gypsum at sub-micron scale: First evidence of oriented bassanite crystallization.* European Journal of Mineralogy, 32(4), 379-391. DOI: 10.5194/ejm-32-379-2020
   - Key finding: Dehydration cracks form parallel to the {010} cleavage plane due to structural water loss and lattice shrinkage along [001] and [10-1] directions.

2. **Sipple, E. M., et al.** (2020). *Formation of bassanite during the dehydration of gypsum at dry and low-pressure conditions.* American Mineralogist, 105(3), 324-333.
   - Key finding: Water molecules coordinated to Ca-SO₄ chains within (010) planes; H₂O diffusion along [001] is hindered.

