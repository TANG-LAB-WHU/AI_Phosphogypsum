# 6. LAMMPS Large-Scale & Long-Time Hydrothermal Simulation Center

This directory hosts the **large-scale ($10^4 \sim 10^5$ atoms) and long-time ($10 \sim 100\text{ ns}$) molecular dynamics simulations** of phosphogypsum hydrothermal dissolution and hydroxyapatite crystallization using the production-grade Deep Potential trained in Stage 5.

---

## Workflow Structure

```
6.LAMMPS_ScalingUp/
├── 01.system_setup/        # Supercell and mesoporous system generation (data.phosphogypsum_hydrothermal)
├── 02.lammps_simulation/   # Production LAMMPS input script (in.phosphogypsum_hydrothermal.lammps), launch scripts
├── 03.trajectory_analysis/ # Reaction kinetics analyzer (dissolution rates, PNC growth, diffusion coefficients)
└── README.md
```

---

## Key Physical Settings

1. **Hydrothermal Reaction Conditions**:
   - Temperature: $T = 453.15\text{ K}$ ($180^\circ\text{C}$ strictly aligned with experimental findings).
   - Autoclave Pressure: $P = 50.0\text{ bar}$.
   - Timestep: $\Delta t = 0.5\text{ fs}$.
2. **Deep Potential Loading**:
   - `pair_style deepmd frozen_model_compressed.pb`
   - Tabulated spline acceleration for high-throughput GPU inference.
3. **In-situ Reaction Diagnostics**:
   - Radial Distribution Functions (RDFs: $Ca-O$, $Ca-P$, $P-O$, $N-H$, $S-O$);
   - Mean Squared Displacement (MSD) for ion diffusion kinetics;
   - Coordination number evolution and prenucleation cluster tracking.

---

## Execution Guide

### 1. Build Large-Scale Hydrothermal System
```bash
cd 01.system_setup
python3 build_large_scale_hydrothermal_box.py --repeat-x 2 --repeat-y 2 --repeat-z 1
```

### 2. Run Production Simulation
```bash
cd ../02.lammps_simulation
./run_lammps.sh
```

### 3. Analyze Reaction Kinetics
```bash
cd ../03.trajectory_analysis
python3 analyze_reaction_kinetics.py
```
