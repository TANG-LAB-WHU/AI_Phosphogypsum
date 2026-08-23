#!/usr/bin/env python3
"""
build_large_scale_hydrothermal_box.py
=====================================
Generates large-scale (tens of thousands to hundreds of thousands of atoms)
mesoporous/slit-pore phosphogypsum hydrothermal reaction systems for LAMMPS.

Atom types mapping:
  1: Ca (40.078)
  2: O  (15.999)
  3: S  (32.065)
  4: H  (1.008)
  5: N  (14.007)
  6: P  (30.974)
"""

import os
import argparse
import numpy as np
from ase.io import read
from ase.atoms import Atoms

TYPE_MAP = ["Ca", "O", "S", "H", "N", "P"]
TYPE_DICT = {elem: idx + 1 for idx, elem in enumerate(TYPE_MAP)}
MASSES = [40.078, 15.999, 32.065, 1.008, 14.007, 30.974]

def write_lammps_data(atoms, filename):
    """Write an ASE Atoms object to standard LAMMPS data format."""
    n_atoms = len(atoms)
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    
    # Calculate box bounds
    xlo, ylo, zlo = 0.0, 0.0, 0.0
    xhi, yhi, zhi = cell[0, 0], cell[1, 1], cell[2, 2]
    xy, xz, yz = cell[1, 0], cell[2, 0], cell[2, 1]
    
    with open(filename, "w") as f:
        f.write("# LAMMPS data file generated for Phosphogypsum Hydrothermal Reaction\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{len(TYPE_MAP)} atom types\n\n")
        f.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
        f.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
        f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n")
        if abs(xy) > 1e-5 or abs(xz) > 1e-5 or abs(yz) > 1e-5:
            f.write(f"{xy:.6f} {xz:.6f} {yz:.6f} xy xz yz\n")
        f.write("\nMasses\n\n")
        for i, mass in enumerate(MASSES, 1):
            f.write(f"{i} {mass:.4f} # {TYPE_MAP[i-1]}\n")
        f.write("\nAtoms # atomic\n\n")
        for i, (sym, pos) in enumerate(zip(symbols, positions), 1):
            atype = TYPE_DICT[sym]
            f.write(f"{i} {atype} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            
    print(f"[Export] Successfully generated LAMMPS data file: {filename} ({n_atoms} atoms)")

def main():
    parser = argparse.ArgumentParser(description="Build large-scale hydrothermal simulation box.")
    parser.add_argument("--repeat-x", type=int, default=2, help="Supercell repeat in X")
    parser.add_argument("--repeat-y", type=int, default=2, help="Supercell repeat in Y")
    parser.add_argument("--repeat-z", type=int, default=1, help="Supercell repeat in Z")
    parser.add_argument("--output", default="data.phosphogypsum_hydrothermal", help="Output filename")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ref_struct = os.path.join(base_dir, "..", "..", "3.CP2K_AIMD", "3.2.2CSO-0.625H2O+NH4", "geoopt_optimized_structure_extxyz_wrap.xyz")
    if not os.path.exists(ref_struct):
        ref_struct = os.path.join(base_dir, "..", "..", "1.ModelConfig", "1.2.2CSO-0.625H2O_improved+NH4_3", "conventional_cell_slab_020_L1_2x2_packed_w20_nh3_3_nh4_4_hpo4_2.xyz")
        
    print("=" * 60)
    print("BUILDING LARGE-SCALE HYDROTHERMAL SIMULATION BOX")
    print(f"Base unit: {ref_struct}")
    print(f"Supercell replication: {args.repeat_x} x {args.repeat_y} x {args.repeat_z}")
    print("=" * 60)
    
    atoms = read(ref_struct)
    print(f"Base slab has {len(atoms)} atoms ({atoms.get_chemical_formula()}).")
    
    # Replicate supercell
    supercell = atoms.repeat((args.repeat_x, args.repeat_y, args.repeat_z))
    print(f"Replicated box has {len(supercell)} atoms ({supercell.get_chemical_formula()}).")
    print(f"Box dimensions: {supercell.cell.lengths().round(2)} Angstroms, Volume: {supercell.cell.volume:.1f} A^3")
    
    # Save LAMMPS data file
    out_path = os.path.join(base_dir, args.output)
    write_lammps_data(supercell, out_path)
    
    # Copy to simulation dir
    sim_dir = os.path.join(base_dir, "..", "02.lammps_simulation")
    if os.path.exists(sim_dir):
        dst_path = os.path.join(sim_dir, args.output)
        write_lammps_data(supercell, dst_path)

if __name__ == "__main__":
    main()
