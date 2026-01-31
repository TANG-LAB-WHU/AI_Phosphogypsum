#!/usr/bin/env python3
"""

This script generates a series of CaSO4·nH2O structures with varying water content (n = 2 to 0)
for AIMD simulations with CP2K. It implements three physically meaningful water removal strategies:

1. RANDOM: Uniform random removal of water molecules
2. LAYER_ORDERED: Preferential removal from (010) inter-layer regions (physically motivated)
3. HBOND_WEAKEST: Remove water molecules with weakest hydrogen bonding (requires bond analysis)

Usage:
    python generate_dehydration_series.py --cif-input gypsum_n2.cif --n-values 2.0 1.5 1.0 0.5 0.0
    python generate_dehydration_series.py --cif-input gypsum.cif --n-step 0.25
    
"""

import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict
import random

from ase import Atoms
from ase.io import read, write
from ase.build import make_supercell
from ase.neighborlist import neighbor_list


def identify_water_molecules(atoms: Atoms, oh_cutoff: float = 1.3) -> List[Tuple[int, int, int]]:
    """
    Identify water molecules in the structure based on O-H bond distances.
    
    Returns:
        List of tuples (O_index, H1_index, H2_index) for each water molecule
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    
    # Find all O and H atoms
    o_indices = [i for i, s in enumerate(symbols) if s == 'O']
    h_indices = [i for i, s in enumerate(symbols) if s == 'H']
    
    if len(h_indices) == 0:
        print("  Warning: No hydrogen atoms found in structure!")
        return []
    
    # Build neighbor list to find O-H bonds
    i_list, j_list, d_list = neighbor_list('ijd', atoms, cutoff=oh_cutoff)
    
    # For each O, find bonded H atoms
    water_molecules = []
    o_to_h = defaultdict(list)
    
    for i, j, d in zip(i_list, j_list, d_list):
        si, sj = symbols[i], symbols[j]
        if si == 'O' and sj == 'H':
            o_to_h[i].append(j)
        elif si == 'H' and sj == 'O':
            o_to_h[j].append(i)
    
    # Filter for O with exactly 2 H neighbors (water molecules)
    for o_idx, h_list in o_to_h.items():
        if len(h_list) == 2:
            water_molecules.append((o_idx, h_list[0], h_list[1]))
        elif len(h_list) > 2:
            # Take the two closest H atoms
            h_distances = [(h, np.linalg.norm(positions[h] - positions[o_idx])) for h in h_list]
            h_distances.sort(key=lambda x: x[1])
            water_molecules.append((o_idx, h_distances[0][0], h_distances[1][0]))
    
    return water_molecules


def calculate_hbond_strength(atoms: Atoms, water_molecules: List[Tuple[int, int, int]], 
                             hbond_cutoff: float = 3.5) -> List[float]:
    """
    Estimate hydrogen bond strength for each water molecule.
    
    Higher score = more hydrogen bonds = harder to remove.
    Returns a list of scores (number of H-bonds) for each water molecule.
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # Find O atoms that are NOT water (sulfate oxygens)
    water_o_indices = set(wm[0] for wm in water_molecules)
    sulfate_o_indices = [i for i, s in enumerate(symbols) if s == 'O' and i not in water_o_indices]
    
    hbond_scores = []
    
    for o_idx, h1_idx, h2_idx in water_molecules:
        score = 0
        o_pos = positions[o_idx]
        h1_pos = positions[h1_idx]
        h2_pos = positions[h2_idx]
        
        # Count H-bonds: water H to sulfate O
        for so_idx in sulfate_o_indices:
            so_pos = positions[so_idx]
            d_h1 = np.linalg.norm(h1_pos - so_pos)
            d_h2 = np.linalg.norm(h2_pos - so_pos)
            
            if d_h1 < hbond_cutoff:
                score += 1
            if d_h2 < hbond_cutoff:
                score += 1
        
        # Count H-bonds: water O to other water H (water-water H-bonds)
        for other_o, other_h1, other_h2 in water_molecules:
            if other_o == o_idx:
                continue
            other_h1_pos = positions[other_h1]
            other_h2_pos = positions[other_h2]
            
            d1 = np.linalg.norm(o_pos - other_h1_pos)
            d2 = np.linalg.norm(o_pos - other_h2_pos)
            
            if d1 < hbond_cutoff:
                score += 0.5  # Water-water H-bonds are weaker
            if d2 < hbond_cutoff:
                score += 0.5
        
        hbond_scores.append(score)
    
    return hbond_scores


def get_layer_positions(atoms: Atoms, water_molecules: List[Tuple[int, int, int]], 
                        layer_axis: int = 1) -> List[float]:
    """
    Get the position of each water molecule along the layer axis (default: y-axis for (010) plane).
    
    In gypsum, water is located in the (010) inter-layer region. 
    Removing from surfaces (high/low y) first simulates directional dehydration.
    """
    positions = atoms.get_positions()
    layer_positions = []
    
    for o_idx, _, _ in water_molecules:
        layer_positions.append(positions[o_idx, layer_axis])
    
    return layer_positions


def remove_water_molecules(atoms: Atoms, water_molecules: List[Tuple[int, int, int]], 
                           indices_to_remove: List[int]) -> Atoms:
    """
    Remove specified water molecules from the structure.
    
    Args:
        atoms: Original structure
        water_molecules: List of (O, H1, H2) tuples
        indices_to_remove: Indices into water_molecules list to remove
        
    Returns:
        New Atoms object with water molecules removed
    """
    # Collect all atom indices to remove
    atoms_to_remove = set()
    for idx in indices_to_remove:
        o, h1, h2 = water_molecules[idx]
        atoms_to_remove.add(o)
        atoms_to_remove.add(h1)
        atoms_to_remove.add(h2)
    
    # Keep atoms not in removal set
    keep_indices = [i for i in range(len(atoms)) if i not in atoms_to_remove]
    
    new_atoms = atoms[keep_indices]
    return new_atoms


def select_waters_random(water_molecules: List, n_remove: int, seed: Optional[int] = None) -> List[int]:
    """Random selection strategy."""
    if seed is not None:
        random.seed(seed)
    
    indices = list(range(len(water_molecules)))
    random.shuffle(indices)
    return indices[:n_remove]


def select_waters_layer_ordered(water_molecules: List, n_remove: int, 
                                 layer_positions: List[float], from_surface: bool = True) -> List[int]:
    """
    Layer-ordered selection: remove from surfaces (high/low y) first.
    
    This is physically motivated by the observation that dehydration cracks 
    form parallel to {010} cleavage planes.
    """
    # Sort by distance from center of layer
    y_min, y_max = min(layer_positions), max(layer_positions)
    y_center = (y_min + y_max) / 2
    
    indexed_distances = [(i, abs(layer_positions[i] - y_center)) for i in range(len(water_molecules))]
    
    if from_surface:
        # Remove from surfaces first (largest distance from center)
        indexed_distances.sort(key=lambda x: -x[1])
    else:
        # Remove from center first (smallest distance from center)
        indexed_distances.sort(key=lambda x: x[1])
    
    return [idx for idx, _ in indexed_distances[:n_remove]]


def select_waters_hbond_weakest(water_molecules: List, n_remove: int, 
                                 hbond_scores: List[float]) -> List[int]:
    """
    H-bond based selection: remove weakest bonded water first.
    
    This is physically motivated by the fact that loosely bound water 
    molecules are easier to remove during dehydration.
    """
    indexed_scores = [(i, hbond_scores[i]) for i in range(len(water_molecules))]
    indexed_scores.sort(key=lambda x: x[1])  # Lowest score first
    
    return [idx for idx, _ in indexed_scores[:n_remove]]


def generate_dehydration_series(
    input_cif: str,
    output_dir: str,
    n_values: List[float],
    supercell_dims: Tuple[int, int, int] = (2, 1, 2),
    strategies: List[str] = ['random', 'layer_ordered', 'hbond_weakest'],
    random_seed: int = 42,
    output_format: str = 'xyz',
    verbose: bool = True
):
    """
    Generate a series of dehydrated gypsum structures.
    
    Args:
        input_cif: Path to input CIF file (gypsum with H atoms)
        output_dir: Directory to save output structures
        n_values: List of target n values (water content per CaSO4)
        supercell_dims: Supercell dimensions (a, b, c)
        strategies: List of removal strategies to apply
        random_seed: Seed for random number generator
        output_format: Output file format (xyz, cif, extxyz)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and create supercell
    if verbose:
        print(f"\n{'='*60}")
        print("GYPSUM DEHYDRATION SERIES GENERATOR")
        print(f"{'='*60}")
        print(f"\nInput CIF: {input_cif}")
        print(f"Output directory: {output_dir}")
        print(f"Supercell: {supercell_dims[0]}x{supercell_dims[1]}x{supercell_dims[2]}")
    
    bulk = read(input_cif)
    supercell_matrix = np.diag(supercell_dims)
    atoms = make_supercell(bulk, supercell_matrix)
    
    if verbose:
        print(f"Total atoms in supercell: {len(atoms)}")
    
    # Identify water molecules
    water_molecules = identify_water_molecules(atoms)
    n_water_total = len(water_molecules)
    
    # Count CaSO4 units
    symbols = atoms.get_chemical_symbols()
    n_ca = symbols.count('Ca')
    n_s = symbols.count('S')
    n_caso4 = min(n_ca, n_s)  # Number of CaSO4 formula units
    
    if verbose:
        print(f"CaSO4 formula units: {n_caso4}")
        print(f"Water molecules found: {n_water_total}")
        print(f"Initial n value: {n_water_total / n_caso4:.3f}")
    
    # Verify we have roughly 2 waters per CaSO4
    expected_water = n_caso4 * 2
    if abs(n_water_total - expected_water) > 1:
        print(f"  Warning: Expected {expected_water} water molecules, found {n_water_total}")
    
    # Calculate H-bond scores and layer positions (for all strategies)
    hbond_scores = calculate_hbond_strength(atoms, water_molecules)
    layer_positions = get_layer_positions(atoms, water_molecules)
    
    # Generate structures for each n value and strategy
    if verbose:
        print(f"\nGenerating structures for n values: {n_values}")
        print(f"Strategies: {strategies}")
        print("-" * 60)
    
    for strategy in strategies:
        strategy_dir = output_path / strategy
        strategy_dir.mkdir(exist_ok=True)
        
        if verbose:
            print(f"\nStrategy: {strategy.upper()}")
        
        for n_target in n_values:
            # Calculate how many waters to keep/remove
            n_water_target = int(round(n_target * n_caso4))
            n_remove = n_water_total - n_water_target
            
            if n_remove < 0:
                print(f"  Skipping n={n_target}: Would need to add water (n_target > n_initial)")
                continue
            if n_remove > n_water_total:
                n_remove = n_water_total
            
            # Select waters to remove based on strategy
            if strategy == 'random':
                remove_indices = select_waters_random(water_molecules, n_remove, seed=random_seed)
            elif strategy == 'layer_ordered':
                remove_indices = select_waters_layer_ordered(water_molecules, n_remove, layer_positions)
            elif strategy == 'hbond_weakest':
                remove_indices = select_waters_hbond_weakest(water_molecules, n_remove, hbond_scores)
            else:
                print(f"  Unknown strategy: {strategy}, skipping")
                continue
            
            # Create new structure
            new_atoms = remove_water_molecules(atoms.copy(), water_molecules, remove_indices)
            
            # Calculate actual n value
            new_water = identify_water_molecules(new_atoms)
            n_actual = len(new_water) / n_caso4
            
            # Generate filename
            n_str = f"{n_target:.2f}".replace('.', 'p')
            # Use .xyz extension for extxyz format to avoid confusion
            file_ext = 'xyz' if output_format == 'extxyz' else output_format
            filename = f"CaSO4_n{n_str}.{file_ext}"
            filepath = strategy_dir / filename
            
            # Save structure
            write(str(filepath), new_atoms, format=output_format)
            
            if verbose:
                print(f"  n={n_target:.2f}: Removed {n_remove}/{n_water_total} waters -> {filepath.name} ({len(new_atoms)} atoms)")
    
    # Save original (n=2) as reference
    # Use .xyz extension for extxyz format to avoid confusion
    ref_ext = 'xyz' if output_format == 'extxyz' else output_format
    ref_path = output_path / f"CaSO4_n2p00_reference.{ref_ext}"
    write(str(ref_path), atoms, format=output_format)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Generation complete!")
        print(f"Reference structure saved: {ref_path}")
        print(f"{'='*60}")
    
    # Write log file with parameters
    log_path = output_path / "generation_log.txt"
    with open(log_path, 'w') as f:
        f.write("Gypsum Dehydration Series Generation Log\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input CIF: {input_cif}\n")
        f.write(f"Supercell: {supercell_dims}\n")
        f.write(f"Total atoms: {len(atoms)}\n")
        f.write(f"CaSO4 units: {n_caso4}\n")
        f.write(f"Water molecules: {n_water_total}\n")
        f.write(f"n values: {n_values}\n")
        f.write(f"Strategies: {strategies}\n")
        f.write(f"Random seed: {random_seed}\n")
        f.write(f"Output format: {output_format}\n")
    
    return output_path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate gypsum dehydration series for AIMD simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate with specific n values
    python generate_dehydration_series.py --cif-input gypsum.cif --n-values 2.0 1.5 1.0 0.5 0.0
    
    # Generate with step size
    python generate_dehydration_series.py --cif-input gypsum.cif --n-step 0.25
    
    # Use larger supercell for finer granularity
    python generate_dehydration_series.py --cif-input gypsum.cif --supercell 3 1 3 --n-step 0.125

Water Removal Strategies:
    random        : Uniform random selection (statistical sampling)
    layer_ordered : Remove from (010) inter-layer surfaces first (directional dehydration)
    hbond_weakest : Remove weakly H-bonded water first (thermodynamic preference)
        """
    )
    
    parser.add_argument(
        "--cif-input", "-c",
        type=str,
        default="gypsum_n2.cif",
        help="Path to input CIF file (default: gypsum_n2_clean.cif, a cleaned version of 33.cif)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./dehydration_series",
        help="Output directory (default: ./dehydration_series)"
    )
    
    # n value specification (mutually exclusive)
    n_group = parser.add_mutually_exclusive_group()
    n_group.add_argument(
        "--n-values",
        type=float,
        nargs='+',
        help="Explicit list of n values (e.g., 2.0 1.5 1.0 0.5 0.0)"
    )
    n_group.add_argument(
        "--n-step",
        type=float,
        help="Step size for n values from 2 to 0 (e.g., 0.25 -> [2.0, 1.75, 1.5, ...])"
    )
    
    parser.add_argument(
        "--supercell",
        type=int,
        nargs=3,
        default=[2, 1, 2],
        metavar=('A', 'B', 'C'),
        help="Supercell dimensions (default: 2 1 2 for CP2K efficiency)"
    )
    
    parser.add_argument(
        "--strategies",
        type=str,
        nargs='+',
        default=['random', 'layer_ordered', 'hbond_weakest'],
        choices=['random', 'layer_ordered', 'hbond_weakest'],
        help="Water removal strategies to use (default: all three)"
    )
    
    parser.add_argument(
        "--format", "-f",
        type=str,
        default="extxyz",
        choices=['cif', 'extxyz'],
        help="Output file format (default: extxyz, saves with .xyz extension)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Print detailed progress"
    )
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Determine n values
    if args.n_values:
        n_values = sorted(args.n_values, reverse=True)
    elif args.n_step:
        n_values = list(np.arange(2.0, -args.n_step/2, -args.n_step))
        n_values = [round(n, 4) for n in n_values]
    else:
        # Default: 0.5 step
        n_values = [2.0, 1.5, 1.0, 0.5, 0.0]
    
    generate_dehydration_series(
        input_cif=args.cif_input,
        output_dir=args.output_dir,
        n_values=n_values,
        supercell_dims=tuple(args.supercell),
        strategies=args.strategies,
        random_seed=args.seed,
        output_format=args.format,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
