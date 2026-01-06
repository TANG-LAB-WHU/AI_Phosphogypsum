#!/usr/bin/env python3
"""
Create CaSO4 Slab Model from CIF File

This script reads a CIF file and creates a slab model with specified Miller indices.
The output is suitable for AIMD (Ab Initio Molecular Dynamics) and MLIP 
(Machine Learning Interatomic Potentials) calculations.

Dependencies:
    - ase (Atomic Simulation Environment): pip install ase
    - pymatgen (optional, for advanced CIF parsing): pip install pymatgen

Usage:
    python create_slab_from_cif.py --cif-input input.cif --miller 0 1 0 --layers 3 --vacuum 15.0
    
    
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

try:
    from ase import Atoms
    from ase.io import read, write
    from ase.build import surface, make_supercell
except ImportError:
    print("Error: ASE (Atomic Simulation Environment) is required.")
    print("Install with: pip install ase")
    sys.exit(1)


def check_orthogonality(atoms: Atoms, tolerance: float = 0.1) -> Tuple[bool, np.ndarray]:
    """
    Check if the cell is orthogonal (all angles are 90 degrees).
    
    Args:
        atoms: ASE Atoms object
        tolerance: Tolerance in degrees for angle comparison (default: 0.1)
        
    Returns:
        Tuple of (is_orthogonal, cell_angles)
    """
    cell_params = atoms.cell.cellpar()
    angles = cell_params[3:6]  # alpha, beta, gamma
    
    is_orthogonal = all(abs(angle - 90.0) < tolerance for angle in angles)
    
    return is_orthogonal, angles


def orthogonalize_cell(atoms: Atoms, verbose: bool = False) -> Atoms:
    """
    Attempt to orthogonalize a non-orthogonal cell.
    
    This function uses the Niggli reduction and then attempts to find
    an equivalent orthogonal representation if possible.
    
    Args:
        atoms: ASE Atoms object with potentially non-orthogonal cell
        verbose: Print detailed information
        
    Returns:
        ASE Atoms object with orthogonalized cell (or original if not possible)
    """
    from ase.build import niggli_reduce
    
    # Make a copy to avoid modifying the original
    atoms_copy = atoms.copy()
    
    # Try Niggli reduction first to get a more standard cell
    try:
        niggli_reduce(atoms_copy)
        if verbose:
            print("  Applied Niggli reduction to standardize the cell")
    except Exception as e:
        if verbose:
            print(f"  Niggli reduction not applicable: {e}")
    
    # Check if the cell is now orthogonal
    is_orth, angles = check_orthogonality(atoms_copy)
    
    if is_orth:
        if verbose:
            print("  Cell is now orthogonal after Niggli reduction")
        return atoms_copy
    
    # If still not orthogonal, try to find an orthogonal supercell
    # This is a more aggressive approach
    if verbose:
        print("  Attempting to find orthogonal supercell...")
    
    try:
        from ase.build import find_optimal_cell_shape
        
        # Find a more orthogonal cell shape
        # Target size is approximately the same as original
        target_size = len(atoms)
        
        cell = atoms.get_cell()
        
        # Try different supercell matrices
        best_atoms = atoms_copy
        best_deviation = sum(abs(a - 90.0) for a in angles)
        
        # Try simple supercell transformations
        test_matrices = [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [1, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 0], [1, 0, 1]],
            [[1, 0, 1], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 1, 1], [0, 0, 1]],
        ]
        
        for matrix in test_matrices:
            try:
                test_atoms = make_supercell(atoms, matrix)
                is_orth_test, angles_test = check_orthogonality(test_atoms)
                deviation = sum(abs(a - 90.0) for a in angles_test)
                
                if deviation < best_deviation:
                    best_deviation = deviation
                    best_atoms = test_atoms
                    if verbose:
                        print(f"    Found better cell with angles: {angles_test}")
                
                if is_orth_test:
                    if verbose:
                        print(f"    Found orthogonal supercell with matrix {matrix}")
                    return test_atoms
            except Exception:
                continue
        
        return best_atoms
        
    except ImportError:
        if verbose:
            print("  Could not find orthogonal supercell")
        return atoms_copy


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Create a slab model from a CIF file with specified Miller indices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create a (010) slab with 3 layers and 15 Angstrom vacuum
    python create_slab_from_cif.py --cif-input gypsum.cif --miller 0 1 0 --layers 3 --vacuum 15.0
    
    # Create a (001) slab with 2x2 supercell
    python create_slab_from_cif.py --cif-input caso4.cif --miller 0 0 1 --layers 4 --supercell 2 2 1
    
    # Output in VASP POSCAR format
    python create_slab_from_cif.py --cif-input input.cif --miller 1 0 0 --format vasp
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--cif-input", "-c",
        type=str,
        required=True,
        help="Path to the input CIF file"
    )
    
    parser.add_argument(
        "--miller", "-m",
        type=int,
        nargs=3,
        required=True,
        metavar=("H", "K", "L"),
        help="Miller indices (h k l) for the surface"
    )
    
    # Optional arguments
    parser.add_argument(
        "--layers", "-l",
        type=int,
        default=3,
        help="Number of layers in the slab (default: 3)"
    )
    
    parser.add_argument(
        "--vacuum", "-v",
        type=float,
        default=15.0,
        help="Vacuum thickness in Angstroms (default: 15.0)"
    )
    
    parser.add_argument(
        "--supercell", "-s",
        type=int,
        nargs=3,
        default=[1, 1, 1],
        metavar=("A", "B", "C"),
        help="Supercell dimensions (default: 1 1 1)"
    )
    
    parser.add_argument(
        "--output-dir", "-od",
        type=str,
        default=None,
        help="Output directory (default: same as input CIF file)"
    )
    
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        default=None,
        help="Output filename (default: auto-generated with Miller indices)"
    )
    
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["xyz", "vasp", "cif", "extxyz", "lammps-data"],
        default="extxyz",
        help="Output file format (default: extxyz)"
    )
    
    parser.add_argument(
        "--periodic", "-p",
        action="store_true",
        default=False,
        help="Keep the slab periodic in all directions (no vacuum)"
    )
    
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.001,
        help="Tolerance for symmetry detection (default: 0.001)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information during processing"
    )
    
    parser.add_argument(
        "--ortho-tolerance",
        type=float,
        default=0.01,
        help="Tolerance in degrees for orthogonality check (default: 0.01)"
    )
    
    parser.add_argument(
        "--require-ortho",
        action="store_true",
        default=False,
        help="Exit with error if cell cannot be made fully orthogonal"
    )
    
    parser.add_argument(
        "--skip-ortho",
        action="store_true",
        default=False,
        help="Skip automatic orthogonalization (use original cell even if non-orthogonal)"
    )
    
    return parser.parse_args()


def load_structure_from_cif(cif_path: str, verbose: bool = False) -> Atoms:
    """
    Load crystal structure from a CIF file.
    
    Args:
        cif_path: Path to the CIF file
        verbose: Print detailed information
        
    Returns:
        ASE Atoms object with the crystal structure
    """
    cif_file = Path(cif_path)
    
    if not cif_file.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
    
    if verbose:
        print(f"Loading CIF file: {cif_file}")
    
    # Read the CIF file using ASE
    atoms = read(str(cif_file), format="cif")
    
    if verbose:
        print(f"  Number of atoms: {len(atoms)}")
        print(f"  Chemical formula: {atoms.get_chemical_formula()}")
        print(f"  Cell parameters: {atoms.cell.cellpar()}")
    
    return atoms


def create_slab(
    bulk: Atoms,
    miller_indices: Tuple[int, int, int],
    layers: int,
    vacuum: float,
    periodic: bool = False,
    center: bool = True,
    tolerance: float = 0.001,
    verbose: bool = False
) -> Atoms:
    """
    Create a slab model from bulk structure.
    
    Args:
        bulk: Bulk ASE Atoms object
        miller_indices: Miller indices (h, k, l)
        layers: Number of layers
        vacuum: Vacuum thickness in Angstroms
        periodic: If True, keep periodic in all directions
        center: If True, center the slab in the cell
        tolerance: Tolerance for structure operations
        verbose: Print detailed information
        
    Returns:
        ASE Atoms object representing the slab
    """
    h, k, l = miller_indices
    
    if verbose:
        print(f"\nCreating slab with Miller indices ({h} {k} {l})")
        print(f"  Layers: {layers}")
        print(f"  Vacuum: {vacuum} Angstrom")
    
    # Create the surface slab
    # The surface() function creates a slab with the specified Miller indices
    slab = surface(
        bulk,
        indices=(h, k, l),
        layers=layers,
        vacuum=vacuum if not periodic else 0.0,
        tol=tolerance,
        periodic=periodic
    )
    
    if center and not periodic:
        # Center the slab in the unit cell
        slab.center(vacuum=vacuum / 2, axis=2)
    
    if verbose:
        print(f"  Slab atoms: {len(slab)}")
        print(f"  Slab cell: {slab.cell.cellpar()}")
    
    return slab


def apply_supercell(
    slab: Atoms,
    supercell_dims: Tuple[int, int, int],
    verbose: bool = False
) -> Atoms:
    """
    Apply supercell expansion to the slab.
    
    Args:
        slab: ASE Atoms object
        supercell_dims: Supercell dimensions (a, b, c)
        verbose: Print detailed information
        
    Returns:
        Expanded ASE Atoms object
    """
    a, b, c = supercell_dims
    
    if a == 1 and b == 1 and c == 1:
        return slab
    
    if verbose:
        print(f"\nApplying supercell: {a}x{b}x{c}")
    
    # Create the supercell matrix
    supercell_matrix = np.diag([a, b, c])
    
    expanded_slab = make_supercell(slab, supercell_matrix)
    
    if verbose:
        print(f"  Expanded atoms: {len(expanded_slab)}")
        print(f"  Expanded cell: {expanded_slab.cell.cellpar()}")
    
    return expanded_slab


def get_output_filename(
    cif_path: str,
    miller_indices: Tuple[int, int, int],
    layers: int,
    supercell_dims: Tuple[int, int, int],
    output_format: str,
    output_dir: Optional[str] = None,
    output_file_name: Optional[str] = None
) -> str:
    """
    Generate output filename based on parameters.
    
    Args:
        cif_path: Original CIF file path
        miller_indices: Miller indices
        layers: Number of layers
        supercell_dims: Supercell dimensions
        output_format: Output file format
        output_dir: Optional output directory
        output_file_name: Optional explicit output filename
        
    Returns:
        Generated output filename
    """
    cif_file = Path(cif_path)
    base_name = cif_file.stem
    
    h, k, l = miller_indices
    a, b, c = supercell_dims
    
    # Format extension mapping
    ext_map = {
        "xyz": ".xyz",
        "extxyz": ".xyz",
        "vasp": ".vasp",
        "cif": ".cif",
        "lammps-data": ".lmp"
    }
    
    ext = ext_map.get(output_format, ".xyz")
    
    # Create descriptive filename
    if output_file_name:
        output_name = output_file_name
    else:
        miller_str = f"{h}{k}{l}"
        supercell_str = f"_{a}x{b}x{c}" if not (a == 1 and b == 1 and c == 1) else ""
        output_name = f"{base_name}_slab_{miller_str}_L{layers}{supercell_str}{ext}"
    
    if output_dir:
        out_path = Path(output_dir)
        if not out_path.exists():
            try:
                out_path.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        return str(out_path / output_name)
    
    return str(cif_file.parent / output_name)


def save_slab(
    slab: Atoms,
    output_path: str,
    output_format: str,
    verbose: bool = False
) -> None:
    """
    Save the slab to a file.
    
    Args:
        slab: ASE Atoms object
        output_path: Output file path
        output_format: Output file format
        verbose: Print detailed information
    """
    if verbose:
        print(f"\nSaving slab to: {output_path}")
        print(f"  Format: {output_format}")
    
    # Write the output file
    write(output_path, slab, format=output_format)
    
    print(f"\nOutput saved: {output_path}")
    print(f"  Total atoms: {len(slab)}")
    print(f"  Chemical formula: {slab.get_chemical_formula()}")
    print(f"  Cell dimensions: {slab.cell.cellpar()[:3].round(4)}")
    print(f"  Cell angles: {slab.cell.cellpar()[3:].round(2)}")


def print_slab_info(slab: Atoms) -> None:
    """
    Print detailed information about the slab.
    
    Args:
        slab: ASE Atoms object
    """
    print("\n" + "=" * 60)
    print("SLAB MODEL INFORMATION")
    print("=" * 60)
    
    # Atom counts by element
    symbols = slab.get_chemical_symbols()
    unique_symbols = sorted(set(symbols))
    
    print("\nAtom Counts:")
    for symbol in unique_symbols:
        count = symbols.count(symbol)
        print(f"  {symbol:3s}: {count:5d}")
    
    print(f"\n  Total: {len(slab):5d} atoms")
    
    # Cell information
    print("\nCell Parameters:")
    cell_params = slab.cell.cellpar()
    print(f"  a = {cell_params[0]:.4f} Angstrom")
    print(f"  b = {cell_params[1]:.4f} Angstrom")
    print(f"  c = {cell_params[2]:.4f} Angstrom")
    print(f"  alpha = {cell_params[3]:.2f} degrees")
    print(f"  beta  = {cell_params[4]:.2f} degrees")
    print(f"  gamma = {cell_params[5]:.2f} degrees")
    
    # Cell volume
    volume = slab.get_volume()
    print(f"\nCell Volume: {volume:.2f} Angstrom^3")
    
    # Position ranges
    positions = slab.get_positions()
    print("\nPosition Ranges:")
    print(f"  x: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}]")
    print(f"  y: [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}]")
    print(f"  z: [{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}]")
    
    print("=" * 60)


def main():
    """Main function to create slab model from CIF file."""
    args = parse_arguments()
    
    print("\n" + "=" * 60)
    print("CaSO4 SLAB MODEL GENERATOR")
    print("=" * 60)
    
    try:
        # Load the bulk crystal structure
        bulk = load_structure_from_cif(args.cif_input, verbose=args.verbose)
        
        # Create the slab
        slab = create_slab(
            bulk=bulk,
            miller_indices=tuple(args.miller),
            layers=args.layers,
            vacuum=args.vacuum,
            periodic=args.periodic,
            center=True,
            tolerance=args.tolerance,
            verbose=args.verbose
        )
        
        # Apply supercell expansion
        slab = apply_supercell(
            slab=slab,
            supercell_dims=tuple(args.supercell),
            verbose=args.verbose
        )
        
        # Check orthogonality and auto-orthogonalize if needed
        is_orthogonal, angles = check_orthogonality(slab, tolerance=args.ortho_tolerance)
        
        print("\n" + "-" * 60)
        print("ORTHOGONALITY CHECK")
        print("-" * 60)
        print(f"Cell angles: alpha={angles[0]:.4f}, beta={angles[1]:.4f}, gamma={angles[2]:.4f}")
        print(f"Tolerance: {args.ortho_tolerance} degrees")
        
        if is_orthogonal:
            print("Status: ORTHOGONAL - Cell satisfies orthogonality criteria")
            print("Action: Direct output (no orthogonalization needed)")
        else:
            print("Status: NON-ORTHOGONAL")
            print(f"  Deviation from 90: alpha={abs(angles[0]-90):.4f}, beta={abs(angles[1]-90):.4f}, gamma={abs(angles[2]-90):.4f}")
            
            if args.skip_ortho:
                # User requested to skip orthogonalization
                print("\nAction: Skipping orthogonalization (--skip-ortho flag set)")
                print("Warning: Non-orthogonal cell may cause issues in some AIMD/MLIP codes.")
            else:
                # Automatically perform orthogonalization
                print("\nPerforming automatic orthogonalization...")
                original_atom_count = len(slab)
                slab = orthogonalize_cell(slab, verbose=args.verbose)
                
                # Re-check orthogonality
                is_orthogonal_after, angles_after = check_orthogonality(slab, tolerance=args.ortho_tolerance)
                print(f"\nAfter orthogonalization:")
                print(f"  Cell angles: alpha={angles_after[0]:.4f}, beta={angles_after[1]:.4f}, gamma={angles_after[2]:.4f}")
                print(f"  Atom count: {original_atom_count} -> {len(slab)}")
                
                if is_orthogonal_after:
                    print("  Result: SUCCESS - Cell is now orthogonal")
                else:
                    deviation_after = sum(abs(a - 90.0) for a in angles_after)
                    deviation_before = sum(abs(a - 90.0) for a in angles)
                    improvement = (1 - deviation_after / deviation_before) * 100 if deviation_before > 0 else 0
                    print(f"  Result: IMPROVED (total deviation reduced by {improvement:.1f}%)")
                    
                    if args.require_ortho:
                        print("\nError: --require-ortho is set but cell could not be fully orthogonalized.")
                        print("The output cell is still non-orthogonal.")
                        sys.exit(1)
                    else:
                        print("  Note: Cell is more orthogonal but not fully. Output will proceed.")
        
        print("-" * 60)
        
        # Print slab information
        print_slab_info(slab)
        
        # Determine output path
        output_path = get_output_filename(
            cif_path=args.cif_input,
            miller_indices=tuple(args.miller),
            layers=args.layers,
            supercell_dims=tuple(args.supercell),
            output_format=args.format,
            output_dir=args.output_dir,
            output_file_name=args.output_file
        )
        
        # Save the slab
        save_slab(
            slab=slab,
            output_path=output_path,
            output_format=args.format,
            verbose=args.verbose
        )
        
        print("\nSlab model creation completed successfully!")
        print("The output file can be used for AIMD and MLIP calculations.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating slab: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
