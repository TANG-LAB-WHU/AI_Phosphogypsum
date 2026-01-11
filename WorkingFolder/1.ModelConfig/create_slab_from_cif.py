#!/usr/bin/env python3
"""

This script reads a CIF file and creates a slab model with specified Miller indices.
The output is suitable for AIMD (Ab Initio Molecular Dynamics) and MLIP 
(Machine Learning Interatomic Potentials) calculations.

Usage:
    python create_slab_from_cif.py --cif-input input.cif --miller 0 1 0 --layers 3 --vacuum 15.0
    
    
"""

import argparse
import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, TextIO

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import surface, make_supercell

# Write output to both terminal and log file
class TeeOutput:
    """
    A class that duplicates output to both terminal and a log file.
    
    This allows all print statements to be captured in a log file
    while still displaying them in the terminal.
    """
    
    def __init__(self, log_file: TextIO, terminal: TextIO):
        self.log_file = log_file
        self.terminal = terminal
    
    def write(self, message: str) -> int:
        self.terminal.write(message)
        self.log_file.write(message)
        return len(message)
    
    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

# Check if the cell is orthogonal (all angles are 90 degrees)
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

# Attempt to orthogonalize a non-orthogonal cell
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

# Parse command line arguments
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
                
                # Create a (001) slab with 2x2 in-plane supercell (layers control c direction)
                python create_slab_from_cif.py --cif-input caso4.cif --miller 0 0 1 --layers 4 --supercell 2 2
                
                # Output in VASP POSCAR format
                python create_slab_from_cif.py --cif-input input.cif --miller 1 0 0 --format vasp
        """
    )
    
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
        nargs=2,
        default=[1, 1],
        metavar=("A", "B"),
        help="Supercell dimensions in a and b directions (default: 1 1). Note: c direction is controlled by --layers."
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
        default=0.001,
        help="Tolerance in degrees for orthogonality check (default: 0.01)"
    )
    
    parser.add_argument(
        "--z-offset",
        type=float,
        default=1.0,
        help="Set the minimum z-coordinate of the slab in Angstrom (distance from z=0). Default: 1.0"
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
    
    parser.add_argument(
        "--log-file",
        type=str,
        default="auto",
        help="Path to save log file. Default: 'auto' (generates log file alongside output). Set to 'none' to disable logging."
    )
    
    return parser.parse_args()

def clean_cif_for_ase(cif_path: str, output_path: str = None, verbose: bool = False) -> str:
    """
    Clean a CIF file to make it compatible with ASE parser.
    
    Removes problematic multi-line comments (_cgraph_comments, _cgraph_title)
    that cause parsing errors in ASE.
    
    Args:
        cif_path: Path to the original CIF file
        output_path: Path to save the cleaned CIF file (default: adds '_ase' suffix)
        verbose: Print detailed information
        
    Returns:
        Path to the cleaned CIF file
    """
    import re
    
    cif_file = Path(cif_path)
    if output_path is None:
        output_path = cif_file.parent / f"{cif_file.stem}_ase.cif"
    else:
        output_path = Path(output_path)
    
    with open(cif_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Remove problematic multi-line entries
    # Pattern matches: _cgraph_comments 'any text including newlines until closing quote'
    # or _cgraph_title 'any text'
    patterns_to_remove = [
        r"_cgraph_comments\s+'[^']*'\s*\n?",  # Multi-line comments
        r"_cgraph_comments\s+\"[^\"]*\"\s*\n?",
        r"_cgraph_title\s+'[^']*'\s*\n?",
        r"_cgraph_title\s+\"[^\"]*\"\s*\n?",
        r"_eof\s*\n?",  # End of file marker
        r"#### End of Crystallographic Information File ####\s*\n?",
    ]
    
    cleaned_content = content
    for pattern in patterns_to_remove:
        cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.DOTALL | re.MULTILINE)
    
    # Also handle malformed multi-line strings (lines starting with continuation of previous string)
    # This catches cases where the comment spans multiple lines with embedded newlines
    lines = cleaned_content.split('\n')
    cleaned_lines = []
    in_multiline_comment = False
    
    for line in lines:
        # Check if we're starting a problematic entry
        if '_cgraph_comments' in line or '_cgraph_title' in line:
            # Check if it's a single-line entry (starts and ends with quotes)
            quote_count = line.count("'") + line.count('"')
            if quote_count % 2 == 0:
                # Complete entry, skip it
                continue
            else:
                # Start of multi-line, skip and set flag
                in_multiline_comment = True
                continue
        
        # If in multi-line comment, look for closing quote
        if in_multiline_comment:
            if "'" in line or '"' in line:
                in_multiline_comment = False
            continue
        
        cleaned_lines.append(line)
    
    cleaned_content = '\n'.join(cleaned_lines)
    
    # Write cleaned file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    if verbose:
        print(f"  Cleaned CIF saved to: {output_path}")
    
    return str(output_path)


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
    
    # Try to read the CIF file using ASE
    try:
        atoms = read(str(cif_file), format="cif")
    except ValueError as e:
        error_msg = str(e)
        if "Unexpected CIF file entry" in error_msg:
            # CIF has problematic multi-line comments, attempt auto-clean
            print(f"  Warning: CIF parsing error detected: {error_msg}")
            print(f"  Attempting automatic CIF cleaning...")
            
            cleaned_path = clean_cif_for_ase(cif_path, verbose=verbose)
            print(f"  Retrying with cleaned CIF: {cleaned_path}")
            
            # Retry with cleaned file
            atoms = read(cleaned_path, format="cif")
            print(f"  Success! CIF loaded after cleaning.")
        else:
            # Different error, re-raise
            raise
    
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
    supercell_dims: Tuple[int, int],
    verbose: bool = False
) -> Atoms:
    """
    Apply supercell expansion to the slab in the a and b directions only.
    
    The c direction (perpendicular to the surface) is not expanded here,
    as it is controlled by the --layers parameter during slab creation.
    
    Args:
        slab: ASE Atoms object
        supercell_dims: Supercell dimensions (a, b) - in-plane expansion only
        verbose: Print detailed information
        
    Returns:
        Expanded ASE Atoms object
    """
    a, b = supercell_dims
    
    if a == 1 and b == 1:
        return slab
    
    if verbose:
        print(f"\nApplying in-plane supercell: {a}x{b}")
    
    # Create the supercell matrix (only expand in a and b, keep c as 1)
    supercell_matrix = np.diag([a, b, 1])
    
    expanded_slab = make_supercell(slab, supercell_matrix)
    
    if verbose:
        print(f"  Expanded atoms: {len(expanded_slab)}")
        print(f"  Expanded cell: {expanded_slab.cell.cellpar()}")
    
    return expanded_slab

def apply_z_offset(
    slab: Atoms,
    z_offset: float,
    verbose: bool = False
) -> Atoms:
    """
    Shift the slab along the z-axis so that the minimum z-coordinate equals z_offset.
    
    Args:
        slab: ASE Atoms object
        z_offset: Target minimum z-coordinate (distance from z=0)
        verbose: Print detailed information
        
    Returns:
        ASE Atoms object with adjusted z positions
    """
    positions = slab.get_positions()
    current_z_min = positions[:, 2].min()
    shift = z_offset - current_z_min
    
    if verbose:
        print(f"\nApplying z-offset:")
        print(f"  Current z_min: {current_z_min:.4f} Angstrom")
        print(f"  Target z_min: {z_offset:.4f} Angstrom")
        print(f"  Shift: {shift:.4f} Angstrom")
    
    # Apply the shift
    positions[:, 2] += shift
    slab.set_positions(positions)
    
    if verbose:
        print(f"  New z range: [{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}]")
    
    return slab

def get_output_filename(
    cif_path: str,
    miller_indices: Tuple[int, int, int],
    layers: int,
    supercell_dims: Tuple[int, int],
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
        supercell_dims: Supercell dimensions (a, b) - in-plane only
        output_format: Output file format
        output_dir: Optional output directory
        output_file_name: Optional explicit output filename
        
    Returns:
        Generated output filename
    """
    cif_file = Path(cif_path)
    base_name = cif_file.stem
    
    h, k, l = miller_indices
    a, b = supercell_dims
    
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
        supercell_str = f"_{a}x{b}" if not (a == 1 and b == 1) else ""
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
    cell_params = slab.cell.cellpar()
    print(f"  Lattice parameters: {cell_params[0]:.4f} {cell_params[1]:.4f} {cell_params[2]:.4f} {cell_params[3]:.2f} {cell_params[4]:.2f} {cell_params[5]:.2f}")

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
    
    # Setup logging if requested
    log_file_handle = None
    original_stdout = sys.stdout
    
    try:
        # Determine output path first (needed for auto log file naming)
        output_path = get_output_filename(
            cif_path=args.cif_input,
            miller_indices=tuple(args.miller),
            layers=args.layers,
            supercell_dims=tuple(args.supercell),
            output_format=args.format,
            output_dir=args.output_dir,
            output_file_name=args.output_file
        )
        
        # Setup log file if requested (skip if set to 'none')
        if args.log_file and args.log_file.lower() != "none":
            if args.log_file.lower() == "auto":
                # Generate log file path alongside output file
                output_path_obj = Path(output_path)
                log_path = output_path_obj.with_suffix(".log")
            else:
                log_path = Path(args.log_file)
            
            # Ensure parent directory exists
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file_handle = open(log_path, "w", encoding="utf-8")
            sys.stdout = TeeOutput(log_file_handle, original_stdout)
            
            # Write log header with all parameters
            print(f"# Log generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"# Command: python create_slab_from_cif.py \\")
            print(f"#   --cif-input {args.cif_input} \\")
            print(f"#   --miller {' '.join(map(str, args.miller))} \\")
            print(f"#   --layers {args.layers} \\")
            print(f"#   --vacuum {args.vacuum} \\")
            print(f"#   --supercell {' '.join(map(str, args.supercell))} \\")
            print(f"#   --z-offset {args.z_offset} \\")
            print(f"#   --format {args.format} \\")
            print(f"#   --ortho-tolerance {args.ortho_tolerance}", end="")
            if args.require_ortho:
                print(" \\")
                print("#   --require-ortho", end="")
            if args.skip_ortho:
                print(" \\")
                print("#   --skip-ortho", end="")
            if args.periodic:
                print(" \\")
                print("#   --periodic", end="")
            if args.verbose:
                print(" \\")
                print("#   --verbose", end="")
            print("")  # End of command line
        
        print("\n" + "=" * 60)
        print("CaSO4 SLAB MODEL GENERATOR")
        print("=" * 60)
        
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
        
        # Apply z-offset if specified
        if args.z_offset is not None:
            print("\n" + "-" * 60)
            print("Z-OFFSET ADJUSTMENT")
            print("-" * 60)
            positions = slab.get_positions()
            current_z_min = positions[:, 2].min()
            current_z_max = positions[:, 2].max()
            print(f"Before: z range = [{current_z_min:.4f}, {current_z_max:.4f}] Angstrom")
            
            slab = apply_z_offset(slab, args.z_offset, verbose=args.verbose)
            
            positions = slab.get_positions()
            new_z_min = positions[:, 2].min()
            new_z_max = positions[:, 2].max()
            print(f"After:  z range = [{new_z_min:.4f}, {new_z_max:.4f}] Angstrom")
            print(f"Shift applied: {args.z_offset - current_z_min:.4f} Angstrom")
            print("-" * 60)
        
        # Print slab information
        print_slab_info(slab)
        
        # Save the slab
        save_slab(
            slab=slab,
            output_path=output_path,
            output_format=args.format,
            verbose=args.verbose
        )
        
        print("\nSlab model creation completed successfully!")
        print("The output file can be used for AIMD and MLIP calculations.")
        
        if log_file_handle:
            print(f"\nLog saved to: {log_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating slab: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        if log_file_handle:
            log_file_handle.close()


if __name__ == "__main__":
    main()

