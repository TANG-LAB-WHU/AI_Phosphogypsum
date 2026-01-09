#!/usr/bin/env python3
"""
This script adds reactant molecules (H2O, NH3, NH4+, HPO4^2-) to the vacuum region
of CaSO4 slab models for AIMD simulations of phosphogypsum chemical conversion.

Reaction system: CaSO4 + (NH4)2HPO4 + NH3·H2O → Ca5(PO4)3OH + (NH4)2SO4

Usage:
    python NH4_packing_CSOslab.py --slab-input slab.xyz --n-water 30 --n-nh3 5 --n-nh4 5 --n-hpo4 3
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from ase import Atoms
from ase.io import read, write


# =============================================================================
# Molecule Geometry Definitions
# =============================================================================

def create_water() -> Atoms:
    """
    Create a water molecule (H2O).
    
    Geometry: O-H bond 0.9572 Å, H-O-H angle 104.52°
    
    Returns:
        ASE Atoms object for H2O
    """
    # H2O geometry with O at origin
    angle = np.radians(104.52 / 2)
    d_oh = 0.9572
    
    positions = [
        [0.0, 0.0, 0.0],  # O
        [d_oh * np.sin(angle), 0.0, d_oh * np.cos(angle)],  # H1
        [-d_oh * np.sin(angle), 0.0, d_oh * np.cos(angle)]  # H2
    ]
    
    return Atoms('OH2', positions=positions)


def create_ammonia() -> Atoms:
    """
    Create an ammonia molecule (NH3).
    
    Geometry: N-H bond 1.017 Å, H-N-H angle 107.8°, pyramidal structure
    
    Returns:
        ASE Atoms object for NH3
    """
    d_nh = 1.017
    angle = np.radians(107.8)
    
    # Calculate H positions in pyramidal geometry
    # N at origin, H atoms form a triangular base
    h_z = d_nh * np.cos(np.radians(68.0))  # Height from N plane
    h_r = d_nh * np.sin(np.radians(68.0))  # Radial distance
    
    positions = [
        [0.0, 0.0, 0.0],  # N
        [h_r, 0.0, -h_z],  # H1
        [-h_r * 0.5, h_r * np.sqrt(3) / 2, -h_z],  # H2
        [-h_r * 0.5, -h_r * np.sqrt(3) / 2, -h_z]  # H3
    ]
    
    return Atoms('NH3', positions=positions)


def create_ammonium() -> Atoms:
    """
    Create an ammonium ion (NH4+).
    
    Geometry: Tetrahedral, N-H bond 1.02 Å
    
    Returns:
        ASE Atoms object for NH4+
    """
    d_nh = 1.02
    
    # Tetrahedral coordinates: N at center, 4 H at vertices
    # Using normalized tetrahedral vertices
    t = d_nh / np.sqrt(3)
    
    positions = [
        [0.0, 0.0, 0.0],  # N (center)
        [t, t, t],        # H1
        [t, -t, -t],      # H2
        [-t, t, -t],      # H3
        [-t, -t, t]       # H4
    ]
    
    return Atoms('NH4', positions=positions)


def create_hydrogen_phosphate() -> Atoms:
    """
    Create a hydrogen phosphate ion (HPO4^2-).
    
    Geometry: Tetrahedral around P
    - P=O (double bond): 1.52 Å
    - P-O (anionic): 1.54 Å  
    - P-OH: 1.57 Å
    - O-H: 0.97 Å
    
    Returns:
        ASE Atoms object for HPO4^2-
    """
    # Tetrahedral geometry around phosphorus
    d_po_double = 1.52
    d_po_anion = 1.54
    d_poh = 1.57
    d_oh = 0.97
    
    # Use tetrahedral angles
    t = 1.0 / np.sqrt(3)
    
    # P at origin
    # O1 (double bond) at +x+y+z direction
    # O2, O3 (anionic) at other tetrahedral positions
    # O4 (OH group) at remaining position
    
    positions = [
        [0.0, 0.0, 0.0],  # P (center)
        [d_po_double * t, d_po_double * t, d_po_double * t],  # O1 (=O)
        [d_po_anion * t, -d_po_anion * t, -d_po_anion * t],   # O2 (O-)
        [-d_po_anion * t, d_po_anion * t, -d_po_anion * t],   # O3 (O-)
        [-d_poh * t, -d_poh * t, d_poh * t],                  # O4 (OH)
        [-d_poh * t - d_oh * t, -d_poh * t - d_oh * t, d_poh * t + d_oh * t]  # H (on O4)
    ]
    
    return Atoms('PO4H', positions=positions)


# =============================================================================
# Packing Functions
# =============================================================================

def random_rotation_matrix() -> np.ndarray:
    """
    Generate a random 3D rotation matrix.
    
    Returns:
        3x3 rotation matrix
    """
    # Random axis
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    
    # Random angle
    angle = np.random.uniform(0, 2 * np.pi)
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return R


def rotate_molecule(mol: Atoms) -> Atoms:
    """
    Apply random rotation to a molecule.
    
    Args:
        mol: ASE Atoms object
        
    Returns:
        Rotated molecule (new Atoms object)
    """
    rotated = mol.copy()
    R = random_rotation_matrix()
    
    positions = rotated.get_positions()
    center = positions.mean(axis=0)
    
    # Rotate around center
    rotated_positions = np.dot(positions - center, R.T) + center
    rotated.set_positions(rotated_positions)
    
    return rotated


def check_collision(
    new_positions: np.ndarray,
    existing_positions: np.ndarray,
    min_distance: float
) -> bool:
    """
    Check if new positions collide with existing atoms.
    
    Args:
        new_positions: Positions of new molecule atoms
        existing_positions: Positions of existing atoms
        min_distance: Minimum allowed distance between atoms
        
    Returns:
        True if collision detected, False otherwise
    """
    if len(existing_positions) == 0:
        return False
    
    for new_pos in new_positions:
        distances = np.linalg.norm(existing_positions - new_pos, axis=1)
        if np.any(distances < min_distance):
            return True
    
    return False


def estimate_required_vacuum(
    molecules: List[Tuple[Atoms, int]],
    min_distance: float,
    z_buffer: float,
    packing_efficiency: float = 0.4
) -> float:
    """
    Estimate the required vacuum layer height based on molecules to be packed.
    
    Args:
        molecules: List of (molecule, count) tuples
        min_distance: Minimum distance between atoms
        z_buffer: Buffer distance from surfaces
        packing_efficiency: Expected packing efficiency (0-1), lower = more space
        
    Returns:
        Estimated required vacuum height in Angstrom
    """
    total_volume = 0.0
    max_mol_height = 0.0
    
    for mol_template, count in molecules:
        positions = mol_template.get_positions()
        mol_extent = positions.max(axis=0) - positions.min(axis=0)
        
        # Approximate molecular volume as a sphere with radius = half of max extent + min_distance
        mol_radius = max(mol_extent) / 2 + min_distance / 2
        mol_volume = (4/3) * np.pi * mol_radius**3
        total_volume += mol_volume * count
        
        # Track largest molecule height
        max_mol_height = max(max_mol_height, max(mol_extent))
    
    # The minimum height should accommodate at least a few layers of the largest molecule
    # plus buffers on both ends
    min_height = max_mol_height + 2 * z_buffer + min_distance * 2
    
    # Estimate based on packing efficiency (assuming we have a and b cell dimensions)
    # This will be refined when we know the actual cell dimensions
    estimated_height = max(min_height, 15.0)  # At least 15 Angstrom
    
    return estimated_height, total_volume, max_mol_height


def pack_molecules(
    slab: Atoms,
    molecules: List[Tuple[Atoms, int]],
    min_distance: float = 2.5,
    z_buffer: float = 2.0,
    packing_vacuum: float = 15.0,
    target_vacuum: float = None,
    auto_vacuum: bool = True,
    max_attempts: int = 1000,
    no_boundary_cross: bool = False,
    verbose: bool = False
) -> Tuple[Atoms, dict]:
    """
    Pack molecules into the vacuum region above the slab.
    
    Args:
        slab: ASE Atoms object representing the slab
        molecules: List of (molecule, count) tuples
        min_distance: Minimum distance between atoms
        z_buffer: Minimum distance from slab surface to molecules
        packing_vacuum: Height of molecule packing region
        target_vacuum: Target final vacuum thickness above molecules (adjusts cell after packing)
        auto_vacuum: If True, automatically expand packing region to fit all molecules
        max_attempts: Maximum placement attempts per molecule
        no_boundary_cross: If True, ensure molecules don't cross x/y boundaries
        verbose: Print detailed information
        
    Returns:
        Tuple of (combined structure, packing statistics)
    """
    # Get cell parameters
    cell = slab.get_cell().copy()
    a, b = cell[0, 0], cell[1, 1]
    
    # Determine slab surface position
    slab_positions = slab.get_positions()
    z_max_slab = slab_positions[:, 2].max()
    z_min_slab = slab_positions[:, 2].min()
    slab_thickness = z_max_slab - z_min_slab
    
    # Calculate packing region height
    if auto_vacuum and molecules:
        # Estimate based on molecules
        est_height, est_volume, max_mol_h = estimate_required_vacuum(
            molecules, min_distance, z_buffer
        )
        # Refine based on actual cell dimensions
        cell_area = a * b
        # Volume-based height estimation with packing efficiency
        volume_based_height = est_volume / (cell_area * 0.4) + max_mol_h + z_buffer
        packing_height = max(packing_vacuum, est_height, volume_based_height)
        
        if verbose:
            print(f"\nAuto-vacuum estimation:")
            print(f"  Estimated molecular volume: {est_volume:.1f} Angstrom^3")
            print(f"  Max molecule height: {max_mol_h:.2f} Angstrom")
            print(f"  Calculated packing height: {packing_height:.2f} Angstrom")
    else:
        packing_height = packing_vacuum
    
    # Set packing region (molecules go between z_buffer above slab and packing_height)
    z_min_pack = z_max_slab + z_buffer
    z_max_pack = z_min_pack + packing_height
    
    # Calculate initial cell height (will be adjusted later if target_vacuum specified)
    # Add z_buffer at top for initial packing space
    new_c = z_max_pack + z_buffer
    
    # Create working copy with adjusted cell
    combined = slab.copy()
    new_cell = cell.copy()
    new_cell[2, 2] = new_c
    combined.set_cell(new_cell)
    
    if verbose:
        print(f"\nPacking region: z = [{z_min_pack:.2f}, {z_max_pack:.2f}] Angstrom")
        print(f"Cell dimensions: a={a:.2f}, b={b:.2f}, c={new_c:.2f}")
        print(f"Packing region height: {packing_height:.2f} Angstrom")
    
    # Start with slab atoms positions for collision detection
    all_positions = slab_positions.copy()
    
    stats = {
        'total_placed': 0,
        'total_failed': 0,
        'by_molecule': {}
    }
    
    for mol_template, count in molecules:
        mol_name = mol_template.get_chemical_formula()
        placed = 0
        failed = 0
        
        if verbose:
            print(f"\nPacking {count} x {mol_name}...")
        
        for i in range(count):
            success = False
            
            for attempt in range(max_attempts):
                # Random rotation
                mol = rotate_molecule(mol_template)
                mol_positions = mol.get_positions()
                
                # Random position in packing region
                mol_center = mol_positions.mean(axis=0)
                mol_extent = mol_positions.max(axis=0) - mol_positions.min(axis=0)
                mol_min = mol_positions.min(axis=0)
                mol_max = mol_positions.max(axis=0)
                
                # Calculate placement bounds
                if no_boundary_cross:
                    # Ensure entire molecule stays within [0, a] x [0, b]
                    x_min_bound = -mol_min[0] + mol_center[0] + 0.1
                    x_max_bound = a - mol_max[0] + mol_center[0] - 0.1
                    y_min_bound = -mol_min[1] + mol_center[1] + 0.1
                    y_max_bound = b - mol_max[1] + mol_center[1] - 0.1
                else:
                    # Allow molecules to cross boundaries (will be wrapped)
                    x_min_bound = mol_extent[0] / 2
                    x_max_bound = a - mol_extent[0] / 2
                    y_min_bound = mol_extent[1] / 2
                    y_max_bound = b - mol_extent[1] / 2
                
                # Skip if bounds are invalid
                if x_min_bound >= x_max_bound or y_min_bound >= y_max_bound:
                    continue
                
                # Generate random position
                x = np.random.uniform(x_min_bound, x_max_bound)
                y = np.random.uniform(y_min_bound, y_max_bound)
                z = np.random.uniform(z_min_pack + mol_extent[2] / 2, 
                                     z_max_pack - mol_extent[2] / 2)
                
                # Translate molecule
                new_center = np.array([x, y, z])
                translated_positions = mol_positions - mol_center + new_center
                
                # Check collision
                if not check_collision(translated_positions, all_positions, min_distance):
                    # Place molecule
                    mol.set_positions(translated_positions)
                    combined += mol
                    all_positions = np.vstack([all_positions, translated_positions])
                    placed += 1
                    success = True
                    break
            
            if not success:
                failed += 1
                if verbose:
                    print(f"  Warning: Failed to place molecule {i+1}/{count} after {max_attempts} attempts")
        
        stats['by_molecule'][mol_name] = {'placed': placed, 'failed': failed}
        stats['total_placed'] += placed
        stats['total_failed'] += failed
        
        if verbose:
            print(f"  Placed: {placed}/{count}")
    
    return combined, stats


# =============================================================================
# I/O and Main Functions
# =============================================================================

class TeeOutput:
    """Duplicate output to both terminal and log file."""
    
    def __init__(self, log_file, terminal):
        self.log_file = log_file
        self.terminal = terminal
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        return len(message)
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pack reactant molecules onto CaSO4 slab models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Pack 30 water, 5 NH3, 5 NH4+ and 3 HPO4^2- molecules
    python NH4_packing_CSOslab.py --slab-input slab.xyz --n-water 30 --n-nh3 5 --n-nh4 5 --n-hpo4 3
    
    # Custom minimum distance and output format
    python NH4_packing_CSOslab.py --slab-input slab.xyz --n-water 50 --min-distance 3.0 --format vasp
        """
    )
    
    parser.add_argument(
        "--slab-input", "-i",
        type=str,
        required=True,
        help="Path to input slab structure file (xyz, extxyz, etc.)"
    )
    
    parser.add_argument(
        "--n-water",
        type=int,
        default=0,
        help="Number of H2O molecules to add (default: 0)"
    )
    
    parser.add_argument(
        "--n-nh3",
        type=int,
        default=0,
        help="Number of NH3 molecules to add (default: 0)"
    )
    
    parser.add_argument(
        "--n-nh4",
        type=int,
        default=0,
        help="Number of NH4+ ions to add (default: 0)"
    )
    
    parser.add_argument(
        "--n-hpo4",
        type=int,
        default=0,
        help="Number of HPO4^2- ions to add (default: 0)"
    )
    
    parser.add_argument(
        "--min-distance",
        type=float,
        default=2.5,
        help="Minimum distance between atoms in Angstrom (default: 2.5)"
    )
    
    parser.add_argument(
        "--z-buffer",
        type=float,
        default=2.0,
        help="Minimum distance from slab surface to molecules in Angstrom (default: 2.0)"
    )
    
    parser.add_argument(
        "--packing-vacuum",
        type=float,
        default=15.0,
        help="Height of molecule packing region in Angstrom (default: 15.0)"
    )
    
    parser.add_argument(
        "--target-vacuum",
        type=float,
        default=None,
        help="Target vacuum thickness above molecules in Angstrom. If specified, the cell will be adjusted to achieve this vacuum. If not specified, uses packing-vacuum to determine cell height."
    )
    
    parser.add_argument(
        "--auto-vacuum",
        action="store_true",
        default=True,
        help="Automatically adjust packing vacuum height to fit all molecules (default: True)"
    )
    
    parser.add_argument(
        "--no-auto-vacuum",
        action="store_true",
        default=False,
        help="Disable auto-vacuum, use original slab cell height"
    )
    
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1000,
        help="Maximum placement attempts per molecule (default: 1000)"
    )
    
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)"
    )
    
    parser.add_argument(
        "--output-dir", "-od",
        type=str,
        default=None,
        help="Output directory (default: same as input slab file)"
    )
    
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["xyz", "extxyz", "vasp", "cif", "lammps-data"],
        default="extxyz",
        help="Output file format (default: extxyz)"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        default="auto",
        help="Log file path. 'auto' generates alongside output, 'none' disables (default: auto)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed information"
    )
    
    parser.add_argument(
        "--z-reoffset",
        type=float,
        default=None,
        help="Re-offset the final structure so that minimum z equals this value (in Angstrom). If not specified, no re-offset is applied."
    )
    
    parser.add_argument(
        "--wrap",
        action="store_true",
        default=False,
        help="Wrap atoms into the periodic cell (recommended for AIMD). Only wraps in x and y directions for slab models."
    )
    
    parser.add_argument(
        "--pbc-xyz",
        action="store_true",
        default=False,
        help="Set periodic boundary conditions to [True, True, True] (for DeepMD-kit training). Default is [True, True, False] for slab models."
    )
    
    parser.add_argument(
        "--no-boundary-cross",
        action="store_true",
        default=False,
        help="Ensure molecules do not cross x/y periodic boundaries. Molecules will be placed entirely within [0, a] x [0, b]."
    )
    
    return parser.parse_args()


def get_output_filename(
    slab_path: str,
    n_water: int,
    n_nh3: int,
    n_nh4: int,
    n_hpo4: int,
    output_format: str,
    output_dir: Optional[str] = None
) -> str:
    """Generate output filename based on parameters."""
    slab_file = Path(slab_path)
    base_name = slab_file.stem
    
    ext_map = {
        "xyz": ".xyz",
        "extxyz": ".xyz",
        "vasp": ".vasp",
        "cif": ".cif",
        "lammps-data": ".lmp"
    }
    ext = ext_map.get(output_format, ".xyz")
    
    # Create descriptive name
    parts = [base_name, "packed"]
    if n_water > 0:
        parts.append(f"w{n_water}")
    if n_nh3 > 0:
        parts.append(f"nh3_{n_nh3}")
    if n_nh4 > 0:
        parts.append(f"nh4_{n_nh4}")
    if n_hpo4 > 0:
        parts.append(f"hpo4_{n_hpo4}")
    
    output_name = "_".join(parts) + ext
    
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        return str(out_path / output_name)
    
    return str(slab_file.parent / output_name)


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Determine output path
    if args.output_file:
        output_path = args.output_file
    else:
        output_path = get_output_filename(
            args.slab_input,
            args.n_water,
            args.n_nh3,
            args.n_nh4,
            args.n_hpo4,
            args.format,
            args.output_dir
        )
    
    # Setup logging
    log_file_handle = None
    original_stdout = sys.stdout
    
    try:
        if args.log_file and args.log_file.lower() != "none":
            if args.log_file.lower() == "auto":
                log_path = Path(output_path).with_suffix(".log")
            else:
                log_path = Path(args.log_file)
            
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file_handle = open(log_path, "w", encoding="utf-8")
            sys.stdout = TeeOutput(log_file_handle, original_stdout)
            
            print(f"# Log generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"# Command: python NH4_packing_CSOslab.py \\")
            print(f"#   --slab-input {args.slab_input} \\")
            if args.output_dir:
                print(f"#   --output-dir {args.output_dir} \\")
            print(f"#   --n-water {args.n_water} \\")
            print(f"#   --n-nh3 {args.n_nh3} \\")
            print(f"#   --n-nh4 {args.n_nh4} \\")
            print(f"#   --n-hpo4 {args.n_hpo4} \\")
            print(f"#   --packing-vacuum {args.packing_vacuum} \\")
            print(f"#   --min-distance {args.min_distance} \\")
            print(f"#   --z-buffer {args.z_buffer} \\")
            auto_vac_str = "OFF" if args.no_auto_vacuum else "ON"
            print(f"#   --auto-vacuum {auto_vac_str}", end="")
            if args.target_vacuum is not None:
                print(" \\")
                print(f"#   --target-vacuum {args.target_vacuum}", end="")
            if args.z_reoffset is not None:
                print(" \\")
                print(f"#   --z-reoffset {args.z_reoffset}", end="")
            if args.wrap:
                print(" \\")
                print("#   --wrap", end="")
            if args.pbc_xyz:
                print(" \\")
                print("#   --pbc-xyz", end="")
            if args.no_boundary_cross:
                print(" \\")
                print("#   --no-boundary-cross", end="")
            print("")  # End of command
        
        print("\n" + "=" * 60)
        print("MOLECULE PACKING FOR CaSO4 SLAB")
        print("=" * 60)
        
        # Load slab
        print(f"\nLoading slab: {args.slab_input}")
        slab = read(args.slab_input)
        print(f"  Slab atoms: {len(slab)}")
        print(f"  Formula: {slab.get_chemical_formula()}")
        
        cell_params = slab.cell.cellpar()
        print(f"  Cell: {cell_params[0]:.2f} x {cell_params[1]:.2f} x {cell_params[2]:.2f} Angstrom")
        
        positions = slab.get_positions()
        z_min, z_max = positions[:, 2].min(), positions[:, 2].max()
        print(f"  Slab z-range: [{z_min:.2f}, {z_max:.2f}] Angstrom")
        
        # Prepare molecules to pack
        molecules = []
        if args.n_water > 0:
            molecules.append((create_water(), args.n_water))
        if args.n_nh3 > 0:
            molecules.append((create_ammonia(), args.n_nh3))
        if args.n_nh4 > 0:
            molecules.append((create_ammonium(), args.n_nh4))
        if args.n_hpo4 > 0:
            molecules.append((create_hydrogen_phosphate(), args.n_hpo4))
        
        if not molecules:
            print("\nWarning: No molecules specified. Output will be same as input.")
            combined = slab
            stats = {'total_placed': 0, 'total_failed': 0, 'by_molecule': {}}
        else:
            # Pack molecules
            print("\n" + "-" * 60)
            print("PACKING MOLECULES")
            print("-" * 60)
            
            combined, stats = pack_molecules(
                slab=slab,
                molecules=molecules,
                min_distance=args.min_distance,
                z_buffer=args.z_buffer,
                packing_vacuum=args.packing_vacuum,
                auto_vacuum=not args.no_auto_vacuum,
                max_attempts=args.max_attempts,
                no_boundary_cross=args.no_boundary_cross,
                verbose=args.verbose
            )
        
        # Print statistics
        print("\n" + "-" * 60)
        print("PACKING RESULTS")
        print("-" * 60)
        print(f"Molecules placed: {stats['total_placed']}")
        print(f"Placement failures: {stats['total_failed']}")
        for mol_name, mol_stats in stats['by_molecule'].items():
            print(f"  {mol_name}: {mol_stats['placed']} placed, {mol_stats['failed']} failed")
        
        # Apply wrap if requested (for AIMD compatibility)
        if args.wrap:
            print("\n" + "-" * 60)
            print("WRAP OPERATION")
            print("-" * 60)
            # For slab models, only wrap in x and y (pbc=[True, True, False])
            pbc = combined.get_pbc()
            if not pbc[2]:  # z is not periodic (slab model)
                combined.wrap(pbc=[True, True, False])
                print("Wrapped atoms in x and y directions (slab model detected)")
            else:
                combined.wrap()
                print("Wrapped atoms in all periodic directions")
        
        # Apply z-reoffset if specified
        if args.z_reoffset is not None:
            print("\n" + "-" * 60)
            print("Z-REOFFSET ADJUSTMENT")
            print("-" * 60)
            positions = combined.get_positions()
            current_z_min = positions[:, 2].min()
            shift = args.z_reoffset - current_z_min
            print(f"Current z_min: {current_z_min:.4f} Angstrom (in pure CSO slab model)")
            print(f"Target z_min: {args.z_reoffset:.4f} Angstrom")
            print(f"Shift applied: {shift:.4f} Angstrom")
            positions[:, 2] += shift
            combined.set_positions(positions)
            
            # Also adjust cell c to maintain vacuum at top
            cell = combined.get_cell()
            cell[2, 2] += shift
            combined.set_cell(cell)
            print(f"New cell c: {cell[2, 2]:.4f} Angstrom")
        
        # Apply target_vacuum if specified (adjust cell to achieve target vacuum above molecules)
        if args.target_vacuum is not None:
            print("\n" + "-" * 60)
            print("TARGET VACUUM ADJUSTMENT")
            print("-" * 60)
            positions = combined.get_positions()
            z_max = positions[:, 2].max()
            current_vacuum = combined.cell.cellpar()[2] - z_max
            print(f"Current vacuum above molecules: {current_vacuum:.2f} Angstrom")
            print(f"Target vacuum: {args.target_vacuum:.2f} Angstrom")
            
            # Calculate new cell height
            new_c = z_max + args.target_vacuum
            cell = combined.get_cell()
            cell[2, 2] = new_c
            combined.set_cell(cell)
            print(f"New cell c: {new_c:.4f} Angstrom")
        
        # Calculate final vacuum thickness
        positions = combined.get_positions()
        z_min = positions[:, 2].min()
        z_max = positions[:, 2].max()
        cell_c = combined.cell.cellpar()[2]
        vacuum_thickness = cell_c - z_max
        
        print(f"\nFinal structure:")
        print(f"  Total atoms: {len(combined)}")
        print(f"  Formula: {combined.get_chemical_formula()}")
        print(f"  Z-range: [{z_min:.4f}, {z_max:.4f}] Angstrom")
        print(f"  Final vacuum thickness: {vacuum_thickness:.2f} Angstrom")
        
        # Set PBC if pbc_xyz is requested
        if args.pbc_xyz:
            combined.set_pbc([True, True, True])
            print(f"  PBC: [True, True, True] (for DeepMD-kit)")
        else:
            print(f"  PBC: {list(combined.get_pbc())}")
        
        # Save output
        print(f"\nSaving to: {output_path}")
        write(output_path, combined, format=args.format)
        
        cell_params = combined.cell.cellpar()
        print(f"  Lattice parameters: {cell_params[0]:.4f} {cell_params[1]:.4f} {cell_params[2]:.4f} {cell_params[3]:.2f} {cell_params[4]:.2f} {cell_params[5]:.2f}")
        
        print("\n" + "=" * 60)
        print("Packing completed successfully!")
        print("=" * 60)
        
        if log_file_handle:
            print(f"\nLog saved to: {log_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        sys.stdout = original_stdout
        if log_file_handle:
            log_file_handle.close()


if __name__ == "__main__":
    main()
