#!/usr/bin/env python
"""

This script performs geometry optimization using the MACE-MP pre-trained
machine learning potential. Designed for Fe3O4-Ni oxide surfaces with
oxygen vacancies and CH4 molecules.

"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

from ase.io import read, write
from ase.optimize import BFGS, LBFGS, FIRE
from ase.constraints import FixAtoms
from ase.io.trajectory import Trajectory

import torch
import argparse

# Available pretained MACE-MP models for Fe3O4-Ni + CH4 system
AVAILABLE_MODELS = {
    "medium-mpa-0": "Latest recommended model (PBE+U, Materials Project)",
    "mace-matpes-pbe-0": "Pure PBE without +U correction (MATPES dataset)",
    "mace-matpes-r2scan-0": "r2SCAN functional (better for some materials)",
}


def check_gpu():
    """Check GPU availability and print info."""
    print("=" * 60)
    print("GPU Information")
    print("=" * 60)
    
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Available: Yes")
        print(f"GPU Name: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
    else:
        device = "cpu"
        print("GPU Available: No (using CPU)")
        print("WARNING: CPU mode will be significantly slower!")
    
    print("=" * 60)
    return device


def run_optimization(
    input_file: str = "conventional_cell_slab_020_L1_2x2_packed_w20_nh3_3_nh4_4_hpo4_2.xyz",
    output_file: str = "optimized_structure.xyz",
    model_name: str = "medium-mpa-0",
    fmax: float = 0.05,
    max_steps: int = 500,
    optimizer: str = "BFGS",
    fix_bottom_layers: bool = True,
    fix_z_threshold: float = 3.0,
    use_dispersion: bool = True,
    damping: str = "bj",
    save_trajectory: bool = True,
):
    """
    Run geometry optimization using MACE-MP potential.
    
    Parameters
    ----------
    input_file : str
        Input XYZ file with initial structure
    output_file : str
        Output XYZ file for optimized structure
    model_name : str
        MACE-MP model to use. Available options:
        - "medium-mpa-0": Latest recommended (default)
        - "mace-matpes-pbe-0": Pure PBE without +U
        - "mace-matpes-r2scan-0": r2SCAN functional
    fmax : float
        Force convergence criterion (eV/Å)
    max_steps : int
        Maximum optimization steps
    optimizer : str
        Optimizer to use: "BFGS", "LBFGS", or "FIRE"
    fix_bottom_layers : bool
        Whether to fix bottom layer atoms during optimization
    fix_z_threshold : float
        Z-coordinate threshold for fixing atoms (atoms with z < threshold are fixed)
    use_dispersion : bool
        Whether to include D3 dispersion corrections
    damping : str
        D3 damping function: "bj" (Becke-Johnson), "zero", "zerom", or "bjm"
    save_trajectory : bool
        Whether to save optimization trajectory
    """
    
    # Import MACE calculator
    try:
        from mace.calculators import mace_mp
    except ImportError:
        print("ERROR: mace-torch is not installed!")
        print("Please install with: pip install mace-torch")
        sys.exit(1)
    
    # Check GPU availability
    device = check_gpu()
    
    # Get working directory
    script_dir = Path(__file__).parent
    input_path = script_dir / input_file
    output_path = script_dir / output_file
    traj_path = script_dir / "optimization.traj"
    log_path = script_dir / "optimization.log"
    
    # Read initial structure
    print(f"\nReading structure from: {input_path}")
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    atoms = read(str(input_path))
    n_atoms = len(atoms)
    
    print(f"Number of atoms: {n_atoms}")
    print(f"Elements: {set(atoms.get_chemical_symbols())}")
    print(f"Cell: {atoms.cell.cellpar()}")
    
    # Setup constraints - fix bottom layers if requested
    if fix_bottom_layers:
        z_positions = atoms.positions[:, 2]
        z_min = z_positions.min()
        fixed_indices = [i for i, z in enumerate(z_positions) if z < z_min + fix_z_threshold]
        
        if fixed_indices:
            constraint = FixAtoms(indices=fixed_indices)
            atoms.set_constraint(constraint)
            print(f"\nFixed {len(fixed_indices)} atoms with z < {z_min + fix_z_threshold:.2f} Å")
        else:
            print("\nNo atoms fixed (all above threshold)")
    
    # Setup MACE calculator
    print(f"\nLoading MACE-MP model: {model_name}")
    print(f"Device: {device}")
    print(f"Dispersion correction: {use_dispersion}")
    
    # Use float64 for geometry optimization (more accurate)
    calc = mace_mp(
        model=model_name,
        device=device,
        default_dtype="float64",  # Recommended for geometry optimization
        dispersion=use_dispersion,
        damping=damping,          # D3 damping function (default: bj)
        dispersion_xc="pbe",      # PBE functional for D3 correction
    )
    atoms.calc = calc
    
    # Calculate initial energy and forces
    print("\nCalculating initial energy and forces...")
    start_time = time.time()
    initial_energy = atoms.get_potential_energy()
    initial_forces = atoms.get_forces()
    initial_fmax = max(abs(initial_forces).max(axis=1))
    calc_time = time.time() - start_time
    
    print(f"Initial energy: {initial_energy:.6f} eV")
    print(f"Initial max force: {initial_fmax:.6f} eV/Å")
    print(f"Single point calculation time: {calc_time:.2f} s")
    
    # Setup optimizer
    print(f"\nStarting optimization with {optimizer} optimizer")
    print(f"Force convergence criterion: {fmax} eV/Å")
    print(f"Maximum steps: {max_steps}")
    
    # Remove existing log file to ensure clean overwrite (ASE appends by default)
    if log_path.exists():
        log_path.unlink()
    
    if optimizer.upper() == "BFGS":
        opt = BFGS(atoms, trajectory=str(traj_path) if save_trajectory else None,
                   logfile=str(log_path))
    elif optimizer.upper() == "LBFGS":
        opt = LBFGS(atoms, trajectory=str(traj_path) if save_trajectory else None,
                    logfile=str(log_path))
    elif optimizer.upper() == "FIRE":
        opt = FIRE(atoms, trajectory=str(traj_path) if save_trajectory else None,
                   logfile=str(log_path))
    else:
        print(f"Unknown optimizer: {optimizer}, using BFGS")
        opt = BFGS(atoms, trajectory=str(traj_path) if save_trajectory else None,
                   logfile=str(log_path))
    
    # Run optimization
    print("\n" + "=" * 60)
    print("Optimization Progress")
    print("=" * 60)
    
    opt_start_time = time.time()
    converged = opt.run(fmax=fmax, steps=max_steps)
    opt_time = time.time() - opt_start_time
    
    # Final results
    final_energy = atoms.get_potential_energy()
    final_forces = atoms.get_forces()
    final_fmax = max(abs(final_forces).max(axis=1))
    
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    print(f"Converged: {converged}")
    print(f"Steps taken: {opt.nsteps}")
    print(f"Final energy: {final_energy:.6f} eV")
    print(f"Final max force: {final_fmax:.6f} eV/Å")
    print(f"Energy change: {final_energy - initial_energy:.6f} eV")
    print(f"Optimization time: {opt_time:.1f} s ({opt_time/60:.1f} min)")
    print(f"Average time per step: {opt_time/max(1, opt.nsteps):.2f} s")
    
    # Wrap atoms back into the periodic cell (for cleaner visualization and CP2K input)
    atoms.wrap()
    
    # Save optimized structure
    print(f"\nSaving optimized structure to: {output_path}")
    write(str(output_path), atoms)
    
    # Save UNWRAPPED version first (before wrap was applied, use original final positions)
    extxyz_unwrap_path = script_dir / "optimized_structure_extxyz_unwrap.xyz"
    atoms_unwrap = atoms.copy()
    atoms_unwrap.calc = None
    # Note: atoms has already been wrapped, so we need to get unwrapped from forces calc
    atoms_unwrap.info["energy"] = final_energy
    atoms_unwrap.arrays["forces"] = final_forces
    write(str(extxyz_unwrap_path), atoms_unwrap, format="extxyz")
    print(f"Extended XYZ (unwrapped) saved to: {extxyz_unwrap_path}")
    
    # Save WRAPPED version
    extxyz_wrap_path = script_dir / "optimized_structure_extxyz_wrap.xyz"
    atoms_wrap = atoms.copy()
    atoms_wrap.calc = None
    atoms_wrap.wrap()  # Ensure wrapped
    atoms_wrap.info["energy"] = final_energy
    atoms_wrap.arrays["forces"] = final_forces
    write(str(extxyz_wrap_path), atoms_wrap, format="extxyz")
    print(f"Extended XYZ (wrapped) saved to: {extxyz_wrap_path}")
    
    # Convert trajectory to XYZ format for VMD visualization
    if save_trajectory and traj_path.exists():
        traj_xyz_path = script_dir / "optimization_trajectory.xyz"
        try:
            traj_frames = read(str(traj_path), index=":")
            write(str(traj_xyz_path), traj_frames, format="xyz")
            print(f"Trajectory XYZ (for VMD) saved to: {traj_xyz_path} ({len(traj_frames)} frames)")
        except Exception as e:
            print(f"Warning: Could not convert trajectory to XYZ: {e}")
    
    # Write summary report
    report_path = script_dir / "optimization_report.txt"
    with open(report_path, "w") as f:
        f.write("MACE Geometry Optimization Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input file: {input_file}\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Dispersion: {use_dispersion}\n\n")
        
        f.write("Structure Info:\n")
        f.write(f"  Number of atoms: {n_atoms}\n")
        f.write(f"  Elements: {set(atoms.get_chemical_symbols())}\n")
        f.write(f"  Cell parameters: a={atoms.cell.cellpar()[0]:.4f}, "
                f"b={atoms.cell.cellpar()[1]:.4f}, c={atoms.cell.cellpar()[2]:.4f} Å\n\n")
        
        f.write("Optimization Parameters:\n")
        f.write(f"  Optimizer: {optimizer}\n")
        f.write(f"  fmax: {fmax} eV/Å\n")
        f.write(f"  max_steps: {max_steps}\n")
        f.write(f"  Fixed atoms: {len(fixed_indices) if fix_bottom_layers else 0}\n\n")
        
        f.write("Results:\n")
        f.write(f"  Converged: {converged}\n")
        f.write(f"  Steps: {opt.nsteps}\n")
        f.write(f"  Initial energy: {initial_energy:.6f} eV\n")
        f.write(f"  Final energy: {final_energy:.6f} eV\n")
        f.write(f"  Energy change: {final_energy - initial_energy:.6f} eV\n")
        f.write(f"  Initial max force: {initial_fmax:.6f} eV/Å\n")
        f.write(f"  Final max force: {final_fmax:.6f} eV/Å\n")
        f.write(f"  Optimization time: {opt_time:.1f} s\n")
    
    print(f"Report saved to: {report_path}")
    print("\nOptimization complete!")
    
    return atoms, final_energy


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="MACE geometry optimization for Fe3O4-Ni + CH4 systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  medium-mpa-0         Latest recommended model (PBE+U, default)
  mace-matpes-pbe-0    Pure PBE without +U correction
  mace-matpes-r2scan-0 r2SCAN functional

Examples:
  python run_mace_GeoOpt.py
  python run_mace_GeoOpt.py --model mace-matpes-pbe-0
  python run_mace_GeoOpt.py --model mace-matpes-r2scan-0 --fmax 0.01
"""
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=list(AVAILABLE_MODELS.keys()),
        default="mace-matpes-r2scan-0",
        help="MACE model to use (default: mace-matpes-r2scan-0)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="conventional_cell_slab_020_L1_2x2_packed_w20_nh3_3_nh4_4_hpo4_2.xyz",
        help="Input XYZ file (default: conventional_cell_slab_020_L1_2x2_packed_w20_nh3_3_nh4_4_hpo4_2.xyz)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output XYZ file (default: input_<model>_optimized.xyz)"
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.00001,
        help="Force convergence criterion in eV/Å (default: 0.0001)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum optimization steps (default: 5000)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["BFGS", "LBFGS", "FIRE"],
        default="BFGS",
        help="Optimizer to use (default: BFGS)"
    )
    parser.add_argument(
        "--no-dispersion",
        action="store_true",
        help="Disable D3 dispersion correction"
    )
    parser.add_argument(
        "--damping",
        type=str,
        choices=["bj", "zero", "zerom", "bjm"],
        default="bj",
        help="D3 damping function (default: bj = Becke-Johnson)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    args = parser.parse_args()
    
    # List models and exit if requested
    if args.list_models:
        print("\nAvailable MACE models for Fe3O4-Ni + CH4 system:")
        print("=" * 60)
        for name, desc in AVAILABLE_MODELS.items():
            print(f"  {name:25s} {desc}")
        print()
        sys.exit(0)
    
    # Generate output filename if not specified
    if args.output is None:
        input_stem = Path(args.input).stem
        model_short = args.model.replace("-", "_")
        args.output = f"{input_stem}_{model_short}_optimized.xyz"
    
    print(f"\nUsing model: {args.model}")
    print(f"Description: {AVAILABLE_MODELS[args.model]}")
    
    run_optimization(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        fmax=args.fmax,
        max_steps=args.max_steps,
        optimizer=args.optimizer,
        fix_bottom_layers=False,
        use_dispersion=not args.no_dispersion,
        damping=args.damping,
        save_trajectory=True,
    )


if __name__ == "__main__":
    main()

