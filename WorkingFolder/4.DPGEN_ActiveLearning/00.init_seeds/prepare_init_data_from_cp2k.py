#!/usr/bin/env python3
"""
prepare_init_data_from_cp2k.py
==============================
Automated dataset extraction tool to convert CP2K GeoOpt and AIMD output trajectories
(coordinates, forces, energies, virials, and cells) into DeePMD-kit / DP-GEN standard format.

Type map for the phosphogypsum-ammonium conversion system:
  0: Ca, 1: O, 2: S, 3: H, 4: N, 5: P
"""

import os
import glob
import re
import argparse
import numpy as np
from collections import Counter
from ase.io import read

# Standard Type Map
TYPE_MAP = ["Ca", "O", "S", "H", "N", "P"]
TYPE_DICT = {elem: idx for idx, elem in enumerate(TYPE_MAP)}

def parse_cp2k_log_energies(log_path):
    """Extract SCF energies (in eV) from CP2K log file."""
    energies = []
    if not os.path.exists(log_path):
        return energies
    with open(log_path, "r") as f:
        for line in f:
            if "ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.):" in line or "Total energy (a.u.):" in line:
                val = float(line.split()[-1])
                energies.append(val * 27.211386245988) # Convert Hartree to eV
            elif "E =" in line and "i =" in line:
                # GeoOpt/MD step header
                m = re.search(r'E\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                if m:
                    val = float(m.group(1))
                    energies.append(val * 27.211386245988)
    return energies

def extract_system_data(system_dir, output_dir, name):
    """Convert a CP2K calculation directory into DeePMD raw/npy dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Look for trajectory / optimized structures
    pos_files = glob.glob(os.path.join(system_dir, "*-pos-1.xyz"))
    struct_file = os.path.join(system_dir, "geoopt_optimized_structure_extxyz_wrap.xyz")
    if not os.path.exists(struct_file):
        struct_file = os.path.join(system_dir, "optimized_structure_extxyz_wrap.xyz")
        
    frames = []
    if pos_files and os.path.getsize(pos_files[0]) > 0:
        try:
            frames = read(pos_files[0], index=":")
        except Exception as e:
            print(f"[{name}] Warning reading trajectory {pos_files[0]}: {e}")
            
    if not frames and os.path.exists(struct_file):
        try:
            frames = read(struct_file, index=":")
        except Exception as e:
            print(f"[{name}] Error reading structure {struct_file}: {e}")
            return False

    if not frames:
        print(f"[{name}] No frames found in {system_dir}")
        return False
        
    n_frames = len(frames)
    first_frame = frames[0]
    symbols = first_frame.get_chemical_symbols()
    n_atoms = len(symbols)
    
    # Verify all elements in type map
    types = []
    for s in symbols:
        if s not in TYPE_DICT:
            raise ValueError(f"Unknown element {s} in {name} not in TYPE_MAP {TYPE_MAP}")
        types.append(TYPE_DICT[s])
    types = np.array(types, dtype=int)
    
    # Parse forces if available
    frc_files = glob.glob(os.path.join(system_dir, "*-frc-1.xyz"))
    forces_list = []
    if frc_files and os.path.getsize(frc_files[0]) > 0:
        try:
            with open(frc_files[0], "r") as f:
                lines = f.readlines()
            frame_len = n_atoms + 2
            n_frc_frames = len(lines) // frame_len
            for i in range(n_frc_frames):
                flines = lines[i*frame_len + 2 : (i+1)*frame_len]
                f_frame = []
                for l in flines:
                    parts = l.split()
                    # CP2K frc file is in Hartree/Bohr -> convert to eV/Angstrom
                    # 1 Hartree/Bohr = 51.4220674763 eV/Angstrom
                    fx = float(parts[1]) * 51.4220674763
                    fy = float(parts[2]) * 51.4220674763
                    fz = float(parts[3]) * 51.4220674763
                    f_frame.append([fx, fy, fz])
                forces_list.append(f_frame)
        except Exception as e:
            print(f"[{name}] Warning parsing forces: {e}")
            
    # Collect arrays
    coords = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    boxes = np.zeros((n_frames, 9), dtype=np.float32)
    energies = np.zeros((n_frames,), dtype=np.float32)
    forces = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    
    # Look for energy in log
    log_files = glob.glob(os.path.join(system_dir, "*.log"))
    scf_energies = []
    if log_files:
        scf_energies = parse_cp2k_log_energies(log_files[0])
        
    for i, fr in enumerate(frames):
        coords[i] = fr.get_positions()
        # Cell 3x3 -> 9
        cell_mat = fr.get_cell().array
        boxes[i] = cell_mat.flatten()
        if i < len(scf_energies):
            energies[i] = scf_energies[i]
        elif "E" in fr.info:
            energies[i] = float(fr.info["E"]) * 27.211386245988
        if i < len(forces_list):
            forces[i] = forces_list[i]
            
    # Reshape coords and forces to (n_frames, n_atoms*3)
    coords_flat = coords.reshape(n_frames, -1)
    forces_flat = forces.reshape(n_frames, -1)
    
    # Write DeePMD format
    # type.raw and type_map.raw
    with open(os.path.join(output_dir, "type.raw"), "w") as f:
        f.write(" ".join(map(str, types)) + "\n")
    with open(os.path.join(output_dir, "type_map.raw"), "w") as f:
        for t in TYPE_MAP:
            f.write(f"{t}\n")
            
    # Write set.000 directory with npy arrays
    set_dir = os.path.join(output_dir, "set.000")
    os.makedirs(set_dir, exist_ok=True)
    np.save(os.path.join(set_dir, "box.npy"), boxes)
    np.save(os.path.join(set_dir, "coord.npy"), coords_flat)
    np.save(os.path.join(set_dir, "energy.npy"), energies)
    np.save(os.path.join(set_dir, "force.npy"), forces_flat)
    
    print(f"[{name}] Exported {n_frames} frames ({n_atoms} atoms) -> {output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert CP2K GeoOpt and AIMD runs into DeePMD/DP-GEN seed dataset.")
    parser.add_argument("--repo-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
                        help="Repository root path")
    args = parser.parse_args()
    
    seeds_root = os.path.dirname(os.path.abspath(__file__))
    aimd_root = os.path.join(args.repo_root, "WorkingFolder", "3.CP2K_AIMD")
    geoopt_root = os.path.join(args.repo_root, "WorkingFolder", "2.CP2K_GeoOpt")
    
    print("=" * 60)
    print("CP2K TO DEEPMD / DP-GEN SEED DATASET CONVERTER")
    print(f"Type Map: {TYPE_MAP}")
    print("=" * 60)
    
    count = 0
    # Process 3.CP2K_AIMD systems
    if os.path.exists(aimd_root):
        subdirs = sorted([d for d in os.listdir(aimd_root) if os.path.isdir(os.path.join(aimd_root, d)) and d.startswith("3.")])
        for sub in subdirs:
            src_dir = os.path.join(aimd_root, sub)
            dst_dir = os.path.join(seeds_root, f"init_{sub}")
            if extract_system_data(src_dir, dst_dir, f"AIMD/{sub}"):
                count += 1
                
    # Process 2.CP2K_GeoOpt systems
    if os.path.exists(geoopt_root):
        subdirs = sorted([d for d in os.listdir(geoopt_root) if os.path.isdir(os.path.join(geoopt_root, d)) and d.startswith("2.")])
        for sub in subdirs:
            src_dir = os.path.join(geoopt_root, sub)
            dst_dir = os.path.join(seeds_root, f"init_geo_{sub}")
            if extract_system_data(src_dir, dst_dir, f"GeoOpt/{sub}"):
                count += 1
                
    print("=" * 60)
    print(f"Extraction complete! Successfully generated {count} seed dataset directories in {seeds_root}.")
    print("=" * 60)

if __name__ == "__main__":
    main()
