#!/usr/bin/env python3
"""
freeze_and_compress.py
======================
Automated script to:
1. Freeze trained DeePMD-kit model checkpoint into `frozen_model.pb`.
2. Compress `frozen_model.pb` using tabulated grid approximation into `frozen_model_compressed.pb`
   for 3x - 10x inference speedup in LAMMPS.
3. Automatically copy the compressed production model to `6.LAMMPS_ScalingUp/02.lammps_simulation/`.
"""

import os
import shutil
import subprocess

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, "..", "01.training")
    lammps_dir = os.path.join(base_dir, "..", "..", "6.LAMMPS_ScalingUp", "02.lammps_simulation")
    
    frozen_model = os.path.join(base_dir, "frozen_model.pb")
    compressed_model = os.path.join(base_dir, "frozen_model_compressed.pb")
    
    print("=" * 60)
    print("DEEPMD-KIT MODEL FREEZE & COMPRESSION WORKFLOW")
    print("=" * 60)
    
    # 1. dp freeze
    cmd_freeze = ["dp", "freeze", "-o", frozen_model]
    print(f"[Step 1] Freezing model checkpoint: {' '.join(cmd_freeze)}")
    res = subprocess.run(cmd_freeze, cwd=train_dir, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"[Warning] Freeze encountered status {res.returncode}: {res.stderr}")
        
    # 2. dp compress
    if os.path.exists(frozen_model):
        cmd_compress = ["dp", "compress", "-i", frozen_model, "-o", compressed_model]
        print(f"[Step 2] Compressing model for accelerated inference: {' '.join(cmd_compress)}")
        res2 = subprocess.run(cmd_compress, cwd=base_dir, capture_output=True, text=True)
        print(res2.stdout)
        if res2.returncode != 0:
            print(f"[Warning] Compression output: {res2.stderr}")
            
        # 3. Copy to LAMMPS directory
        if os.path.exists(compressed_model) and os.path.exists(lammps_dir):
            dst_path = os.path.join(lammps_dir, "frozen_model_compressed.pb")
            shutil.copy2(compressed_model, dst_path)
            print(f"[Step 3] Successfully deployed compressed model to LAMMPS: {dst_path}")
        elif os.path.exists(frozen_model) and os.path.exists(lammps_dir):
            dst_path = os.path.join(lammps_dir, "frozen_model.pb")
            shutil.copy2(frozen_model, dst_path)
            print(f"[Step 3] Deployed uncompressed model to LAMMPS: {dst_path}")
            
    print("=" * 60)
    print("Model freeze and compression workflow completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
