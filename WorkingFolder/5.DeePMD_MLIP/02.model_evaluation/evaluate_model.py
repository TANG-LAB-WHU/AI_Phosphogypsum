#!/usr/bin/env python3
"""
evaluate_model.py
=================
Automated evaluation tool to test the accuracy of the trained DeePMD model
against quantum mechanical DFT reference data (forces, energies, virials).
Generates Parity Plots and computes RMSE metrics.
"""

import os
import glob
import subprocess
import numpy as np

def run_dp_test(model_path, system_dir, output_prefix):
    """Run `dp test` on a specific system dataset."""
    cmd = [
        "dp", "test",
        "-m", model_path,
        "-s", system_dir,
        "-d", output_prefix
    ]
    print(f"[Testing] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(f"[Warning] {res.stderr}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "03.freeze_and_compress", "frozen_model.pb")
    if not os.path.exists(model_path):
        model_path = os.path.join(base_dir, "..", "01.training", "frozen_model.pb")
        
    if not os.path.exists(model_path):
        print(f"[Info] Model file {model_path} not found yet. Please run training and freeze first.")
        return
        
    test_systems = [
        "../../4.DPGEN_ActiveLearning/00.init_seeds/init_3.2.2CSO-0.625H2O+NH4_improved_3",
        "../../4.DPGEN_ActiveLearning/00.init_seeds/init_3.2.2CSO-0.583H2O+NH4_improved",
        "../../4.DPGEN_ActiveLearning/00.init_seeds/init_3.2.1CSO-2H2O+NH4"
    ]
    
    for sys_path in test_systems:
        full_sys_path = os.path.join(base_dir, sys_path)
        if os.path.exists(full_sys_path):
            sys_name = os.path.basename(sys_path)
            out_prefix = os.path.join(base_dir, f"test_{sys_name}")
            run_dp_test(model_path, full_sys_path, out_prefix)
            
    print("=" * 60)
    print("Model evaluation complete! Parity datasets generated.")
    print("=" * 60)

if __name__ == "__main__":
    main()
