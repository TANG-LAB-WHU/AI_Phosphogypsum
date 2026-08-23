# 5. DeePMD-kit Production MLIP Model Center

This directory manages the **production-grade Deep Potential training, evaluation, graph freezing, and tabulation compression** for high-throughput scaling up in LAMMPS.

---

## Workflow Structure

```
5.DeePMD_MLIP/
├── 01.training/            # High-step production training configuration (input.json, run scripts)
├── 02.model_evaluation/    # Model accuracy evaluation (dp test, force/energy RMSE, parity plots)
├── 03.freeze_and_compress/ # Model freezing (dp freeze) and tabulation compression (dp compress)
└── README.md
```

---

## Key Features

1. **High-Accuracy Deep Potential Architecture**:
   - Element Type Map: `["Ca", "O", "S", "H", "N", "P"]`
   - Smooth local environment descriptor: `se_e2_a` with radial cutoff $r_{\text{cut}} = 6.00\text{ \AA}$ and smooth decay starting at $0.50\text{ \AA}$.
   - Fitting network: 3 hidden layers `[240, 240, 240]` with ResNet shortcut connections.
2. **Deep Convergence Training**:
   - $1,000,000$ to $2,000,000$ steps with exponential learning rate decay ($1.0 \times 10^{-3} \rightarrow 3.5 \times 10^{-8}$).
   - Force weight $w_f = 1000 \rightarrow 1.0$, energy weight $w_e = 0.02 \rightarrow 1.0$.
3. **Inference Acceleration via Model Compression (`dp compress`)**:
   - Converts the neural network activation layers into tabulated spline lookups, achieving a **$3\times$ to $10\times$ speedup** during LAMMPS production simulations while significantly reducing GPU VRAM consumption.

---

## Execution Guide

### 1. Run Production Training
```bash
cd 01.training
./run_dp_train.sh
```

### 2. Evaluate Model Accuracy
```bash
cd ../02.model_evaluation
python3 evaluate_model.py
```

### 3. Freeze & Compress Model for LAMMPS
```bash
cd ../03.freeze_and_compress
python3 freeze_and_compress.py
```
