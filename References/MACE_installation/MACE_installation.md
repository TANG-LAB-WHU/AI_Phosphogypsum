# MACE + cuEquivariance 安装指南

本文档记录了在 NVIDIA Blackwell 架构 GPU（RTX 5080/5090）上成功安装 MACE 和 cuEquivariance 加速库的完整过程。

> **适用 GPU**：RTX 5080、RTX 5090 及其他 Blackwell 架构显卡

---

## 目录

1. [环境要求](#1-环境要求)
2. [创建 Python 虚拟环境](#2-创建-python-虚拟环境)
3. [安装 PyTorch](#3-安装-pytorch)
4. [安装 MACE](#4-安装-mace)
5. [安装 cuEquivariance](#5-安装-cuequivariance)
6. [配置 CUDA 库路径](#6-配置-cuda-库路径)
7. [验证安装](#7-验证安装)
8. [常见问题排查](#8-常见问题排查)
9. [RTX 5080/5090 (Blackwell) 特别说明](#9-rtx-50805090-blackwell-特别说明)
10. [完整安装命令汇总](#10-完整安装命令汇总)
11. [参考链接](#11-参考链接)

---

## 1. 环境要求

### 1.1 硬件

- NVIDIA GPU（支持 CUDA 12.x）
- 推荐显存：≥ 8GB

### 1.2 软件

- 操作系统：Ubuntu 24.04 (WSL2) 或原生 Linux
- Python：3.10 或 3.11
- NVIDIA 驱动：≥ 535（对于 RTX 5080 需要 ≥ 575）
- CUDA：12.x（驱动自带，无需单独安装 CUDA Toolkit）

### 1.3 测试环境配置

| 组件           | 版本                           |
| -------------- | ------------------------------ |
| GPU            | NVIDIA GeForce RTX 5080 (16GB) |
| 驱动版本       | 576.02                         |
| CUDA 版本      | 12.9                           |
| Python         | 3.10                           |
| PyTorch        | 2.9.1+cu129                    |
| MACE           | 0.3.15                         |
| cuEquivariance | 0.8.1                          |

---

## 2. 创建 Python 虚拟环境

### 2.1 使用 venv（推荐）

```bash
# 创建项目目录
mkdir -p ~/documents4projects/MACE
cd ~/documents4projects/MACE

# 创建虚拟环境
python3.10 -m venv mace_env

# 激活环境
source mace_env/bin/activate

# 升级 pip
pip install --upgrade pip
```

### 2.2 使用 Conda（可选）

```bash
conda create -n mace_env python=3.10
conda activate mace_env
pip install --upgrade pip
```

---

## 3. 安装 PyTorch

### 3.1 安装 PyTorch with CUDA 12.x

```bash
# 安装 PyTorch (CUDA 12.9) (这一步断开VPN，安装速度快)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

### 3.2 验证 PyTorch CUDA 支持

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

预期输出：

```
PyTorch: 2.9.1+cu129
CUDA available: True
CUDA version: 12.9
GPU: NVIDIA GeForce RTX 5080 or NVIDIA GeForce RTX 5090
```

---

## 4. 安装 MACE

### 4.1 安装 mace-torch

```bash
pip install mace-torch
```

这会自动安装以下依赖：

- e3nn
- torch-dftd（D3 色散校正）
- ASE（原子模拟环境）
- 其他科学计算库

### 4.2 验证 MACE 安装

```bash
python -c "from mace.calculators import mace_mp; print('MACE installed successfully')"
```

---

## 5. 安装 cuEquivariance

cuEquivariance 为 MACE 提供 GPU 加速的等变张量运算。

### 5.1 安装所有组件

```bash
# 安装核心库和 PyTorch 绑定
pip install cuequivariance==0.8.1 cuequivariance-torch==0.8.1

# 安装 CUDA 12 优化的操作库
pip install cuequivariance-ops-cu12==0.8.1 cuequivariance-ops-torch-cu12==0.8.1
```

> **重要**：确保所有 cuEquivariance 相关包版本一致（都是 0.8.1），版本不匹配会导致运行错误。

### 5.2 版本兼容性

| cuEquivariance 版本 | 支持的功能                         |
| ------------------- | ---------------------------------- |
| 0.8.1 (cu12)        | 基本 GPU 加速，RTX 5080 可用       |
| 0.8.1 (cu13)        | Blackwell 优化内核（需要 CUDA 13） |

---

## 6. 配置 CUDA 库路径

### 6.1 问题说明

cuEquivariance-ops 需要访问 NVIDIA CUDA 运行时库（如 `libnvrtc.so.12`、`libcublas.so.12`）。
这些库通过 pip 安装在 Python 包目录中，需要配置 `LD_LIBRARY_PATH` 才能正确加载。

### 6.2 查找库路径

```bash
# 查找 CUDA 库位置
find ~/documents4projects/MACE/mace_env -name "libnvrtc*" 2>/dev/null
find ~/documents4projects/MACE/mace_env -name "libcublas*" 2>/dev/null
```

### 6.3 设置环境变量

```bash
# 设置 LD_LIBRARY_PATH（临时）
export LD_LIBRARY_PATH=/home/siqi/documents4projects/MACE/mace_env/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/home/siqi/documents4projects/MACE/mace_env/lib/python3.10/site-packages/nvidia/cublas/lib:/home/siqi/documents4projects/MACE/mace_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/home/siqi/documents4projects/MACE/mace_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/siqi/documents4projects/MACE/mace_env/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
```

### 6.4 永久配置（添加到 ~/.bashrc，推荐进行永久配置；LD_LIBRARY_PATH根据6.2中找到的进行修改，此处为示例）

```bash
# 添加到 ~/.bashrc
cat >> ~/.bashrc << 'EOF'

# cuEquivariance CUDA libraries
export LD_LIBRARY_PATH=/home/siqi/documents4projects/MACE/mace_env/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:/home/siqi/documents4projects/MACE/mace_env/lib/python3.10/site-packages/nvidia/cublas/lib:/home/siqi/documents4projects/MACE/mace_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/home/siqi/documents4projects/MACE/mace_env/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:/home/siqi/documents4projects/MACE/mace_env/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
EOF

# 使配置生效
source ~/.bashrc
```

> **注意**：执行 `source ~/.bashrc` 后，虚拟环境会被取消激活，需要重新激活：
>
> ```bash
> source ~/documents4projects/MACE/mace_env/bin/activate
> ```

### 6.5 通用方法（或者自动查找路径，之后运行 `source ~/.bashrc`并再次激活mace_env虚拟环境）

```bash
# 自动查找所有 nvidia 包的 lib 目录
export LD_LIBRARY_PATH=$(find ~/documents4projects/MACE/mace_env/lib/python3.10/site-packages/nvidia -name "lib" -type d 2>/dev/null | tr '\n' ':'):$LD_LIBRARY_PATH
```

---

## 7. 验证安装

### 7.1 验证 cuEquivariance 加载

```bash
python -c "import cuequivariance_ops_torch; print('cuequivariance_ops_torch loaded successfully')"
```

预期输出：

```
Failed to get GPU information from pynvml: Not Supported. Using default values for NVIDIA GeForce RTX 5080.
GPU information: NVIDIA RTX A6000, 8, 6, 45, 84, 300, 2100
cuequivariance_ops_torch loaded successfully
```

> **注意**：`pynvml` 警告可以忽略，不影响功能。

### 7.2 验证可用的 CUDA 操作

```bash
python -c "import torch; import cuequivariance_ops_torch; print(dir(torch.ops.cuequivariance))"
```

应该看到类似：

```
['tensor_product_uniform_1d_jit', 'triangle_attention', 'tri_mul_update', ...]
```

### 7.3 完整功能测试

创建测试脚本 `test_mace_cueq.py`：

```python
#!/usr/bin/env python
"""测试 MACE + cuEquivariance 是否正常工作"""

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# 测试 cuEquivariance
import cuequivariance_ops_torch
print("cuequivariance_ops_torch: OK")

# 测试 MACE
from mace.calculators import mace_mp
from ase import Atoms

# 创建简单的水分子
water = Atoms('H2O', positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]])
water.center(vacuum=5.0)

# 加载 MACE 模型（启用 cuEquivariance）
calc = mace_mp(
    model="medium-mpa-0",
    device="cuda",
    default_dtype="float32",
    enable_cueq=True,  # 启用 cuEquivariance 加速
)
water.calc = calc

# 计算能量和力
energy = water.get_potential_energy()
forces = water.get_forces()

print(f"\nTest Results:")
print(f"Energy: {energy:.6f} eV")
print(f"Max force: {abs(forces).max():.6f} eV/Å")
print("\n✓ MACE + cuEquivariance 测试通过!")
```

运行测试：

```bash
python test_mace_cueq.py
```

---

## 8. 常见问题排查

### 8.1 问题: `libnvrtc.so.12: cannot open shared object file`

**原因**：CUDA 运行时库路径未配置。

**解决方案**：配置 `LD_LIBRARY_PATH`，参见 [6. 配置 CUDA 库路径](#6-配置-cuda-库路径)。

### 8.2 问题: `ImportError: uniform_1d extension is not available`

**原因**：cuEquivariance 包版本不匹配。

**解决方案**：

```bash
# 卸载所有版本
pip uninstall cuequivariance cuequivariance-torch cuequivariance-ops-cu12 cuequivariance-ops-torch-cu12

# 重新安装统一版本
pip install cuequivariance==0.8.1 cuequivariance-torch==0.8.1 \
            cuequivariance-ops-cu12==0.8.1 cuequivariance-ops-torch-cu12==0.8.1
```

### 8.3 问题: `AttributeError: buffer_num_segments`

**原因**：使用了 editable 安装的 cuequivariance，与 ops 包 API 不兼容。

**解决方案**：使用 pip 安装而非 editable 安装：

```bash
pip uninstall cuequivariance cuequivariance-torch
pip install cuequivariance==0.8.1 cuequivariance-torch==0.8.1
```

### 8.4 问题: `source ~/.bashrc` 后虚拟环境消失

**原因**：`source ~/.bashrc` 会重置 shell 状态。

**解决方案**：重新激活虚拟环境：

```bash
source ~/documents4projects/MACE/mace_env/bin/activate
```

### 8.5 问题: 首次运行非常慢（>30秒）

**原因**：首次运行需要进行 JIT (Just-In-Time) 编译。

**说明**：这是正常的，后续运行会使用缓存，速度会大幅提升（~90ms/步）。

---

## 9. RTX 5080/5090 (Blackwell) 特别说明

RTX 5080 和 RTX 5090 均使用 **NVIDIA Blackwell 架构**（计算能力 10.x，sm_120）。

### 9.1 支持的 GPU 型号

| GPU         | 显存 | CUDA 核心 | 架构               | 支持状态 |
| ----------- | ---- | --------- | ------------------ | -------- |
| RTX 5090    | 32GB | ~21,760   | Blackwell (sm_120) | ✅ 支持  |
| RTX 5080    | 16GB | ~10,752   | Blackwell (sm_120) | ✅ 支持  |
| RTX 5070 Ti | 16GB | ~8,960    | Blackwell (sm_120) | ✅ 支持  |
| RTX 5070    | 12GB | ~6,144    | Blackwell (sm_120) | ✅ 支持  |

> **注意**：所有 Blackwell 架构 GPU 的安装步骤完全相同。

### 9.2 当前支持状态（cuEquivariance 0.8.1 + cu12）

| 功能                           | 状态         |
| ------------------------------ | ------------ |
| 基本 GPU 加速                  | ✅ 支持      |
| MACE 推理/训练                 | ✅ 支持      |
| cuEquivariance tensor products | ✅ 支持      |
| Blackwell 优化内核             | ❌ 需要 cu13 |

### 9.3 已知限制

1. **pynvml 警告**：`Failed to get GPU information from pynvml: Not Supported`

   - 这只是信息警告，pynvml 尚未完全支持 RTX 50 系列
   - 不影响计算功能
2. **Blackwell 优化内核**：

   - `cuet.triangle_attention` 的 Blackwell 优化需要 CUDA 13 (cu13) 构建版本
   - 当前使用的 cu12 版本功能正常，但无法使用专属优化

### 9.4 性能参考

在 RTX 5080 上使用 cuEquivariance (cu12) 的测试结果：

| 测试项目       | 性能                  |
| -------------- | --------------------- |
| 原子数         | 187                   |
| 首次单点计算   | ~32 秒（含 JIT 编译） |
| 后续每步优化   | ~90 毫秒              |
| 500 步几何优化 | ~43 秒                |

> **提示**：RTX 5090 拥有更大显存（32GB）和更多 CUDA 核心，可处理更大的分子系统，性能预计比 RTX 5080 提升约 50-80%。

### 9.5 未来升级

当以下条件满足时，可升级到完整 Blackwell 支持：

1. NVIDIA 发布 CUDA 13 驱动
2. PyTorch 发布 cu13 版本
3. cuEquivariance 发布 cu13 包

届时安装命令：

```bash
pip install cuequivariance-ops-cu13 cuequivariance-ops-torch-cu13
```

---

## 10. 完整安装命令汇总

```bash
# 1. 创建并激活虚拟环境
mkdir -p ~/documents4projects/MACE && cd ~/documents4projects/MACE
python3.10 -m venv mace_env
source mace_env/bin/activate
pip install --upgrade pip

# 2. 安装 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# 3. 安装 MACE
pip install mace-torch

# 4. 安装 cuEquivariance（版本一致）
pip install cuequivariance==0.8.1 cuequivariance-torch==0.8.1
pip install cuequivariance-ops-cu12==0.8.1 cuequivariance-ops-torch-cu12==0.8.1

# 5. 配置 CUDA 库路径
export LD_LIBRARY_PATH=$(find ~/documents4projects/MACE/mace_env/lib/python3.10/site-packages/nvidia -name "lib" -type d 2>/dev/null | tr '\n' ':'):$LD_LIBRARY_PATH

# 6. 永久保存配置
echo 'export LD_LIBRARY_PATH=$(find ~/documents4projects/MACE/mace_env/lib/python3.10/site-packages/nvidia -name "lib" -type d 2>/dev/null | tr "\n" ":"):$LD_LIBRARY_PATH' >> ~/.bashrc

# 7. 验证安装
python -c "import cuequivariance_ops_torch; print('Success')"
```

---

## 11. 参考链接

- [MACE GitHub](https://github.com/ACEsuit/mace)
- [cuEquivariance GitHub](https://github.com/NVIDIA/cuEquivariance)
- [cuEquivariance CHANGELOG](https://github.com/NVIDIA/cuEquivariance/blob/main/CHANGELOG.md)
- [PyTorch 安装](https://pytorch.org/get-started/locally/)

---

*文档更新日期：2026-01-26*
*测试环境：RTX 5080, Ubuntu 24.04 (WSL2), Python 3.10*
*适用于：RTX 5090, RTX 5080, RTX 5070 Ti, RTX 5070 及其他 Blackwell 架构 GPU*
