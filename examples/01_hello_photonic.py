"""
例 1：最小可运行示例 —— 把一个矩阵编译上光子芯片，跑一遍，和 CPU 对答案。

    python examples/01_hello_photonic.py
"""
from __future__ import annotations

try:
    import photonic_mzi  # noqa: F401
except ImportError:      # 未 pip install 时直接从源码目录跑
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from photonic_mzi import PhotonicMatrixProcessor

M = np.array([[0.65, -0.42, 0.18, 0.91],
              [-0.12, 0.88, -0.54, 0.33],
              [0.47, 0.21, -0.76, 0.15],
              [-0.83, 0.35, 0.62, -0.49]])
x = np.array([1.0, 0.5, -0.8, 0.2])

opu = PhotonicMatrixProcessor(M, seed=42)
print(opu.report())

y_cpu = M @ x
y_opt = opu.read_coherent(x)

print("\nCPU 数字计算 :", np.round(y_cpu, 6))
print("理想光子芯片 :", np.round(y_opt, 6))
print(f"绝对误差     : {np.linalg.norm(y_opt - y_cpu):.2e}")

# 非方阵、批量输入都支持
W = np.random.default_rng(0).standard_normal((3, 7))
X = np.random.default_rng(1).standard_normal((7, 128))
err = np.linalg.norm(PhotonicMatrixProcessor(W).read_coherent(X) - W @ X)
print(f"\n非方阵 (3x7) + 批量 128 路：误差 {err:.2e}")

# 光电二极管只能测 |E|^2，符号在平方里丢了
print("\n相干（零差）探测 :", np.round(opu.read_coherent(x), 4))
print("直接光强探测     :", np.round(opu.read_intensity(x), 4), " <- 全为正，符号没了")
