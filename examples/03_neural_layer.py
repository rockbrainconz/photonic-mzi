"""
例 3：光子线性层。Example 3: a photonic linear layer.

「相对误差 10%、只剩 3 bit 精度」听上去很吓人，但分类任务真正在意的是 argmax
有没有翻，不是每个 logit 的小数点后几位。这个例子把两者分开量化。

任务是自造的：16 维空间里 4 团高斯，最近类心分类器（本质就是个线性层 W: 4x16）。
把 W 编译上光子芯片做推理，看噪声要多大才会真的分错。

    python examples/03_neural_layer.py
"""
from __future__ import annotations

import pathlib
import sys

# 直接运行仓库示例时始终使用同仓库源码，避免误导入环境中残留的旧安装版本。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from photonic_mzi import NoiseModel, PhotonicMatrixProcessor

DIM, N_CLASS, N_TEST = 16, 4, 400
rng = np.random.default_rng(0)

# ---- 造数据：4 团分得开的高斯 ----
centroids = rng.standard_normal((N_CLASS, DIM)) * 2.0
labels = rng.integers(0, N_CLASS, N_TEST)
Xtest = centroids[labels] + rng.standard_normal((N_TEST, DIM)) * 0.8

# ---- 最近类心分类器 = 一个线性层 ----
# argmax_c (-||x-mu_c||^2 / 2) = argmax_c (mu_c . x - ||mu_c||^2 / 2)
W = centroids                                  # (4, 16)，光路只做这个矩阵乘
bias = -0.5 * np.sum(centroids ** 2, axis=1)   # 偏置留在电域

logits_digital = Xtest @ W.T + bias
acc_digital = float(np.mean(np.argmax(logits_digital, axis=1) == labels))


def evaluate(opu, ideal=False):
    logits = opu.read_coherent(Xtest.T, ideal=ideal).T + bias
    rel = np.linalg.norm(logits - logits_digital) / np.linalg.norm(logits_digital)
    acc = float(np.mean(np.argmax(logits, axis=1) == labels))
    return rel, acc


print(f"数字基线 / Digital baseline: accuracy {acc_digital:.3f}; "
      f"W={W.shape[0]}x{W.shape[1]}\n")

opu_ideal = PhotonicMatrixProcessor(W)
print(opu_ideal.report())
rel, acc = evaluate(opu_ideal, ideal=True)
print(f"\n理想光处理器 / Ideal OPU: accuracy {acc:.3f}; relative logit error {rel:.2e}")
print("4x16 矩阵占用 16 个模式但仅有 4 个非零奇异值 / "
      "A 4x16 matrix uses 16 modes but has only four non-zero singular values.\n")

# ------------------------------------------------------------------ #
# A. 只扫独立动态相位抖动，插损置零，隔离随机误差影响
# ------------------------------------------------------------------ #
print("=" * 70)
print("A. 每样本动态相位抖动 / Per-sample dynamic phase jitter")
print("=" * 70)
print(f"{'jitter/rad':>12} {'rel. logit err.':>16} {'effective bit':>14} {'accuracy':>10}")
for drift in [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]:
    opu = PhotonicMatrixProcessor(
        W, noise=NoiseModel(fab_theta=0.02, fab_phi=0.02,
                            drift_theta=drift, drift_phi=drift), seed=7)
    opu.calibrate()
    rel, acc = evaluate(opu)
    print(f"{drift:>12.3f} {rel * 100:>15.2f}% "
          f"{-np.log2(max(rel, 1e-18)):>10.1f} {acc:>10.3f}")

print("\n-> 本合成任务在 2~3 bit logit 精度下仍保持分类 / This synthetic task retains classification.")
print("   不能外推到其他模型或真实热漂移 / Do not generalize to other models or physical drift.")

# ------------------------------------------------------------------ #
# B. 插损是另一回事：它是确定性的，不是随机噪声
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("B. 插损 / Insertion loss (deterministic mismatch, not random noise)")
print("=" * 70)
print(f"{'loss dB/MZI':>14} {'bound/dB':>10} {'rel. logit err.':>16} {'accuracy':>10}")
for loss in [0.0, 0.02, 0.05, 0.1, 0.2]:
    opu = PhotonicMatrixProcessor(W, noise=NoiseModel(mzi_loss_db=loss), seed=7)
    rel, acc = evaluate(opu)
    worst = opu.mode_mzi_count().max() * loss
    print(f"{loss:>14.2f} {worst:>9.1f}dB {rel * 100:>15.2f}% {acc:>10.3f}")

cnt = PhotonicMatrixProcessor(W).mode_mzi_count()
print(f"\n模式 MZI 参与数 / MZIs per mode: {cnt.tolist()}")
print(f"最大/最小 / max/min: {cnt.max()}/{cnt.min()} — 拓扑代理 / topology proxy.")
print("不是端到端路径计数 / This is not an end-to-end path count.")
print("插损误差是确定性的 / Insertion-loss error is deterministic.")
print("calibrate() 只处理相移偏置，不补偿插损 / calibrate() does not compensate loss.")
print("\nClements 可缩短光学深度，但不会消除器件离散性 / Clements reduces depth, not variation.")
