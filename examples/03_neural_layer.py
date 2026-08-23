"""
例 3：拿光子芯片跑一层真实的神经网络推理。

「相对误差 10%、只剩 3 bit 精度」听上去很吓人，但分类任务真正在意的是 argmax
有没有翻，不是每个 logit 的小数点后几位。这个例子把两者分开量化。

任务是自造的：16 维空间里 4 团高斯，最近类心分类器（本质就是个线性层 W: 4x16）。
把 W 编译上光子芯片做推理，看噪声要多大才会真的分错。

    python examples/03_neural_layer.py
"""
from __future__ import annotations

try:
    import photonic_mzi  # noqa: F401
except ImportError:      # 未 pip install 时直接从源码目录跑
    import pathlib
    import sys
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


print(f"数字基线：准确率 {acc_digital:.3f}    权重矩阵 W: {W.shape[0]}x{W.shape[1]}\n")

opu_ideal = PhotonicMatrixProcessor(W)
print(opu_ideal.report())
rel, acc = evaluate(opu_ideal, ideal=True)
print(f"\n理想光子芯片：准确率 {acc:.3f}，logit 相对误差 {rel:.2e}")
print("注意 4x16 的矩阵占满了 16 根波导，但只有 4 个非零奇异值 —— "
      "方形网格跑扁矩阵，大半芯片是空转的。\n")

# ------------------------------------------------------------------ #
# A. 只扫随机相位噪声（热漂移），插损置零，隔离出随机误差的影响
# ------------------------------------------------------------------ #
print("=" * 70)
print("A. 只有随机热漂移（已出厂校准，无插损）")
print("=" * 70)
print(f"{'漂移 (rad)':>12} {'logit 相对误差':>16} {'有效 bit':>10} {'准确率':>10}")
for drift in [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]:
    opu = PhotonicMatrixProcessor(
        W, noise=NoiseModel(fab_theta=0.02, fab_phi=0.02,
                            drift_theta=drift, drift_phi=drift), seed=7)
    opu.calibrate()
    rel, acc = evaluate(opu)
    print(f"{drift:>12.3f} {rel * 100:>15.2f}% "
          f"{-np.log2(max(rel, 1e-18)):>10.1f} {acc:>10.3f}")

print("\n-> logit 掉到 2~3 bit，分类准确率仍然纹丝不动。")
print("   低精度对推理是可接受的 —— 这正是光计算敢做 AI 加速器的前提。")

# ------------------------------------------------------------------ #
# B. 插损是另一回事：它是确定性的，不是随机噪声
# ------------------------------------------------------------------ #
print("\n" + "=" * 70)
print("B. 插损（确定性通道失配，不是随机噪声）")
print("=" * 70)
print(f"{'插损(dB/MZI)':>14} {'最坏路径':>10} {'logit 相对误差':>16} {'准确率':>10}")
for loss in [0.0, 0.02, 0.05, 0.1, 0.2]:
    opu = PhotonicMatrixProcessor(W, noise=NoiseModel(mzi_loss_db=loss), seed=7)
    rel, acc = evaluate(opu)
    worst = opu.path_mzi_count().max() * loss
    print(f"{loss:>14.2f} {worst:>9.1f}dB {rel * 100:>15.2f}% {acc:>10.3f}")

cnt = PhotonicMatrixProcessor(W).path_mzi_count()
print(f"\n各波导经过的 MZI 台数: {cnt.tolist()}")
print(f"最长路径 {cnt.max()} 台 vs 最短 {cnt.min()} 台 —— Reck 三角网格的固有缺陷。")
print("这部分误差是**系统性**的：每次都一样，不会随机抖动。")
print("真实系统会通过反向调低低损通道的 VOA 把它拉平（代价是牺牲整体光功率），")
print("本模拟器的 calibrate() 只处理相移误差，没有做这个补偿，所以这里看到的是裸值。")
print("\n换成 Clements 矩形网格能从根本上消除路径不均匀 —— 见 docs/review.md 的 P7。")
