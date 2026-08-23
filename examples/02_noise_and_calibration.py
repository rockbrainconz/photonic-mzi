"""
例 2：静态相移控制偏置可以标定，独立动态相位抖动标不掉。

这是把 fab_* 和 drift_* 分开建模的全部意义 —— 如果像对照样例那样把两者
混进一个 noise_std，就无法区分可表征的控制偏置与运行时随机抖动。

注意：这里的 fab_* 不是分束器制造误差，drift_* 也只是每个样本独立的敏感度
模型，不描述真实热漂移的慢时间相关性。

    python examples/02_noise_and_calibration.py
"""
from __future__ import annotations

import pathlib
import sys

# 直接运行仓库示例时始终使用同仓库源码，避免误导入环境中残留的旧安装版本。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from photonic_mzi import NoiseModel, PhotonicMatrixProcessor

rng = np.random.default_rng(0)
M = rng.standard_normal((8, 8))
x = rng.standard_normal(8)
truth = M @ x


def bits(y) -> float:
    rel = np.linalg.norm(y - truth) / np.linalg.norm(truth)
    return -np.log2(max(rel, 1e-18))


def run(label: str, noise: NoiseModel, calibrate: bool) -> None:
    opu = PhotonicMatrixProcessor(M, noise=noise, seed=7)
    if calibrate:
        opu.calibrate()
    # 多次独立调用取平均，体现每样本 i.i.d. 抖动的统计特性
    b = np.mean([bits(opu.read_coherent(x, ideal=False)) for _ in range(32)])
    print(f"  {label:<34} {b:5.1f} bit")


print("有效精度（越高越好）\n")

print("只有静态相移控制偏置 (fab 0.02 rad):")
run("未校准", NoiseModel(fab_theta=0.02, fab_phi=0.02), calibrate=False)
run("校准后", NoiseModel(fab_theta=0.02, fab_phi=0.02), calibrate=True)
print("  -> 在本模型的理想表征假设下，加性相移偏置可以精确抵消\n")

print("只有独立动态相位抖动 (drift 0.02 rad):")
run("未校准", NoiseModel(drift_theta=0.02, drift_phi=0.02), calibrate=False)
run("校准后", NoiseModel(drift_theta=0.02, drift_phi=0.02), calibrate=True)
print("  -> 每个样本独立变化，静态校准不能消除\n")

print("两者都有 (fab 0.02 + drift 0.005):")
nz = NoiseModel(fab_theta=0.02, fab_phi=0.02, drift_theta=0.005, drift_phi=0.005)
run("未校准", nz, calibrate=False)
run("校准后", nz, calibrate=True)
print("  -> 校准后误差由动态抖动主导；这不是完整硬件精度上限\n")

print("再叠加插损与探测噪声（校准也管不了这些）:")
full = NoiseModel(fab_theta=0.02, fab_phi=0.02, drift_theta=0.005, drift_phi=0.005,
                  mzi_loss_db=0.2, voa_rel_err=0.01, detector_snr_db=40)
run("校准后", full, calibrate=True)

opu = PhotonicMatrixProcessor(M, noise=full, seed=7)
cnt = opu.mode_mzi_count()
print(f"\n  各空间模式参与的 MZI 数: {cnt.tolist()}")
print(f"  参与数范围相差 {cnt.max() - cnt.min()} 台；按统一器件损耗折算为 "
      f"{(cnt.max() - cnt.min()) * full.mzi_loss_db:.1f} dB 的拓扑不均匀性代理")
print("  光会在 MZI 中分束并迁移模式，因此这不是端到端路径追踪。")
print("  Clements 矩形网格通常可缩短并均衡光学深度，但不会消除所有器件误差。")
