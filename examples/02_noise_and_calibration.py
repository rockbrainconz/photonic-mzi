"""
例 2：静态偏置与动态抖动。Example 2: static offsets and dynamic jitter.

这是把 fab_* 和 drift_* 分开建模的意义：混进一个 noise_std，就无法区分
可表征的控制偏置与运行时随机抖动。

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


print("有效精度 / Effective precision (higher is better)\n")

print("只有静态相移控制偏置 / Static phase offset only (fab 0.02 rad):")
run("未校准 / uncalibrated", NoiseModel(fab_theta=0.02, fab_phi=0.02), calibrate=False)
run("校准后 / calibrated", NoiseModel(fab_theta=0.02, fab_phi=0.02), calibrate=True)
print("  -> 理想表征可抵消加性偏置 / ideal characterization cancels additive offsets\n")

print("只有独立动态相位抖动 / Dynamic jitter only (drift 0.02 rad):")
run("未校准 / uncalibrated", NoiseModel(drift_theta=0.02, drift_phi=0.02), calibrate=False)
run("校准后 / calibrated", NoiseModel(drift_theta=0.02, drift_phi=0.02), calibrate=True)
print("  -> 静态校准无法消除 / static calibration cannot remove per-sample jitter\n")

print("两者都有 / Both (fab 0.02 + drift 0.005):")
nz = NoiseModel(fab_theta=0.02, fab_phi=0.02, drift_theta=0.005, drift_phi=0.005)
run("未校准 / uncalibrated", nz, calibrate=False)
run("校准后 / calibrated", nz, calibrate=True)
print("  -> 动态抖动主导 / dynamic jitter dominates; not a hardware precision limit\n")

print("再叠加插损与探测噪声 / Add loss and readout noise:")
full = NoiseModel(fab_theta=0.02, fab_phi=0.02, drift_theta=0.005, drift_phi=0.005,
                  mzi_loss_db=0.2, voa_rel_err=0.01, detector_snr_db=40)
run("校准后 / calibrated", full, calibrate=True)

opu = PhotonicMatrixProcessor(M, noise=full, seed=7)
cnt = opu.mode_mzi_count()
print(f"\n  模式 MZI 参与数 / MZIs per mode: {cnt.tolist()}")
print(f"  参与数范围相差 {cnt.max() - cnt.min()} 台；按统一器件损耗折算为 "
      f"{(cnt.max() - cnt.min()) * full.mzi_loss_db:.1f} dB 的拓扑代理 / "
      f"participation spread gives a {(cnt.max() - cnt.min()) * full.mzi_loss_db:.1f} dB topology proxy")
print("  这不是端到端路径追踪 / This is not end-to-end path tracing.")
print("  Clements 可缩短光学深度，但不会消除器件误差 / Clements reduces depth, not device errors.")
