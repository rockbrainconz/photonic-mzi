"""
例 2：静态制造误差可以标定，动态热漂移标不掉。

这是把 fab_* 和 drift_* 分开建模的全部意义 —— 如果像参考实现那样把两者
混进一个 noise_std，就会得出「光计算精度天生只有 3 bit」这种过于悲观的结论。

    python examples/02_noise_and_calibration.py
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
    # 多发几枪取平均，把逐脉冲抖动的统计特性体现出来
    b = np.mean([bits(opu.read_coherent(x, ideal=False)) for _ in range(32)])
    print(f"  {label:<34} {b:5.1f} bit")


print("有效精度（越高越好）\n")

print("只有静态制造误差 (fab 0.02 rad):")
run("未校准", NoiseModel(fab_theta=0.02, fab_phi=0.02), calibrate=False)
run("校准后", NoiseModel(fab_theta=0.02, fab_phi=0.02), calibrate=True)
print("  -> 静态误差是确定性的，表征一次就能整个抵消掉\n")

print("只有动态热漂移 (drift 0.02 rad):")
run("未校准", NoiseModel(drift_theta=0.02, drift_phi=0.02), calibrate=False)
run("校准后", NoiseModel(drift_theta=0.02, drift_phi=0.02), calibrate=True)
print("  -> 逐脉冲在变，校准完全无能为力\n")

print("两者都有 (fab 0.02 + drift 0.005):")
nz = NoiseModel(fab_theta=0.02, fab_phi=0.02, drift_theta=0.005, drift_phi=0.005)
run("未校准", nz, calibrate=False)
run("校准后", nz, calibrate=True)
print("  -> 校准把地板从 fab 抬到 drift，剩下的才是真实的物理极限\n")

print("再叠加插损与探测噪声（校准也管不了这些）:")
full = NoiseModel(fab_theta=0.02, fab_phi=0.02, drift_theta=0.005, drift_phi=0.005,
                  mzi_loss_db=0.2, voa_rel_err=0.01, detector_snr_db=40)
run("校准后", full, calibrate=True)

opu = PhotonicMatrixProcessor(M, noise=full, seed=7)
cnt = opu.path_mzi_count()
print(f"\n  各波导经过的 MZI 台数: {cnt.tolist()}")
print(f"  最长与最短路径相差 {cnt.max() - cnt.min()} 台 "
      f"= {(cnt.max() - cnt.min()) * full.mzi_loss_db:.1f} dB 的确定性通道失配")
print("  这是 Reck 三角网格的固有缺陷，换 Clements 矩形网格可以消除。")
