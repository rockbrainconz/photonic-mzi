"""Experimental incoherent-sunlight MAC with signed dual rails and reference readout."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from photonic_mzi import IncoherentSolarProcessor, SolarNoiseModel

M = np.array([
    [0.8, -0.3, 0.0, 0.5],
    [-0.2, 0.6, 0.4, -0.1],
    [0.1, 0.0, -0.7, 0.9],
])
bias = np.array([0.05, -0.10, 0.20])
x = np.array([0.7, -0.4, 0.2, 0.9])
truth = M @ x + bias

ideal = IncoherentSolarProcessor(M, bias=bias)
print(ideal.report())
print("\nCPU / 电域真值       :", np.round(truth, 6))
print("Ideal solar / 理想日光:", np.round(ideal.read(x), 6))

noise = SolarNoiseModel(
    irradiance=2.0,
    common_fluctuation=0.50,
    spatial_nonuniformity=0.01,
    spectral_weight_error=0.01,
    differential_gain_error=0.005,
    photons_per_unit=200_000,
    detector_noise=1e-4,
    reference_noise=1e-4,
)
solar = IncoherentSolarProcessor(M, bias=bias, noise=noise, seed=7)

# Every batch column is a simultaneous exposure with independently fluctuating sunlight.
X = np.repeat(x[:, None], 256, axis=1)
powers = solar.optical_powers(X, ideal=False)
observed = solar.detect(powers, ideal=False)
raw = solar.decode(observed, normalize=False)
normalized = solar.decode(observed, normalize=True)

mean = normalized.mean(axis=1)
relative_error = np.linalg.norm(mean - truth) / np.linalg.norm(truth)
print("\nRaw output std / 未归一化波动  :", np.round(raw.std(axis=1), 6))
print("Normalized mean / 归一化均值 :", np.round(mean, 6))
print(f"Mean relative error / 均值相对误差: {relative_error:.3%}")

ideal_powers = ideal.optical_powers(x)
print("Positive rail / 正轨功率:", np.round(ideal_powers.positive, 6))
print("Negative rail / 负轨功率:", np.round(ideal_powers.negative, 6))
print("Reference / 参考功率    :", round(ideal_powers.reference, 6))
