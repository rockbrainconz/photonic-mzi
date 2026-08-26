"""
photonic-mzi — MZI mesh photonic-computing simulator / MZI 网格光计算模拟器
============================================================================

Validate circuit-level photonic matrix multiply-accumulate by compiling any real
matrix through SVD into Mach-Zehnder interferometer meshes and checking ideal
propagation against ``M @ x``. This is a teaching model, not a hardware replica.

验证光处理器执行矩阵乘加的电路级可行性：把任意实矩阵通过 SVD 编译成
马赫-曾德尔干涉仪（MZI）网格，并检查理想传播与 ``M @ x`` 一致。
它是电路级教学模型，不是商用芯片硬件复刻。

    [x] -> [V^T unitary mesh / 酉变换网格] -> [Sigma attenuators / 光衰减器]
        -> [U unitary mesh / 酉变换网格] -> [y]

Quick start / 快速上手::

    import numpy as np
    from photonic_mzi import PhotonicMatrixProcessor

    M = np.random.randn(8, 5)
    opu = PhotonicMatrixProcessor(M, seed=42)
    y = opu.read_coherent(np.random.randn(5))   # agrees with M @ x to about 1e-15
    E = opu.optical_field(np.random.randn(5))   # complex field before detection
    print(opu.report())

Animation / 动画 (requires ``pip install "photonic-mzi[viz]"``)::

    python -m photonic_mzi

Experimental incoherent-sunlight backend / 实验性非相干日光后端::

    from photonic_mzi import IncoherentSolarProcessor

    solar = IncoherentSolarProcessor(
        M, fanout_efficiency=0.8, input_full_scale=1.0)
    y = solar.read(x)

This backend uses signed intensity rails, passive uniform fan-out, direct detection,
and simultaneous reference normalization. It is separate from the coherent MZI field
model. / 此后端使用强度双轨、被动均匀扇出、直接探测与同时参考归一化，不属于相干
MZI 光场模型。
"""
from .mesh import MZI, decompose_unitary, mzi_transfer_matrix, recompose_unitary
from .processor import NoiseModel, PhotonicMatrixProcessor
from .solar import IncoherentSolarProcessor, SolarNoiseModel, SolarPowerReadout

__version__ = "1.0.1"

__all__ = [
    "MZI",
    "NoiseModel",
    "PhotonicMatrixProcessor",
    "IncoherentSolarProcessor",
    "SolarNoiseModel",
    "SolarPowerReadout",
    "decompose_unitary",
    "mzi_transfer_matrix",
    "recompose_unitary",
    "__version__",
]
