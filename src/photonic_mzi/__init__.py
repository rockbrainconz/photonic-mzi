"""
photonic-mzi — MZI 网格光计算模拟器
====================================

演示集成光子矩阵乘法器使用的线性光学原理：把任意实矩阵通过 SVD 编译成
马赫-曾德尔干涉仪（MZI）网格。它是电路级教学模型，不是商用芯片硬件复刻。

    [x] -> [V^T 酉变换网格] -> [Sigma 光衰减器] -> [U 酉变换网格] -> [y]

快速上手::

    import numpy as np
    from photonic_mzi import PhotonicMatrixProcessor

    M = np.random.randn(8, 5)
    opu = PhotonicMatrixProcessor(M, seed=42)
    y = opu.read_coherent(np.random.randn(5))   # 与 M @ x 一致到 ~1e-15
    E = opu.optical_field(np.random.randn(5))   # 探测前复光场，不含电域增益
    print(opu.report())

动画（需要装可选依赖 ``pip install "photonic-mzi[viz]"``）::

    python -m photonic_mzi
"""
from .mesh import MZI, decompose_unitary, mzi_transfer_matrix, recompose_unitary
from .processor import NoiseModel, PhotonicMatrixProcessor

__version__ = "1.0.0"

__all__ = [
    "MZI",
    "NoiseModel",
    "PhotonicMatrixProcessor",
    "decompose_unitary",
    "mzi_transfer_matrix",
    "recompose_unitary",
    "__version__",
]
