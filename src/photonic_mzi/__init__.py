"""
photonic-mzi — MZI 网格光计算模拟器
====================================

模拟 Lightmatter Envise / 曦智 PACE 这类集成光子芯片的底层计算原理：
把任意矩阵编译成一张马赫-曾德尔干涉仪（MZI）网格，让光走一趟就完成矩阵乘法。

    [x] -> [V^T 酉变换网格] -> [Sigma 光衰减器] -> [U 酉变换网格] -> [y]

快速上手::

    import numpy as np
    from photonic_mzi import PhotonicMatrixProcessor

    M = np.random.randn(8, 5)
    opu = PhotonicMatrixProcessor(M, seed=42)
    y = opu.read_coherent(np.random.randn(5))   # 与 M @ x 一致到 ~1e-15
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
