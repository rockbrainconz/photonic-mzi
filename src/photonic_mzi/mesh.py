"""
器件层与编译层：2x2 马赫-曾德尔干涉仪，以及「酉矩阵 -> MZI 相移角」的分解。

这一层不依赖 matplotlib，也不涉及任何物理噪声模型，只做纯粹的线性代数。
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "MZI",
    "mzi_transfer_matrix",
    "decompose_unitary",
    "recompose_unitary",
]


# =============================================================================
# 1. 器件层：单个 2x2 马赫-曾德尔干涉仪
# =============================================================================
def mzi_transfer_matrix(theta: float, phi: float) -> np.ndarray:
    """
    2x2 MZI 传输矩阵（幺正）。

        theta : 内相移，控制分光比（振幅路由）
        phi   : 外相移，控制两路的相对相位

    注意 ``theta=0`` 时本矩阵是 **交换阵** 而非单位阵；单位阵对应
    ``(theta=pi/2, phi=pi)``。这个反直觉的事实正是参考实现里那个 bug 的根源，
    详见 docs/review.md 的 B1。
    """
    s, c = np.sin(theta), np.cos(theta)
    e = np.exp(1j * phi)
    return np.array([[-e * s, c],
                     [e * c, s]], dtype=complex)


def apply_T_left(A: np.ndarray, m: int, theta: float, phi: float) -> None:
    """``A <- T_m(theta,phi) @ A``，就地更新第 m/m+1 两行。O(列数) 而非 O(N^2)。"""
    s, c = np.sin(theta), np.cos(theta)
    e = np.exp(1j * phi)
    a, b = A[m].copy(), A[m + 1]
    A[m] = -e * s * a + c * b
    A[m + 1] = e * c * a + s * b


def apply_T_dagger_left(E: np.ndarray, m: int, theta: float | np.ndarray,
                        phi: float | np.ndarray,
                        amp: float = 1.0) -> None:
    """``E <- T_m(theta,phi)^H @ E``，就地更新。

    ``theta`` / ``phi`` 可以是标量，也可以是与批量列数相同的一维数组；后者用于让
    每个输入样本拥有独立的动态相位抖动。``amp`` 是该器件的幅度透过率（插损）。
    """
    s, c = np.sin(theta), np.cos(theta)
    e = np.exp(-1j * phi)
    a, b = E[m].copy(), E[m + 1]
    E[m] = amp * e * (-s * a + c * b)
    E[m + 1] = amp * (c * a + s * b)


# =============================================================================
# 2. 编译层：酉矩阵 -> MZI 相移角
# =============================================================================
@dataclass(frozen=True)
class MZI:
    """网格中的一台物理干涉仪。``mode`` 是它占用的低编号波导（占用 mode 与 mode+1）。"""

    mode: int
    theta: float
    phi: float
    layer: int = 0      #: 光路上的列号，同一列的 MZI 互不相交、天然并行
    index: int = 0      #: 在正向传播顺序中的序号


def decompose_unitary(U: np.ndarray) -> tuple[list[MZI], np.ndarray]:
    """
    Reck 三角分解：把 N x N 酉矩阵拆成 N(N-1)/2 台 MZI + N 个输出相移。

    .. math::
        T_k \\cdots T_1 U = \\mathrm{diag}(e^{i\\delta})
        \\;\\Longrightarrow\\;
        U = T_1^H \\cdots T_k^H \\, \\mathrm{diag}(e^{i\\delta})

    返回 ``(mzi_list, diag_phases)``，``mzi_list`` 已按 **正向光传播顺序** 排好，
    并带上贪心分层得到的 ``layer``。
    """
    U = np.asarray(U, dtype=complex)
    if U.ndim != 2 or U.shape[0] != U.shape[1] or U.shape[0] == 0:
        raise ValueError(f"U 必须是非空方阵，收到 shape={U.shape}")
    if not np.all(np.isfinite(U)):
        raise ValueError("U 必须只包含有限数值")
    N = U.shape[0]
    if not np.allclose(U.conj().T @ U, np.eye(N), atol=1e-10, rtol=1e-10):
        raise ValueError("U 必须是酉矩阵（U^H @ U = I）")

    A = U.copy()
    elim: list[tuple[int, float, float]] = []

    for col in range(N - 1):
        for row in range(N - 1, col, -1):
            m = row - 1
            x, y = A[m, col], A[row, col]

            # 目标: (T@A)[row,col] = exp(i*phi)*cos(theta)*x + sin(theta)*y = 0
            #   =>  tan(theta) = -exp(i*phi) * x / y
            # 选 phi 让右边变成非负实数，再用 arctan2 取 theta。
            # arctan2 天然覆盖 y->0 (theta->pi/2) 与 x->0 (theta->0) 两个退化极限，
            # 不需要 if/else 特判 —— 参考实现的特判恰恰写反了。
            phi = np.angle(y) - np.angle(x) - np.pi
            # 相移器本质上模 2*pi，把 phi 折回 (-pi, pi]：既是物理事实
            # （热调相移器行程有限），也避免报出 -360 度这种没意义的数。
            phi = (phi + np.pi) % (2 * np.pi) - np.pi
            theta = np.arctan2(np.abs(x), np.abs(y))

            apply_T_left(A, m, theta, phi)
            elim.append((m, theta, phi))

    diag_phases = np.diag(A).copy()

    # 正向传播顺序 = 消元顺序的逆序；在该顺序上贪心分层得到网格的物理列号
    forward = list(reversed(elim))
    busy = np.zeros(N, dtype=int)
    mzis: list[MZI] = []
    for i, (m, theta, phi) in enumerate(forward):
        layer = int(max(busy[m], busy[m + 1]))
        busy[m] = busy[m + 1] = layer + 1
        mzis.append(MZI(mode=m, theta=theta, phi=phi, layer=layer, index=i))
    return mzis, diag_phases


def recompose_unitary(mzis: Iterable[MZI], diag_phases: np.ndarray, N: int) -> np.ndarray:
    """由相移角重建酉矩阵。用于验证编译是否忠实（回环自检）。"""
    A = np.zeros((N, N), dtype=complex)
    for k in range(N):
        A[k, k] = diag_phases[k]
    for z in mzis:
        apply_T_dagger_left(A, z.mode, z.theta, z.phi)
    return A
