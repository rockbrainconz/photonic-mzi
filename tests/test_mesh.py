"""器件层与编译层：MZI 传输矩阵、Reck 分解、回环重建。"""
from __future__ import annotations

import numpy as np
import pytest

from photonic_mzi import decompose_unitary, mzi_transfer_matrix, recompose_unitary
from photonic_mzi.mesh import apply_T_left


def random_unitary(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


# --------------------------------------------------------------------------- #
# 器件
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("theta", np.linspace(0, np.pi, 7))
@pytest.mark.parametrize("phi", np.linspace(-np.pi, np.pi, 5))
def test_transfer_matrix_is_unitary(theta, phi):
    T = mzi_transfer_matrix(theta, phi)
    assert np.allclose(T.conj().T @ T, np.eye(2), atol=1e-14)


def test_theta_zero_is_a_swap_not_identity():
    """这条反直觉的性质正是参考实现那个 bug 的根源，锁死它防止回退。"""
    assert np.allclose(mzi_transfer_matrix(0.0, 0.0), [[0, 1], [1, 0]], atol=1e-15)
    assert not np.allclose(mzi_transfer_matrix(0.0, 0.0), np.eye(2))


def test_identity_requires_theta_half_pi_phi_pi():
    assert np.allclose(mzi_transfer_matrix(np.pi / 2, np.pi), np.eye(2), atol=1e-15)


@pytest.mark.parametrize("x, y, theta, phi", [
    # (x, y) 退化时，消元必须仍然把下面那个元素干掉
    (0.7 + 0.2j, 0.0, np.pi / 2, np.pi),      # y ~ 0：应保持不动
    (0.0, 0.7 + 0.2j, 0.0, 0.0),              # x ~ 0：应把 y 换上去
])
def test_degenerate_elimination_targets(x, y, theta, phi):
    """new_row = e^{i phi} cos(theta) x + sin(theta) y 必须为 0。"""
    residual = np.exp(1j * phi) * np.cos(theta) * x + np.sin(theta) * y
    assert abs(residual) < 1e-15


def test_apply_T_left_matches_full_matmul():
    """就地两行更新必须与「构造 N×N 全矩阵再相乘」完全等价。"""
    rng = np.random.default_rng(3)
    N, m, theta, phi = 6, 2, 0.7, -1.3
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    full = np.eye(N, dtype=complex)
    full[m:m + 2, m:m + 2] = mzi_transfer_matrix(theta, phi)
    expected = full @ A
    apply_T_left(A, m, theta, phi)
    assert np.allclose(A, expected, atol=1e-14)


# --------------------------------------------------------------------------- #
# 分解
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 11])
def test_decompose_recompose_roundtrip(n):
    U = random_unitary(n, seed=n)
    mzis, phases = decompose_unitary(U)
    assert len(mzis) == n * (n - 1) // 2
    assert np.linalg.norm(recompose_unitary(mzis, phases, n) - U) < 1e-12


@pytest.mark.parametrize("U, name", [
    (np.eye(4), "identity"),
    (np.eye(4)[[2, 0, 3, 1]], "permutation"),
    (np.diag([1, -1, 1j, -1j]), "diagonal-phases"),
    (np.fliplr(np.eye(5)), "anti-diagonal"),
])
def test_decompose_handles_structured_unitaries(U, name):
    """结构化酉矩阵会命中退化分支 —— 参考实现在这里全线崩溃。"""
    U = np.asarray(U, dtype=complex)
    n = U.shape[0]
    mzis, phases = decompose_unitary(U)
    assert np.linalg.norm(recompose_unitary(mzis, phases, n) - U) < 1e-12, name


@pytest.mark.parametrize("n", [2, 4, 7])
def test_phi_wrapped_into_principal_branch(n):
    """相移器模 2π，不该报出 -360° 这种值。"""
    mzis, _ = decompose_unitary(random_unitary(n, seed=n + 100))
    assert all(-np.pi - 1e-9 <= z.phi <= np.pi + 1e-9 for z in mzis)


@pytest.mark.parametrize("n", [2, 4, 6, 9])
def test_layers_are_conflict_free_and_bounded(n):
    """同一列内的 MZI 必须占用互不相交的波导；深度不超过 Reck 的 2N-3。"""
    mzis, _ = decompose_unitary(random_unitary(n, seed=n + 7))
    depth = max(z.layer for z in mzis) + 1
    assert depth <= max(2 * n - 3, 1)
    for layer in range(depth):
        modes = [z.mode for z in mzis if z.layer == layer]
        occupied = [m for z in modes for m in (z, z + 1)]
        assert len(occupied) == len(set(occupied)), f"第 {layer} 列有波导冲突"


def test_indices_are_sequential_in_forward_order():
    mzis, _ = decompose_unitary(random_unitary(5, seed=1))
    assert [z.index for z in mzis] == list(range(len(mzis)))
