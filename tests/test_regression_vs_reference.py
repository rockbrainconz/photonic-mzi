"""
与独立对照样例（tests/_reference_comparison.py）的回归比较。

这些用例把 docs/review.md 里 B1 那个 bug 的表现锁死：
对照样例在结构化矩阵上静默算错，本实现必须算对。
"""
from __future__ import annotations

import numpy as np
import pytest

from photonic_mzi import PhotonicMatrixProcessor

from ._reference_comparison import PhotonicMatrixProcessor as ComparisonOPU

# 对照样例只支持方阵，且不接受非方阵输入
CASES = {
    "identity(4)": np.eye(4),
    "permutation(4)": np.eye(4)[[2, 0, 3, 1]],
    "diagonal(4)": np.diag([2.0, 1.0, 0.5, 3.0]),
    "block-diag(4)": np.array([[1, 2, 0, 0], [3, 4, 0, 0],
                               [0, 0, 5, 6], [0, 0, 7, 8]], float),
    "rank-deficient": np.array([[1, 2, 3, 4], [2, 4, 6, 8],
                                [1, 0, 1, 0], [0, 1, 0, 1]], float),
    "all-ones(4)": np.ones((4, 4)),
}


def _x(n: int, tag: str) -> np.ndarray:
    return np.random.default_rng(abs(hash(tag)) % 2**32).standard_normal(n)


@pytest.mark.parametrize("name", list(CASES))
def test_reference_is_wrong_and_we_are_right(name):
    M = CASES[name]
    x = _x(M.shape[1], name)
    truth = M @ x

    ref_err = np.linalg.norm(
        ComparisonOPU(M).forward_optical_simulation(x) - truth)
    our_err = np.linalg.norm(
        PhotonicMatrixProcessor(M).read_coherent(x) - truth)

    assert ref_err > 1e-3, f"{name}: 对照样例本应在这里出错，回归用例可能失效了"
    assert our_err < 1e-9, f"{name}: 本实现算错了 (err={our_err:.2e})"


@pytest.mark.parametrize("n", [2, 3, 5, 7, 10])
def test_reference_and_ours_agree_on_random_dense(n):
    """随机稠密矩阵碰不到退化分支，两版都应达到机器精度 —— 原验证结论成立。"""
    rng = np.random.default_rng(n * 31)
    M, x = rng.standard_normal((n, n)), rng.standard_normal(n)
    truth = M @ x
    scale = np.linalg.norm(truth)
    assert np.linalg.norm(
        ComparisonOPU(M).forward_optical_simulation(x) - truth) / scale < 1e-11
    assert np.linalg.norm(
        PhotonicMatrixProcessor(M).read_coherent(x) - truth) / scale < 1e-11


def test_reference_uses_global_rng_and_ours_does_not():
    """对照样例走全局 np.random，会污染调用方的随机流；本实现不会。"""
    M = np.random.default_rng(0).standard_normal((4, 4))
    x = np.random.default_rng(1).standard_normal(4)

    np.random.seed(1234)
    before = np.random.random()
    np.random.seed(1234)
    ComparisonOPU(M).forward_optical_simulation(x, add_noise=True)
    assert np.random.random() != before          # 全局流被推进了

    np.random.seed(1234)
    from photonic_mzi import NoiseModel
    PhotonicMatrixProcessor(M, noise=NoiseModel(fab_theta=0.1, drift_theta=0.1),
                            seed=7).read_coherent(x, ideal=False)
    assert np.random.random() == before          # 全局流原封不动
