"""
复杂度对比：对照样例（每台 MZI 构造 N×N 全矩阵再相乘）vs 本实现（就地两行更新）。

    python benchmarks/bench_decomposition.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tests._reference_comparison import (  # noqa: E402
    PhotonicMatrixProcessor as ComparisonOPU,
)

from photonic_mzi import PhotonicMatrixProcessor  # noqa: E402


def timeit(fn, rep: int = 3) -> float:
    best = float("inf")
    for _ in range(rep):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def main() -> None:
    print(f"{'N':>4} | {'编译(参考)':>12} {'编译(本实现)':>13} {'加速':>7} | "
          f"{'前向(参考)':>12} {'前向(本实现)':>13} {'加速':>7}")
    print("-" * 88)
    for N in [8, 16, 32, 64, 128]:
        rng = np.random.default_rng(0)
        A = rng.standard_normal((N, N))
        v = rng.standard_normal(N)
        rep = 3 if N <= 32 else 1

        t_ref_c = timeit(lambda A=A: ComparisonOPU(A), rep)
        t_new_c = timeit(lambda A=A: PhotonicMatrixProcessor(A), rep)

        ref, new = ComparisonOPU(A), PhotonicMatrixProcessor(A)
        t_ref_f = timeit(lambda ref=ref, v=v: ref.forward_optical_simulation(v), rep)
        t_new_f = timeit(lambda new=new, v=v: new.forward(v), rep)

        assert np.allclose(new.read_coherent(v), A @ v, atol=1e-9)

        print(f"{N:>4} | {t_ref_c * 1e3:>10.1f}ms {t_new_c * 1e3:>11.1f}ms "
              f"{t_ref_c / t_new_c:>6.1f}x | {t_ref_f * 1e3:>10.1f}ms "
              f"{t_new_f * 1e3:>11.1f}ms {t_ref_f / t_new_f:>6.1f}x")

    # 批量吞吐：对照样例完全不支持
    N, B = 64, 256
    A = np.random.default_rng(1).standard_normal((N, N))
    X = np.random.default_rng(2).standard_normal((N, B))
    opu = PhotonicMatrixProcessor(A)
    t_batch = timeit(lambda: opu.forward(X), 1)
    t_loop = timeit(lambda: [opu.forward(X[:, i]) for i in range(B)], 1)
    print(f"\n批量 {N}x{N} @ {B} 路输入："
          f"一次性 {t_batch * 1e3:.1f}ms  vs  逐条循环 {t_loop * 1e3:.1f}ms"
          f"  ({t_loop / t_batch:.0f}x)")
    print("对照样例没有批量接口，只能走逐条循环。")


if __name__ == "__main__":
    main()
