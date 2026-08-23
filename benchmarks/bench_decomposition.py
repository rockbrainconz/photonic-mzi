"""
Scaling and batch-throughput benchmark for this implementation.
本项目的规模扩展与批处理基准。

    python benchmarks/bench_decomposition.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from photonic_mzi import PhotonicMatrixProcessor  # noqa: E402


def timeit(fn, rep: int = 3) -> float:
    """Return the best of repeated timings. / 返回多次运行中的最短时间。"""
    best = float("inf")
    for _ in range(rep):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


def main() -> None:
    print("Scaling / 规模扩展")
    print(f"{'N':>4} | {'compile / 编译':>18} | {'forward / 单次前向':>22} | "
          f"{'rel. error / 相对误差':>23}")
    print("-" * 78)
    for n in [8, 16, 32, 64, 128]:
        rng = np.random.default_rng(n)
        matrix = rng.standard_normal((n, n))
        vector = rng.standard_normal(n)
        rep = 3 if n <= 32 else 1

        compile_time = timeit(lambda matrix=matrix: PhotonicMatrixProcessor(matrix), rep)
        opu = PhotonicMatrixProcessor(matrix)
        forward_time = timeit(lambda opu=opu, vector=vector: opu.forward(vector), rep)
        relative_error = (
            np.linalg.norm(opu.read_coherent(vector) - matrix @ vector)
            / np.linalg.norm(matrix @ vector)
        )

        print(f"{n:>4} | {compile_time * 1e3:>15.1f} ms | "
              f"{forward_time * 1e3:>19.1f} ms | {relative_error:>23.2e}")

    n, batch = 64, 256
    matrix = np.random.default_rng(1).standard_normal((n, n))
    inputs = np.random.default_rng(2).standard_normal((n, batch))
    opu = PhotonicMatrixProcessor(matrix)
    batch_time = timeit(lambda: opu.forward(inputs), 1)
    loop_time = timeit(lambda: [opu.forward(inputs[:, i]) for i in range(batch)], 1)
    print(f"\nBatch / 批处理: {n}x{n} @ {batch} inputs")
    print(f"one call / 一次调用: {batch_time * 1e3:.1f} ms")
    print(f"Python loop / 逐条循环: {loop_time * 1e3:.1f} ms")
    print(f"batch speedup / 批处理加速: {loop_time / batch_time:.0f}x")


if __name__ == "__main__":
    main()
