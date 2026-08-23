"""处理器层：编译正确性、非方阵、批量、能量守恒、噪声语义、校准。"""
from __future__ import annotations

import numpy as np
import pytest

from photonic_mzi import NoiseModel, PhotonicMatrixProcessor

STRUCTURED = {
    "identity(4)": np.eye(4),
    "permutation(4)": np.eye(4)[[2, 0, 3, 1]],
    "diagonal(4)": np.diag([2.0, 1.0, 0.5, 3.0]),
    "swap-sparse(4)": np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                                [0, 0, 0, 1], [0, 0, 1, 0]], float),
    "block-diag(4)": np.array([[1, 2, 0, 0], [3, 4, 0, 0],
                               [0, 0, 5, 6], [0, 0, 7, 8]], float),
    "rank-deficient": np.array([[1, 2, 3, 4], [2, 4, 6, 8],
                                [1, 0, 1, 0], [0, 1, 0, 1]], float),
    "upper-triangular(5)": np.triu(np.ones((5, 5))),
    "all-ones(4)": np.ones((4, 4)),
    "one-hot(3x3)": np.array([[0, 0, 1.0], [0, 0, 0], [0, 0, 0]]),
    "zeros(3)": np.zeros((3, 3)),
}


# --------------------------------------------------------------------------- #
# 核心正确性
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(STRUCTURED))
def test_structured_matrices(name):
    """结构化矩阵覆盖零元素对应的退化极限。"""
    M = STRUCTURED[name]
    x = np.random.default_rng(abs(hash(name)) % 2**32).standard_normal(M.shape[1])
    y = PhotonicMatrixProcessor(M).read_coherent(x)
    assert np.linalg.norm(y - M @ x) < 1e-9, name


@pytest.mark.parametrize("n", [2, 3, 5, 8, 12])
def test_random_dense(n):
    rng = np.random.default_rng(n)
    M, x = rng.standard_normal((n, n)), rng.standard_normal(n)
    y = PhotonicMatrixProcessor(M).read_coherent(x)
    assert np.linalg.norm(y - M @ x) / np.linalg.norm(M @ x) < 1e-12


@pytest.mark.parametrize("shape", [(2, 5), (5, 2), (3, 7), (7, 3), (1, 4), (4, 1), (1, 1)])
def test_rectangular(shape):
    rng = np.random.default_rng(sum(shape))
    M, x = rng.standard_normal(shape), rng.standard_normal(shape[1])
    y = PhotonicMatrixProcessor(M).read_coherent(x)
    assert y.shape == (shape[0],)
    assert np.linalg.norm(y - M @ x) < 1e-9


def test_batch_matches_loop():
    rng = np.random.default_rng(0)
    M, X = rng.standard_normal((6, 6)), rng.standard_normal((6, 32))
    opu = PhotonicMatrixProcessor(M)
    assert np.allclose(opu.read_coherent(X), M @ X, atol=1e-12)
    assert np.allclose(opu.read_coherent(X[:, 3]), opu.read_coherent(X)[:, 3], atol=1e-12)


def test_dynamic_jitter_is_independent_across_batch_samples():
    """batch 的每列代表独立输入样本，不能共享一份所谓“逐样本”动态抖动。"""
    M = np.eye(3)
    x = np.array([1.0, 0.5, -0.2])
    X = np.repeat(x[:, None], 8, axis=1)
    opu = PhotonicMatrixProcessor(
        M, noise=NoiseModel(drift_theta=0.05, drift_phi=0.05), seed=42)
    Y = opu.read_coherent(X, ideal=False)
    assert not np.allclose(Y, Y[:, [0]])


def test_static_voa_error_is_fixed_across_reads():
    opu, _, x = _setup(NoiseModel(voa_rel_err=0.05))
    assert np.array_equal(opu.read_coherent(x, ideal=False),
                          opu.read_coherent(x, ideal=False))


def test_orthogonal_matrix_conserves_energy():
    """正交阵奇异值全为 1，整条光路应当无损。"""
    rng = np.random.default_rng(5)
    Q = np.linalg.qr(rng.standard_normal((7, 7)))[0]
    x = rng.standard_normal(7)
    E = PhotonicMatrixProcessor(Q).forward(x)
    assert abs(np.sum(np.abs(E) ** 2) - np.sum(x ** 2)) < 1e-10


@pytest.mark.parametrize("n", [3, 6, 9])
def test_unitary_roundtrip_error(n):
    eu, ev = PhotonicMatrixProcessor(
        np.random.default_rng(n).standard_normal((n, n))).unitary_error()
    assert max(eu, ev) < 1e-10


def test_singular_values_normalised_below_one():
    """VOA 只能衰减，物理透过率必须 <= 1，增益记在电域。"""
    M = np.array([[3.0, 1.0], [0.0, 2.0]])
    opu = PhotonicMatrixProcessor(M)
    assert opu.S_phys.max() <= 1.0 + 1e-12
    assert opu.gain == pytest.approx(opu.S_target[0])


def test_optical_field_excludes_electrical_gain():
    """光场、相干电读出和平方律读出必须位于清晰分开的物理层。"""
    opu = PhotonicMatrixProcessor(np.array([[2.0]]))
    assert opu.gain == pytest.approx(2.0)
    assert opu.optical_field([1.0]) == pytest.approx([1.0 + 0j])
    assert opu.read_coherent([1.0]) == pytest.approx([2.0])
    assert opu.read_intensity([1.0]) == pytest.approx([4.0])


def test_detector_noise_is_post_detection_not_optical_field_noise():
    nz = NoiseModel(detector_noise_floor=0.1)
    clean = PhotonicMatrixProcessor(np.eye(2), noise=nz, seed=3)
    noisy = PhotonicMatrixProcessor(np.eye(2), noise=nz, seed=3)
    x = np.array([0.0, 0.0])
    assert np.array_equal(clean.optical_field(x, ideal=False), np.zeros(2))
    assert not np.array_equal(noisy.read_coherent(x, ideal=False), np.zeros(2))


def test_intensity_readout_loses_sign():
    """直接光电探测测的是 |E|^2，符号信息必然丢失。"""
    M, x = np.array([[-1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 1.0])
    opu = PhotonicMatrixProcessor(M)
    assert np.all(opu.read_intensity(x) >= 0)
    assert opu.read_coherent(x)[0] < 0


@pytest.mark.parametrize("bad", [np.array(1.0), np.zeros((2, 2, 2)), np.zeros((0, 3))])
def test_input_validation(bad):
    with pytest.raises(ValueError):
        PhotonicMatrixProcessor(bad)


@pytest.mark.parametrize("bad_x", [
    np.arange(4.0),
    np.zeros((1, 4)),
    np.zeros((4, 1)),
    np.zeros((2, 2, 1)),
    np.array([1.0, np.nan]),
])
def test_forward_rejects_wrong_input_shape_or_nonfinite_values(bad_x):
    with pytest.raises(ValueError):
        PhotonicMatrixProcessor(np.eye(2)).optical_field(bad_x)


def test_complex_matrix_is_rejected_instead_of_silently_truncated():
    with pytest.raises(ValueError, match="实矩阵"):
        PhotonicMatrixProcessor(np.array([[1.0 + 1.0j]]))


@pytest.mark.parametrize("kwargs", [
    {"mzi_loss_db": -0.1},
    {"drift_theta": -0.1},
    {"voa_rel_err": float("nan")},
    {"detector_snr_db": 0},
    {"detector_snr_db": -float("inf")},
    {"detector_noise_floor": -1.0},
])
def test_noise_model_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        NoiseModel(**kwargs)


# --------------------------------------------------------------------------- #
# 噪声语义
# --------------------------------------------------------------------------- #
def _setup(noise, seed=1):
    rng = np.random.default_rng(11)
    M, x = rng.standard_normal((5, 5)), rng.standard_normal(5)
    return PhotonicMatrixProcessor(M, noise=noise, seed=seed), M, x


def test_same_seed_is_reproducible():
    """独立 Generator，不依赖也不污染全局 np.random。"""
    nz = NoiseModel(fab_theta=0.03, fab_phi=0.03)
    a, _, x = _setup(nz)
    b, _, _ = _setup(nz)
    assert np.allclose(a.read_coherent(x, ideal=False), b.read_coherent(x, ideal=False))


def test_static_fab_error_is_repeatable_shot_to_shot():
    """静态相移控制偏置固定：同一处理器重复读出必须一致。"""
    opu, _, x = _setup(NoiseModel(fab_theta=0.03, fab_phi=0.03))
    assert np.allclose(opu.read_coherent(x, ideal=False),
                       opu.read_coherent(x, ideal=False))


def test_thermal_drift_varies_shot_to_shot():
    """i.i.d. 动态相位抖动按输入样本重采样。"""
    opu, _, x = _setup(NoiseModel(drift_theta=0.02))
    assert not np.allclose(opu.read_coherent(x, ideal=False),
                           opu.read_coherent(x, ideal=False))


def test_insertion_loss_attenuates():
    opu, M, x = _setup(NoiseModel(mzi_loss_db=0.5))
    lossy = np.sum(np.abs(opu.forward(x, ideal=False)) ** 2)
    ideal = np.sum(np.abs(opu.forward(x, ideal=True)) ** 2)
    assert lossy < ideal


def test_ideal_path_ignores_noise_model():
    """ideal=True 必须完全绕开噪声，否则动画和基线就不可信了。"""
    nz = NoiseModel(fab_theta=0.1, drift_theta=0.1, mzi_loss_db=1.0,
                    voa_rel_err=0.1, detector_snr_db=10)
    opu, M, x = _setup(nz)
    assert np.allclose(opu.read_coherent(x, ideal=True), M @ x, atol=1e-12)


# --------------------------------------------------------------------------- #
# 校准
# --------------------------------------------------------------------------- #
def test_calibration_removes_static_error():
    """理想表征可以精确抵消模型内的加性相移控制偏置。"""
    opu, M, x = _setup(NoiseModel(fab_theta=0.03, fab_phi=0.03))
    before = np.linalg.norm(opu.read_coherent(x, ideal=False) - M @ x)
    opu.calibrate()
    after = np.linalg.norm(opu.read_coherent(x, ideal=False) - M @ x)
    assert after < before / 100, f"校准前 {before:.3e} -> 校准后 {after:.3e}"
    assert after < 1e-12


def test_calibration_also_corrects_output_phase_screens():
    """fab_phi 同时覆盖网格外相移器和输出相位屏，不能漏掉后者。"""
    opu, M, x = _setup(NoiseModel(fab_phi=0.03))
    assert np.linalg.norm(opu.read_coherent(x, ideal=False) - M @ x) > 1e-6
    opu.calibrate()
    assert np.linalg.norm(opu.read_coherent(x, ideal=False) - M @ x) < 1e-12


def test_calibration_cannot_fix_drift():
    """动态漂移标不掉 —— 校准前后误差量级应当相当。"""
    opu, M, x = _setup(NoiseModel(drift_theta=0.02, drift_phi=0.02))
    before = np.linalg.norm(opu.read_coherent(x, ideal=False) - M @ x)
    opu.calibrate()
    after = np.linalg.norm(opu.read_coherent(x, ideal=False) - M @ x)
    assert after > before / 10


def test_reset_calibration():
    opu, M, x = _setup(NoiseModel(fab_theta=0.03))
    opu.calibrate()
    assert opu.calibrated
    opu.reset_calibration()
    assert not opu.calibrated
    assert np.linalg.norm(opu.read_coherent(x, ideal=False) - M @ x) > 1e-9


# --------------------------------------------------------------------------- #
# 报告
# --------------------------------------------------------------------------- #
def test_report_contains_key_metrics():
    opu = PhotonicMatrixProcessor(np.random.default_rng(0).standard_normal((4, 4)))
    text = opu.report()
    for key in ["MZI 总数", "网格深度", "奇异值", "编译回环误差", "校准状态"]:
        assert key in text
    assert opu.n_mzi == 2 * (4 * 3 // 2)
    assert opu.mode_mzi_count().sum() == 2 * opu.n_mzi
    assert np.array_equal(opu.path_mzi_count(), opu.mode_mzi_count())
