"""Experimental incoherent-sunlight backend: algebra, noise semantics, and boundaries."""
from __future__ import annotations

import numpy as np
import pytest

from photonic_mzi import IncoherentSolarProcessor, SolarNoiseModel


def test_signed_matrix_vector_is_exact_in_ideal_model():
    rng = np.random.default_rng(10)
    M = rng.standard_normal((7, 5))
    x = rng.standard_normal(5)
    solar = IncoherentSolarProcessor(M)
    assert np.allclose(solar.read(x), M @ x, atol=1e-12)


def test_signed_batch_is_exact_in_ideal_model():
    rng = np.random.default_rng(11)
    M = rng.standard_normal((4, 6))
    X = rng.standard_normal((6, 23))
    solar = IncoherentSolarProcessor(M)
    assert np.allclose(solar.read(X), M @ X, atol=1e-12)


def test_bias_uses_constant_optical_channel():
    M = np.array([[1.0, -2.0], [-0.5, 0.25]])
    bias = np.array([0.75, -1.25])
    X = np.array([[2.0, 0.0, -1.0], [1.0, -3.0, 0.5]])
    solar = IncoherentSolarProcessor(M, bias=bias)
    assert solar.n_optical_inputs == 3
    assert np.allclose(solar.read(X), M @ X + bias[:, None], atol=1e-12)


def test_dual_rail_powers_are_nonnegative_and_difference_decodes():
    M = np.array([[2.0, -1.0], [-3.0, 4.0]])
    x = np.array([-0.5, 2.0])
    solar = IncoherentSolarProcessor(M, noise=SolarNoiseModel(irradiance=2.5))
    powers = solar.optical_powers(x)
    assert np.all(powers.positive >= 0)
    assert np.all(powers.negative >= 0)
    decoded = ((powers.positive - powers.negative) * solar.weight_scale *
               powers.input_scale / powers.reference)
    assert np.allclose(decoded, M @ x, atol=1e-12)
    assert np.allclose(solar.decode(solar.detect(powers)), M @ x, atol=1e-12)


def test_physical_transmissions_never_exceed_one():
    M = np.array([[100.0, -4.0], [0.0, 25.0]])
    solar = IncoherentSolarProcessor(M)
    assert solar.weight_positive.min() >= 0
    assert solar.weight_negative.min() >= 0
    assert solar.weight_positive.max() <= 1
    assert solar.weight_negative.max() <= 1


def test_reference_normalization_cancels_common_sunlight_fluctuation():
    M = np.array([[1.0, -0.5], [0.25, 2.0]])
    x = np.array([0.3, -0.8])
    X = np.repeat(x[:, None], 64, axis=1)
    solar = IncoherentSolarProcessor(
        M, noise=SolarNoiseModel(irradiance=3.0, common_fluctuation=0.8), seed=4)
    Y = solar.read(X, ideal=False, normalize=True)
    assert np.allclose(Y, (M @ x)[:, None], atol=1e-12)


def test_without_reference_normalization_sunlight_fluctuation_remains():
    M = np.eye(2)
    x = np.array([1.0, 0.5])
    X = np.repeat(x[:, None], 32, axis=1)
    solar = IncoherentSolarProcessor(
        M, noise=SolarNoiseModel(common_fluctuation=0.5), seed=5)
    Y = solar.read(X, ideal=False, normalize=False)
    assert not np.allclose(Y, Y[:, [0]])
    assert np.allclose(Y[1] / Y[0], 0.5, atol=1e-12)


def test_spatial_and_spectral_errors_are_fixed_not_per_read_noise():
    M = np.array([[1.0, -2.0], [0.5, 3.0]])
    x = np.array([0.8, -0.4])
    solar = IncoherentSolarProcessor(
        M,
        noise=SolarNoiseModel(
            spatial_nonuniformity=0.1,
            spectral_weight_error=0.1,
            differential_gain_error=0.05,
        ),
        seed=8,
    )
    y1 = solar.read(x, ideal=False)
    y2 = solar.read(x, ideal=False)
    assert np.array_equal(y1, y2)
    assert not np.allclose(y1, M @ x)


def test_shot_noise_is_reproducible_with_seed():
    M = np.array([[1.0, 0.5], [-0.25, 1.5]])
    x = np.array([0.4, -0.7])
    noise = SolarNoiseModel(photons_per_unit=2000)
    a = IncoherentSolarProcessor(M, noise=noise, seed=9)
    b = IncoherentSolarProcessor(M, noise=SolarNoiseModel(photons_per_unit=2000), seed=9)
    powers = a.optical_powers(x, ideal=False)
    y_a = a.decode(a.detect(powers, ideal=False))
    y_b = b.read(x, ideal=False)
    assert np.array_equal(y_a, y_b)


def test_noisy_reference_must_remain_positive():
    solar = IncoherentSolarProcessor(
        np.eye(1), noise=SolarNoiseModel(reference_noise=10.0), seed=4)
    with pytest.raises(RuntimeError, match="reference detector"):
        solar.read(np.array([1.0]), ideal=False)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_nonnegative_noise_parameters_are_validated(bad):
    with pytest.raises(ValueError):
        SolarNoiseModel(common_fluctuation=bad)


@pytest.mark.parametrize("x", [np.ones(3), np.ones((3, 2)), np.ones((2, 2, 1))])
def test_input_shape_is_validated(x):
    solar = IncoherentSolarProcessor(np.eye(2))
    with pytest.raises(ValueError):
        solar.read(x)


def test_zero_matrix_and_zero_input_are_well_defined():
    solar = IncoherentSolarProcessor(np.zeros((3, 2)))
    assert np.array_equal(solar.read(np.zeros(2)), np.zeros(3))
    assert solar.weight_scale == 1.0


def test_report_states_model_boundary():
    report = IncoherentSolarProcessor(np.eye(2)).report()
    assert "incoherent sunlight" in report
    assert "not wavelength resolved" in report
