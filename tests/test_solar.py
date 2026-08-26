"""Experimental incoherent-sunlight backend: algebra, noise semantics, and boundaries."""
from __future__ import annotations

import numpy as np
import pytest

from photonic_mzi import (
    IncoherentSolarProcessor,
    SolarNoiseModel,
    SolarPowerReadout,
)


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
               powers.input_scale / (powers.reference * solar.fanout_fraction))
    assert np.allclose(decoded, M @ x, atol=1e-12)
    assert np.allclose(solar.decode(solar.detect(powers)), M @ x, atol=1e-12)


def test_uniform_fanout_conserves_total_input_rail_power():
    M = np.ones((5, 2))
    x = np.ones(2)
    solar = IncoherentSolarProcessor(
        M, fanout_efficiency=0.8, input_full_scale=1.0)
    powers = solar.optical_powers(x)
    output_power = np.sum(powers.positive) + np.sum(powers.negative)
    available_input_power = 0.8 * np.sum(np.abs(x))
    assert solar.fanout_fraction == pytest.approx(0.8 / 5)
    assert output_power == pytest.approx(available_input_power)
    assert np.allclose(solar.decode(powers), M @ x, atol=1e-12)


def test_weight_setting_error_cannot_create_transmission_gain():
    M = np.ones((6, 2))
    x = np.ones(2)
    solar = IncoherentSolarProcessor(
        M,
        noise=SolarNoiseModel(spectral_weight_error=2.0),
        seed=2,
        fanout_efficiency=0.7,
        input_full_scale=1.0,
    )
    powers = solar.optical_powers(x, ideal=False)
    output_power = np.sum(powers.positive) + np.sum(powers.negative)
    assert output_power <= 0.7 * np.sum(np.abs(x)) + 1e-12


def test_physical_transmissions_never_exceed_one():
    M = np.array([[100.0, -4.0], [0.0, 25.0]])
    solar = IncoherentSolarProcessor(M)
    assert solar.weight_positive.min() >= 0
    assert solar.weight_negative.min() >= 0
    assert solar.weight_positive.max() <= 1
    assert solar.weight_negative.max() <= 1


def test_fixed_input_full_scale_preserves_dynamic_range_and_rejects_overflow():
    solar = IncoherentSolarProcessor(
        np.eye(1), input_full_scale=2.0)
    small = solar.optical_powers(np.array([0.1]))
    large = solar.optical_powers(np.array([1.0]))
    assert small.input_scale == 2.0
    assert large.input_scale == 2.0
    assert small.positive[0] < large.positive[0]
    assert np.allclose(solar.decode(small), [0.1], atol=1e-12)
    assert np.allclose(solar.decode(large), [1.0], atol=1e-12)
    with pytest.raises(ValueError, match="input_full_scale"):
        solar.read(np.array([2.1]))


def test_default_per_vector_agc_records_the_scale_explicitly():
    solar = IncoherentSolarProcessor(np.eye(1))
    small = solar.optical_powers(np.array([0.1]))
    large = solar.optical_powers(np.array([1.0]))
    assert small.input_scale == pytest.approx(0.1)
    assert large.input_scale == pytest.approx(1.0)
    assert small.positive[0] == pytest.approx(large.positive[0])


@pytest.mark.parametrize("bad", [0.0, -1.0, 1.01, float("nan"), float("inf")])
def test_fanout_efficiency_is_validated(bad):
    with pytest.raises(ValueError, match="fanout_efficiency"):
        IncoherentSolarProcessor(np.eye(1), fanout_efficiency=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_fixed_input_full_scale_is_validated(bad):
    with pytest.raises(ValueError, match="input_full_scale"):
        IncoherentSolarProcessor(np.eye(1), input_full_scale=bad)


def test_bias_requires_fixed_full_scale_of_at_least_one():
    with pytest.raises(ValueError, match="at least 1"):
        IncoherentSolarProcessor(
            np.eye(1), bias=np.zeros(1), input_full_scale=0.5)


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


def test_differential_gain_belongs_to_detection_not_passive_optical_power():
    M = np.array([[1.0, -0.5], [0.25, 2.0]])
    x = np.array([0.3, -0.8])
    solar = IncoherentSolarProcessor(
        M, noise=SolarNoiseModel(differential_gain_error=0.5), seed=12)
    ideal_powers = solar.optical_powers(x, ideal=True)
    physical_powers = solar.optical_powers(x, ideal=False)
    assert np.array_equal(physical_powers.positive, ideal_powers.positive)
    assert np.array_equal(physical_powers.negative, ideal_powers.negative)

    observed = solar.detect(physical_powers, ideal=False)
    assert not np.array_equal(observed.positive, physical_powers.positive)
    assert not np.array_equal(observed.negative, physical_powers.negative)


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


def test_external_readout_rejects_mismatched_or_complex_rails_cleanly():
    solar = IncoherentSolarProcessor(np.eye(1))
    scalar_negative = SolarPowerReadout(np.ones(1), 0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        solar.decode(scalar_negative)

    complex_positive = SolarPowerReadout(
        np.array([1.0 + 1.0j]), np.zeros(1), 1.0, 1.0)
    with pytest.raises(ValueError, match="positive must be real"):
        solar.decode(complex_positive)


def test_report_states_model_boundary():
    report = IncoherentSolarProcessor(np.eye(2)).report()
    assert "incoherent sunlight" in report
    assert "not wavelength resolved" in report
    assert "2-way uniform" in report
    assert "per-vector AGC" in report
