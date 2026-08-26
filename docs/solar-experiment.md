# Experimental sunlight matrix multiply-accumulate

[English](solar-experiment.md) | [简体中文](solar-experiment.zh-CN.md)

See [Incoherent sunlight MAC: theory and processor design](solar-processor-design.md)
for assumptions, broadband limits, hardware, electrical boundaries, calibration, and
acceptance criteria.

## Scope

This backend asks a different question from the coherent MZI processor: can sunlight
act as an incoherent power carrier for an arbitrary real matrix MAC,
`y = M @ x + b`? `IncoherentSolarProcessor` propagates non-negative powers through an
intensity crossbar. It does not propagate complex fields, use phase interference, or
reuse the Reck mesh.

## Signed dual rails

Write `x = x+ - x-` and `M = M+ - M-`. The optical outputs are

```text
P+ = C(t) [M+ x+ + M- x-]
P- = C(t) [M+ x- + M- x+]
```

so `P+ - P- = C(t) Mx`. A bias is represented by an extra constant input `x0=1`.
Known input and weight scales keep all encoded intensities and transmissions in
`[0, 1]`; the readout restores those scales.

## Simultaneous reference

A reference photodiode measures `Pref=C(t)`. Dividing the differential output by the
reference cancels common irradiance variation in the noiseless common-mode limit. It
does not cancel spatial nonuniformity, wavelength-integrated weight mismatch,
differential detector-arm mismatch, shot noise, or read noise.

The noise parameters separate those meanings:

| Parameter | Meaning |
|---|---|
| `common_fluctuation` | Independent per-exposure coefficient of variation of sunlight |
| `spatial_nonuniformity` | Fixed per-input gain mismatch |
| `spectral_weight_error` | Fixed lumped wavelength-integrated error per weight |
| `differential_gain_error` | Fixed positive/negative detector-arm mismatch |
| `photons_per_unit` | Enables Poisson photon-counting noise when finite |
| `detector_noise`, `reference_noise` | Additive Gaussian read noise in power units |

## Spectral boundary

A real detector measures a wavelength integral,

```text
Ii = sum_j integral S(lambda,t) R_i(lambda) T_ij(lambda) x_j d_lambda.
```

The current model collapses this into `spectral_weight_error`. It does not resolve the
solar spectrum, filter curves, SLM dispersion, or detector responsivity, so it cannot
select real filters or predict outdoor ENOB.

## Run

```bash
python examples/04_solar_incoherent.py
```

This is an algebra and sensitivity experiment, not evidence of hardware energy,
precision, aperture, weather tolerance, or commercial viability.
