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
reuse the Reck mesh. The current compiled core uses an explicit uniform passive fan-out;
it does not duplicate one input rail at full power into every output.

## Signed dual rails

Write `x = x+ - x-` and `M = M+ - M-`. The optical outputs are

```text
P+ = C(t) f [M+ x+ + M- x-]
P- = C(t) f [M+ x- + M- x+]
```

Here `f=eta_f/m` is the branch fraction of an `m`-way fan-out with total efficiency
`eta_f`, so `P+ - P- = C(t) f Mx`. A bias is represented by an extra constant input
`x0=1`.
Known input and weight scales keep all encoded intensities and transmissions in
`[0, 1]`; the readout restores those scales and the known `1/f` factor.

## Passive fan-out and input range

`fanout_efficiency` is the fraction of each input rail available across all output
rows. Uniform fan-out gives every row `fanout_efficiency / n_out`, ensuring

```text
sum(P+ + P-) <= irradiance * fanout_efficiency * sum(abs(encoded_input)).
```

`input_full_scale=value` uses one fixed hardware range and rejects overflow.
`input_full_scale=None` uses per-vector AGC and records its scale in
`SolarPowerReadout`; that convenient algebra mode requires an external scale/control
path and must not be treated as free small-signal gain.
`fanout_efficiency` is measured after the reference tap; reference-tap loss belongs in
an absolute incident-power budget even though the normalized reference is not part of
the compute-output bound.

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

Noisy reference division is a ratio estimator and is generally biased. The model
simulates that ratio directly, but it does not yet provide an analytic uncertainty
estimate; experiments must report reference SNR and absolute/full-scale error near
zero.

## Spectral boundary

A real detector measures a wavelength integral,

```text
Ii = sum_j integral S(lambda,t) R_i(lambda) T_ij(lambda) x_j d_lambda.
```

The current model collapses this into `spectral_weight_error`. It does not resolve the
solar spectrum, filter curves, SLM dispersion, or detector responsivity, so it cannot
select real filters or predict outdoor ENOB. A scalar reference is valid only when the
measured effective matrix obeys `M_eff(t)=c(t)M0`; the full design document defines a
residual metric for violations.

## Coherence boundary

Power addition also requires mutual-coherence cross terms to average below the target
error. Broadband sunlight usually helps, but narrow filtering, single-mode coupling, or
matched split paths can restore interference. A real bench must verify fringe
visibility versus path delay, aperture, and filter bandwidth rather than infer the
condition from the source label alone.

## Run

```bash
python examples/04_solar_incoherent.py
```

This is an algebra and sensitivity experiment, not evidence of hardware energy,
precision, aperture, weather tolerance, or commercial viability.
