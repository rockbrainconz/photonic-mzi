# Incoherent sunlight MAC: theory and processor design

[English](solar-processor-design.md) | [简体中文](solar-processor-design.zh-CN.md)

> Status: experimental design specification for `experiment/solar-incoherent`.
> Theory, system boundaries, code semantics, and acceptance criteria come first;
> animation is intentionally deferred.

## Claim and boundary

Sunlight can serve as an incoherent power carrier for matrix multiply-accumulate. An
intensity transmission performs multiplication and a photodetector performs spatial
power summation. This is a separate intensity crossbar, not sunlight injected into the
coherent MZI mesh.

The claim requires common-mode illumination, linear modulators and detectors,
simultaneous signed dual rails, stable wavelength-integrated responses, and explicit
electronic readout boundaries. It does not imply a zero-electricity programmable
computer.

## Algebra

Augment the target with a bias:

```text
A = [M  b],  z = [x; 1],  y = Az.
```

With `alpha=max|A_ij|`, `beta=max|z_j|`, define `W=A/alpha`, `u=z/beta`, then split
`W=W+-W-` and `u=u+-u-` into non-negative rails. The two optical accumulators are

```text
P+ = C(t) [W+u+ + W-u-]
P- = C(t) [W+u- + W-u+].
```

Therefore

```text
P+ - P- = C(t) Az/(alpha beta).
```

If a simultaneous reference measures `Pref=C(t)`, decoding gives

```text
y = alpha beta (P+ - P-) / Pref = Mx + b.
```

The reference cancels only a shared scalar. Per-channel illumination, spectral weight
drift, detector mismatch, local shadows, shot noise, and read noise remain.

## Broadband condition

A real detector measures

```text
P_i(t) = sum_j integral S_j(lambda,t) T_ij(lambda,t) R_i(lambda,t) u_j d_lambda.
```

A fixed matrix exists only when this wavelength-dependent product separates into a
common time-varying scalar and stable channel responses. Otherwise the hardware
computes `M_eff(t)x`; one reference photodiode cannot recover `Mx`.

`spectral_weight_error` is currently a fixed lumped error after wavelength integration.
It is not a resolved solar-spectrum, filter, SLM-dispersion, or responsivity model.

## Hardware chain

```text
sunlight -> collection -> bandpass -> homogenizer
         -> input rails -> fan-out -> weight rails -> row summation
         -> paired photodiodes + simultaneous reference
         -> differential readout -> reference normalization -> scale restoration
```

Each logical `(i,j)` needs the four products `W+u+`, `W-u-`, `W+u-`, and `W-u+`.
The first two feed the positive detector and the other two feed the negative detector.
All rails must be exposed simultaneously.

Three implementation levels are meaningful:

1. Fixed passive mask and optical input: the optical core can be externally unpowered.
2. Fixed matrix and programmable input: the input modulator and readout use electricity.
3. Fully programmable input and weight SLMs: drivers, TIA, ADC, control, and often thermal
   stabilization use electricity.

The current normalized model contains no watts, exposure time, or peripheral energy and
cannot report J/MAC.

## Calibration sequence

1. Dark offsets and detector noise.
2. Reference/rail linearity and saturation limits.
3. Per-input flat-field gain.
4. Per-weight command-to-transmission lookup table.
5. Positive/negative detector-arm balance.
6. Re-characterization under several spectra or filter bands.
7. Basis-vector measurement of `M_eff`, followed by held-out random vectors.
8. Outdoor repetition across irradiance, cloud, solar elevation, and temperature.

Static calibration cannot remove within-exposure local shadows, spectral drift, or
independent detector noise.

## Software-to-hardware mapping

The implementation exposes the physical boundaries directly:

```python
powers = solar.optical_powers(x, ideal=False)  # passive optical powers
observed = solar.detect(powers, ideal=False)   # photon/readout statistics
y = solar.decode(observed, normalize=True)    # differential/reference decoding
```

`read()` composes those three operations. Real measured powers can be packaged as
`SolarPowerReadout` and passed to the same decoder.

## Precision boundary

Shot-noise-only SNR scales as `sqrt(N_photons)`, so an optimistic `b`-bit relative
resolution needs order `2^(2b)` detected photons. Signed cancellation makes the bound
worse when `P+` and `P-` are both large but their difference is small. Experiments must
report absolute/full-scale error and cancellation ratio, not only relative error near
zero.

## Current phase acceptance

- Exact ideal algebra for arbitrary real matrices, inputs, bias, and batches.
- Non-negative pre-detection powers and transmissions bounded by one.
- Exact removal of noiseless common-mode irradiance.
- Separate fixed mismatch and dynamic detector-noise semantics.
- No sunlight animation, hardware energy claim, resolved-spectrum filter selection, or
  commercial-performance claim at this stage.

## Experimental precedent

- Bocker, *Matrix Multiplication Using Incoherent Optical Techniques*, Applied Optics
  13, 1670–1676 (1974), DOI: 10.1364/AO.13.001670.
- Song et al., *Low-power scalable multilayer optoelectronic neural networks enabled
  with incoherent light*, Nature Communications 15, 10692 (2024),
  DOI: 10.1038/s41467-024-55139-4.
- *Quantum-limited stochastic optical neural networks operating at a few quanta per
  activation*, Nature Communications (2024), DOI: 10.1038/s41467-024-55220-y.

These support incoherent intensity-times-transmission dot products. They do not by
themselves validate this project's outdoor sunlight implementation.
