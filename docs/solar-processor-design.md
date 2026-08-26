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

With `alpha=max|A_ij|` and a declared input full scale `beta`, define `W=A/alpha`,
`u=z/beta`, then split `W=W+-W-` and `u=u+-u-` into non-negative rails. Let a
uniform passive fan-out have total efficiency `eta_f` and `m` output rows, giving
each row the branch fraction `f=eta_f/m`. The two optical accumulators are

```text
P+ = C(t) f [W+u+ + W-u-]
P- = C(t) f [W+u- + W-u+].
```

Therefore

```text
P+ - P- = C(t) f Az/(alpha beta).
```

If a simultaneous reference measures `Pref=C(t)`, decoding gives

```text
y = alpha beta (P+ - P-) / (f Pref) = Mx + b.
```

The reference cancels only a shared scalar. Per-channel illumination, spectral weight
drift, detector mismatch, local shadows, shot noise, and read noise remain.

### Fixed full scale versus AGC

A physical input encoder normally uses one fixed `beta_FS` for a characterized
operating range. Inputs with `max|z_j| > beta_FS` must saturate, clip, or be rejected,
and small inputs use proportionally fewer photons. This is the mode represented by
`input_full_scale=beta_FS`.

`input_full_scale=None` instead chooses `beta=max|z_j|` independently for every
vector. That is an explicit automatic-gain-control (AGC) mode: it requires measuring or
knowing the vector scale, reprogramming the input encoder, and carrying `beta` to the
decoder for every exposure. It is useful for algebra tests but must not be used to infer
small-signal photon efficiency without including that control path. With a bias channel,
a fixed full scale must be at least one because the augmented input contains `z0=1`.

## Passive realizability and power conservation

Bounding every mask transmission by one is necessary but not sufficient. A passive
fan-out cannot copy one input rail at full power into every output row. For general
branch fractions `s_ij`,

```text
s_ij >= 0,   sum_i s_ij <= eta_j <= 1.
```

The implemented uniform architecture uses `s_ij=f=eta_f/m`. Therefore, before fixed
gain mismatch and detector noise,

```text
sum_i (P_i+ + P_i-) <= C(t) eta_f sum_j |u_j|.
```

Weight masks only discard branch power. The decoder restores the known `1/f` scale,
which recovers the mathematical result but not the lost photons; shot-noise performance
must use the attenuated detector powers. A non-uniform splitter may improve utilization
for a fixed matrix, but then its measured `s_ij` belongs in the compiled effective
matrix. An architecture that illuminates `m` independently modulated copies instead
of splitting one input is also possible, but its aperture, incident solar power, and
input-control resources scale with those copies and cannot be called free fan-out.
`eta_f` is the compute-path efficiency after the reference tap; the separately
normalized reference power is not included in the compute-output bound, and its tap
loss must be included when converting incident sunlight to watts.

## Condition for incoherent power addition

Calling the source incoherent does not by itself remove every interference term. For
fields coupled from inputs `j` and `k`, a detector generally measures

```text
P_i = sum_j |h_ij|^2 I_j
    + sum_(j != k) 2 Re[h_ij h_ik* Gamma_jk],
```

where `Gamma_jk` is the mutual-coherence function after collection, filtering,
splitting, and propagation. The intensity-crossbar equation is valid only when the
cross terms are negligible over the detector area and integration time. Sufficient
implementations include mutually incoherent source pixels, non-overlapping spots whose
photocurrents sum inside one detector, path delays well beyond the filtered coherence
time, or enough spatial/temporal modes to average residual fringes.

Broadband sunlight has a short temporal coherence time, but narrow filters lengthen it,
and split copies of one spatial mode can remain mutually coherent. The bench must
measure fringe visibility or output variance while sweeping path delay and aperture;
`Gamma_jk ~= 0` is an acceptance condition, not an assumption inferred from the word
“sunlight”.

## Broadband condition

A real detector measures

```text
P_i(t) = sum_j integral S_j(lambda,t) T_ij(lambda,t) R_i(lambda,t) u_j d_lambda.
```

A fixed matrix exists only when this wavelength-dependent product separates into a
common time-varying scalar and stable channel responses. Define the measured effective
matrix

```text
M_eff(t)[i,j] = integral S_j(lambda,t) T_ij(lambda,t) R_i(lambda,t) d lambda.
```

One scalar reference is sufficient only when `M_eff(t)=c(t) M0`. A useful
non-separability metric is

```text
epsilon_spec(t) = min_c ||M_eff(t) - c M0||_F / ||M0||_F.
```

Otherwise the hardware computes `M_eff(t)x`; scalar normalization cannot recover
`M0 x`. A row-, channel-, polarization-, or wavelength-dependent reference may reduce
specific structured errors, but it increases detector and calibration resources.

`spectral_weight_error` is currently a fixed lumped error after wavelength integration.
It is not a resolved solar-spectrum, filter, SLM-dispersion, polarization, or
responsivity model, and it cannot predict `epsilon_spec(t)` under changing weather or
solar elevation.

## Reference-ratio statistics

The exact cancellation above assumes noiseless `D=P+-P-` and `R=Pref`. With
zero-mean detector/reference errors `n_D` and `n_R`,

```text
y_hat = (alpha beta / f) (D + n_D) / (C + n_R).
```

For a nonzero output and small noise, the first-order relative error is
`n_D/D - n_R/C`. At second order the reference produces an approximate relative bias

```text
Var(n_R)/C^2 - Cov(n_D,n_R)/(D C).
```

Consequently, merely checking that the noisy reference is positive is not a precision
guarantee. Measurements must dark-subtract all detectors, keep the reference away from
zero and saturation, report its SNR, and match its spectrum, polarization, aperture,
and time window to the compute paths. Near a true zero output, use absolute or
full-scale error rather than relative error.

## Hardware chain

```text
sunlight -> collection -> bandpass -> homogenizer
         -> input rails -> fan-out -> weight rails -> row summation
         -> paired photodiodes + simultaneous reference
         -> differential readout -> reference normalization -> scale restoration
```

Each logical `(i,j)` needs the four algebraic products `W+u+`, `W-u-`, `W+u-`,
and `W-u+`.
The first two feed the positive detector and the other two feed the negative detector.
All rails and the reference must be exposed simultaneously. The uniform splitter sends
only `eta_f/m` of each input-rail power toward a row before weight attenuation.

Three implementation levels are meaningful:

1. Fixed passive mask and optical input: the optical core can be externally unpowered.
2. Fixed matrix and programmable input: the input modulator and readout use electricity.
3. Fully programmable input and weight SLMs: drivers, TIA, ADC, control, and often thermal
   stabilization use electricity.

The current normalized model contains no watts, exposure time, or peripheral energy and
cannot report J/MAC.

## Calibration sequence

1. Dark offsets, detector noise, and reference-ratio bias.
2. Source/rail/reference linearity, fixed input full scale, and saturation limits.
3. Fan-out fractions, total power conservation, and path insertion loss.
4. Residual mutual coherence by path-delay, aperture, and filter-band sweeps.
5. Per-input flat-field gain and local-shadow sensitivity.
6. Per-weight command-to-transmission lookup table.
7. Positive/negative detector-arm balance and zero-output cancellation ratio.
8. Re-characterization under several spectra, polarizations, and filter bands.
9. Basis-vector measurement of `M_eff`, followed by held-out random vectors.
10. Outdoor repetition across irradiance, cloud, solar elevation, and temperature.

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
`SolarPowerReadout` and passed to the same decoder, provided they use the processor's
compiled fan-out fraction and input-scale convention.

## Precision boundary

For spectral detector power `P_i(lambda)`, integration time `tau`, and detector
quantum efficiency `eta_q(lambda)`, the mean detected photon count is

```text
N_i = tau integral eta_q(lambda) P_i(lambda) lambda/(h c) d lambda.
```

Independent Poisson rails give `Var(N+ - N-)=N+ + N-`. Shot-noise-only SNR therefore
scales as `sqrt(N_photons)`, so an optimistic `b`-bit relative resolution needs order
`2^(2b)` detected photons. Signed cancellation makes the bound worse when `P+` and
`P-` are both large but their difference is small, and fan-out reduces the photons
available at each row.

`photons_per_unit` implements only this normalized Poisson baseline. It does not map
to watts without `P(lambda)`, `tau`, collection area, optical efficiency, and quantum
efficiency. It also omits dark counts, detector NEP/bandwidth, saturation, and thermal-
light bunching or other excess source noise. Experiments must report absolute/full-scale
error and cancellation ratio, not only relative error near zero.

## Quantitative acceptance metrics

Numerical thresholds depend on the target application and must be declared before data
collection, but the measured quantities are not optional:

| Boundary | Metric |
|---|---|
| Passive core | `sum(P+ + P-) / [C eta_f sum|u|]` and insertion loss |
| Incoherent addition | worst-case residual fringe visibility / mutual coherence |
| Matrix fidelity | `||M_meas-c M0||_F / ||M0||_F` on basis vectors |
| Spectral stability | distribution of `epsilon_spec(t)` across test spectra |
| Common-mode rejection | raw versus normalized output variation under irradiance sweeps |
| Signed cancellation | `|P+-P-|/(P++P-)` for commanded zero outputs |
| Linearity | full-scale residual over the declared input range |
| Precision | full-scale RMSE or ENOB versus irradiance and integration time |
| Repeatability | drift versus time, temperature, solar elevation, and cloud condition |

Held-out random vectors test prediction after calibration; they must not be reused to
fit `M_eff` or select correction parameters.

## Current phase acceptance

- Exact ideal algebra for arbitrary real matrices, inputs, bias, and batches.
- Non-negative pre-detection powers, mask transmissions bounded by one, and explicit
  passive fan-out power conservation.
- Fixed-full-scale hardware mode plus explicit per-vector AGC mode.
- Exact removal of noiseless common-mode irradiance.
- Separate fixed mismatch and dynamic detector-noise semantics.
- Explicitly deferred mutual-coherence, wavelength-resolved, thermal-light, and
  physical-unit detector models.
- No sunlight animation, hardware energy claim, resolved-spectrum filter selection, or
  commercial-performance claim at this stage.

## Experimental precedent

- Bocker, *Matrix Multiplication Using Incoherent Optical Techniques*, Applied Optics
  13, 1670–1676 (1974), DOI: 10.1364/AO.13.001670.
- Song et al., *Low-power scalable multilayer optoelectronic neural networks enabled
  with incoherent light*, Nature Communications 15, 10692 (2024),
  DOI: 10.1038/s41467-024-55139-4.
- Ma et al., *Quantum-limited stochastic optical neural networks operating at a few
  quanta per activation*, Nature Communications 16, 359 (2025),
  DOI: 10.1038/s41467-024-55220-y.
- Ricketti et al., *The coherence time of sunlight in the context of natural and
  artificial light-harvesting*, Scientific Reports 12, 5438 (2022),
  DOI: 10.1038/s41598-022-08693-0.
- Mashaal et al., *First direct measurement of the spatial coherence of sunlight*,
  Optics Letters 37, 3516–3518 (2012), DOI: 10.1364/OL.37.003516.
- Wang et al., *An optical neural network using less than 1 photon per multiplication*,
  Nature Communications 13, 123 (2022), DOI: 10.1038/s41467-021-27774-8.

These support incoherent intensity-times-transmission dot products, fan-in/fan-out
architectures, coherence analysis, and photon-limited readout. They do not by themselves
validate this project's outdoor sunlight implementation.
