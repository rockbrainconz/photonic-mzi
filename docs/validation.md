# Model and Validation Notes

[English](validation.md) | [简体中文](validation.zh-CN.md)

## 1. Validation target

This project validates the **circuit-level feasibility** of photonic matrix
multiply-accumulate. The evidence chain is:

1. compile a target real matrix into realizable MZI and VOA parameters;
2. encode inputs as coherent complex optical amplitudes and propagate them;
3. recover signed results through coherent detection and compare them with `M @ x` or `M @ X`;
4. inject simplified non-idealities independently and quantify sensitivity.

“Feasible” is limited to the mathematical mapping, numerical implementation, and
simplified linear optical circuit model. It does not establish fabrication, packaging,
energy, latency, bandwidth, precision, or economic feasibility for a physical chip.

## 2. Mathematical mapping

Any real matrix has an SVD:

```text
M = U · Σ · Vᵀ
```

`U` and `Vᵀ` are orthogonal transforms that can be embedded in unitary transforms and
compiled into triangular Reck MZI meshes. `Σ` is non-negative diagonal scaling and maps
to a VOA bank. Because a passive VOA can only attenuate, singular values are normalized
as `S_phys = S / S_max`, and the overall gain `S_max` is restored in the electrical
readout domain.

Unitary compilation uses column elimination. For a target pair `(x, y)`, the parameters
are:

```python
phi = np.angle(y) - np.angle(x) - np.pi
theta = np.arctan2(np.abs(x), np.abs(y))
```

`arctan2` continuously covers the `y -> 0` and `x -> 0` limits without an absolute
threshold. Under this project's transfer-matrix convention, `theta=0, phi=0` is a swap,
while the identity is `theta=pi/2, phi=pi`.

## 3. Optical propagation and readout

The input is encoded into complex amplitudes split from one coherent source. The two
lossless unitary meshes redistribute energy among modes; the VOA bank applies the
singular-value scaling. In the ideal model, the pre-detection field satisfies:

```text
E = (M @ x) / gain
```

`read_coherent()` represents coherent or homodyne detection with a local oscillator and
recovers a signed real component. `read_intensity()` represents square-law direct
detection and returns calibrated `gain²·|E|²`, which loses sign information.

## 4. Simplified non-idealities

`NoiseModel` deliberately separates different statistical and physical meanings:

| Parameter | Model meaning | Calibration |
|---|---|---|
| `fab_theta`, `fab_phi` | Fixed additive phase-control offsets sampled once at construction | Cancelled by the ideal characterization model |
| `drift_theta`, `drift_phi` | Independent per-input i.i.d. phase jitter | Not removable by a static table |
| `mzi_loss_db` | Uniform insertion loss per MZI | Not handled by `calibrate()` |
| `voa_rel_err` | Fixed per-channel relative VOA setting error | Not handled by `calibrate()` |
| `detector_snr_db`, `detector_noise_floor` | Post-detection equivalent AWGN | Not handled by `calibrate()` |

These parameters support sensitivity analysis; they are not a complete device model.
In particular, `fab_*` does not model beam-splitter errors that restrict reachable
splitting ratios, and `drift_*` does not model slow temporal correlation, spatial
correlation, or thermal crosstalk.

## 5. Topology and complexity

The implementation uses a triangular Reck mesh. Each `N x N` unitary requires
`N(N-1)/2` MZIs and has worst-case optical depth `2N-3`. A rectangular Clements mesh
uses the same MZI count with depth `N` and is generally more tolerant of uniform loss;
it is not implemented yet.

In-place two-row updates give `O(N³)` compilation and `O(N²)` single-input forward
propagation. Batched inputs use shape `(n_in, B)`. MZIs in one mesh layer occupy disjoint
modes and can act in parallel physically, although the current Python model still loops
over devices.

## 6. Executable validation

```bash
pytest -m "not slow"
pytest
python benchmarks/bench_decomposition.py
```

The fast suite contains 128 tests and the full suite contains 133. Coverage includes:

- single-MZI unitarity, swap behavior, and degenerate elimination limits;
- random dense, structured, rank-deficient, rectangular, and batched matrices;
- compilation round trips, ideal propagation accuracy, and energy conservation;
- static offsets, dynamic jitter, insertion loss, VOA errors, and readout-noise semantics;
- coherent/direct detection boundaries, calibration boundaries, and input validation;
- all-frame animation glyph checks, field consistency, interaction, and GIF export.

## 7. System metrics not validated

The model does not include Maxwell/FDTD fields, wavelength-dependent couplers,
polarization, nonlinearities, thermal crosstalk, lasers, modulators, DAC/ADC, TIA,
shot noise, packaging, or control systems. It therefore cannot quantitatively predict
system TOPS, energy per operation, end-to-end latency, physical ENOB, chip area, yield,
or commercial product performance.
