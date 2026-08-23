# Changelog

[English](CHANGELOG.md) | [简体中文](CHANGELOG.zh-CN.md)

This file records notable changes following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Dynamic phase jitter is sampled independently for every input in a batch.
- `optical_field()` returns only the pre-detection field; equivalent noise is added in
  the correct coherent or direct-detection domain.
- Static offsets, dynamic jitter, and phase calibration apply to output phase screens;
  VOA error is a fixed per-channel setting offset.
- Input shapes, finite values, real-matrix boundaries, noise parameters, and unitary
  preconditions are validated explicitly.
- Repository examples always import the checkout's `src/` tree.

### Documentation

- Made English the default language for repository metadata, unsuffixed documentation,
  package metadata, CLI output, examples, CI labels, and generated visual assets; Chinese
  translations now use the `.zh-CN` suffix.
- Added Chinese and English versions of the README, model notes, contribution guide,
  changelog, public API text, examples, and animation.
- Defined the core goal as circuit-level validation of photonic matrix
  multiply-accumulate, with compilation, propagation, readout, and sensitivity stages.
- Clarified that `fab_*` is a controllable phase offset and `drift_*` is an i.i.d.
  sensitivity model.
- Removed unsupported system-level energy, latency, full detector-noise, calibration,
  and fixed neural-network workload claims.
- Corrected path terminology to mode participation and documented the compatibility API.
- Clarified SVD non-uniqueness, reflections in orthogonal transforms, coherent-source
  requirements, and disjoint-mode parallelism.

## [1.0.0] - 2026-08-23

Initial release with MZI mesh decomposition, optical propagation, simplified
non-idealities, and a bilingual teaching animation.

### Core implementation

- Uses `arctan2` to cover the `x -> 0` and `y -> 0` elimination limits continuously,
  without an absolute threshold.
- Wraps `phi` to the physically equivalent range `(-pi, pi]`.
- Includes regression coverage for structured, sparse, rank-deficient, and very small
  scale matrices.

### Added

- Rectangular matrices through zero-padding to `N = max(n_out, n_in)`.
- Batched inputs with shape `(n_in, B)`.
- `NoiseModel` with static phase-control offsets, per-sample phase jitter, insertion
  loss, VOA setting error, and equivalent detection SNR.
- `calibrate()` and `reset_calibration()` for the ideal characterization model.
- Separate coherent and direct-detection readout APIs.
- Passive singular-value normalization with electrical-domain gain restoration.
- Chip reports, compilation round-trip checks, mesh depth, and mode participation metrics.
- A 494-frame, nine-stage teaching animation with pause and stepping controls.

### Changed

- In-place two-row updates provide `O(N^3)` compilation and `O(N^2)` forward propagation.
- Randomness uses an independent `numpy.random.Generator`.
- The package is split into linear-algebra, processor-model, and optional visualization layers.

### Known limitations

- Reck topology only; Clements topology is not implemented.
- Calibration does not compensate insertion-loss channel mismatch.
- Forward propagation still loops over MZIs in Python.
- Wavelength dependence, polarization, nonlinearities, and device crosstalk are not modeled.

[1.0.0]: https://github.com/rockbrainconz/photonic-mzi/releases/tag/v1.0.0
