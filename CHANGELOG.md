# Changelog

[English](CHANGELOG.md) | [简体中文](CHANGELOG.zh-CN.md)

This file records notable changes following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Added the isolated experimental `IncoherentSolarProcessor` backend for signed
  dual-rail intensity MAC with simultaneous reference normalization.
- Added separate spatial, wavelength-integrated weight, differential-arm, photon-count,
  detector, and reference non-idealities for the sunlight experiment.
- Added a bilingual model-boundary document, runnable example, and 34 focused tests.
- Documented the full signed derivation, common-mode normalization theorem, broadband
  failure conditions, passive/programmed hardware variants, electrical boundary,
  calibration sequence, and experimental acceptance criteria.
- Split the sunlight API into explicit optical-power, detection, and decoding stages so
  measured hardware powers can reuse the same decoder.
- Added an explicit uniform passive fan-out efficiency and decoder scale restoration,
  with a tested total-power conservation bound instead of implicit full-power copying.
- Added fixed-input-full-scale hardware mode alongside the explicitly reported
  per-vector AGC mode, including overflow validation.
- Added symmetric real/shape validation for externally supplied dual-rail readouts.
- Moved fixed differential detector-arm gain from passive `optical_powers()` into the
  explicit `detect()` boundary.
- Extended the theory with passive realizability, mutual-coherence conditions,
  reference-ratio bias, a spectral non-separability metric, physical photon-count
  equations, and quantitative experimental acceptance metrics.
- Corrected the publication metadata for the 2025 quantum-limited stochastic optical
  neural-network reference and added primary coherence/fan-out references.
- Made the branch-root English and Chinese READMEs sunlight-specific; links point back
  to the unchanged coherent-MZI READMEs on `main`.

## [1.0.1] - 2026-08-24

### Fixed

- Escaped the intensity-expression separators so the readout comparison table renders
  completely on PyPI and GitHub.

## [1.0.0] - 2026-08-24

Initial release with MZI mesh decomposition, optical propagation, simplified
non-idealities, and a bilingual teaching animation.

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

- Updated canonical repository, CI badge, issue tracker, and release links after the
  transfer to `yaoniming3k/photonic-mzi`.
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
- Added a PyPI Trusted Publishing workflow with release-tag/version validation and
  isolated build and publish jobs.
- Made README links, images, and installation instructions render correctly on PyPI.

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

[Unreleased]: https://github.com/yaoniming3k/photonic-mzi/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/yaoniming3k/photonic-mzi/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/yaoniming3k/photonic-mzi/releases/tag/v1.0.0
