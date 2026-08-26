> [!IMPORTANT]
> You are viewing the experimental `experiment/solar-incoherent` branch. Its sunlight
> intensity processor uses a different physical principle from the coherent MZI backend.
> Start with the [sunlight branch overview](README.solar.md).

<div align="center">

# photonic-mzi

[English](https://github.com/yaoniming3k/photonic-mzi/blob/main/README.md) | [简体中文](https://github.com/yaoniming3k/photonic-mzi/blob/main/README.zh-CN.md)

**An electrical-circuit-level feasibility demonstration of photonic matrix multiply-accumulate — compile any real matrix into MZI meshes and execute the linear transform through optical propagation**

[![CI](https://github.com/yaoniming3k/photonic-mzi/actions/workflows/ci.yml/badge.svg)](https://github.com/yaoniming3k/photonic-mzi/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/photonic-mzi.svg)](https://pypi.org/project/photonic-mzi/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/yaoniming3k/photonic-mzi/blob/main/LICENSE)

The project asks one central question: **can a photonic processor execute matrix
multiply-accumulate?** It uses SVD, unitary transforms, MZI interference meshes,
and optical attenuators to build a runnable, device-by-device-verifiable chain for
`y = Mx` (or batched `Y = MX`) that can be checked against NumPy. It also includes
an animation that connects each line of code with propagation through each device.

```text
input [x] -> [V^T unitary MZI mesh] -> [Sigma attenuators] -> [U unitary MZI mesh] -> detector [y]
```

</div>

---

## What the project validates

An optical circuit does not execute electronic instructions one multiply at a time.
Instead, the multiplications and sums for each output element are mapped to complex
amplitude encoding, interference, per-channel scaling, and coherent readout. The
project closes four parts of that chain:

1. **Compilation:** any real matrix can be decomposed by SVD into two orthogonal transforms and non-negative scaling, then compiled into two MZI meshes and one VOA bank.
2. **Propagation:** after encoding an input vector as coherent complex amplitudes, ideal propagation satisfies `E = (M @ x) / gain`; the same relation holds for batched inputs.
3. **Readout:** coherent detection with a local oscillator recovers signed outputs that agree with `M @ x` to floating-point precision.
4. **Sensitivity:** static phase-control offsets, per-sample phase jitter, insertion loss, VOA setting error, and equivalent readout noise can be injected and measured independently.

This establishes feasibility inside the mathematical mapping, software implementation,
and simplified linear optical circuit model. It does not by itself establish the
energy, latency, precision, scale, or manufacturability of a physical chip.

---

## Animation

<div align="center">
<img src="https://raw.githubusercontent.com/yaoniming3k/photonic-mzi/main/docs/images/demo.gif" alt="Bilingual MZI mesh photonic-computing animation" width="100%">
</div>

The **left panel shows the executing code** with the active line highlighted. The
**right panel shows what that line does inside the optical circuit**. Color encodes
phase (red at 0 degrees, cyan at 180 degrees), while bar height encodes amplitude.

```bash
python -m pip install "photonic-mzi[viz]"
python -m photonic_mzi
```

| Key | Action |
|:---:|---|
| `Space` | Pause / resume |
| `Left` `Right` | Step backward / forward |
| `,` `.` | Previous / next stage |
| `r` | Restart |

The animation has nine stages:

| Stage | Content |
|---|---|
| 0 Core question | Can a photonic processor execute `y = Mx`? |
| 1 SVD | Why `M = U · Sigma · V^T` maps a real matrix to transform-scale-transform |
| 2 Compile MZIs | Eliminate matrix elements and fill the MZI parameter table |
| 3 Inject light | Encode the input as complex amplitudes; a negative value is a pi phase shift |
| 4 V^T mesh | Propagate layer by layer and inspect interference and energy conservation |
| 5 Sigma attenuation | Apply the programmed attenuation corresponding to singular values |
| 6 U mesh | Apply the second orthogonal transform |
| 7 Detect output | Compare coherent readout against NumPy |
| 8 Non-idealities | Measure sensitivity to phase offsets, jitter, loss, VOA error, and readout noise |

<table>
<tr>
<td width="50%"><img src="https://raw.githubusercontent.com/yaoniming3k/photonic-mzi/main/docs/images/stage2-compile.png" alt="Compilation stage"><br><sub><b>Stage 2:</b> matrix elimination and MZI parameter programming</sub></td>
<td width="50%"><img src="https://raw.githubusercontent.com/yaoniming3k/photonic-mzi/main/docs/images/stage4-interference.png" alt="Interference stage"><br><sub><b>Stage 4:</b> interference and energy conservation in one MZI</sub></td>
</tr>
<tr>
<td width="50%"><img src="https://raw.githubusercontent.com/yaoniming3k/photonic-mzi/main/docs/images/stage1-svd.png" alt="SVD stage"><br><sub><b>Stage 1:</b> M = U · Sigma · V^T</sub></td>
<td width="50%"><img src="https://raw.githubusercontent.com/yaoniming3k/photonic-mzi/main/docs/images/stage8-noise.png" alt="Non-ideality stage"><br><sub><b>Stage 8:</b> NumPy, ideal photonic, and simplified non-ideal results</sub></td>
</tr>
</table>

---

## Installation

```bash
python -m pip install "photonic-mzi[viz]"
```

For the NumPy-only compute kernel without the animation:

```bash
python -m pip install photonic-mzi
```

To develop from a source checkout, editable installs require **pip 21.3 or later**
because this is a `pyproject.toml`-only package using PEP 660:

```bash
python -m pip install --upgrade pip && pip install -e ".[dev]"
```

You can also run directly from the checkout:

```bash
PYTHONPATH=src python -m photonic_mzi
```

## Usage

```python
import numpy as np
from photonic_mzi import PhotonicMatrixProcessor

M = np.random.randn(8, 5)  # rectangular matrices are supported
opu = PhotonicMatrixProcessor(M, seed=42)

x = np.random.randn(5)
y = opu.read_coherent(x)  # agrees with M @ x to about 1e-15
Y = opu.read_coherent(np.random.randn(5, 256))  # batched inputs

E = opu.optical_field(x)  # pre-detection field: (M @ x) / opu.gain when ideal
print(opu.report())
```

Add the simplified non-ideality model:

```python
from photonic_mzi import NoiseModel

noise = NoiseModel(
    fab_theta=0.02,       # static phase-control offset; calibratable in the ideal model
    drift_theta=0.005,    # independent per-sample phase jitter
    mzi_loss_db=0.2,      # insertion loss per MZI
    voa_rel_err=0.01,
    detector_snr_db=40,   # post-detection equivalent relative AWGN
)
opu = PhotonicMatrixProcessor(M, noise=noise, seed=7)
opu.read_coherent(x, ideal=False)
opu.calibrate()
opu.read_coherent(x, ideal=False)
```

The two readout methods represent different detection schemes:

| Method | Physical meaning |
|---|---|
| `read_coherent(x)` | Coherent or homodyne detection with a local oscillator; preserves the signed real component |
| `read_intensity(x)` | Square-law direct detection; returns calibrated `gain²·\|E\|²` and loses sign information |

## Examples

```bash
python examples/01_hello_photonic.py
python examples/02_noise_and_calibration.py
python examples/03_neural_layer.py
python examples/04_solar_incoherent.py
```

The third example uses a synthetic, well-separated classification problem. Its
accuracy under a selected noise model must not be generalized to other models,
datasets, or time-correlated thermal drift.

## Tests and benchmark

```bash
pytest -m "not slow"
python benchmarks/bench_decomposition.py
```

The fast suite contains 146 tests; the full suite contains 151 tests, including
all-frame glyph checks and GIF export. Coverage includes degenerate and structured
matrices, random dense and rectangular matrices, batching, energy conservation,
noise semantics, detection boundaries, calibration boundaries, and input validation.

## Experimental sunlight backend

`IncoherentSolarProcessor` is a separate experimental model, not an MZI source option.
It treats sunlight as an incoherent power carrier, uses intensity transmission for
multiplication, detector power accumulation for addition, and signed dual rails for
arbitrary real matrices. A simultaneous reference cancels only common-mode irradiance.

See [Experimental sunlight matrix multiply-accumulate](docs/solar-experiment.md) for
the API and noise semantics, and
[Incoherent sunlight MAC: theory and processor design](docs/solar-processor-design.md)
for the derivation and hardware boundaries.

## Known limitations

- The implementation uses a triangular **Reck mesh**, not a rectangular Clements mesh. Both use `N(N-1)/2` MZIs, but Clements has lower optical depth and is usually more tolerant of uniform loss.
- `fab_*` models additive phase-control offsets, not beam-splitter fabrication errors that may restrict the reachable splitting ratio.
- `drift_*` is independent per-input phase jitter; it does not model temporal or spatial correlation or thermal crosstalk.
- `detector_snr_db` is post-detection equivalent AWGN; there is no detailed shot-noise, local-oscillator, responsivity, TIA, or bandwidth model.
- `mode_mzi_count()` is a topology proxy, not end-to-end path tracing.
- Forward propagation still loops over MZIs in Python; devices in one mesh layer could be vectorized.
- Input modulators, laser power, DAC/ADC, system energy and latency, wavelength dependence, polarization, nonlinearities, and device crosstalk are not modeled.

See [Model and validation notes](https://github.com/yaoniming3k/photonic-mzi/blob/main/docs/validation.md) for the full scope.

## References

- Reck et al., *Experimental realization of any discrete unitary operator*, PRL 73, 58 (1994)
- Clements et al., *Optimal design for universal multiport interferometers*, Optica 3, 1460 (2016)
- Shen et al., *Deep learning with coherent nanophotonic circuits*, Nature Photonics 11, 441 (2017)

## License

[MIT](https://github.com/yaoniming3k/photonic-mzi/blob/main/LICENSE)
