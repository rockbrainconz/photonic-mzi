# photonic-mzi: experimental sunlight matrix MAC

[English](README.md) | [简体中文](README.zh-CN.md) |
[Main coherent-MZI branch](https://github.com/yaoniming3k/photonic-mzi/tree/main)

> **Status: experimental.** This is the default README for
> `experiment/solar-incoherent`. The branch studies incoherent sunlight intensity
> computing; it is not a source replacement for the coherent MZI processor on `main`.

## Purpose

The target is a real matrix multiply-accumulate:

```text
y = Mx + b
```

Its physical path is deliberately different from the main backend:

```text
sunlight -> homogenization/filtering -> signed input intensity rails
         -> passive uniform fan-out -> non-negative weight transmission
         -> detector power summation
         -> differential readout -> reference normalization -> y
```

Multiplication is performed by intensity transmission and addition by spatial power
accumulation. No controlled complex field or MZI phase interference is required.

## Difference from `main`

| Property | Main `PhotonicMatrixProcessor` | This branch `IncoherentSolarProcessor` |
|---|---|---|
| Optical variable | Coherent complex field | Non-negative optical power |
| Typical source | Laser | Sunlight, LED, or solar simulator |
| MAC mechanism | MZI interference, unitary meshes, VOA | Transmission, fan-out, detector summation |
| Signed values | π phase difference | Positive/negative dual rails |
| Readout | Coherent/homodyne with local oscillator | Paired direct detectors and differencing |
| Source variation | Not applicable | Reference cancels common mode only |
| Main class | `PhotonicMatrixProcessor` | `IncoherentSolarProcessor` |

The two backends share the high-level matrix-MAC objective, not propagation equations
or device topology. The original MZI project remains documented on the
[main branch](https://github.com/yaoniming3k/photonic-mzi/blob/main/README.md).

## Core algebra

Augment bias as `A=[M b]`, `z=[x;1]`, normalize to `W` and `u`, then split
`W=W+-W-` and `u=u+-u-`. For an `m`-way passive fan-out with total efficiency
`eta_f`, each row receives `f=eta_f/m` and the optical rails are

```text
P+ = C(t) f [W+u+ + W-u-]
P- = C(t) f [W+u- + W-u+].
```

Thus `P+-P-=C(t)fWu`. The decoder restores the known `1/f`, while the attenuated
detector powers still determine shot noise. A simultaneous `Pref=C(t)` removes only
the shared irradiance scalar. It cannot remove local shadows, channel nonuniformity,
spectral mismatch, differential-arm mismatch, or independent detector noise.

See [Theory and processor design](docs/solar-processor-design.md) for the complete
derivation and hardware conditions.

## API

The implementation exposes physical boundaries explicitly:

```python
from photonic_mzi import IncoherentSolarProcessor, SolarNoiseModel

solar = IncoherentSolarProcessor(
    M,
    bias=b,
    fanout_efficiency=0.8,
    input_full_scale=1.0,
    noise=SolarNoiseModel(...),
    seed=7,
)

powers = solar.optical_powers(x, ideal=False)
observed = solar.detect(powers, ideal=False)
y = solar.decode(observed, normalize=True)
```

`read()` composes those stages. Measured hardware powers can be packaged as
`SolarPowerReadout` and passed to the same decoder.

`input_full_scale` selects a fixed characterized hardware range and rejects overflow.
Leaving it as `None` enables explicit per-vector AGC; that mode carries its scale in
the readout and should not be treated as free small-signal optical gain.

Implementation: [src/photonic_mzi/solar.py](src/photonic_mzi/solar.py)

## Modeled non-idealities

- common irradiance fluctuation per exposure;
- fixed spatial gain mismatch per input channel;
- fixed lumped wavelength-integrated error per weight;
- fixed positive/negative detector-arm mismatch;
- Poisson photon-counting noise;
- additive detector and reference read noise.

These support algebra and sensitivity experiments. They are not a complete outdoor
device model.

The compiled core also enforces a uniform passive fan-out budget: summed output-rail
power cannot exceed `fanout_efficiency` times the available encoded input-rail power.
Mutual coherence, wavelength-resolved drift, thermal-light excess noise, and absolute
detector units remain deferred to measured-device models.

## Electrical boundary

Fixed lenses, filters, diffusers, and masks can be passive. Programmable input/weight
modulators, TIA, differential/reference electronics, ADC, control, and stabilization
normally consume electricity. The optical MAC core can be passive; a programmable
numerical system is not a zero-electricity computer.

The current model uses normalized power and does not report watts, exposure time, or
J/MAC.

## Run and verify

```bash
python examples/04_solar_incoherent.py
python -m pytest tests/test_solar.py -q
python -m pytest -m "not slow" -q
```

Current status: 34 focused sunlight tests, 162 fast tests with 5 pre-existing animation
tests deselected, and a clean Ruff check.

## Documentation

- [API, noise semantics, and model boundary](docs/solar-experiment.md)
- [Full theory and processor design](docs/solar-processor-design.md)
- [Runnable example](examples/04_solar_incoherent.py)
- [Focused tests](tests/test_solar.py)
- [Main coherent-MZI README](https://github.com/yaoniming3k/photonic-mzi/blob/main/README.md)

## Explicitly deferred

- No sunlight animation or GIF.
- No sunlight mode inside the coherent MZI field model.
- No real filter selection without wavelength-resolved device data.
- No assumption that the word “sunlight” alone guarantees zero mutual-coherence terms;
  a bench must verify residual fringe visibility.
- No zero-electricity, outdoor-ENOB, TOPS, or energy-advantage claim.
- No substitution of ideal floating-point agreement for hardware evidence.

The next physical milestone is a controlled incoherent-source or solar-simulator bench,
followed by outdoor sunlight only after dual-rail balance, reference normalization,
linearity, and calibration are demonstrated.
