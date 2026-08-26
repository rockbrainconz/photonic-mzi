"""
Experimental incoherent-sunlight matrix multiply-accumulate model.

实验性非相干日光矩阵乘加模型。

This module deliberately does not reuse the coherent MZI field model.  It propagates
non-negative optical powers through a passive, dual-rail intensity crossbar:

    sunlight -> input intensity -> passive fan-out -> non-negative transmission
             -> power sum

Signed inputs and weights use positive/negative rails.  A reference photodiode removes
only common-mode irradiance variation; it cannot remove spatial, spectral, or detector
mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SolarNoiseModel", "SolarPowerReadout", "IncoherentSolarProcessor"]


def _relative_gain(rng: np.random.Generator, cv: float, shape) -> np.ndarray:
    """Positive unit-mean log-normal gain with the requested coefficient of variation."""
    if cv == 0:
        return np.ones(shape, dtype=float)
    sigma = np.sqrt(np.log1p(cv * cv))
    return rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=shape)


@dataclass
class SolarNoiseModel:
    """Simplified non-idealities for the experimental sunlight backend.

    All optical powers are expressed in normalized units.  ``irradiance`` sets the
    mean power of the simultaneous reference channel.

    ``common_fluctuation``
        Per-exposure coefficient of variation of a positive, unit-mean irradiance
        multiplier.  Reference normalization cancels it in the noiseless common-mode
        limit.
    ``spatial_nonuniformity``
        Fixed coefficient of variation of the gain seen by each input channel.
    ``spectral_weight_error``
        Fixed coefficient of variation of each wavelength-integrated weight.  This is
        a lumped filtered-spectrum model, not wavelength-resolved propagation.
    ``differential_gain_error``
        Fixed coefficient of variation of the positive and negative detector arms.
    ``photons_per_unit``
        Photon-count conversion used for Poisson shot noise.  ``inf`` disables it.
    ``detector_noise`` / ``reference_noise``
        Additive Gaussian read noise in normalized power units.
    """

    irradiance: float = 1.0
    common_fluctuation: float = 0.0
    spatial_nonuniformity: float = 0.0
    spectral_weight_error: float = 0.0
    differential_gain_error: float = 0.0
    photons_per_unit: float = float("inf")
    detector_noise: float = 0.0
    reference_noise: float = 0.0

    def __post_init__(self) -> None:
        finite_positive = {"irradiance": self.irradiance}
        for name, value in finite_positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive / 必须是有限正数")

        nonnegative = {
            "common_fluctuation": self.common_fluctuation,
            "spatial_nonuniformity": self.spatial_nonuniformity,
            "spectral_weight_error": self.spectral_weight_error,
            "differential_gain_error": self.differential_gain_error,
            "detector_noise": self.detector_noise,
            "reference_noise": self.reference_noise,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative / 必须是有限非负数")

        if not (self.photons_per_unit == float("inf") or
                np.isfinite(self.photons_per_unit) and self.photons_per_unit > 0):
            raise ValueError(
                "photons_per_unit must be positive or inf / 必须是正数或 inf")


@dataclass(frozen=True)
class SolarPowerReadout:
    """Dual-rail/reference powers plus the input encoding scale.

    ``optical_powers()`` returns this structure before detector noise; ``detect()``
    returns the same structure after photon counting and read noise.  Keeping the
    encoding scale with the measurement makes hardware-like external readout decoding
    explicit and testable.  The processor separately knows the fixed passive fan-out
    fraction used by the compiled optical core.
    """

    positive: np.ndarray
    negative: np.ndarray
    reference: np.ndarray | float
    input_scale: np.ndarray | float


class IncoherentSolarProcessor:
    """Experimental signed matrix MAC using incoherent sunlight intensity.

    The optical core supports arbitrary real ``M`` through dual-rail decomposition:

    ``M = M_pos - M_neg`` and ``x = x_pos - x_neg``.

    Positive and negative output powers are accumulated separately and differenced at
    readout.  Optional ``bias`` is implemented as an additional constant optical input.
    Every input rail is uniformly fanned out across the output rows, so the summed ideal
    pre-detection output power cannot exceed the configured passive fan-out efficiency.
    Relative illumination/detector mismatch is reported against the reference and can
    perturb that normalized bound. This is an analog intensity-crossbar model, not an
    MZI or coherent-field model.

    ``input_full_scale=None`` uses per-vector automatic gain control (AGC), which is
    convenient for algebra tests but requires an externally known scale for every
    exposure.  Set a positive fixed full scale for hardware-like photon and dynamic-
    range studies; inputs outside that range are rejected instead of silently clipped.
    """

    def __init__(self, M: np.ndarray, bias: np.ndarray | None = None,
                 noise: SolarNoiseModel | None = None, seed: int | None = None, *,
                 fanout_efficiency: float = 1.0,
                 input_full_scale: float | None = None):
        raw_M = np.asarray(M)
        if np.iscomplexobj(raw_M) and np.any(np.imag(raw_M) != 0):
            raise ValueError("M must be real / M 必须是实矩阵")
        M = np.asarray(np.real(raw_M), dtype=float)
        if M.ndim != 2 or M.size == 0:
            raise ValueError("M must be a non-empty 2-D matrix / M 必须是非空二维矩阵")
        if not np.all(np.isfinite(M)):
            raise ValueError("M must contain only finite values / M 必须只包含有限数值")

        self.M = M
        self.n_out, self.n_in = M.shape
        self.noise = noise or SolarNoiseModel()
        self.rng = np.random.default_rng(seed)

        if not np.isfinite(fanout_efficiency) or not 0 < fanout_efficiency <= 1:
            raise ValueError(
                "fanout_efficiency must be in (0, 1] / 扇出效率必须在 (0, 1] 内")
        self.fanout_efficiency = float(fanout_efficiency)
        self.fanout_fraction = self.fanout_efficiency / self.n_out

        if input_full_scale is not None:
            if not np.isfinite(input_full_scale) or input_full_scale <= 0:
                raise ValueError(
                    "input_full_scale must be finite and positive / 输入满量程必须是有限正数")
            input_full_scale = float(input_full_scale)
        self.input_full_scale = input_full_scale

        if bias is None:
            self.bias = None
            self.A = M.copy()
        else:
            raw_bias = np.asarray(bias)
            if np.iscomplexobj(raw_bias) and np.any(np.imag(raw_bias) != 0):
                raise ValueError("bias must be real / bias 必须是实数")
            bias = np.asarray(np.real(raw_bias), dtype=float)
            if bias.shape != (self.n_out,) or not np.all(np.isfinite(bias)):
                raise ValueError(
                    f"bias must have shape ({self.n_out},) with finite values / "
                    f"bias 形状必须为 ({self.n_out},)")
            self.bias = bias
            self.A = np.column_stack([M, bias])

        if self.bias is not None and self.input_full_scale is not None:
            if self.input_full_scale < 1.0:
                raise ValueError(
                    "input_full_scale must be at least 1 when bias is present / "
                    "含偏置时输入满量程至少为 1")

        self.n_optical_inputs = self.A.shape[1]
        self.weight_scale = float(np.max(np.abs(self.A))) if np.any(self.A) else 1.0
        normalized = self.A / self.weight_scale
        self.weight_positive = np.maximum(normalized, 0.0)
        self.weight_negative = np.maximum(-normalized, 0.0)

        nz = self.noise
        self._channel_gain = _relative_gain(
            self.rng, nz.spatial_nonuniformity, self.n_optical_inputs)
        self._weight_gain = _relative_gain(
            self.rng, nz.spectral_weight_error, self.A.shape)
        self._arm_gain = _relative_gain(
            self.rng, nz.differential_gain_error, (self.n_out, 2))

    def _prepare_input(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        raw_x = np.asarray(x)
        if np.iscomplexobj(raw_x) and np.any(np.imag(raw_x) != 0):
            raise ValueError("x must be real / x 必须是实数")
        x = np.asarray(np.real(raw_x), dtype=float)
        if x.ndim == 1:
            if x.shape[0] != self.n_in:
                raise ValueError(f"x must have length {self.n_in} / x 长度不正确")
            squeeze = True
            X = x[:, None]
        elif x.ndim == 2:
            if x.shape[0] != self.n_in:
                raise ValueError(
                    f"x must have first dimension {self.n_in} / x 第一维不正确")
            squeeze = False
            X = x
        else:
            raise ValueError("x must have shape (n_in,) or (n_in, B) / x 形状不正确")
        if not np.all(np.isfinite(X)):
            raise ValueError("x must contain only finite values / x 必须只包含有限数值")

        if self.bias is not None:
            X = np.vstack([X, np.ones((1, X.shape[1]))])
        if self.input_full_scale is None:
            scale = np.max(np.abs(X), axis=0)
            scale = np.where(scale > 0, scale, 1.0)
        else:
            if np.any(np.abs(X) > self.input_full_scale):
                raise ValueError(
                    f"x exceeds fixed input_full_scale={self.input_full_scale:g} / "
                    "x 超出固定输入满量程")
            scale = np.full(X.shape[1], self.input_full_scale)
        return X / scale[None, :], scale, squeeze

    def _powers_2d(self, x: np.ndarray, ideal: bool) -> tuple[SolarPowerReadout, bool]:
        X, input_scale, squeeze = self._prepare_input(x)
        Xp, Xn = np.maximum(X, 0.0), np.maximum(-X, 0.0)

        if ideal:
            channel_gain = np.ones(self.n_optical_inputs)
            weight_gain = np.ones_like(self.A)
            irradiance = np.full(X.shape[1], self.noise.irradiance)
        else:
            channel_gain = self._channel_gain
            weight_gain = self._weight_gain
            fluctuation = _relative_gain(
                self.rng, self.noise.common_fluctuation, X.shape[1])
            irradiance = self.noise.irradiance * fluctuation

        Xp = Xp * channel_gain[:, None]
        Xn = Xn * channel_gain[:, None]
        # A multiplicative setting error cannot turn an attenuator into optical gain.
        Wp = np.clip(self.weight_positive * weight_gain, 0.0, 1.0)
        Wn = np.clip(self.weight_negative * weight_gain, 0.0, 1.0)

        # Uniform passive fan-out sends only eta/n_out of each input-rail power to
        # each row.  Weight masks can attenuate a branch but never create power.
        positive = (self.fanout_fraction * irradiance[None, :] *
                    (Wp @ Xp + Wn @ Xn))
        negative = (self.fanout_fraction * irradiance[None, :] *
                    (Wp @ Xn + Wn @ Xp))
        reference = irradiance.copy()
        return SolarPowerReadout(positive, negative, reference, input_scale), squeeze

    @staticmethod
    def _squeeze_readout(readout: SolarPowerReadout) -> SolarPowerReadout:
        return SolarPowerReadout(
            readout.positive[:, 0],
            readout.negative[:, 0],
            float(np.asarray(readout.reference)[0]),
            float(np.asarray(readout.input_scale)[0]),
        )

    def optical_powers(self, x: np.ndarray, ideal: bool = True) -> SolarPowerReadout:
        """Return non-negative dual-rail and reference powers before detector noise."""
        readout, squeeze = self._powers_2d(x, ideal)
        return self._squeeze_readout(readout) if squeeze else readout

    def _coerce_readout(self, readout: SolarPowerReadout, *, physical: bool
                        ) -> tuple[SolarPowerReadout, bool]:
        """Validate a vector/batch readout and convert it to the internal 2-D layout."""
        if not isinstance(readout, SolarPowerReadout):
            raise TypeError("readout must be SolarPowerReadout / 读出类型不正确")

        raw_positive = np.asarray(readout.positive)
        raw_negative = np.asarray(readout.negative)
        raw_reference = np.asarray(readout.reference)
        raw_input_scale = np.asarray(readout.input_scale)
        named_raw = {
            "positive": raw_positive,
            "negative": raw_negative,
            "reference": raw_reference,
            "input_scale": raw_input_scale,
        }
        for name, raw in named_raw.items():
            if np.iscomplexobj(raw) and np.any(np.imag(raw) != 0):
                raise ValueError(f"{name} must be real / 必须是实数")

        positive = np.asarray(np.real(raw_positive), dtype=float)
        negative = np.asarray(np.real(raw_negative), dtype=float)
        if positive.ndim not in (1, 2) or negative.ndim not in (1, 2):
            raise ValueError("power rails must be 1-D or 2-D / 功率轨必须是一维或二维")
        if positive.ndim != negative.ndim:
            raise ValueError("power rails must have matching dimensions / 功率轨维度不匹配")
        if positive.ndim == 1:
            squeeze = True
            positive = positive[:, None]
            negative = negative[:, None]
        elif positive.ndim == 2:
            squeeze = False
        if positive.shape != negative.shape or positive.shape[0] != self.n_out:
            raise ValueError(
                f"power rails must share shape ({self.n_out}, B) / 功率轨形状不匹配")

        batch = positive.shape[1]
        reference = np.asarray(np.real(raw_reference), dtype=float)
        input_scale = np.asarray(np.real(raw_input_scale), dtype=float)
        if reference.ndim == 0:
            reference = reference.reshape(1)
        if input_scale.ndim == 0:
            input_scale = input_scale.reshape(1)
        if reference.shape != (batch,) or input_scale.shape != (batch,):
            raise ValueError(
                f"reference and input_scale must have shape ({batch},) / "
                "参考与输入标度形状不匹配")
        arrays = (positive, negative, reference, input_scale)
        if not all(np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("readout must contain only finite values / 读出必须为有限数值")
        if np.any(input_scale <= 0):
            raise ValueError("input_scale must be positive / 输入标度必须为正")
        if physical and (np.any(positive < 0) or np.any(negative < 0) or
                         np.any(reference < 0)):
            raise ValueError("pre-detection optical powers must be non-negative / "
                             "探测前光功率必须非负")
        return SolarPowerReadout(positive, negative, reference, input_scale), squeeze

    def _detect_channel(self, power: np.ndarray, gaussian_noise: float) -> np.ndarray:
        observed = power.copy()
        if np.isfinite(self.noise.photons_per_unit):
            counts = self.rng.poisson(
                np.maximum(observed, 0.0) * self.noise.photons_per_unit)
            observed = counts / self.noise.photons_per_unit
        if gaussian_noise:
            observed += self.rng.normal(0.0, gaussian_noise, observed.shape)
        return observed

    def detect(self, powers: SolarPowerReadout, ideal: bool = True) -> SolarPowerReadout:
        """Apply detector-arm gain, photon-counting, and read/reference noise.

        This is the explicit boundary between passive optical power propagation and
        photodetection.  Additive read noise is allowed to make a background-subtracted
        observed value negative even though the incoming optical power is non-negative.
        """
        powers, squeeze = self._coerce_readout(powers, physical=True)
        if ideal:
            observed = SolarPowerReadout(
                powers.positive.copy(), powers.negative.copy(),
                powers.reference.copy(), powers.input_scale.copy())
        else:
            positive = powers.positive * self._arm_gain[:, [0]]
            negative = powers.negative * self._arm_gain[:, [1]]
            observed = SolarPowerReadout(
                self._detect_channel(positive, self.noise.detector_noise),
                self._detect_channel(negative, self.noise.detector_noise),
                self._detect_channel(powers.reference, self.noise.reference_noise),
                powers.input_scale.copy(),
            )
        return self._squeeze_readout(observed) if squeeze else observed

    def decode(self, readout: SolarPowerReadout,
               normalize: bool = True) -> np.ndarray:
        """Difference dual rails, restore known scales, and optionally normalize.

        This method can decode either simulated detector output or externally measured
        hardware powers packaged as :class:`SolarPowerReadout`.
        """
        readout, squeeze = self._coerce_readout(readout, physical=False)
        decoded = ((readout.positive - readout.negative) * self.weight_scale *
                   readout.input_scale[None, :] / self.fanout_fraction)
        if normalize:
            if np.any(readout.reference <= 0):
                raise RuntimeError(
                    "reference detector is non-positive; cannot normalize / 参考探测器非正")
            decoded = decoded / readout.reference[None, :]
        return decoded[:, 0] if squeeze else decoded

    def read(self, x: np.ndarray, ideal: bool = True,
             normalize: bool = True) -> np.ndarray:
        """Detect and decode ``M @ x + bias``.

        With ``normalize=True`` the simultaneous reference channel removes common
        irradiance.  Fixed spatial/spectral imbalance and independent detector noise
        remain.  ``normalize=False`` intentionally leaves the irradiance multiplier in
        the result.
        """
        powers = self.optical_powers(x, ideal=ideal)
        observed = self.detect(powers, ideal=ideal)
        return self.decode(observed, normalize=normalize)

    __call__ = read

    def report(self) -> str:
        """Human-readable scope and configuration summary."""
        return "\n".join([
            "Backend / 后端          : incoherent sunlight intensity / 非相干日光强度",
            f"Matrix / 矩阵尺寸       : {self.n_out} x {self.n_in}",
            f"Optical inputs / 光输入 : {self.n_optical_inputs}"
            f"{' (includes bias / 含偏置)' if self.bias is not None else ''}",
            "Dual rail / 正负双轨    : implemented / 已实现",
            f"Weight scale / 权重标度 : {self.weight_scale:.6g}",
            f"Fan-out / 被动扇出      : {self.n_out}-way uniform, "
            f"eta={self.fanout_efficiency:.6g}, branch={self.fanout_fraction:.6g}",
            "Input scale / 输入标度  : " + (
                "per-vector AGC / 逐向量自动增益"
                if self.input_full_scale is None else
                f"fixed full scale {self.input_full_scale:.6g} / 固定满量程"),
            f"Irradiance / 日光标度   : {self.noise.irradiance:.6g}",
            "Reference / 参考归一化  : simultaneous common-mode only / 仅同时公共模",
            "Spectrum / 光谱模型     : lumped effective error, not wavelength resolved / "
            "等效误差，非逐波长",
            "Electrical / 用电边界  : modulators, TIA, ADC and control excluded / "
            "不含调制器、TIA、ADC 与控制",
        ])

    def __repr__(self) -> str:
        return (f"IncoherentSolarProcessor({self.n_out}x{self.n_in}, "
                f"bias={self.bias is not None}, fanout={self.fanout_efficiency:g})")
