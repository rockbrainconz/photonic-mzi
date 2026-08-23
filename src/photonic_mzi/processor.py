"""
Processor layer: SVD compilation, forward optical propagation, and non-idealities.
处理器层：SVD 编译、正向光学仿真、物理非理想性模型。

    M  = U @ diag(S) @ Vt                       (SVD)
    U, Vt  -> triangular Reck MZI meshes / Reck 三角 MZI 网格
    diag(S) -> variable optical attenuator bank / 可变光衰减器阵列

    [x] -> [V^T mesh] -> [Sigma VOA] -> [U mesh] -> [y]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mesh import MZI, apply_T_dagger_left, decompose_unitary, recompose_unitary

__all__ = ["NoiseModel", "PhotonicMatrixProcessor"]


# =============================================================================
# 物理非理想性模型
# =============================================================================
@dataclass
class NoiseModel:
    """
    Simplified physical non-ideality model. / 简化物理非理想性模型。

    The model separates fixed phase-control offsets, independent per-sample phase
    jitter, insertion loss, fixed VOA setting error, and post-detection equivalent
    noise. Combining them into one ``noise_std`` would erase their distinct physical
    and statistical meanings. All angular quantities are 1-sigma values in radians.

    ``fab_*``
        Fixed phase-control offsets, sampled once and exactly cancellable by the ideal
        characterization model. These do not represent beam-splitter fabrication errors.
        静态**相移控制偏置**。它在处理器构造时采样一次，并可由理想化的表征流程
        精确抵消。它不代表分束器制造偏差；后者可能限制 MZI 的可达分光比，本模型
        尚未描述，不能用 :meth:`PhotonicMatrixProcessor.calibrate` 简单消除。
    ``drift_*``
        Independent per-input phase jitter for sensitivity analysis, not time- or
        space-correlated physical thermal drift.
        每个输入样本独立采样的动态相位抖动。它是用于敏感度分析的 i.i.d. 简化模型，
        不等同于具有慢时间尺度和空间相关性的真实热漂移。
    ``mzi_loss_db``
        Insertion loss per MZI. Unequal Reck depth creates deterministic mismatch.
        每台 MZI 的插入损耗。Reck 三角网格路径长度不均匀，它会让不同波导衰减不同，
        形成确定性的通道失配 —— 真实芯片的主要误差源之一。
    ``voa_rel_err``
        Fixed relative VOA setting error, sampled once per channel.
        衰减器的固定相对设定误差，构造处理器时每通道采样一次。
    ``detector_snr_db``
        Post-detection equivalent RMS SNR; not a separate model of shot noise, TIA,
        electrical bandwidth, or a coherent local oscillator.
        探测后读出量的等效 RMS 信噪比。它只是相对 AWGN 参数，不声称分别建模
        散粒噪声、TIA、电带宽或相干本振。
    ``detector_noise_floor``
        Absolute post-detection Gaussian noise floor in output units.
        探测后、以返回值单位表示的绝对高斯噪声底。相对噪声与它按方差相加。
    """

    fab_theta: float = 0.0
    fab_phi: float = 0.0
    drift_theta: float = 0.0
    drift_phi: float = 0.0
    mzi_loss_db: float = 0.0
    voa_rel_err: float = 0.0
    detector_snr_db: float = float("inf")
    detector_noise_floor: float = 0.0

    def __post_init__(self) -> None:
        nonnegative = {
            "fab_theta": self.fab_theta,
            "fab_phi": self.fab_phi,
            "drift_theta": self.drift_theta,
            "drift_phi": self.drift_phi,
            "mzi_loss_db": self.mzi_loss_db,
            "voa_rel_err": self.voa_rel_err,
            "detector_noise_floor": self.detector_noise_floor,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"{name} must be finite and non-negative / 必须是有限非负数; got {value!r}")
        if not (self.detector_snr_db == float("inf") or
                np.isfinite(self.detector_snr_db) and self.detector_snr_db > 0):
            raise ValueError("detector_snr_db must be positive or inf / 必须是正数或 inf")

    @property
    def amp_transmission(self) -> float:
        """Convert loss in dB to amplitude transmission. / 插损 dB 转幅度透过率。"""
        return 10.0 ** (-self.mzi_loss_db / 20.0)


# =============================================================================
# 集成光子处理器
# =============================================================================
class PhotonicMatrixProcessor:
    """
    Compile any real matrix ``M`` (n_out x n_in) onto a photonic processor model.
    把任意实数矩阵 ``M`` (n_out x n_in) 编译到光子处理器模型上。

    Parameters
    ----------
    M
        Target matrix; rectangular shapes are zero-padded internally to
        ``N = max(n_out, n_in)``. / 目标矩阵；支持非方阵，内部零填充。
    noise
        Physical non-ideality model; defaults to an ideal circuit. /
        物理非理想性模型；默认全零（理想电路）。
    seed
        Seed for an independent ``np.random.Generator``. /
        独立随机数生成器的种子，不污染全局 ``np.random``。

    Examples
    --------
    >>> import numpy as np
    >>> from photonic_mzi import PhotonicMatrixProcessor
    >>> M = np.array([[0.5, -0.2], [0.1, 0.9]])
    >>> opu = PhotonicMatrixProcessor(M, seed=0)
    >>> bool(np.allclose(opu.read_coherent([1.0, 2.0]), M @ [1.0, 2.0]))
    True
    """

    def __init__(self, M: np.ndarray, noise: NoiseModel | None = None,
                 seed: int | None = None):
        raw_M = np.asarray(M)
        if np.iscomplexobj(raw_M) and np.any(np.imag(raw_M) != 0):
            raise ValueError(
                "PhotonicMatrixProcessor currently supports real matrices only / 当前只支持实矩阵")
        M = np.asarray(np.real(raw_M), dtype=float)
        if M.ndim != 2:
            raise ValueError(f"M must be two-dimensional / M 必须是二维矩阵; shape={M.shape}")
        if M.size == 0:
            raise ValueError("M must not be empty / M 不能是空矩阵")
        if not np.all(np.isfinite(M)):
            raise ValueError("M must contain only finite values / M 必须只包含有限数值")
        self.M = M
        self.n_out, self.n_in = M.shape
        self.N = max(self.n_out, self.n_in)
        self.rng = np.random.default_rng(seed)
        self.noise = noise or NoiseModel()

        # --- 步骤 1: 零填充成方阵后做 SVD ---
        Mp = np.zeros((self.N, self.N))
        Mp[:self.n_out, :self.n_in] = M
        U, S, Vt = np.linalg.svd(Mp)
        self.U_target, self.S_target, self.Vt_target = U, S, Vt

        # --- 步骤 1b: VOA 只能衰减，把奇异值归一化到 [0,1]，增益留给电域 ---
        self.gain = float(S[0]) if S[0] > 0 else 1.0
        self.S_phys = S / self.gain

        # --- 步骤 2: 编译成 MZI 相移角 ---
        self.vt_mzis, self.vt_phases = decompose_unitary(self.Vt_target)
        self.u_mzis, self.u_phases = decompose_unitary(self.U_target)

        # --- 步骤 3: 采样一次静态相移控制偏置与 VOA 设定误差 ---
        self._fab_vt = self._sample_fab(len(self.vt_mzis))
        self._fab_u = self._sample_fab(len(self.u_mzis))
        self._fab_vt_phases = self.rng.normal(0, self.noise.fab_phi, self.N)
        self._fab_u_phases = self.rng.normal(0, self.noise.fab_phi, self.N)
        self._voa_rel_offset = self.rng.normal(0, self.noise.voa_rel_err, self.N)
        # 校准修正量，calibrate() 之前恒为 0
        self._cal_vt = np.zeros_like(self._fab_vt)
        self._cal_u = np.zeros_like(self._fab_u)
        self._cal_vt_phases = np.zeros(self.N)
        self._cal_u_phases = np.zeros(self.N)
        self.calibrated = False

    # ------------------------------------------------------------------ 内部
    def _sample_fab(self, n: int) -> np.ndarray:
        nz = self.noise
        return np.stack([self.rng.normal(0, nz.fab_theta, n),
                         self.rng.normal(0, nz.fab_phi, n)], axis=1)

    def _run_mesh(self, E, mzis: list[MZI], phases, fab, cal,
                  fab_phases, cal_phases, ideal, trace):
        nz = self.noise
        amp = 1.0 if ideal else nz.amp_transmission
        if ideal:
            phase_factor = phases[:, None]
        else:
            # 每一列代表一个独立输入样本；动态抖动按样本独立采样。
            phase_jitter = self.rng.normal(0, nz.drift_phi, E.shape)
            phase_error = fab_phases[:, None] + cal_phases[:, None] + phase_jitter
            phase_factor = phases[:, None] * np.exp(1j * phase_error)
        E = E * phase_factor
        if trace is not None:
            trace.append(("phase", None, E.copy()))
        for k, z in enumerate(mzis):
            th, ph = z.theta, z.phi
            if not ideal:
                th += fab[k, 0] + cal[k, 0] + self.rng.normal(
                    0, nz.drift_theta, E.shape[1])
                ph += fab[k, 1] + cal[k, 1] + self.rng.normal(
                    0, nz.drift_phi, E.shape[1])
            apply_T_dagger_left(E, z.mode, th, ph, amp)
            if trace is not None:
                trace.append(("mzi", z, E.copy()))
        return E

    # -------------------------------------------------------------- 公开 API
    def optical_field(self, x: np.ndarray, ideal: bool = True,
                      trace: list | None = None) -> np.ndarray:
        """
        Forward optical propagation returning the physical complex field before detection.
        正向光学传播，返回探测前的**物理复数光场**。

        奇异值整体增益 ``self.gain`` 不在这里补回；因此理想情况下返回
        ``(M @ x) / gain``。电域标度和读出噪声分别由 :meth:`read_coherent` 与
        :meth:`read_intensity` 处理。

        Parameters
        ----------
        x
            Input vector ``(n_in,)`` or batch ``(n_in, B)``. /
            输入向量或批量输入。
        ideal
            ``True`` for ideal propagation; ``False`` enables :class:`NoiseModel`. /
            ``True`` 使用理想传播；``False`` 引入全部非理想性。
        trace
            Optional list receiving the field after every device for animation. /
            可选列表，用于记录每个器件后的光场供动画使用。
        """
        x = np.asarray(x, dtype=complex)
        if x.ndim == 1:
            if x.shape[0] != self.n_in:
                raise ValueError(
                    f"1-D x must have length {self.n_in} / 一维 x 的长度必须是 {self.n_in}; "
                    f"got {x.shape[0]}")
        elif x.ndim == 2:
            if x.shape[0] != self.n_in:
                raise ValueError(
                    f"first dimension of 2-D x must be {self.n_in} / "
                    f"二维 x 的第一维必须是 {self.n_in}; shape={x.shape}")
        else:
            raise ValueError(
                f"x must have shape (n_in,) or (n_in, B) / x 形状必须符合要求; got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values / x 必须只包含有限数值")
        squeeze = x.ndim == 1
        x = x.reshape(self.n_in, -1)
        E = np.zeros((self.N, x.shape[1]), dtype=complex)
        E[:self.n_in] = x                       # 多余输入端口保持黑暗
        if trace is not None:
            trace.append(("input", None, E.copy()))

        E = self._run_mesh(E, self.vt_mzis, self.vt_phases,
                           self._fab_vt, self._cal_vt,
                           self._fab_vt_phases, self._cal_vt_phases,
                           ideal, trace)

        s = self.S_phys
        if not ideal and self.noise.voa_rel_err:
            s = np.clip(s * (1 + self._voa_rel_offset), 0.0, 1.0)
        E = E * s[:, None]
        if trace is not None:
            trace.append(("voa", None, E.copy()))

        E = self._run_mesh(E, self.u_mzis, self.u_phases,
                           self._fab_u, self._cal_u,
                           self._fab_u_phases, self._cal_u_phases,
                           ideal, trace)

        y = E[:self.n_out]

        if trace is not None:
            trace.append(("output", None, y.copy()))
        return y[:, 0] if squeeze else y

    forward = optical_field
    __call__ = optical_field

    def _add_readout_noise(self, y: np.ndarray, ideal: bool) -> np.ndarray:
        """Add equivalent AWGN after detection. / 在探测后加入等效 AWGN。"""
        if ideal:
            return y
        nz = self.noise
        sigma2 = np.full(y.shape[1], nz.detector_noise_floor ** 2)
        if np.isfinite(nz.detector_snr_db):
            rms = np.sqrt(np.mean(y ** 2, axis=0))
            sigma2 += (rms * 10 ** (-nz.detector_snr_db / 20.0)) ** 2
        if not np.any(sigma2):
            return y
        return y + self.rng.normal(0, np.sqrt(sigma2)[None, :], y.shape)

    def read_coherent(self, x, **kw) -> np.ndarray:
        """
        Coherent (homodyne) detection that recovers the signed real component.
        相干（零差）探测：获取带符号的实部。

        物理上这需要引一路本振光提供相位参考，不是「测一下光强」那么简单。
        """
        ideal = kw.get("ideal", True)
        E = self.optical_field(x, **kw)
        squeeze = E.ndim == 1
        y = np.real(E) * self.gain
        y2 = y[:, None] if squeeze else y
        y2 = self._add_readout_noise(y2, ideal)
        return y2[:, 0] if squeeze else y2

    def read_intensity(self, x, **kw) -> np.ndarray:
        """Direct detection returning calibrated ``gain^2 * |E|^2``.
        直接探测并返回标定后的功率量 ``gain^2 * |E|^2``。

        噪声在平方检波之后加入，因此不会把探测器电噪声误当成光场扰动。返回值是
        经过系统标度校准的读出量，不是未标定的瓦特数；符号信息仍然丢失。
        """
        ideal = kw.get("ideal", True)
        E = self.optical_field(x, **kw)
        squeeze = E.ndim == 1
        y = np.abs(E) ** 2 * self.gain ** 2
        y2 = y[:, None] if squeeze else y
        y2 = self._add_readout_noise(y2, ideal)
        return y2[:, 0] if squeeze else y2

    # ------------------------------------------------------------------ 校准
    def calibrate(self) -> None:
        """
        Simulate ideal characterization that cancels static phase-control offsets.
        模拟一次理想表征校准：抵消静态相移控制偏置。

        真实芯片的做法是逐台 MZI 扫描控制电压、测出实际相移曲线，再反推出
        控制量的修正表。这里直接用已知的 ``fab`` 偏移取反 —— 等价于「表征做得
        足够准」这一理想假设。它不能校正分束器制造偏差，也不能据此推断真实芯片
        可以达到浮点精度。

        注意校准对 ``drift_*``、``voa_rel_err``、``mzi_loss_db`` 和读出噪声无效：
        它们要么按样本变化，要么根本不是加性相移偏置。
        """
        self._cal_vt = -self._fab_vt
        self._cal_u = -self._fab_u
        self._cal_vt_phases = -self._fab_vt_phases
        self._cal_u_phases = -self._fab_u_phases
        self.calibrated = True

    def reset_calibration(self) -> None:
        """Reset to the uncalibrated state. / 撤销校准。"""
        self._cal_vt = np.zeros_like(self._fab_vt)
        self._cal_u = np.zeros_like(self._fab_u)
        self._cal_vt_phases = np.zeros(self.N)
        self._cal_u_phases = np.zeros(self.N)
        self.calibrated = False

    # ------------------------------------------------------------ 自检与报告
    def unitary_error(self) -> tuple[float, float]:
        """Compilation round-trip errors for U and V^T. / 编译回环误差。"""
        eu = np.linalg.norm(
            recompose_unitary(self.u_mzis, self.u_phases, self.N) - self.U_target)
        ev = np.linalg.norm(
            recompose_unitary(self.vt_mzis, self.vt_phases, self.N) - self.Vt_target)
        return float(eu), float(ev)

    @property
    def mesh_depth(self) -> tuple[int, int]:
        """Mesh depths for V^T and U; Reck worst case is ``2N-3``. / 两张网格的深度。"""
        return (max(z.layer for z in self.vt_mzis) + 1 if self.vt_mzis else 0,
                max(z.layer for z in self.u_mzis) + 1 if self.u_mzis else 0)

    @property
    def n_mzi(self) -> int:
        return len(self.vt_mzis) + len(self.u_mzis)

    def mode_mzi_count(self) -> np.ndarray:
        """Number of MZIs touching each spatial mode. / 每个空间模式参与的 MZI 数量。

        这是 Reck 拓扑不均匀性的简单代理量，不是端到端光路追踪：光会在 MZI 中
        分束并迁移到其他模式，所以不能把最大计数严格解释为某条光路的真实插损。
        """
        cnt = np.zeros(self.N, dtype=int)
        for z in self.vt_mzis + self.u_mzis:
            cnt[z.mode] += 1
            cnt[z.mode + 1] += 1
        return cnt

    def path_mzi_count(self) -> np.ndarray:
        """Compatibility alias; prefer :meth:`mode_mzi_count`. / 兼容旧 API。"""
        return self.mode_mzi_count()

    def report(self) -> str:
        """Human-readable English-first bilingual chip report. / 英文优先的双语报告。"""
        dv, du = self.mesh_depth
        eu, ev = self.unitary_error()
        cnt = self.mode_mzi_count()
        loss = self.noise.mzi_loss_db
        return "\n".join([
            f"Matrix / 矩阵尺寸       : {self.n_out} x {self.n_in}   (modes / 模式 N={self.N})",
            f"MZI count / MZI 总数    : {self.n_mzi}  "
            f"(V^T {len(self.vt_mzis)} + U {len(self.u_mzis)})",
            f"Mesh depth / 网格深度  : V^T {dv} layers/列 + U {du} layers/列  "
            f"(Reck worst {max(2 * self.N - 3, 0)}; Clements {self.N})",
            f"Mode count / 模式参与   : {cnt.tolist()}  "
            f"(max/min={cnt.max()}/{cnt.min()}, topology proxy / 拓扑代理)",
            f"Loss bound / 损耗上界   : {cnt.max() * loss:.2f} dB @ {loss} dB/MZI  "
            f"(not path tracing / 非路径追踪)",
            f"Singular values / 奇异值: {np.array2string(self.S_target, precision=4)}",
            f"VOA gain / 增益         : {self.gain:.4f}  (transmission <=1; electrical gain)",
            f"Round trip / 编译回环误差: ||U_rebuild-U||={eu:.2e}   ||Vt_rebuild-Vt||={ev:.2e}",
            f"Calibration / 校准状态  : "
            f"{'calibrated / 已校准' if self.calibrated else 'uncalibrated / 未校准'}",
        ])

    def __repr__(self) -> str:
        return (f"PhotonicMatrixProcessor({self.n_out}x{self.n_in}, N={self.N}, "
                f"n_mzi={self.n_mzi}, calibrated={self.calibrated})")
