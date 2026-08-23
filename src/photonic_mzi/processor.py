"""
处理器层：SVD 编译、正向光学仿真、物理非理想性模型。

    M  = U @ diag(S) @ Vt                       (SVD)
    U, Vt  -> Reck 三角 MZI 网格
    diag(S) -> 一列可变光衰减器 VOA

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
    区分两类本质不同的误差 —— 把它们混成一个 ``noise_std`` 会得出错误的结论：

    ``fab_*``
        静态**相移控制偏置**。它在处理器构造时采样一次，并可由理想化的表征流程
        精确抵消。它不代表分束器制造偏差；后者可能限制 MZI 的可达分光比，本模型
        尚未描述，不能用 :meth:`PhotonicMatrixProcessor.calibrate` 简单消除。
    ``drift_*``
        每个输入样本独立采样的动态相位抖动。它是用于敏感度分析的 i.i.d. 简化模型，
        不等同于具有慢时间尺度和空间相关性的真实热漂移。
    ``mzi_loss_db``
        每台 MZI 的插入损耗。Reck 三角网格路径长度不均匀，它会让不同波导衰减不同，
        形成确定性的通道失配 —— 真实芯片的主要误差源之一。
    ``voa_rel_err``
        衰减器的固定相对设定误差，构造处理器时每通道采样一次。
    ``detector_snr_db``
        探测后读出量的等效 RMS 信噪比。它只是相对 AWGN 参数，不声称分别建模
        散粒噪声、TIA、电带宽或相干本振。
    ``detector_noise_floor``
        探测后、以返回值单位表示的绝对高斯噪声底。相对噪声与它按方差相加。

    所有角度量的单位都是 **弧度**（1-sigma），不是百分比。
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
                raise ValueError(f"{name} 必须是有限非负数，收到 {value!r}")
        if not (self.detector_snr_db == float("inf") or
                np.isfinite(self.detector_snr_db) and self.detector_snr_db > 0):
            raise ValueError("detector_snr_db 必须是正数或 inf")

    @property
    def amp_transmission(self) -> float:
        """插损 dB -> 幅度透过率：功率 ``10**(-L/10)``，幅度开根号即 ``10**(-L/20)``。"""
        return 10.0 ** (-self.mzi_loss_db / 20.0)


# =============================================================================
# 集成光子处理器
# =============================================================================
class PhotonicMatrixProcessor:
    """
    把任意实数矩阵 ``M`` (n_out x n_in) 编译到光子芯片上。

    Parameters
    ----------
    M
        目标矩阵。支持非方阵，内部零填充到 ``N = max(n_out, n_in)``，多余端口暗置。
    noise
        物理非理想性模型。默认全零（理想芯片）。
    seed
        随机种子。走独立的 ``np.random.Generator``，不污染全局 ``np.random``。

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
            raise ValueError("PhotonicMatrixProcessor 当前只支持实矩阵，不能静默丢弃虚部")
        M = np.asarray(np.real(raw_M), dtype=float)
        if M.ndim != 2:
            raise ValueError(f"M 必须是二维矩阵，收到 shape={M.shape}")
        if M.size == 0:
            raise ValueError("M 不能是空矩阵")
        if not np.all(np.isfinite(M)):
            raise ValueError("M 必须只包含有限数值")
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
        正向光学传播，返回探测前的**物理复数光场**。

        奇异值整体增益 ``self.gain`` 不在这里补回；因此理想情况下返回
        ``(M @ x) / gain``。电域标度和读出噪声分别由 :meth:`read_coherent` 与
        :meth:`read_intensity` 处理。

        Parameters
        ----------
        x
            输入向量 ``(n_in,)`` 或批量 ``(n_in, B)``。
        ideal
            ``True`` 走理想物理；``False`` 引入 :class:`NoiseModel` 的全部非理想性。
        trace
            若传入 list，会记录每一步之后的光场（供动画使用）。
        """
        x = np.asarray(x, dtype=complex)
        if x.ndim == 1:
            if x.shape[0] != self.n_in:
                raise ValueError(f"一维 x 的长度必须是 {self.n_in}，收到 {x.shape[0]}")
        elif x.ndim == 2:
            if x.shape[0] != self.n_in:
                raise ValueError(f"二维 x 的第一维必须是 {self.n_in}，收到 shape={x.shape}")
        else:
            raise ValueError(f"x 必须是 (n_in,) 或 (n_in, B)，收到 shape={x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("x 必须只包含有限数值")
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
        """在探测后的实数量上加入等效 AWGN；不把电噪声误加到复光场上。"""
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
        相干（零差）探测：拿到带符号的实部。

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
        """直接探测并返回标定后的功率量 ``gain^2 * |E|^2``。

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
        """撤销 :meth:`calibrate`，回到未校准状态。"""
        self._cal_vt = np.zeros_like(self._fab_vt)
        self._cal_u = np.zeros_like(self._fab_u)
        self._cal_vt_phases = np.zeros(self.N)
        self._cal_u_phases = np.zeros(self.N)
        self.calibrated = False

    # ------------------------------------------------------------ 自检与报告
    def unitary_error(self) -> tuple[float, float]:
        """编译回环误差：由相移角重建的 U / V^T 与目标的差距。"""
        eu = np.linalg.norm(
            recompose_unitary(self.u_mzis, self.u_phases, self.N) - self.U_target)
        ev = np.linalg.norm(
            recompose_unitary(self.vt_mzis, self.vt_phases, self.N) - self.Vt_target)
        return float(eu), float(ev)

    @property
    def mesh_depth(self) -> tuple[int, int]:
        """``(V^T 网格列数, U 网格列数)``。Reck 三角最坏为 ``2N-3``。"""
        return (max(z.layer for z in self.vt_mzis) + 1 if self.vt_mzis else 0,
                max(z.layer for z in self.u_mzis) + 1 if self.u_mzis else 0)

    @property
    def n_mzi(self) -> int:
        return len(self.vt_mzis) + len(self.u_mzis)

    def mode_mzi_count(self) -> np.ndarray:
        """每个空间模式参与的 MZI 数量。

        这是 Reck 拓扑不均匀性的简单代理量，不是端到端光路追踪：光会在 MZI 中
        分束并迁移到其他模式，所以不能把最大计数严格解释为某条光路的真实插损。
        """
        cnt = np.zeros(self.N, dtype=int)
        for z in self.vt_mzis + self.u_mzis:
            cnt[z.mode] += 1
            cnt[z.mode + 1] += 1
        return cnt

    def path_mzi_count(self) -> np.ndarray:
        """兼容旧 API；请使用名称更准确的 :meth:`mode_mzi_count`。"""
        return self.mode_mzi_count()

    def report(self) -> str:
        """人类可读的芯片规格报告。"""
        dv, du = self.mesh_depth
        eu, ev = self.unitary_error()
        cnt = self.mode_mzi_count()
        loss = self.noise.mzi_loss_db
        return "\n".join([
            f"矩阵尺寸        : {self.n_out} x {self.n_in}   (芯片模式数 N={self.N})",
            f"MZI 总数        : {self.n_mzi}  "
            f"(V^T {len(self.vt_mzis)} + U {len(self.u_mzis)})",
            f"网格深度        : V^T {dv} 列 + U {du} 列  "
            f"(Reck 三角最坏 {max(2 * self.N - 3, 0)} 列; Clements 矩形可降到 {self.N} 列)",
            f"空间模式参与数  : {cnt.tolist()}  "
            f"(max/min={cnt.max()}/{cnt.min()}, 仅作拓扑不均匀性代理)",
            f"串联损耗上界估计: {cnt.max() * loss:.2f} dB @ {loss} dB/MZI  "
            f"(不是端到端路径追踪)",
            f"奇异值          : {np.array2string(self.S_target, precision=4)}",
            f"VOA 归一化增益  : {self.gain:.4f}  (物理透过率 <=1，增益在电域补回)",
            f"编译回环误差    : ||U_rebuild-U||={eu:.2e}   ||Vt_rebuild-Vt||={ev:.2e}",
            f"校准状态        : {'已校准（静态相移偏置已抵消）' if self.calibrated else '未校准'}",
        ])

    def __repr__(self) -> str:
        return (f"PhotonicMatrixProcessor({self.n_out}x{self.n_in}, N={self.N}, "
                f"n_mzi={self.n_mzi}, calibrated={self.calibrated})")
