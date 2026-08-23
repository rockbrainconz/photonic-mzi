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
        静态制造误差（波导刻蚀偏差、耦合器分光比偏离 50:50）。流片后就固定不变，
        开机跑一次表征就能标定掉，所以仿真中只应在构造时采样一次。
        见 :meth:`PhotonicMatrixProcessor.calibrate`。
    ``drift_*``
        动态热漂移（相邻加热器串扰、环境温度）。每发一个光脉冲都在变，标不掉。
    ``mzi_loss_db``
        每台 MZI 的插入损耗。Reck 三角网格路径长度不均匀，它会让不同波导衰减不同，
        形成确定性的通道失配 —— 真实芯片的主要误差源之一。
    ``voa_rel_err``
        衰减器设定值的相对误差。
    ``detector_snr_db``
        探测端等效信噪比（含散粒噪声与 TIA 电噪声）。

    所有角度量的单位都是 **弧度**（1-sigma），不是百分比。
    """

    fab_theta: float = 0.0
    fab_phi: float = 0.0
    drift_theta: float = 0.0
    drift_phi: float = 0.0
    mzi_loss_db: float = 0.0
    voa_rel_err: float = 0.0
    detector_snr_db: float = float("inf")

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
        M = np.asarray(M, dtype=float)
        if M.ndim != 2:
            raise ValueError(f"M 必须是二维矩阵，收到 shape={M.shape}")
        if M.size == 0:
            raise ValueError("M 不能是空矩阵")
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

        # --- 步骤 3: 采样一次静态制造误差（流片后固定不变） ---
        self._fab_vt = self._sample_fab(len(self.vt_mzis))
        self._fab_u = self._sample_fab(len(self.u_mzis))
        # 校准修正量，calibrate() 之前恒为 0
        self._cal_vt = np.zeros_like(self._fab_vt)
        self._cal_u = np.zeros_like(self._fab_u)
        self.calibrated = False

    # ------------------------------------------------------------------ 内部
    def _sample_fab(self, n: int) -> np.ndarray:
        nz = self.noise
        return np.stack([self.rng.normal(0, nz.fab_theta, n),
                         self.rng.normal(0, nz.fab_phi, n)], axis=1)

    def _run_mesh(self, E, mzis: list[MZI], phases, fab, cal, ideal, trace):
        nz = self.noise
        amp = 1.0 if ideal else nz.amp_transmission
        E = E * phases[:, None]
        if trace is not None:
            trace.append(("phase", None, E.copy()))
        for k, z in enumerate(mzis):
            th, ph = z.theta, z.phi
            if not ideal:
                th += fab[k, 0] + cal[k, 0] + self.rng.normal(0, nz.drift_theta)
                ph += fab[k, 1] + cal[k, 1] + self.rng.normal(0, nz.drift_phi)
            apply_T_dagger_left(E, z.mode, th, ph, amp)
            if trace is not None:
                trace.append(("mzi", z, E.copy()))
        return E

    # -------------------------------------------------------------- 公开 API
    def forward(self, x: np.ndarray, ideal: bool = True,
                trace: list | None = None) -> np.ndarray:
        """
        正向光学仿真，返回**复数**光场。

        Parameters
        ----------
        x
            输入向量 ``(n_in,)`` 或批量 ``(n_in, B)``。
        ideal
            ``True`` 走理想物理；``False`` 引入 :class:`NoiseModel` 的全部非理想性。
        trace
            若传入 list，会记录每一步之后的光场（供动画使用）。
        """
        x = np.asarray(x)
        squeeze = x.ndim == 1
        x = x.reshape(self.n_in, -1)
        E = np.zeros((self.N, x.shape[1]), dtype=complex)
        E[:self.n_in] = x                       # 多余输入端口保持黑暗
        if trace is not None:
            trace.append(("input", None, E.copy()))

        E = self._run_mesh(E, self.vt_mzis, self.vt_phases,
                           self._fab_vt, self._cal_vt, ideal, trace)

        s = self.S_phys
        if not ideal and self.noise.voa_rel_err:
            s = np.clip(s * (1 + self.rng.normal(0, self.noise.voa_rel_err, self.N)),
                        0.0, 1.0)
        E = E * s[:, None]
        if trace is not None:
            trace.append(("voa", None, E.copy()))

        E = self._run_mesh(E, self.u_mzis, self.u_phases,
                           self._fab_u, self._cal_u, ideal, trace)

        y = E[:self.n_out] * self.gain          # 电域增益补偿

        if not ideal and np.isfinite(self.noise.detector_snr_db):
            rms = float(np.sqrt(np.mean(np.abs(y) ** 2))) or 1.0
            sigma = rms * 10 ** (-self.noise.detector_snr_db / 20.0)
            y = y + self.rng.normal(0, sigma, y.shape)

        if trace is not None:
            trace.append(("output", None, y.copy()))
        return y[:, 0] if squeeze else y

    __call__ = forward

    def read_coherent(self, x, **kw) -> np.ndarray:
        """
        相干（零差）探测：拿到带符号的实部。

        物理上这需要引一路本振光提供相位参考，不是「测一下光强」那么简单。
        """
        return np.real(self.forward(x, **kw))

    def read_intensity(self, x, **kw) -> np.ndarray:
        """直接光电探测：光电二极管只能测 ``|E|^2``，符号信息在这里是丢失的。"""
        return np.abs(self.forward(x, **kw)) ** 2

    # ------------------------------------------------------------------ 校准
    def calibrate(self) -> None:
        """
        模拟一次出厂表征校准：抵消静态制造误差。

        真实芯片的做法是逐台 MZI 扫描控制电压、测出实际相移曲线，再反推出
        控制量的修正表。这里直接用已知的 ``fab`` 偏移取反 —— 等价于「表征做得
        足够准」这一理想假设，用来说明**静态误差可以标定、动态漂移不能**。

        注意校准对 ``drift_*``、``mzi_loss_db``、``detector_snr_db`` 无效：
        它们要么逐脉冲在变，要么根本不是相移误差。
        """
        self._cal_vt = -self._fab_vt
        self._cal_u = -self._fab_u
        self.calibrated = True

    def reset_calibration(self) -> None:
        """撤销 :meth:`calibrate`，回到未校准状态。"""
        self._cal_vt = np.zeros_like(self._fab_vt)
        self._cal_u = np.zeros_like(self._fab_u)
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

    def path_mzi_count(self) -> np.ndarray:
        """每根波导在整条光路上经过的 MZI 台数，体现 Reck 三角的路径不均匀。"""
        cnt = np.zeros(self.N, dtype=int)
        for z in self.vt_mzis + self.u_mzis:
            cnt[z.mode] += 1
            cnt[z.mode + 1] += 1
        return cnt

    def report(self) -> str:
        """人类可读的芯片规格报告。"""
        dv, du = self.mesh_depth
        eu, ev = self.unitary_error()
        cnt = self.path_mzi_count()
        loss = self.noise.mzi_loss_db
        return "\n".join([
            f"矩阵尺寸        : {self.n_out} x {self.n_in}   (芯片模式数 N={self.N})",
            f"MZI 总数        : {self.n_mzi}  "
            f"(V^T {len(self.vt_mzis)} + U {len(self.u_mzis)})",
            f"网格深度        : V^T {dv} 列 + U {du} 列  "
            f"(Reck 三角最坏 {max(2 * self.N - 3, 0)} 列; Clements 矩形可降到 {self.N} 列)",
            f"各波导过 MZI 数 : {cnt.tolist()}  "
            f"(max/min={cnt.max()}/{cnt.min()}, 路径不均匀 -> 插损失配)",
            f"最坏路径插损    : {cnt.max() * loss:.2f} dB @ {loss} dB/MZI",
            f"奇异值          : {np.array2string(self.S_target, precision=4)}",
            f"VOA 归一化增益  : {self.gain:.4f}  (物理透过率 <=1，增益在电域补回)",
            f"编译回环误差    : ||U_rebuild-U||={eu:.2e}   ||Vt_rebuild-Vt||={ev:.2e}",
            f"校准状态        : {'已校准（静态误差已抵消）' if self.calibrated else '未校准'}",
        ])

    def __repr__(self) -> str:
        return (f"PhotonicMatrixProcessor({self.n_out}x{self.n_in}, N={self.N}, "
                f"n_mzi={self.n_mzi}, calibrated={self.calibrated})")
