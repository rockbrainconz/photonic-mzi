"""
photonic_mzi.animation — MZI 网格光计算「逐行 + 逐器件」教学动画
=========================================================
左边是正在执行的代码（高亮当前行），右边是这一行代码在光子芯片上
物理发生了什么。整个流程拆成 9 个阶段：

    0 问题      1 SVD 分解   2 编译成 MZI 角度   3 光注入
    4 V^T 网格   5 Σ 衰减器   6 U 网格            7 探测输出   8 非理想性

用法
----
    python -m photonic_mzi                 # 交互播放（推荐）
    python -m photonic_mzi --save mzi.gif  # 导出 GIF
    python -m photonic_mzi -n 5            # 换成 5x5 随机矩阵

交互键
------
    空格   暂停 / 继续
    ← →    单步回退 / 前进（暂停时最有用）
    , .    上一阶段 / 下一阶段
    r      从头重播
"""
from __future__ import annotations

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from .mesh import apply_T_dagger_left, apply_T_left, mzi_transfer_matrix
from .processor import NoiseModel, PhotonicMatrixProcessor

# --------------------------------------------------------------------------- #
# 配色与字体
# --------------------------------------------------------------------------- #
CN = ["Microsoft YaHei", "SimHei", "sans-serif"]
MONO = ["Consolas", "Microsoft YaHei", "monospace"]
mpl.rcParams["font.sans-serif"] = CN
mpl.rcParams["axes.unicode_minus"] = False

BG = "#0b1020"
PANEL = "#131a2e"
FG = "#e6edf3"
DIM = "#7d8aa5"
WG_IDLE = "#2b3550"
CYAN = "#22d3ee"
AMBER = "#fbbf24"
GREEN = "#4ade80"
PINK = "#f472b6"
VIOLET = "#a78bfa"
RED = "#f87171"

# 用 hsv 这类「全饱和」的循环色图：0 相位=红，π 相位=青，
# 在深色背景上每个相位都同样醒目（twilight 在 0 附近发白，看不清）。
PHASE_CMAP = plt.get_cmap("hsv")


def phase_color(z):
    """复振幅 -> 颜色：色相编码相位，这样一眼能看出干涉是相长还是相消。"""
    return PHASE_CMAP((np.angle(z) + np.pi) / (2 * np.pi) % 1.0)


def text_on(fc):
    """按底色亮度选字色，避免 RdBu_r 中间那段近白色底上再写浅字。"""
    r, g, b = mpl.colors.to_rgb(fc)
    return "#0b1020" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.58 else FG


# --------------------------------------------------------------------------- #
# 代码片段（左侧面板显示的内容）
# --------------------------------------------------------------------------- #
SNIPPETS = {
    "svd": ("photonic_mzi/processor.py  ·  PhotonicMatrixProcessor.__init__", [
        "class PhotonicMatrixProcessor:",
        "    def __init__(self, M):",
        "        self.N = max(M.shape)         # 芯片波导根数",
        "        Mp = zero_pad(M, self.N)      # 非方阵先补成方阵",
        "",
        "        # 关键一步：任意实矩阵都能拆成 正交变换·缩放·正交变换",
        "        U, S, Vt = np.linalg.svd(Mp)",
        "",
        "        self.gain   = S[0]            # VOA 只能衰减不能放大",
        "        self.S_phys = S / self.gain   # 归一化到 [0, 1]",
        "",
        "        # 两个酉矩阵各自编译成一张 MZI 网格",
        "        self.vt_mzis, self.vt_phases = decompose_unitary(Vt)",
        "        self.u_mzis,  self.u_phases  = decompose_unitary(U)",
    ]),
    "compile": ("photonic_mzi/mesh.py  ·  decompose_unitary", [
        "def decompose_unitary(U):",
        "    N = U.shape[0]",
        "    A = U.copy()          # 边消元边记录用了哪些 MZI",
        "    elim = []",
        "",
        "    for col in range(N - 1):",
        "        for row in range(N - 1, col, -1):",
        "            m = row - 1           # 这台 MZI 占用 m / m+1 两根波导",
        "            x, y = A[m, col], A[row, col]",
        "",
        "            # 解出一对 (theta, phi)，让 A[row, col] 干涉相消为 0",
        "            phi   = angle(y) - angle(x) - pi",
        "            theta = arctan2(abs(x), abs(y))",
        "",
        "            apply_T_left(A, m, theta, phi)   # 只动两行",
        "            elim.append((m, theta, phi))",
        "",
        "    return elim, np.diag(A)   # A 已被消成对角阵",
    ]),
    "forward": ("photonic_mzi/processor.py  ·  optical_field  （芯片内传播）", [
        "def optical_field(self, x):",
        "    E = pad(x)                     # 光场 = 复振幅向量",
        "",
        "    # ---- 阶段 1：V^T 酉变换网格 ----",
        "    E = E * self.vt_phases         # N 个输出移相器",
        "    for z in self.vt_mzis:",
        "        apply_T_dagger_left(E, z.mode, z.theta, z.phi)",
        "",
        "    # ---- 阶段 2：Sigma 可变光衰减器 ----",
        "    E = E * self.S_phys            # 逐通道衰减",
        "",
        "    # ---- 阶段 3：U 酉变换网格 ----",
        "    E = E * self.u_phases",
        "    for z in self.u_mzis:",
        "        apply_T_dagger_left(E, z.mode, z.theta, z.phi)",
        "",
        "    return E[:n_out]               # 探测前物理复光场",
    ]),
    "mzi": ("photonic_mzi/mesh.py  ·  apply_T_dagger_left  （一台 MZI 内部）", [
        "def apply_T_dagger_left(E, m, theta, phi, amp=1.0):",
        "    s, c = sin(theta), cos(theta)   # theta 决定分光比",
        "    e    = exp(-1j * phi)           # phi   决定相对相位",
        "",
        "    a, b = E[m], E[m + 1]           # 进入这台 MZI 的两束光",
        "",
        "    # 两束光在 3dB 耦合器里叠加：同相相长，反相相消",
        "    E[m]     = amp * e * (-s * a + c * b)",
        "    E[m + 1] = amp *     ( c * a + s * b)",
    ]),
    "detect": ("photonic_mzi/processor.py  ·  读出", [
        "    # 相干（零差）探测：能拿到带符号的实部",
        "    y = self.gain * np.real(E[:n_out])",
        "    y = add_readout_noise(y)       # 噪声加在探测后的实数量上",
        "",
        "    # 直接光电探测就只剩 |E|^2，符号会丢失",
        "    # y = self.gain**2 * np.abs(E[:n_out]) ** 2",
        "",
        "    assert np.allclose(y, M @ x)    # 与 CPU 结果完全一致",
    ]),
    "noise": ("photonic_mzi/processor.py  ·  NoiseModel  （简化敏感度模型）", [
        "nz = NoiseModel(",
        "    fab_theta = 0.02,      # 固定相移控制偏置，可理想标定",
        "    drift_theta = 0.005,   # 每样本独立相位抖动（简化模型）",
        "    mzi_loss_db = 0.2,     # 每台 MZI 的插入损耗",
        "    voa_rel_err = 0.01,    # 衰减器设定误差",
        "    detector_snr_db = 40,  # 探测后等效相对 AWGN",
        ")",
        "y = opu.read_coherent(x, ideal=False)",
    ]),
}

STAGES = ["0 问题", "1 SVD 分解", "2 编译 MZI", "3 光注入",
          "4 V^T 网格", "5 Σ 衰减", "6 U 网格", "7 探测输出", "8 非理想性"]


# --------------------------------------------------------------------------- #
# 芯片几何
# --------------------------------------------------------------------------- #
class ChipLayout:
    def __init__(self, opu: PhotonicMatrixProcessor):
        self.N = N = opu.N
        self.dv, self.du = opu.mesh_depth
        self.x_in = 0.5
        self.x_vt_ph = 1.5
        self.dx = 1.15
        self.x_vt0 = 2.6
        self.x_voa = self.x_vt0 + self.dv * self.dx + 0.35
        self.x_u_ph = self.x_voa + 1.15
        self.x_u0 = self.x_u_ph + 1.1
        self.x_out = self.x_u0 + (self.du - 1) * self.dx + 1.3
        self.ylim = (-0.9, N - 0.1)

    def y(self, mode):
        return self.N - 1 - mode

    def x_vt(self, layer):
        return self.x_vt0 + layer * self.dx

    def x_u(self, layer):
        return self.x_u0 + layer * self.dx


# --------------------------------------------------------------------------- #
# 脚本：把整个流程编成一串帧
# --------------------------------------------------------------------------- #
def build_script(opu, M, x, geo):
    """返回 list[dict]，每个 dict 完整描述一帧要画什么。"""
    F = []

    def add(n, **kw):
        for i in range(n):
            F.append(dict(kw, _i=i, _n=n))

    N = opu.N
    y_cpu = M @ x

    # ---------------- 阶段 0：问题 ----------------
    for k, txt in enumerate([
        "我们要算的是最普通的一件事：矩阵 × 向量  y = M x —— 现代神经网络的大量算术工作都由这类线性变换组成，具体占比随模型和运行阶段而变。",
        f"M 是 {M.shape[0]}×{M.shape[1]} 的权重矩阵，x 是输入激活向量。电子芯片要做 {M.size} 次乘加。",
        "光子芯片的思路完全不同：不去「算」，而是让光自己走一遍，出口的光强就是答案。",
        "问题是——理想线性光学擅长做「正交／酉变换」和「衰减」，怎么表达一个任意实矩阵？答案是 SVD。",
    ]):
        add(9, stage=0, snip="svd", line=-1, narr=txt, intro=k)

    # ---------------- 阶段 1：SVD ----------------
    add(10, stage=1, snip="svd", line=6, narr="np.linalg.svd(M) —— 任意实矩阵都存在三段分解：M = U · Σ · V^T；重复或零奇异值时，分解不唯一。", svd=0)
    add(10, stage=1, snip="svd", line=6, narr="V^T 是正交变换：保持向量长度，但也可能包含反射，不一定只是旋转。光学上对应一张理想无损干涉网格。", svd=1)
    add(10, stage=1, snip="svd", line=9, narr=f"Σ 是对角阵，只有 {N} 个数（奇异值），负责各方向的缩放。光学上 = 一列可调衰减器。", svd=2)
    add(10, stage=1, snip="svd", line=6, narr="U 是第二个正交变换，对应另一张干涉网格。于是任意实矩阵 = 正交变换 → 缩放 → 正交变换。", svd=3)
    add(12, stage=1, snip="svd", line=9, narr=f"注意 σmax={opu.S_target[0]:.2f} > 1，而 VOA 只会衰减不会放大："
             f"整体归一化到 ≤1，增益放回电域补。", svd=4)

    # ---------------- 阶段 2：编译 ----------------
    A = opu.Vt_target.astype(complex).copy()
    elim = list(reversed(opu.vt_mzis))          # 消元顺序
    add(12, stage=2, snip="compile", line=2,
        narr="现在把 V^T「编译」成硬件参数：用一串 2×2 干涉，把矩阵一点点消成对角阵。",
        comp=dict(A=A.copy(), target=None, placed=0, z=None))
    # 按 decompose_unitary 里完全相同的顺序重放消元过程
    order = []
    for c in range(N - 1):
        for r in range(N - 1, c, -1):
            order.append((r - 1, r, c))
    placed = 0
    for (m, r, c), z in zip(order, elim, strict=True):
        add(6, stage=2, snip="compile", line=8,
            narr=f"轮到 A[{r},{c}]。它上面那个元素是 A[{m},{c}]。这两根波导要送进同一台 MZI。",
            comp=dict(A=A.copy(), target=(r, c), pair=(m, c), placed=placed, z=None))
        add(6, stage=2, snip="compile", line=12,
            narr=f"解方程得 θ={np.degrees(z.theta):6.1f}°  φ={np.degrees(z.phi):7.1f}°  "
                 f"—— θ 定分光比，φ 定相位，让两束光在下臂恰好反相。",
            comp=dict(A=A.copy(), target=(r, c), pair=(m, c), placed=placed, z=z))
        apply_T_left(A, m, z.theta, z.phi)     # 消元是左乘 T（不是 T^H）
        placed += 1
        add(6, stage=2, snip="compile", line=14,
            narr=f"相消完成 → A[{r},{c}] 变成 0。已确定 {placed}/{len(elim)} 台 MZI 的旋钮。",
            comp=dict(A=A.copy(), target=(r, c), pair=None, placed=placed, z=z, done=True))
    add(14, stage=2, snip="compile", line=17,
        narr=f"V^T 消成对角阵，剩下的 {N} 个相位交给输出移相器。U 同理，一共 "
             f"{len(opu.vt_mzis) + len(opu.u_mzis)} 台 MZI —— 芯片布局就此确定。",
        comp=dict(A=A.copy(), target=None, pair=None, placed=placed, z=None))

    # ---------------- 阶段 3~6：光传播 ----------------
    E = np.zeros(N, dtype=complex)
    E[:opu.n_in] = x
    stations = [(geo.x_in, E.copy())]

    add(12, stage=3, snip="forward", line=1,
        narr="编译完了，开始发光。输入向量 x 被编码到同一相干光源分出的 N 路复振幅上（负数 = 相位差 π）。",
        prop=dict(E=E.copy(), front=geo.x_in, act=[], st=list(stations)))

    def run_mesh(mzis, phases, x_ph, xfun, depth, stage, tag):
        nonlocal E, stations
        E = E * phases
        stations.append((x_ph, E.copy()))
        add(8, stage=stage, snip="forward", line=4 if tag == "V^T" else 12,
            narr=f"先过 {N} 个输出移相器：只改相位不改光强，给 {tag} 网格摆好初始相位。",
            prop=dict(E=E.copy(), front=x_ph, act=[], st=list(stations)))
        for layer in range(depth):
            here = [z for z in mzis if z.layer == layer]
            xl = xfun(layer)
            add(3, stage=stage, snip="forward", line=5 if tag == "V^T" else 13,
                narr=f"{tag} 网格第 {layer + 1}/{depth} 列：光同时抵达 {len(here)} 台 MZI；它们占用互不重叠的模式，因此可以并行作用。",
                prop=dict(E=E.copy(), front=xl - 0.35, act=[z.index for z in here], st=list(stations)))
            for z in here:
                before = E[[z.mode, z.mode + 1]].copy()
                add(4, stage=stage, snip="mzi", line=4,
                    narr=f"波导 {z.mode} 与 {z.mode + 1} 的两束光进入 MZI："
                         f"a={before[0].real:+.3f}{before[0].imag:+.3f}j, b={before[1].real:+.3f}{before[1].imag:+.3f}j",
                    prop=dict(E=E.copy(), front=xl, act=[z.index], st=list(stations)),
                    mzi=dict(z=z, before=before, after=None))
                apply_T_dagger_left(E, z.mode, z.theta, z.phi)
                after = E[[z.mode, z.mode + 1]].copy()
                add(5, stage=stage, snip="mzi", line=7,
                    narr=f"干涉输出：θ={np.degrees(z.theta):.1f}° 把能量按 sin/cos 重新分配 → "
                         f"{np.abs(before[0])**2 + np.abs(before[1])**2:.3f} 的总光强守恒地重排。",
                    prop=dict(E=E.copy(), front=xl, act=[z.index], st=list(stations)),
                    mzi=dict(z=z, before=before, after=after))
            stations.append((xl, E.copy()))
        add(4, stage=stage, snip="forward", line=6 if tag == "V^T" else 14,
            narr=f"{tag} 网格走完。这一整块光路做的事，数学上正好等于乘一个正交矩阵 —— 光强总和一分没丢。",
            prop=dict(E=E.copy(), front=xfun(depth - 1) + 0.4, act=[], st=list(stations)))

    run_mesh(opu.vt_mzis, opu.vt_phases, geo.x_vt_ph, geo.x_vt, geo.dv, 4, "V^T")

    E_before_voa = E.copy()
    E = E * opu.S_phys
    stations.append((geo.x_voa, E.copy()))
    add(16, stage=5, snip="forward", line=9,
        narr=f"Σ 衰减器：{N} 个 VOA 各自把对应通道乘上 {np.array2string(opu.S_phys, precision=3)}。"
             f"在理想无插损模型中，这是有意设置的衰减级 —— 矩阵奇异值的物理化身。",
        prop=dict(E=E.copy(), front=geo.x_voa, act=[], st=list(stations)),
        voa=dict(before=E_before_voa, after=E.copy(), s=opu.S_phys))

    run_mesh(opu.u_mzis, opu.u_phases, geo.x_u_ph, geo.x_u, geo.du, 6, "U")

    # ---------------- 阶段 7：探测 ----------------
    y_opt = np.real(E[:opu.n_out] * opu.gain)
    stations.append((geo.x_out, E.copy()))
    add(10, stage=7, snip="detect", line=1,
        narr="光抵达出口。相干探测取实部，再乘回电域增益 —— 就是全部的「计算」。",
        prop=dict(E=E.copy(), front=geo.x_out, act=[], st=list(stations)))
    add(14, stage=7, snip="detect", line=7,
        narr=f"对比 CPU：误差 {np.linalg.norm(y_opt - y_cpu):.1e} —— 逼近双精度浮点的机器精度，两者严格等价。",
        prop=dict(E=E.copy(), front=geo.x_out, act=[], st=list(stations)),
        result=dict(y_cpu=y_cpu, y_opt=y_opt, noisy=None))
    add(14, stage=7, snip="detect", line=7,
        narr="无源干涉核心不需要电子乘法器，传播延迟很低；完整系统仍需要激光、调制器、DAC/ADC、探测器与控制电路，本模型不估算系统延迟或能耗。",
        prop=dict(E=E.copy(), front=geo.x_out, act=[], st=list(stations)),
        result=dict(y_cpu=y_cpu, y_opt=y_opt, noisy=None))

    # ---------------- 阶段 8：简化非理想性 ----------------
    nz = NoiseModel(fab_theta=0.02, fab_phi=0.02, drift_theta=0.005, drift_phi=0.005,
                    mzi_loss_db=0.2, voa_rel_err=0.01, detector_snr_db=40)
    opu_n = PhotonicMatrixProcessor(M, noise=nz, seed=7)
    y_n = opu_n.read_coherent(x, ideal=False)
    rel = np.linalg.norm(y_n - y_cpu) / np.linalg.norm(y_cpu)
    add(12, stage=8, snip="noise", line=1,
        narr="但真实芯片没这么理想。这里先用每个输入样本独立的相位抖动做敏感度分析；真实热漂移通常还有慢时间相关和空间串扰。",
        prop=dict(E=E.copy(), front=geo.x_out, act=[], st=list(stations)),
        result=dict(y_cpu=y_cpu, y_opt=y_opt, noisy=y_n))
    add(12, stage=8, snip="noise", line=4,
        narr=f"再叠加每台 MZI 0.2dB 的统一插损。Reck 拓扑里空间模式参与器件数相差 "
             f"{opu.mode_mzi_count().max() - opu.mode_mzi_count().min()} 台；这是非均匀性的代理，不是端到端路径追踪。",
        prop=dict(E=E.copy(), front=geo.x_out, act=[], st=list(stations)),
        result=dict(y_cpu=y_cpu, y_opt=y_opt, noisy=y_n))
    add(20, stage=8, snip="noise", line=7,
        narr=f"结果：相对误差 {rel * 100:.1f}%，只剩约 {-np.log2(rel):.0f} bit 有效精度。"
             f"这是所选简化参数下的敏感度结果，不是对真实芯片 ENOB、能耗或延迟的预测。",
        prop=dict(E=E.copy(), front=geo.x_out, act=[], st=list(stations)),
        result=dict(y_cpu=y_cpu, y_opt=y_opt, noisy=y_n))
    return F


# --------------------------------------------------------------------------- #
# 绘制
# --------------------------------------------------------------------------- #
class Renderer:
    def __init__(self, opu, M, x, script, geo):
        self.opu, self.M, self.x, self.script, self.geo = opu, M, x, script, geo
        self.cw = None      # 等宽字符宽度（axes 坐标），首帧标定

        self.fig = plt.figure(figsize=(16, 9), facecolor=BG)
        gs = self.fig.add_gridspec(3, 2, width_ratios=[1.0, 1.98],
                                   height_ratios=[1.30, 0.95, 1.05],
                                   left=0.015, right=0.985, top=0.905, bottom=0.105,
                                   wspace=0.05, hspace=0.30)
        self.ax_code = self.fig.add_subplot(gs[:, 0])
        self.ax_chip = self.fig.add_subplot(gs[0, 1])
        self.ax_field = self.fig.add_subplot(gs[1, 1])
        self.ax_math = self.fig.add_subplot(gs[2, 1])
        for ax in (self.ax_code, self.ax_chip, self.ax_field, self.ax_math):
            ax.set_facecolor(PANEL)
            for sp in ax.spines.values():
                sp.set_color("#243049")

        self.t_title = self.fig.text(0.015, 0.955, "", fontsize=19, color=FG,
                                     family=CN, weight="bold", va="center")
        self.t_sub = self.fig.text(0.015, 0.925, "", fontsize=11, color=DIM,
                                   family=CN, va="center")
        self.t_narr = self.fig.text(0.015, 0.045, "", fontsize=13.5, color=FG,
                                    family=CN, va="center", wrap=True)
        self.t_help = self.fig.text(0.985, 0.955, "", fontsize=10, color=DIM,
                                    family=CN, ha="right", va="center")

    # ---------- 小工具 ----------
    def _charw(self, ax, fontsize):
        bb = ax.get_window_extent()
        return 0.5503 * fontsize * (self.fig.dpi / 72.0) / bb.width

    def draw_code(self, key, hl):
        ax = self.ax_code
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(PANEL)

        title, lines = SNIPPETS[key]
        fs = 10.4
        if self.cw is None:
            self.cw = self._charw(ax, fs)
        cw = self.cw

        ax.text(0.035, 0.972, title, fontsize=10, color=VIOLET, family=CN,
                va="top", weight="bold")

        top, step = 0.925, 0.0455
        kw_set = {"class", "def", "for", "in", "range", "return", "import", "if",
                  "else", "self", "np", "assert", "not", "and", "or", "lambda"}
        for i, ln in enumerate(lines):
            y = top - i * step
            if y < 0.02:
                break
            if i == hl:
                ax.add_patch(Rectangle((0.012, y - 0.019), 0.976, 0.036,
                                       facecolor="#1e3a5f", edgecolor=AMBER,
                                       lw=1.2, zorder=0))
                ax.text(0.002, y, ">", fontsize=10, color=AMBER, va="center", zorder=3)
            ax.text(0.030, y, f"{i + 1:>2}", fontsize=8.6, color="#4a5878",
                    family=MONO, va="center", ha="right", zorder=2)

            code, _, comment = ln.partition("#")
            x0 = 0.052
            col = FG if i == hl else "#c3cddd"
            # 关键字上色
            j = 0
            for tok in code.split(" "):
                if tok:
                    c = PINK if tok.strip("():,.[]") in kw_set else col
                    ax.text(x0 + j * cw, y, tok, fontsize=fs, color=c,
                            family=MONO, va="center", zorder=2)
                j += len(tok) + 1
            if comment:
                ax.text(x0 + len(code) * cw, y, "#" + comment, fontsize=fs,
                        color=GREEN if i == hl else "#4d7c5a", family=MONO,
                        va="center", zorder=2)

        # 代码短的时候，下方留白拿来放芯片规格 + 相位色标（自下而上排版，保证放得下）
        y_end = top - len(lines) * step
        self.draw_code_footer(ax, y_end, full=y_end > 0.44)

    def draw_code_footer(self, ax, y_end, full):
        opu, g = self.opu, self.geo

        # --- 底部固定：相位色标 ---
        yb = 0.085
        if y_end > 0.20:
            for i in range(64):
                f = i / 64
                ax.add_patch(Rectangle((0.05 + f * 0.60, yb - 0.017),
                                       0.60 / 64 + 0.003, 0.034,
                                       facecolor=PHASE_CMAP(f), lw=0))
            for f, lab in [(0.0, "-180°"), (0.5, "0°"), (1.0, "+180°")]:
                ax.text(0.05 + f * 0.60, yb - 0.040, lab, fontsize=7.5,
                        color=DIM, ha="center", va="center")
            ax.text(0.05, yb + 0.042, "颜色 = 光的相位", fontsize=9,
                    color=DIM, family=CN, va="center")
        if not full:
            return

        # --- 其上：芯片规格 ---
        rows = [
            ("波导 / 模式数", f"{opu.N}"),
            ("MZI 总数", f"{len(opu.vt_mzis) + len(opu.u_mzis)} = 2×N(N-1)/2"),
            ("网格深度", f"V^T {g.dv} 列 + U {g.du} 列"),
            ("空间模式参与数", f"{opu.mode_mzi_count().tolist()}"),
            ("奇异值 σ", np.array2string(opu.S_target, precision=2)),
        ]
        y0 = yb + 0.10
        for k, (a, b) in enumerate(rows):
            yy = y0 + (len(rows) - 1 - k) * 0.037
            ax.text(0.05, yy, a, fontsize=8.8, color=DIM, family=CN, va="center")
            ax.text(0.45, yy, b, fontsize=8.8, color="#c3cddd", family=MONO, va="center")
        yh = y0 + len(rows) * 0.037
        ax.text(0.04, yh, "本次编译出的芯片", fontsize=10.2, color=VIOLET,
                family=CN, va="center")
        ax.plot([0.03, 0.97], [yh + 0.035, yh + 0.035], color="#243049", lw=1)

    def _cellw(self, ax, ch):
        """把「单元格高度(axes 比例)」换算成等像素正方形的宽度。"""
        bb = ax.get_window_extent()
        return ch * bb.height / bb.width

    def draw_matrix(self, ax, Mx, x0, y0, ch, label, *, vmax=None,
                    highlight=None, pair=None, zeros=True, cmap="RdBu_r",
                    fontsize=7.2):
        """在 ax 上画矩阵热图。ch = 单元格高度；宽度自动按宽高比补正成正方形。"""
        Mx = np.asarray(Mx)
        if Mx.ndim == 1:
            Mx = Mx[:, None]
        r, c = Mx.shape
        v = np.real(Mx)
        vmax = vmax or max(np.abs(v).max(), 1e-9)
        cm = plt.get_cmap(cmap)
        cw = self._cellw(ax, ch)
        for i in range(r):
            for j in range(c):
                val = v[i, j]
                is0 = zeros and abs(val) < 1e-12
                fc = "#0f1729" if is0 else cm(0.5 + 0.5 * val / vmax)
                ax.add_patch(Rectangle((x0 + j * cw, y0 - (i + 1) * ch), cw, ch,
                                       facecolor=fc, edgecolor="#0b1020", lw=0.7))
                ax.text(x0 + (j + .5) * cw, y0 - (i + .5) * ch,
                        "0" if is0 else f"{val:.2f}", ha="center", va="center",
                        fontsize=fontsize, color=text_on(fc))
        for box, col, ls in ((highlight, AMBER, "-"), (pair, CYAN, "--")):
            if box:
                i, j = box
                ax.add_patch(Rectangle((x0 + j * cw, y0 - (i + 1) * ch), cw, ch,
                                       fill=False, edgecolor=col, lw=2.3, ls=ls, zorder=5))
        ax.text(x0 + c * cw / 2, y0 + 0.03, label, ha="center", va="bottom",
                fontsize=10.5, color=FG, family=CN)
        return cw

    # ---------- 芯片 ----------
    def draw_chip(self, fr):
        ax, g, opu = self.ax_chip, self.geo, self.opu
        ax.clear()
        ax.set_xlim(-0.15, g.x_out + 0.85)
        ax.set_ylim(*g.ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(PANEL)
        N = opu.N

        # 区块底色
        for xa, xb, c, name in [
            (g.x_vt_ph - 0.55, g.x_vt(g.dv - 1) + 0.55, "#16233d", "V$^T$  酉变换网格"),
            (g.x_voa - 0.45, g.x_voa + 0.45, "#2a1f3d", "Σ  衰减器"),
            (g.x_u_ph - 0.5, g.x_u(g.du - 1) + 0.55, "#16233d", "U  酉变换网格"),
        ]:
            ax.add_patch(Rectangle((xa, -0.62), xb - xa, N - 0.05,
                                   facecolor=c, edgecolor="#243049", lw=1, zorder=0))
            ax.text((xa + xb) / 2, -0.78, name, ha="center", va="center",
                    fontsize=9.5, color=DIM, family=CN)

        # 波导：底色
        for m in range(N):
            ax.plot([g.x_in, g.x_out], [g.y(m)] * 2, color=WG_IDLE, lw=2.6,
                    solid_capstyle="round", zorder=1)

        p = fr.get("prop")
        if p:
            st, front = p["st"], p["front"]
            # 已通过的区段按当时的光场发光
            for k in range(len(st)):
                xa, Ek = st[k]
                xb = st[k + 1][0] if k + 1 < len(st) else front
                xb = min(xb, front)
                if xb <= xa:
                    continue
                for m in range(N):
                    a = abs(Ek[m])
                    if a < 1e-3:
                        continue
                    w = 2.2 + 5.0 * min(a / 1.2, 1.0)
                    ax.plot([xa, xb], [g.y(m)] * 2, color=phase_color(Ek[m]),
                            lw=w, alpha=0.35, solid_capstyle="round", zorder=2)
                    ax.plot([xa, xb], [g.y(m)] * 2, color=phase_color(Ek[m]),
                            lw=w * 0.42, alpha=0.95, solid_capstyle="round", zorder=3)
            # 波前光斑
            E = p["E"]
            for m in range(N):
                a = abs(E[m])
                if a < 1e-3:
                    continue
                ax.add_patch(Circle((front, g.y(m)), 0.055 + 0.20 * min(a / 1.2, 1),
                                    facecolor=phase_color(E[m]), alpha=0.30, zorder=4))
                ax.add_patch(Circle((front, g.y(m)), 0.03 + 0.10 * min(a / 1.2, 1),
                                    facecolor="white", alpha=0.92, zorder=5))

        act = set(p["act"]) if p else set()
        placed = fr.get("comp", {}).get("placed", None)

        def mzi_box(xc, m, on, ghost=False):
            yb = g.y(m + 1)
            ec = AMBER if on else ("#3d4a68" if not ghost else "#222c46")
            fc = "#3d2f12" if on else ("#1b2440" if not ghost else "#141c30")
            ax.add_patch(FancyBboxPatch((xc - 0.30, yb - 0.16), 0.60, 1.32,
                                        boxstyle="round,pad=0.03,rounding_size=0.10",
                                        facecolor=fc, edgecolor=ec,
                                        lw=2.0 if on else 1.0, zorder=6))
            ax.text(xc, yb + 0.5, "θφ", ha="center", va="center", fontsize=7.5,
                    color=AMBER if on else "#5b6b8f", zorder=7)

        # 编译阶段：还没定下旋钮的 MZI 画成「虚影」。注意消元顺序是 vt_mzis 的逆序，
        # 所以先确定的是列表末尾那些。
        stage = fr.get("stage")
        nv = len(opu.vt_mzis)
        for k, z in enumerate(opu.vt_mzis):
            if stage is not None and stage < 2:
                ghost = True
            elif stage == 2:
                ghost = k < nv - (placed or 0)
            else:
                ghost = False
            mzi_box(g.x_vt(z.layer), z.mode, (z.index in act) and stage == 4, ghost)
        for z in opu.u_mzis:
            mzi_box(g.x_u(z.layer), z.mode, (z.index in act) and stage == 6,
                    ghost=(stage is not None and stage <= 2))

        # 移相器 / VOA
        for xc, lab in [(g.x_vt_ph, "φ"), (g.x_u_ph, "φ")]:
            for m in range(N):
                ax.add_patch(Rectangle((xc - 0.13, g.y(m) - 0.13), 0.26, 0.26,
                                       facecolor="#1b2440", edgecolor="#4b5a80",
                                       lw=1.0, zorder=6))
                ax.text(xc, g.y(m), lab, ha="center", va="center", fontsize=7.5,
                        color="#8ea2c9", zorder=7)
        for m in range(N):
            s = opu.S_phys[m]
            ax.add_patch(Rectangle((g.x_voa - 0.17, g.y(m) - 0.17), 0.34, 0.34,
                                   facecolor=plt.get_cmap("magma")(0.25 + 0.55 * s),
                                   edgecolor=VIOLET, lw=1.3, zorder=6))
            ax.text(g.x_voa, g.y(m) - 0.42, f"{s:.2f}", ha="center", va="center",
                    fontsize=7.2, color=VIOLET, zorder=7)

        # 端口
        for m in range(N):
            ax.text(g.x_in - 0.28, g.y(m), f"x{m}", ha="right", va="center",
                    fontsize=9.5, color=CYAN)
            if m < opu.n_out:
                ax.text(g.x_out + 0.30, g.y(m), f"y{m}", ha="left", va="center",
                        fontsize=9.5, color=GREEN)
        ax.text(g.x_in - 0.28, N - 0.35, "相干光输入", ha="right", va="center",
                fontsize=9, color=DIM, family=CN)
        ax.text(g.x_out + 0.30, N - 0.35, "光电探测", ha="left", va="center",
                fontsize=9, color=DIM, family=CN)

    # ---------- MZI 参数表（编译阶段用） ----------
    def draw_mzi_table(self, fr):
        """把「矩阵 → 硬件旋钮」这一步显式画出来：一台 MZI 一行 θ/φ。"""
        ax = self.ax_field
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        opu = self.opu
        placed = fr.get("comp", {}).get("placed", 0)
        elim = list(reversed(opu.vt_mzis))          # 按消元顺序列出
        n = len(elim)

        ax.text(0.03, 0.90, "V$^T$ 网格的 MZI 参数表　（矩阵 → 硬件旋钮）",
                fontsize=11, color=VIOLET, family=CN, va="center")
        heads = ["MZI", "占用波导", "θ  内相移", "φ  外相移", "分光比 cos²θ"]
        cols = [0.05, 0.17, 0.34, 0.52, 0.72]
        for c, h in zip(cols, heads, strict=True):
            ax.text(c, 0.74, h, fontsize=9.5, color=DIM, family=CN, va="center")
        ax.plot([0.03, 0.90], [0.68, 0.68], color="#2b3550", lw=1)

        for i, z in enumerate(elim):
            y = 0.61 - i * 0.089
            if y < 0.04:
                break
            on = i < placed
            col = FG if on else "#39435e"
            vals = [f"#{i}", f"{z.mode} / {z.mode + 1}",
                    f"{np.degrees(z.theta):8.2f}°" if on else "— — —",
                    f"{np.degrees(z.phi):9.2f}°" if on else "— — —",
                    f"{np.cos(z.theta)**2:.3f}" if on else "—"]
            if i == placed - 1:
                ax.add_patch(Rectangle((0.03, y - 0.042), 0.87, 0.084,
                                       facecolor="#1e3a5f", edgecolor=AMBER, lw=1.1,
                                       zorder=0))
            for c, v in zip(cols, vals, strict=True):
                ax.text(c, y, v, fontsize=9.5, family=MONO,
                        color=AMBER if i == placed - 1 else col, va="center", zorder=2)
        ax.text(0.03, 0.06, f"已锁定 {placed}/{n} 台　（U 网格另有 {len(opu.u_mzis)} 台，同理）",
                fontsize=9.5, color=GREEN if placed == n else DIM, family=CN, va="center")

    # ---------- 光场柱状图 ----------
    def draw_field(self, fr):
        ax = self.ax_field
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=DIM, labelsize=8.5)
        for sp in ax.spines.values():
            sp.set_color("#243049")
        p = fr.get("prop")
        if not p:
            # 还没发光的阶段，这块面板拿来展示正在生成的 MZI 参数表
            self.draw_mzi_table(fr)
            return
        E = p["E"]
        N = len(E)
        idx = np.arange(N)
        amp = np.abs(E)
        ax.bar(idx, amp, width=0.62, color=[phase_color(z) for z in E],
               edgecolor="#0b1020", lw=0.8, zorder=2)
        ax.plot(idx, np.real(E), "o--", color="white", lw=1.2, ms=4.5,
                alpha=0.8, zorder=3)
        ax.axhline(0, color="#3a465f", lw=1)
        for i in range(N):
            ax.text(i, amp[i] + 0.045, f"{np.abs(E[i]):.2f}∠{np.degrees(np.angle(E[i])):.0f}°",
                    ha="center", va="bottom", fontsize=7.6, color=FG)
        ax.set_xticks(idx)
        ax.set_xticklabels([f"波导 {i}" for i in idx], family=CN, fontsize=8.5)
        ax.set_ylim(min(-0.15, np.real(E).min() * 1.35), max(amp.max() * 1.55, 0.5))
        ax.set_ylabel("光场 |E|", color=DIM, family=CN, fontsize=9.5)
        ax.set_title(f"当前光场（柱高=振幅，颜色=相位，白点虚线=实部 Re(E)）   "
                     f"总光强 Σ|E|² = {np.sum(amp**2):.4f}",
                     color=FG, family=CN, fontsize=10.5, pad=6)

    # ---------- 数学面板 ----------
    def draw_math(self, fr):
        ax = self.ax_math
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(PANEL)
        opu, M, x = self.opu, self.M, self.x
        N = opu.N
        ch = min(0.185, 0.70 / N)               # 单元格高度
        cw = self._cellw(ax, ch)                # 等像素正方形对应的宽度
        w = N * cw                              # 一个 N 阶方阵的宽度
        ytop = 0.88
        ymid = ytop - N * ch / 2

        st = fr["stage"]
        if st == 0:
            k = fr.get("intro", 0)
            self.draw_matrix(ax, M, 0.045, ytop, ch, "M  权重矩阵", zeros=False)
            bx = 0.045 + M.shape[1] * cw
            if k >= 1:
                ax.text(bx + 0.022, ymid, "×", fontsize=16, color=FG, va="center", ha="center")
                self.draw_matrix(ax, x, bx + 0.045, ytop, ch, "x  输入", zeros=False)
                bx = bx + 0.045 + cw
            if k >= 2:
                ax.text(bx + 0.022, ymid, "=", fontsize=16, color=FG, va="center", ha="center")
                self.draw_matrix(ax, M @ x, bx + 0.045, ytop, ch, "y  输出", zeros=False)
            if k >= 3:
                ax.text(0.46, 0.62, "M  =  U · Σ · V$^T$", fontsize=22, color=AMBER,
                        family=CN, va="center")
                ax.text(0.46, 0.34, "正交变换  →  缩放  →  正交变换\n理想线性光学可以实现这三部分",
                        fontsize=12.5, color=DIM, family=CN, va="center", linespacing=1.6)
            else:
                ax.text(0.46, 0.5, f"电子芯片：{M.size} 次乘法 + "
                                   f"{M.size - M.shape[0]} 次加法\n"
                                   "光子芯片：0 次 —— 让光自己走一遍",
                        fontsize=12.5, color=DIM, family=CN, va="center", linespacing=1.6)

        elif st == 1:
            k = fr.get("svd", 0)
            self.draw_matrix(ax, M, 0.03, ytop, ch, "M", zeros=False)
            ax.text(0.03 + w + 0.021, ymid, "=", fontsize=15, color=FG,
                    va="center", ha="center")
            xs = 0.03 + w + 0.042
            self.draw_matrix(ax, opu.U_target, xs, ytop, ch, "U  (正交)", zeros=False)
            xs2 = xs + w + 0.042
            ax.text(xs + w + 0.021, ymid, "·", fontsize=15, color=FG, va="center", ha="center")
            self.draw_matrix(ax, np.diag(opu.S_target), xs2, ytop, ch, "Σ  (对角)")
            xs3 = xs2 + w + 0.042
            ax.text(xs2 + w + 0.021, ymid, "·", fontsize=15, color=FG, va="center", ha="center")
            self.draw_matrix(ax, opu.Vt_target, xs3, ytop, ch, "V$^T$  (正交)", zeros=False)
            # 高亮当前正在讲的那一块
            spans = {1: (xs3, w, CYAN), 2: (xs2, w, VIOLET), 3: (xs, w, GREEN)}
            if k in spans:
                sx, sw, sc = spans[k]
                ax.add_patch(Rectangle((sx - 0.009, ytop - N * ch - 0.025),
                                       sw + 0.018, N * ch + 0.105, fill=False,
                                       edgecolor=sc, lw=2.2, zorder=6))
            notes = {
                0: ("任意实矩阵都存在这样的 SVD；重复或零奇异值时分解并不唯一", FG),
                1: ("正交矩阵 V$^T$·V = I：保持长度，可包含旋转与反射  →  对应理想无损干涉网格", CYAN),
                2: ("Σ 只有对角线非零  →  每个通道各自独立衰减，一列 VOA 就够了", VIOLET),
                3: ("U 是第二个正交变换  →  第二张干涉网格。三块合起来 = 任意实矩阵", GREEN),
                4: (f"σ = {np.array2string(opu.S_target, precision=3)}   →   "
                    f"σ/σmax = {np.array2string(opu.S_phys, precision=3)}   "
                    f"电域再补回 ×{opu.gain:.3f}", VIOLET),
            }
            txt, col = notes[k]
            ax.text(0.03, 0.10, txt, fontsize=11, color=col, family=CN, va="center")

        elif st == 2:
            c = fr["comp"]
            self.draw_matrix(ax, c["A"], 0.035, ytop, ch, "消元中的 V$^T$",
                             highlight=c.get("target"), pair=c.get("pair"))
            z = c.get("z")
            bx = 0.035 + w + 0.055
            ax.text(bx, ytop - 0.02, "消元规则：让下面那个元素干涉相消", fontsize=11,
                    color=VIOLET, family=CN, va="top")
            ax.text(bx, ytop - 0.30, "e$^{iφ}$·cosθ·x  +  sinθ·y  =  0",
                    fontsize=14, color=FG, va="center")
            ax.text(bx, ytop - 0.58, "橙框 = 要消成 0 的元素\n蓝虚框 = 与它配对的上一行元素",
                    fontsize=10, color=DIM, family=CN, va="center", linespacing=1.5)
            if z is not None:
                T = mzi_transfer_matrix(z.theta, z.phi)
                ax.text(0.63, ytop - 0.02, f"θ = {np.degrees(z.theta):7.2f}°\n"
                                           f"φ = {np.degrees(z.phi):8.2f}°",
                        fontsize=12.5, color=AMBER, family=MONO, va="top", linespacing=1.7)
                ax.text(0.63, ytop - 0.52,
                        f"T = [{T[0,0].real:+.2f}{T[0,1].real:+.2f}]\n"
                        f"    [{T[1,0].real:+.2f}{T[1,1].real:+.2f}]",
                        fontsize=9.5, color="#9fb3d9", family=MONO, va="center")
            ax.text(0.035, 0.06, f"已确定 {c['placed']}/{len(opu.vt_mzis)} 台 MZI 的旋钮",
                    fontsize=10.5, color=GREEN, family=CN)

        elif fr.get("mzi"):
            d = fr["mzi"]
            z, before, after = d["z"], d["before"], d["after"]
            T = mzi_transfer_matrix(z.theta, z.phi)
            ax.text(0.04, 0.86, f"MZI #{z.index}   波导 {z.mode} / {z.mode+1}   "
                                f"θ={np.degrees(z.theta):.1f}°  φ={np.degrees(z.phi):.1f}°",
                    fontsize=12, color=AMBER, family=CN)
            ax.text(0.04, 0.62, f"a = {before[0].real:+.4f} {before[0].imag:+.4f}j\n"
                                f"b = {before[1].real:+.4f} {before[1].imag:+.4f}j",
                    fontsize=11.5, color=CYAN, family=MONO, va="center")
            ax.text(0.30, 0.62, "---[ MZI ]--->", fontsize=13, color=DIM, va="center")
            if after is not None:
                ax.text(0.46, 0.62, f"a' = {after[0].real:+.4f} {after[0].imag:+.4f}j\n"
                                    f"b' = {after[1].real:+.4f} {after[1].imag:+.4f}j",
                        fontsize=11.5, color=GREEN, family=MONO, va="center")
                pin = np.sum(np.abs(before) ** 2)
                pout = np.sum(np.abs(after) ** 2)
                ax.text(0.04, 0.30, f"能量检查  |a|²+|b|² = {pin:.6f}   →   "
                                    f"|a'|²+|b'|² = {pout:.6f}   (幺正 = 一个光子都没丢)",
                        fontsize=10.5, color=GREEN if abs(pin - pout) < 1e-9 else RED,
                        family=CN)
            ax.text(0.68, 0.86, "传输矩阵 T†", fontsize=10.5, color=VIOLET, family=CN)
            ax.text(0.68, 0.60, f"[{T[0,0]:+.3f}  {T[0,1]:+.3f}]\n"
                                f"[{T[1,0]:+.3f}  {T[1,1]:+.3f}]",
                    fontsize=9.5, color="#9fb3d9", family=MONO, va="center")
            ax.text(0.04, 0.11, "θ 控制两条臂的分光比（能量往哪走），φ 控制相对相位（干涉是相长还是相消）",
                    fontsize=10, color=DIM, family=CN)

        elif fr.get("voa"):
            d = fr["voa"]
            b = np.abs(d["before"])
            a = np.abs(d["after"])
            w = 0.9 / (N + 1)
            for i in range(N):
                xb = 0.06 + i * (0.86 / N)
                scale = 0.5 / max(b.max(), 1e-9)
                ax.add_patch(Rectangle((xb, 0.30), w * 0.42, b[i] * scale,
                                       facecolor=CYAN, alpha=0.9, edgecolor="#0b1020", lw=0.8))
                ax.add_patch(Rectangle((xb + w * 0.46, 0.30), w * 0.42, a[i] * scale,
                                       facecolor=VIOLET, alpha=0.9, edgecolor="#0b1020", lw=0.8))
                ax.text(xb + w * 0.44, 0.20, f"×{d['s'][i]:.3f}", ha="center",
                        fontsize=9.5, color=VIOLET, family=MONO)
                ax.text(xb + w * 0.44, 0.11, f"波导 {i}", ha="center", fontsize=9,
                        color=DIM, family=CN)
            ax.text(0.06, 0.90, "Σ 衰减器：进入（青） vs 出射（紫）", fontsize=12,
                    color=FG, family=CN)
            ax.text(0.06, 0.80, f"总光强 {np.sum(b**2):.4f}  →  {np.sum(a**2):.4f}"
                                f"   （唯一主动丢能量的一级）",
                    fontsize=10.5, color=VIOLET, family=CN)

        elif fr.get("result"):
            d = fr["result"]
            n = len(d["y_cpu"])
            wd = 0.86 / n
            mx = max(np.abs(d["y_cpu"]).max(), 1e-9)
            base, half = 0.50, 0.225           # 零线位置 / 最大半高
            series = [(d["y_cpu"], "#64748b"), (d["y_opt"], GREEN)]
            if d["noisy"] is not None:
                series.append((d["noisy"], RED))
            bw = wd * 0.86 / len(series)
            for i in range(n):
                x0 = 0.07 + i * wd
                for k, (arr, col) in enumerate(series):
                    h = float(arr[i]) / mx * half
                    ax.add_patch(Rectangle((x0 + k * bw, base), bw * 0.84, h,
                                           facecolor=col, alpha=0.92))
                ax.text(x0 + wd * 0.42, 0.17, f"y{i}", ha="center", fontsize=9.5, color=DIM)
                ax.text(x0 + wd * 0.42, 0.07, f"{d['y_cpu'][i]:+.3f}", ha="center",
                        fontsize=8.5, color="#64748b", family=MONO)
            ax.plot([0.05, 0.96], [base, base], color="#3a465f", lw=1)
            e_id = np.linalg.norm(d["y_opt"] - d["y_cpu"])
            ax.text(0.05, 0.94, "CPU（灰）  vs  理想光子（绿）" +
                    ("  vs  简化非理想（红）" if d["noisy"] is not None else ""),
                    fontsize=11.5, color=FG, family=CN, va="center")
            ax.text(0.05, 0.855, f"理想绝对误差 = {e_id:.2e}", fontsize=10.5,
                    color=GREEN, family=CN, va="center")
            if d["noisy"] is not None:
                rel = np.linalg.norm(d["noisy"] - d["y_cpu"]) / np.linalg.norm(d["y_cpu"])
                ax.text(0.45, 0.855, f"非理想相对误差 = {rel*100:.2f}%   ≈ "
                                     f"{-np.log2(rel):.1f} bit 有效精度",
                        fontsize=10.5, color=RED, family=CN, va="center")
        elif st == 3:
            # 输入编码：实数怎么变成一束激光
            ax.text(0.04, 0.90, "输入编码：实数  →  激光的复振幅", fontsize=12,
                    color=CYAN, family=CN, va="center")
            cols = [0.06, 0.20, 0.34, 0.47]
            for c, h in zip(cols, ["波导", "x$_i$", "振幅 |E|", "相位"], strict=True):
                ax.text(c, 0.74, h, fontsize=9.5, color=DIM, family=CN, va="center")
            ax.plot([0.05, 0.60], [0.68, 0.68], color="#2b3550", lw=1)
            for i in range(min(opu.n_in, 6)):
                yy = 0.60 - i * 0.115
                v = float(x[i])
                for c, s in zip(cols, [f"{i}", f"{v:+.2f}", f"{abs(v):.2f}",
                                       "180°" if v < 0 else "0°"],
                                 strict=True):
                    ax.text(c, yy, s, fontsize=10, family=MONO, va="center",
                            color=RED if v < 0 else CYAN)
            ax.text(0.64, 0.52, "负数不是「负的光」——\n"
                                "光强永远非负，负号被编码成\n"
                                "把相位推 π（图上显示为红色）。\n\n"
                                "干涉时，相位差 π 的两束光\n"
                                "相遇就会相消，减法就是这么来的。",
                    fontsize=10.5, color=DIM, family=CN, va="center", linespacing=1.75)

        elif fr.get("prop"):
            # 传播途中但不在某台具体 MZI 上：展示光强台账
            E = fr["prop"]["E"]
            p_in = float(np.sum(np.abs(x) ** 2))
            p_now = float(np.sum(np.abs(E) ** 2))
            ax.text(0.04, 0.86, "光强台账", fontsize=12, color=VIOLET, family=CN, va="center")
            rows = [("注入芯片", f"Σ|E|² = {p_in:.4f}", CYAN),
                    ("当前位置", f"Σ|E|² = {p_now:.4f}   ({p_now / p_in * 100:5.1f}%)",
                     GREEN if p_now > 0.99 * p_in else VIOLET)]
            for k, (a, b, c) in enumerate(rows):
                ax.text(0.06, 0.66 - k * 0.16, a, fontsize=10.5, color=DIM,
                        family=CN, va="center")
                ax.text(0.24, 0.66 - k * 0.16, b, fontsize=11, color=c,
                        family=MONO, va="center")
            ax.text(0.04, 0.24,
                    "酉变换网格（V^T / U）是无损的：光强只在波导之间重新分配，总量不变。\n"
                    "整条光路里唯一主动丢能量的，只有中间那一列 Σ 衰减器。",
                    fontsize=10.5, color=DIM, family=CN, va="center", linespacing=1.7)
        else:
            ax.text(0.5, 0.5, "", ha="center")

    # ---------- 主渲染 ----------
    def render(self, k, live=True):
        fr = self.script[k]
        st = fr["stage"]
        self.draw_code(fr["snip"], fr["line"])
        self.draw_chip(fr)
        self.draw_field(fr)
        self.draw_math(fr)

        bar = "   ".join(f"[{s}]" if i == st else f" {s} "
                         for i, s in enumerate(STAGES))
        self.t_title.set_text(f"MZI 网格光计算模拟器  ·  {STAGES[st]}")
        self.t_sub.set_text(bar)
        self.t_narr.set_text(fr["narr"])
        self.t_help.set_text(
            f"帧 {k+1}/{len(self.script)}   空格=暂停  ←→=单步  ,.=换阶段  r=重播"
             if live else f"帧 {k+1}/{len(self.script)}")
        return []


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(prog="python -m photonic_mzi",
                                 description="MZI 网格光计算教学动画")
    ap.add_argument("--save", metavar="FILE", help="导出为 GIF（如 mzi.gif）")
    ap.add_argument("--fps", type=int, default=14)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--stride", type=int, default=1,
                    help="导出时每 N 帧取 1 帧，用来压 GIF 体积")
    ap.add_argument("-n", type=int, default=0, help="改用 n×n 随机矩阵")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.n:
        rng = np.random.default_rng(args.seed)
        M = np.round(rng.uniform(-1, 1, (args.n, args.n)), 2)
        x = np.round(rng.uniform(-1, 1, args.n), 2)
    else:
        M = np.array([[0.65, -0.42, 0.18, 0.91],
                      [-0.12, 0.88, -0.54, 0.33],
                      [0.47, 0.21, -0.76, 0.15],
                      [-0.83, 0.35, 0.62, -0.49]])
        x = np.array([1.0, 0.5, -0.8, 0.2])

    opu = PhotonicMatrixProcessor(M, seed=args.seed)
    if opu.N < 2:
        raise SystemExit("N 至少要 2：1x1 矩阵没有 MZI，网格是空的，没什么可动画的。")
    geo = ChipLayout(opu)
    script = build_script(opu, M, x, geo)
    print(f"矩阵 {M.shape}，MZI {len(opu.vt_mzis)+len(opu.u_mzis)} 台，"
          f"脚本 {len(script)} 帧 ≈ {len(script)/args.fps:.1f} 秒")

    r = Renderer(opu, M, x, script, geo)

    if args.save:
        import os

        from matplotlib.animation import FuncAnimation, PillowWriter
        keep = list(range(0, len(script), args.stride))
        anim = FuncAnimation(r.fig, lambda k: r.render(k, live=False),
                             frames=keep, interval=1000 // args.fps, blit=False)
        print(f"正在导出 {args.save}（{len(keep)} 帧, dpi={args.dpi}）...")
        anim.save(args.save, writer=PillowWriter(fps=max(1, args.fps // args.stride)),
                  dpi=args.dpi)
        print(f"完成: {args.save}  ({os.path.getsize(args.save) / 1e6:.1f} MB)")
        return

    from matplotlib.animation import FuncAnimation
    state = {"i": 0, "paused": False}

    def step(_):
        if not state["paused"]:
            state["i"] = (state["i"] + 1) % len(script)
        return r.render(state["i"])

    def jump_stage(d):
        cur = script[state["i"]]["stage"]
        tgt = min(max(cur + d, 0), len(STAGES) - 1)
        for j, f in enumerate(script):
            if f["stage"] == tgt:
                state["i"] = j
                break

    def on_key(ev):
        if ev.key == " ":
            state["paused"] = not state["paused"]
        elif ev.key == "right":
            state["paused"] = True
            state["i"] = (state["i"] + 1) % len(script)
        elif ev.key == "left":
            state["paused"] = True
            state["i"] = (state["i"] - 1) % len(script)
        elif ev.key in (".", ">"):
            state["paused"] = True
            jump_stage(1)
        elif ev.key in (",", "<"):
            state["paused"] = True
            jump_stage(-1)
        elif ev.key == "r":
            state["i"] = 0
            state["paused"] = False
        r.render(state["i"])
        r.fig.canvas.draw_idle()

    r.fig.canvas.mpl_connect("key_press_event", on_key)
    _anim = FuncAnimation(r.fig, step, interval=1000 // args.fps,
                          blit=False, cache_frame_data=False)
    plt.show()

