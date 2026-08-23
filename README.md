<div align="center">

# photonic-mzi

**MZI 网格光计算模拟器 —— 把任意矩阵编译成一张干涉仪网格，让光走一趟就算完矩阵乘法**

[![CI](https://github.com/rockbrainconz/photonic-mzi/actions/workflows/ci.yml/badge.svg)](https://github.com/rockbrainconz/photonic-mzi/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

模拟 Lightmatter Envise / 曦智 PACE 这类集成光子芯片的底层计算原理，
**附一个逐行代码 + 逐器件光传播的教学动画**。

```
输入 [x] ──▶ [ Vᵀ 酉变换 MZI 网格 ] ──▶ [ Σ 光衰减器 ] ──▶ [ U 酉变换 MZI 网格 ] ──▶ [y] 探测器
```

</div>

---

## 这个动画在讲什么

<div align="center">
<img src="docs/images/demo.gif" alt="MZI 网格光计算教学动画" width="100%">
</div>

屏幕**左边是正在执行的代码**（高亮当前行），**右边是这一行在光子芯片上物理发生了什么**。
颜色统一编码光的相位（红 0°、青 180°），柱高是振幅 —— 所以「负数怎么用光表示」
「这台干涉仪是相长还是相消」，看颜色就知道。

```bash
pip install -e ".[viz]"
python -m photonic_mzi
```

| 键 | 作用 |
|:--:|---|
| `空格` | 暂停 / 继续 |
| `←` `→` | **单步回退 / 前进**（暂停时最有用） |
| `,` `.` | 跳到上一 / 下一阶段 |
| `r` | 从头重播 |

整个流程拆成 9 个阶段：

| 阶段 | 内容 |
|---|---|
| 0 问题 | `y = M x` 为什么值得用光去算 |
| 1 SVD 分解 | `M = U · Σ · Vᵀ`，为什么任意矩阵都能拆成「旋转→缩放→旋转」 |
| 2 编译 MZI | 逐个元素消元，看 θ/φ 参数表怎么一行行填满、芯片上的 MZI 怎么一台台点亮 |
| 3 光注入 | 输入向量怎么编码成复振幅（负数 = 相位差 π） |
| 4 Vᵀ 网格 | 光波前逐列推进，每台 MZI 内部的 `a,b → a',b'` 干涉过程 + 能量守恒检查 |
| 5 Σ 衰减 | 唯一主动损失能量的一级，奇异值的物理化身 |
| 6 U 网格 | 同上 |
| 7 探测输出 | 与 CPU 对答案，误差 `8.1e-16` |
| 8 噪声 | 加上制造误差 / 热漂移 / 插损 / 探测噪声后，精度掉到多少 |

<table>
<tr>
<td width="50%"><img src="docs/images/stage2-compile.png" alt="编译阶段"><br><sub><b>阶段 2</b>：矩阵消元 ↔ MZI 参数表同步填充</sub></td>
<td width="50%"><img src="docs/images/stage4-interference.png" alt="干涉阶段"><br><sub><b>阶段 4</b>：单台 MZI 的干涉与能量守恒</sub></td>
</tr>
<tr>
<td width="50%"><img src="docs/images/stage1-svd.png" alt="SVD 阶段"><br><sub><b>阶段 1</b>：M = U · Σ · Vᵀ</sub></td>
<td width="50%"><img src="docs/images/stage8-noise.png" alt="噪声阶段"><br><sub><b>阶段 8</b>：CPU / 理想光子 / 含噪光子三方对比</sub></td>
</tr>
</table>

---

## 安装

```bash
pip install -e ".[viz]"
```

只用计算内核、不需要动画的话，`pip install -e .` 即可（仅依赖 numpy）。

## 用法

```python
import numpy as np
from photonic_mzi import PhotonicMatrixProcessor

M = np.random.randn(8, 5)          # 非方阵也可以
opu = PhotonicMatrixProcessor(M, seed=42)

x = np.random.randn(5)
opu.read_coherent(x)               # 与 M @ x 一致到 ~1e-15
opu.read_coherent(np.random.randn(5, 256))   # 批量输入

print(opu.report())                # 芯片规格：MZI 数、网格深度、路径插损、回环误差
```

加上物理非理想性，并演示「静态误差能标定、动态漂移标不掉」：

```python
from photonic_mzi import NoiseModel

nz = NoiseModel(
    fab_theta=0.02,        # 静态制造误差：流片后固定，可校准
    drift_theta=0.005,     # 动态热漂移：逐脉冲抖动，标不掉
    mzi_loss_db=0.2,       # 每台 MZI 插损
    voa_rel_err=0.01,
    detector_snr_db=40,
)
opu = PhotonicMatrixProcessor(M, noise=nz, seed=7)

opu.read_coherent(x, ideal=False)  # 未校准
opu.calibrate()                    # 出厂表征，抵消静态误差
opu.read_coherent(x, ideal=False)  # 校准后
```

两个读出接口对应两种真实的探测方案：

| 方法 | 物理含义 |
|---|---|
| `read_coherent(x)` | 相干（零差）探测，需要本振光提供相位参考，能拿到**带符号**的实部 |
| `read_intensity(x)` | 直接光电探测，光电二极管只能测 `\|E\|²`，**符号信息丢失** |

## 示例

```bash
python examples/01_hello_photonic.py        # 最小可运行示例
python examples/02_noise_and_calibration.py # 静态误差可标定，动态漂移不可
python examples/03_neural_layer.py          # 拿光子芯片跑一层神经网络推理
```

`02` 的实测结果，说明为什么必须把两类误差分开建模：

| 噪声构成 | 未校准 | 校准后 |
|---|---:|---:|
| 只有静态制造误差 (fab 0.02 rad) | 4.2 bit | **48.0 bit** |
| 只有动态热漂移 (drift 0.02 rad) | 3.8 bit | 3.8 bit |
| 两者都有 (fab 0.02 + drift 0.005) | 4.1 bit | 5.8 bit |

`03` 的实测结果，说明低精度对推理其实够用：

| 热漂移 (rad) | logit 相对误差 | 有效 bit | 分类准确率 |
|---:|---:|---:|---:|
| 0.01 | 1.79% | 5.8 | 1.000 |
| 0.05 | 9.05% | 3.5 | 1.000 |
| 0.10 | 20.55% | 2.3 | 1.000 |
| 0.20 | 54.52% | 0.9 | 0.995 |
| 0.40 | 97.69% | 0.0 | 0.285 |

## 测试

```bash
pytest -m "not slow"
```

118 项，约 2 秒。完整套件（含 494 帧字形扫描与 GIF 导出）用 `pytest`，约 6 分钟。

测试覆盖：退化 / 结构化矩阵、随机稠密、非方阵、批量、能量守恒、
噪声语义（静态可复现 vs 动态抖动）、校准、以及与原始参考实现的逐项对照回归。

```bash
python benchmarks/bench_decomposition.py
```

---

## 项目由来：一份代码审查

这个仓库源于对一份广为流传的 MZI 模拟器实现的审查。原实现的**核心数学链条是对的** ——
SVD → 酉矩阵 → MZI 网格的思路、消元公式的推导、正向传播的共轭转置逆序施加全都正确，
随机稠密矩阵能稳定跑到 `1e-14`。但它有一个**会静默算错结果的 bug**。

**根因**：`mzi_transfer_matrix(0, 0)` 并不是单位阵，而是**交换阵**。

```
theta=0,    phi=0   ->  [[0, 1], [1, 0]]     ← SWAP，会把两行对调
theta=pi/2, phi=pi  ->  [[1, 0], [0, 1]]     ← 这才是单位阵
```

原实现两个退化分支的取值恰好互换了，导致任何带结构性零的矩阵都会算错 —— 而且**不抛异常、不报警告**：

| 用例 | 原实现误差 | 本实现误差 |
|---|---:|---:|
| `identity(4)` | 6.7e-01 | 6.2e-17 |
| `permutation(4)` | 1.7e+00 | 1.7e-16 |
| `block-diag(4)` | **1.4e+01** | 2.6e-15 |
| `rank-deficient` | 1.7e+00 | 5.0e-15 |

用随机稠密矩阵测永远碰不到这条路径（`|x|` 或 `|y|` 恰好小于 `1e-15` 的概率为零）。
但剪枝后的神经网络权重、one-hot 嵌入层、注意力掩码，恰恰全是结构化稀疏矩阵。

修法是用 `arctan2` 直接取极限，特判整个删掉：

```python
phi   = np.angle(y) - np.angle(x) - np.pi
theta = np.arctan2(np.abs(x), np.abs(y))     # y→0 得 pi/2，x→0 得 0
```

**完整审查报告见 [docs/review.md](docs/review.md)**，另含 7 项物理建模改进
（VOA 只能衰减、相干探测、插损失配、静态/动态误差分离……）
和 7 项工程改进（非方阵、批量、独立 RNG、O(N⁵)→O(N³)……）。

原始实现保留在 [`tests/_reference_original.py`](tests/_reference_original.py)，
[`tests/test_regression_vs_reference.py`](tests/test_regression_vs_reference.py)
把上面每一条差异都锁成了回归用例。

---

## 已知限制

- 用的是 **Reck 三角网格**，不是 Clements 矩形网格。两者 MZI 数量相同（`N(N-1)/2`），
  但 Clements 深度只有 `N`（Reck 是 `2N-3`），插损更均匀。未实现，理由见
  [docs/review.md](docs/review.md) 的 P7 —— 简单说是不想交付一份没经过充分验证的实现。
- `calibrate()` 只补偿相移误差，不补偿插损造成的通道失配（真实系统会用 VOA 反压拉平，
  代价是牺牲整体光功率）。
- 前向传播仍是 Python 里逐台 MZI 循环。同一列的 MZI 天然可并行，向量化后还能再快一个量级。
- 没有建模波长相关性、偏振、非线性和器件间串扰。

## 参考

- Reck et al., *Experimental realization of any discrete unitary operator*, PRL 73, 58 (1994)
- Clements et al., *Optimal design for universal multiport interferometers*, Optica 3, 1460 (2016)
- Shen et al., *Deep learning with coherent nanophotonic circuits*, Nature Photonics 11, 441 (2017)

## License

[MIT](LICENSE)
