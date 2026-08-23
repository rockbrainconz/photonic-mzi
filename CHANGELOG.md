# Changelog

本文件记录值得注意的变更。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [1.0.0] - 2026-08-23

首个版本。起点是对一份 MZI 网格模拟器实现的代码审查，
完整报告见 [docs/review.md](docs/review.md)。

### 修复

- **退化分支取值写反，导致结构化矩阵静默算错**（审查报告 B1）。
  `mzi_transfer_matrix(0, 0)` 是交换阵而非单位阵，参考实现的两个退化分支恰好互换。
  单位阵、置换阵、对角阵、块对角、秩亏矩阵全部算错且不报任何错误
  （`identity(4)` 误差 `6.7e-01`，`block-diag(4)` 误差 `1.4e+01`）。
  现改用 `arctan2` 直接取极限，特判整个删除。
- 消元判据从绝对阈值 `1e-15` 改为无阈值，不再误判整体尺度很小的矩阵（B2）。
- `phi` 折回 `(-π, π]`，不再输出 `-360°` 这种无意义的值（E5）。

### 新增

- 支持**非方阵**，内部零填充到 `N = max(n_out, n_in)`，多余端口暗置（E1）。
- 支持**批量输入** `(n_in, B)`，实测比逐条循环快约 340 倍（E2）。
- `NoiseModel` 把**静态制造误差**（`fab_*`，可标定）与**动态热漂移**
  （`drift_*`，标不掉）分开建模，另含插损、VOA 误差、探测信噪比（P4、P5）。
- `calibrate()` / `reset_calibration()`：模拟出厂表征校准。
  实测静态误差 4.2 bit → 48 bit，动态漂移 3.8 bit → 3.8 bit。
- `read_coherent()` / `read_intensity()`：区分相干（零差）探测与直接光电探测，
  后者会丢失符号信息（P2）。
- 奇异值归一化到 `[0, 1]`，增益记在电域 —— VOA 只能衰减不能放大（P1）。
- `report()`、`unitary_error()`、`mesh_depth`、`path_mzi_count()`：
  芯片规格报告与编译回环自检（E6）。
- **逐行代码 + 逐器件光传播的教学动画**（`python -m photonic_mzi`），
  9 个阶段、494 帧，支持暂停与单步。

### 变更

- 就地两行更新替代「构造 N×N 全矩阵再相乘」：编译 O(N⁵)→O(N³)，前向 O(N³)→O(N²)。
  N=128 实测编译快 11.4 倍、前向快 9.1 倍（E4）。
- 随机数走独立的 `np.random.Generator`，不再依赖也不污染全局 `np.random`（E3）。
- 拆成 `mesh`（纯线性代数）/ `processor`（物理模型）/ `animation`（可视化）三层，
  matplotlib 变成可选依赖 `[viz]`。

### 已知限制

- 用 Reck 三角网格而非 Clements 矩形网格。两者 MZI 数量相同，
  但 Clements 深度只有 `N`（Reck 是 `2N-3`），插损更均匀。未实现，理由见报告 P7。
- `calibrate()` 不补偿插损造成的通道失配。
- 前向传播仍是逐台 MZI 的 Python 循环，同一列尚未向量化。
- 未建模波长相关性、偏振、非线性与器件间串扰。

[1.0.0]: https://github.com/rockbrainconz/photonic-mzi/releases/tag/v1.0.0
