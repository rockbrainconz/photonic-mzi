# 参与开发

[中文](CONTRIBUTING.md) | [English](CONTRIBUTING.en.md)

## 环境

```bash
python -m pip install --upgrade pip && pip install -e ".[dev]"
```

可编辑安装需要 **pip ≥ 21.3**：本项目只有 `pyproject.toml`、没有 `setup.py`，
旧版 pip 会直接报 `Directory cannot be installed in editable mode`。

不想动全局 pip 的话，不安装也能开发 —— `tests/conftest.py` 会把 `src/` 加进
`sys.path`，`examples/` 会优先导入同仓库 `src/`：

```bash
pytest -m "not slow"
PYTHONPATH=src python -m photonic_mzi
```

## 日常循环

```bash
pytest -m "not slow"
```

128 项、约 3 秒。改完代码先跑这个；完整套件当前共 133 项。

```bash
pytest
```

完整套件包含三类重型渲染用例：逐帧字形扫描、
GIF 导出、多种矩阵尺寸的构建。**提 PR 前至少本地跑一次完整套件。**

```bash
ruff check .
```

CI 只跑 `ruff check`，不跑 `ruff format`。本项目的代码排版（矩阵字面量的对齐、
教学注释的缩进位置）是刻意安排的，交给自动格式化会破坏可读性。

## 几条本项目特有的约定

**动画不能自己编数。** 屏幕上显示的光场必须来自 `photonic_mzi.processor` 的真实计算，
不允许为了画面好看另算一套。`tests/test_animation.py::test_animation_field_matches_real_simulation`
守着这条线。

**中文字形必须全帧扫描，不能抽样。** matplotlib 缺字形时只发一条 `UserWarning`，
渲染出来是方框。这个问题已经漏过一次 —— 抽查了 9 个代表帧，漏掉了 `ᵀ`、`▶`、`₀` 三个符号。
现在 `test_every_frame_renders_without_missing_glyphs` 会把全部 494 帧都画一遍。
加新文案时，优先用 `$V^T$` 这类 mathtext，别直接用生僻 Unicode 符号。

**芯片图按 `MZI.index` 高亮，不能按 `mode`。** 同一根波导上会有多台 MZI 分布在不同列，
按 `mode` 匹配会让整行一起亮，和旁白「光同时抵达 N 台」对不上。

**物理近似必须写进 docstring。** 这个项目的价值有一半在于说清楚
「模型在哪里偏离了真实器件」。比如 `calibrate()` 的 docstring 明确写了它假设表征足够准、
且对插损无能为力。新增近似时请照此办理。

## 加新的噪声源

`NoiseModel` 是个 dataclass，加字段即可。但要先定义其统计和物理语义：

- **固定器件偏置** → 在 `PhotonicMatrixProcessor.__init__` 里采样一次；只有明确属于
  可控相移偏置时，才让 `calibrate()` 抵消它
- **每样本 i.i.d. 抖动** → 在 `_run_mesh` 里按 batch 列独立采样
- **时间/空间相关漂移** → 必须显式保存状态或协方差，不能伪装成 i.i.d. 标量
- **探测噪声** → 在相干或平方律检波之后加入，不能直接加到复光场

新增噪声源请同时补一条测试，验证它的「可重复性语义」符合预期
（参考 `test_static_fab_error_is_repeatable_shot_to_shot`）。

## 想实现 Clements 网格？

这是目前投入产出比最高的改进，背景见 [模型与验证说明](docs/validation.md)。
关键难点是右乘消元之后要把 `T` 穿过对角相位阵对易过去。如果你做了，请：

1. 保留 Reck 作为可选拓扑（`decompose_unitary(..., topology="reck"|"clements")`）
2. 补一条测试，验证 Clements 的网格深度确实是 `N` 而不是 `2N-3`
3. 用真实拓扑深度或传输矩阵验证损耗均衡性，不要把 `mode_mzi_count()` 当端到端路径
