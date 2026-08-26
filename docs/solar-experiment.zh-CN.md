# 实验性日光矩阵乘加

[English](solar-experiment.md) | [简体中文](solar-experiment.zh-CN.md)

完整的成立条件、宽带模型、硬件光路、用电边界、校准和验收标准见
[《非相干日光矩阵乘加：理论与光处理器实现》](solar-processor-design.zh-CN.md)。

## 1. 目标与边界

这个后端研究一个与相干 MZI 处理器不同的问题：能否把日光作为非相干功率载波，
用强度调制完成任意实数矩阵乘加：

```text
y = M @ x + b
```

`IncoherentSolarProcessor` 不传播复光场、不使用相位干涉，也不复用 Reck 网格。
它描述的是自由空间强度交叉阵列：输入调制器编码 `x`，权重掩模编码 `M`，探测器
在功率域求和。当前编译光路包含显式均匀被动扇出，不会把一条输入轨以满功率免费复制
到所有输出。当前实现是实验模型，不是日光硬件性能证明。

## 2. 正负双轨

非相干探测器只能累加非负功率。任意实数输入和权重分解为：

```text
x = x+ - x-
M = M+ - M-
```

两组输出功率为：

```text
P+ = C(t) f [M+ x+ + M- x-]
P- = C(t) f [M+ x- + M- x+]
```

差分读出满足：

```text
P+ - P- = C(t) f Mx
```

其中 `f=eta_f/m` 是总效率为 `eta_f` 的 `m` 路均匀扇出分支比例。偏置 `b`
通过额外的恒定输入通道 `x0=1` 实现。权重和输入分别按已知标度归一化到 `[0,1]`；
读出端补回这些标度和已知的 `1/f`。

## 3. 被动扇出与输入量程

`fanout_efficiency` 表示每条输入轨送达全部输出行的总被动效率。均匀扇出给每行
`fanout_efficiency/n_out`，从而保证：

```text
sum(P+ + P-) <= irradiance * fanout_efficiency * sum(abs(encoded_input))
```

`input_full_scale=value` 使用固定硬件量程并拒绝超量程输入。
`input_full_scale=None` 使用逐向量 AGC，并把标度记录在 `SolarPowerReadout` 中；
这种代数便利模式需要额外测量/控制路径，不能当成免费的小信号增益。
`fanout_efficiency` 从参考抽头之后开始计量；参考功率不计入计算输出上界，但绝对
入射功率预算必须计入参考抽头损耗。

## 4. 同时参考通道

参考探测器测得 `Pref=C(t)`。在所有计算通道共享同一个日照乘数的理想极限下：

```text
(P+ - P-) / Pref = Mx
```

因此同时参考可以消除云层等造成的公共模照度变化，但不能消除：

- 各输入位置不同的照明增益；
- 滤光片、权重单元和探测器的光谱响应差异；
- 正负探测臂失配；
- 散粒噪声、读出噪声或参考通道自身噪声。

`normalize=False` 有意保留日照乘数，可用于验证参考通道的必要性。

有噪声的参考除法是比值估计，一般存在偏差。当前模型会直接模拟这个比值，但尚不提供
解析不确定度；实验必须报告参考 SNR，真实输出接近零时使用绝对或满量程误差。

## 5. 简化非理想性

| 参数 | 统计语义 | 参考归一化能否消除 |
|---|---|---|
| `irradiance` | 参考通道的平均归一化功率 | 仅消除其公共标度 |
| `common_fluctuation` | 每次曝光独立、正值且均值为 1 的照度乘数变异系数 | 理想无噪声时可以 |
| `spatial_nonuniformity` | 每输入通道构造时采样一次的固定增益 | 不可以 |
| `spectral_weight_error` | 每权重一次采样的波长积分等效误差 | 不可以 |
| `differential_gain_error` | 正负输出臂各自的固定增益误差 | 不可以 |
| `photons_per_unit` | 功率到光子计数的换算；有限值启用泊松散粒噪声 | 不可以 |
| `detector_noise` | 正负探测器的加性高斯噪声 | 不可以 |
| `reference_noise` | 参考探测器的加性高斯噪声 | 不可以 |

固定增益采用正值、均值为 1 的对数正态分布，参数是变异系数。动态照度同样采用
对数正态分布，以避免高波动时出现负“日照”。

## 6. 光谱与相干边界

真实探测量应写成：

```text
Ii = sum_j integral S(lambda,t) R_i(lambda) T_ij(lambda) x_j d_lambda
```

当前 `spectral_weight_error` 只把选定滤光条件下的积分偏差压缩成固定乘性误差；
它没有离散波长、太阳光谱、滤光片透射曲线、SLM 色散或探测器响应度。因此本模型
可以验证双轨代数、归一化边界和噪声传播，但不能选择真实滤光片或预测户外 ENOB。
单一参考只有在 `M_eff(t)=c(t)M0` 时才严格有效。

功率相加还要求互相干交叉项平均到目标误差以下。宽带日光通常有利，但窄带滤光、
单模耦合或高度匹配的分束路径可能重新产生干涉。真实台架必须扫描路径差、孔径和
滤光带宽并测量条纹可见度，不能只根据光源名称推断该条件。

## 7. 使用

```python
import numpy as np
from photonic_mzi import IncoherentSolarProcessor, SolarNoiseModel

M = np.array([[1.0, -0.5], [0.25, 2.0]])
x = np.array([0.3, -0.8])

solar = IncoherentSolarProcessor(
    M,
    fanout_efficiency=0.8,
    input_full_scale=1.0,
    noise=SolarNoiseModel(
        common_fluctuation=0.5,
        photons_per_unit=100_000,
    ),
    seed=7,
)

y = solar.read(x, ideal=False, normalize=True)
```

完整示例：

```bash
python examples/04_solar_incoherent.py
```

## 8. 尚未声称的结论

- 没有证明日光系统比激光或电子计算更节能；调制器、TIA 和 ADC 仍可能主导能耗。
- 已加入归一化均匀扇出守恒，但没有模拟集光口径、展度、详细分束器、匀光损耗、
  杂散光、遮挡、饱和、温漂或器件带宽。
- 没有实现逐波长传播，不能把 `spectral_weight_error` 外推为任何真实天气条件。
- 模拟中的理想精确结果只验证代数，不代表模拟精度可以由户外硬件达到。
