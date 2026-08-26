# photonic-mzi：日光矩阵乘加实验分支

[English](README.md) | [简体中文](README.zh-CN.md) |
[主线相干 MZI 分支](https://github.com/yaoniming3k/photonic-mzi/tree/main)

> **分支状态：实验性。** 这是 `experiment/solar-incoherent` 的默认 README。
> 该分支研究非相干日光强度计算，不是给 `main` 上的相干 MZI 处理器更换光源。

## 这个分支在研究什么

目标是执行实数矩阵乘加：

```text
y = Mx + b
```

物理路径和主线有意保持独立：

```text
日光 → 匀光/滤光 → 输入强度双轨 → 被动均匀扇出 → 非负权重透过率
     → 探测器功率汇聚 → 正负差分 → 参考归一化 → y
```

日光在这里是非相干功率载波。乘法由透过率完成，加法由探测器对光功率的空间汇聚完成。
系统不传播可控复振幅，也不依赖 MZI 相位干涉。

## 和 `main` 的区别

| 项目 | 主线 `PhotonicMatrixProcessor` | 本分支 `IncoherentSolarProcessor` |
|---|---|---|
| 光学量 | 相干复光场 | 非负光功率 |
| 典型光源 | 激光器 | 日光、LED 或太阳模拟器 |
| 乘加机制 | MZI 干涉、酉变换、VOA | 强度透过率、空间扇出、探测器汇聚 |
| 负数表示 | π 相位差 | 正负双轨 |
| 读出 | 带本振的相干/零差探测 | 成对直接探测与差分 |
| 光源波动 | 不适用 | 同时参考通道只能消除公共模波动 |
| 主要类 | `PhotonicMatrixProcessor` | `IncoherentSolarProcessor` |

两套处理器共享“验证矩阵乘加”的上层目标，但没有共享传播方程或器件拓扑。原始 MZI
项目说明仍保留在 [main 分支](https://github.com/yaoniming3k/photonic-mzi/blob/main/README.zh-CN.md)。

## 理论核心

把偏置并入矩阵和输入：

```text
A = [M  b]
z = [x; 1]
```

将归一化后的输入和权重拆成非负部分：

```text
W = W+ - W-
u = u+ - u-
```

对总效率为 `eta_f` 的 `m` 路均匀被动扇出，每行分支比例
`f=eta_f/m`。双轨探测功率为：

```text
P+ = C(t) f [W+u+ + W-u-]
P- = C(t) f [W+u- + W-u+]
```

因此 `P+-P-=C(t)fWu`。若同时参考探测器测得 `Pref=C(t)`，并补回输入、权重和
已知的 `1/f` 标度，即可恢复 `Mx+b`。数值标度可以恢复，但分光后损失的光子不能
恢复，实际散粒噪声必须按衰减后的探测功率计算。

这个归一化只能消除所有通道共享的标量变化，不能消除局部阴影、光谱失配、通道
不均匀、正负探测臂失配或独立探测噪声。

完整推导见
[《非相干日光矩阵乘加：理论与光处理器实现》](docs/solar-processor-design.zh-CN.md)。

## 软件实现

处理链按物理边界拆成三段：

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

powers = solar.optical_powers(x, ideal=False)  # P+、P-、Pref：探测前功率
observed = solar.detect(powers, ideal=False)   # 光子计数与探测器噪声
y = solar.decode(observed, normalize=True)    # 差分、参考归一化、标度恢复
```

也可以使用组合接口：

```python
y = solar.read(x, ideal=False, normalize=True)
```

真实硬件测得的三组功率可以封装为 `SolarPowerReadout`，交给同一个 `decode()` 解码。
`input_full_scale` 选择固定硬件量程并拒绝超量程输入；设为 `None` 会启用显式逐向量
AGC，其标度随读出传递，不能当成免费小信号光学增益。

实现文件：[src/photonic_mzi/solar.py](src/photonic_mzi/solar.py)

## 当前建模的非理想性

- 每次曝光的公共日照波动；
- 固定输入通道空间不均匀；
- 固定的波长积分等效权重误差；
- 正负探测臂固定增益失配；
- 泊松光子计数噪声；
- 正负探测器与参考探测器的加性读出噪声。

这些参数用于验证误差语义和敏感度，不是完整的户外器件模型。

编译后的光学核心还执行显式均匀被动扇出守恒：全部输出轨功率之和不会超过
`fanout_efficiency` 乘以可用编码输入轨功率。互相干、逐波长时变、太阳热光过剩
噪声和绝对探测器单位仍等待实测器件模型。

## 是否用电

固定透镜、滤光片、扩散片和固定权重掩模可以是无源的，日光提供光学信号能量。
但可编程输入、可编程权重、TIA、差分电路、参考除法、ADC、控制与温度稳定通常需要电。

所以严谨结论是：

> 光学乘法与空间累加可以由日光和无源器件完成；完整的可编程数值处理器通常仍用电。

当前模型使用归一化功率单位，没有建立瓦特、曝光时间或 J/MAC 模型。

## 快速运行

```bash
python examples/04_solar_incoherent.py
python -m pytest tests/test_solar.py -q
python -m pytest -m "not slow" -q
```

当前验证状态：

- 日光专项：34 项通过；
- 快速套件：162 项通过，5 项既有动画慢测跳过；
- Ruff：全部通过。

## 文档导航

- [日光实验 API、噪声语义与边界](docs/solar-experiment.zh-CN.md)
- [完整理论与光处理器设计](docs/solar-processor-design.zh-CN.md)
- [可运行示例](examples/04_solar_incoherent.py)
- [专项测试](tests/test_solar.py)
- [主线相干 MZI 中文 README](https://github.com/yaoniming3k/photonic-mzi/blob/main/README.zh-CN.md)

## 当前明确不做

- 不把日光塞进相干 MZI 传播模型；
- 不制作日光动画或 GIF；
- 不在缺少真实器件光谱数据时选择滤光片；
- 不假设“日光”二字自动保证互相干交叉项为零，台架必须测量残余条纹；
- 不声称零耗电、户外有效 bit、TOPS 或能效优势；
- 不把理想浮点一致性当作硬件已经可行的证据。

下一阶段应先在受控非相干光源或太阳模拟器台架上验证双轨、参考归一化、线性区和
校准流程，再进入真实日光实验。
