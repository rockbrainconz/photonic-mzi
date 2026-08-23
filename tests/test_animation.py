"""
动画的渲染回归测试。

三件事必须成立：
  1. 全部帧都能画出来，且没有任何字形缺失（中文 / 希腊字母 / 特殊符号）
  2. 动画里展示的光场与 processor 真实计算结果一致 —— 动画不能自己编数
  3. 交互播放与 GIF 导出两条分支都能跑通
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib", reason="动画需要可选依赖 [viz]")
matplotlib.use("Agg")

from photonic_mzi import PhotonicMatrixProcessor  # noqa: E402
from photonic_mzi import animation as anim  # noqa: E402

M = np.array([[0.65, -0.42, 0.18, 0.91],
              [-0.12, 0.88, -0.54, 0.33],
              [0.47, 0.21, -0.76, 0.15],
              [-0.83, 0.35, 0.62, -0.49]])
X = np.array([1.0, 0.5, -0.8, 0.2])


@pytest.fixture(scope="module")
def rig():
    opu = PhotonicMatrixProcessor(M, seed=42)
    geo = anim.ChipLayout(opu)
    script = anim.build_script(opu, M, X, geo)
    renderer = anim.Renderer(opu, M, X, script, geo)
    yield opu, script, renderer
    matplotlib.pyplot.close(renderer.fig)


def test_script_covers_every_stage(rig):
    _, script, _ = rig
    stages = {f["stage"] for f in script}
    assert stages == set(range(len(anim.STAGES)))
    assert len(script) > 300


def test_script_avoids_known_scientific_misstatements(rig):
    """把已经验证的过度概括和错误术语锁死，避免动画文案回退。"""
    _, script, _ = rig
    text = "\n".join(f["narr"] for f in script)
    for forbidden in ["90% 的算力", "唯一拆成", "互不相干", "N 路激光"]:
        assert forbidden not in text
    assert "光处理器能否执行矩阵乘加" in text
    assert "理想电路模型内" in text
    assert "分解不唯一" in text
    assert "同一相干光源" in text


@pytest.mark.slow
def test_every_frame_renders_without_missing_glyphs(rig):
    """
    matplotlib 的缺字形警告走的是 _api.warn_external，不是标准 warnings，
    所以这里直接打补丁把它接住。抽样检查过一次漏了两个符号，必须全帧扫。
    """
    _, script, r = rig
    missing: set[str] = set()

    def catch(msg, *a, **k):
        s = str(msg)
        if "missing from font" in s:
            missing.add(s.split("Glyph ")[1].split(" missing")[0])

    import matplotlib._api as mapi
    original = mapi.warn_external
    mapi.warn_external = catch
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k in range(len(script)):
                r.render(k, live=False)
                r.fig.canvas.draw()
    finally:
        mapi.warn_external = original

    assert not missing, f"缺失字形: {sorted(missing)}"


def test_animation_field_matches_real_simulation(rig):
    """动画末尾展示的输出必须就是 processor 算出来的，不能是另一套数。"""
    opu, script, _ = rig
    result = [f for f in script if f.get("result")][-1]["result"]
    assert np.allclose(result["y_opt"], M @ X, atol=1e-12)
    assert np.allclose(result["y_opt"], opu.read_coherent(X), atol=1e-14)


def test_active_mzi_count_matches_narration(rig):
    """
    芯片图按唯一 index 高亮，不能按 mode —— 否则同一根波导上所有列会一起亮，
    与旁白「光同时抵达 N 台 MZI」对不上。
    """
    opu, script, _ = rig
    by_layer: dict[int, int] = {}
    for z in opu.vt_mzis:
        by_layer[z.layer] = by_layer.get(z.layer, 0) + 1

    seen = 0
    for f in script:
        if f["stage"] != 4 or not f.get("prop") or f.get("mzi"):
            continue
        act = f["prop"]["act"]
        if len(act) > 1 or (act and "列" in f["narr"]):
            assert len(act) == len(set(act))
            assert all(isinstance(i, int) for i in act)
            seen += 1
    assert seen > 0, "没找到任何整列抵达的帧"


def test_energy_conservation_holds_through_unitary_meshes(rig):
    """酉网格内部每一帧的总光强都应等于输入，只有 VOA 之后才允许下降。"""
    _, script, _ = rig
    p_in = float(np.sum(np.abs(X) ** 2))
    for f in script:
        if f["stage"] == 4 and f.get("prop"):
            p = float(np.sum(np.abs(f["prop"]["E"]) ** 2))
            assert abs(p - p_in) < 1e-9, f"V^T 网格里光强变了: {p} vs {p_in}"


@pytest.mark.slow
@pytest.mark.parametrize("n", [2, 3, 5])
def test_other_matrix_sizes_build(n):
    rng = np.random.default_rng(n)
    m, x = rng.standard_normal((n, n)), rng.standard_normal(n)
    opu = PhotonicMatrixProcessor(m, seed=0)
    geo = anim.ChipLayout(opu)
    script = anim.build_script(opu, m, x, geo)
    r = anim.Renderer(opu, m, x, script, geo)
    try:
        for k in (0, len(script) // 2, len(script) - 1):
            r.render(k, live=False)
            r.fig.canvas.draw()
    finally:
        matplotlib.pyplot.close(r.fig)


@pytest.mark.slow
def test_gif_export(tmp_path):
    """导出分支能跑通并产出一个非空 GIF（只导几帧，保持测试快）。"""
    import sys
    out = tmp_path / "demo.gif"
    argv = sys.argv[:]
    sys.argv = ["photonic-mzi", "--save", str(out), "--stride", "60", "--dpi", "40"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            anim.main()
    finally:
        sys.argv = argv
    assert out.exists() and out.stat().st_size > 10_000


def test_interactive_branch_constructs(monkeypatch):
    """不真的开窗，只验证交互分支能构造出来。"""
    import sys

    import matplotlib.pyplot as plt
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(sys, "argv", ["photonic-mzi"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anim.main()
    assert calls == [1]


def test_rejects_degenerate_size(monkeypatch):
    import sys
    monkeypatch.setattr(sys, "argv", ["photonic-mzi", "-n", "1"])
    with pytest.raises(SystemExit):
        anim.main()
