"""Regenerate the English-first bilingual README animation and screenshots."""

# ruff: noqa: E402 -- add the checkout's src tree before importing the package.

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from photonic_mzi import PhotonicMatrixProcessor
from photonic_mzi.animation import ChipLayout, Renderer, build_script


def main() -> None:
    output = ROOT / "docs" / "images"
    output.mkdir(parents=True, exist_ok=True)

    matrix = np.array([
        [0.65, -0.42, 0.18, 0.91],
        [-0.12, 0.88, -0.54, 0.33],
        [0.47, 0.21, -0.76, 0.15],
        [-0.83, 0.35, 0.62, -0.49],
    ])
    vector = np.array([1.0, 0.5, -0.8, 0.2])
    processor = PhotonicMatrixProcessor(matrix, seed=42)
    layout = ChipLayout(processor)
    script = build_script(processor, matrix, vector, layout)
    renderer = Renderer(processor, matrix, vector, script, layout)

    screenshots = {
        "stage1-svd.png": 76,
        "stage2-compile.png": 160,
        "stage4-interference.png": 252,
        "stage8-noise.png": 480,
    }
    for filename, frame in screenshots.items():
        renderer.render(frame, live=False)
        renderer.fig.savefig(output / filename, dpi=100, facecolor=renderer.fig.get_facecolor())

    animation = FuncAnimation(
        renderer.fig,
        lambda frame: renderer.render(frame, live=False),
        frames=range(len(script)),
        interval=1000 // 14,
        blit=False,
    )
    animation.save(output / "demo.gif", writer=PillowWriter(fps=14), dpi=100)
    plt.close(renderer.fig)
    print(f"Generated {len(script)} frames and {len(screenshots)} screenshots in {output}")


if __name__ == "__main__":
    main()
