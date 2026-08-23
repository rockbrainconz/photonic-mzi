"""Entry point for the teaching animation. / 教学动画入口。"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        from .animation import main as animation_main
    except ImportError as exc:  # Matplotlib is an optional dependency.
        print(f"The animation requires matplotlib / 动画需要 matplotlib: {exc}\n\n"
              '    pip install "photonic-mzi[viz]"\n', file=sys.stderr)
        return 1
    animation_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
