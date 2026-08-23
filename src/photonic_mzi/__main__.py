"""``python -m photonic_mzi`` 入口 / entry point for the teaching animation."""
from __future__ import annotations

import sys


def main() -> int:
    try:
        from .animation import main as animation_main
    except ImportError as exc:  # matplotlib 是可选依赖
        print(f"动画需要 matplotlib / The animation requires matplotlib: {exc}\n\n"
              '    pip install "photonic-mzi[viz]"\n', file=sys.stderr)
        return 1
    animation_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
