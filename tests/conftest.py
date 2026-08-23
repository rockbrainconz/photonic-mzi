"""
让测试在「没有 pip install」的情况下也能直接跑：把 src/ 塞进 sys.path。

正式安装（``pip install -e ".[dev]"``）后这段是无害的空操作。
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
