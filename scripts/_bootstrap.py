from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path(script_file: str | Path) -> None:
    script_path = Path(script_file).resolve()
    _prepend_path(script_path.parent)
    _prepend_path(script_path.parents[1] / "src")


def _prepend_path(path: Path) -> None:
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)
