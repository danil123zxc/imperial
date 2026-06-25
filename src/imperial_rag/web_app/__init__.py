from __future__ import annotations

import sys as _sys
from pathlib import Path

_MODULE_NAME = __name__
_PARENT_PACKAGE = __name__.rsplit(".", 1)[0] if "." in __name__ else ""


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[3]
    src = root / "src"
    if str(src) not in _sys.path:
        _sys.path.insert(0, str(src))


_ensure_src_on_path()

from imperial_rag.app import web as _impl

globals().update(_impl.__dict__)
if _MODULE_NAME == "__main__":
    _impl.main()
else:
    _sys.modules[_MODULE_NAME] = _impl
    if _PARENT_PACKAGE:
        setattr(_sys.modules[_PARENT_PACKAGE], _MODULE_NAME.rsplit(".", 1)[-1], _impl)
