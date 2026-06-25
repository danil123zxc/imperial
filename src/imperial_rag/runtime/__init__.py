from __future__ import annotations

import sys as _sys

_MODULE_NAME = __name__
_PARENT_PACKAGE = __name__.rsplit(".", 1)[0] if "." in __name__ else ""

from imperial_rag.answering import runtime as _impl

globals().update(_impl.__dict__)
_sys.modules[_MODULE_NAME] = _impl
if _PARENT_PACKAGE:
    setattr(_sys.modules[_PARENT_PACKAGE], _MODULE_NAME.rsplit(".", 1)[-1], _impl)
