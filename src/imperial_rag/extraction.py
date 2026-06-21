from __future__ import annotations

import sys as _sys

_MODULE_NAME = __name__
_PACKAGE = __package__

from imperial_rag.ingestion import extraction as _impl

globals().update(_impl.__dict__)
_sys.modules[_MODULE_NAME] = _impl
if _PACKAGE:
    setattr(_sys.modules[_PACKAGE], _MODULE_NAME.rsplit(".", 1)[-1], _impl)
