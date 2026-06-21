from __future__ import annotations

import sys
import types

from imperial_rag.indexing import vector as _vector
from imperial_rag.indexing.vector import *  # noqa: F403

for _name in dir(_vector):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_vector, _name))


class _IndexingPackage(types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if hasattr(_vector, name):
            setattr(_vector, name, value)


sys.modules[__name__].__class__ = _IndexingPackage

__all__ = [name for name in globals() if not name.startswith("_")]
