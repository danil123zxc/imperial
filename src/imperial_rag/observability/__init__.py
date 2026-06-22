from __future__ import annotations

import sys
import types

from imperial_rag.observability import logging as _logging
from imperial_rag.observability.eventlog import *  # noqa: F403
from imperial_rag.observability.logging import *  # noqa: F403

for _name in dir(_logging):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_logging, _name))


class _ObservabilityPackage(types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if hasattr(_logging, name):
            setattr(_logging, name, value)


sys.modules[__name__].__class__ = _ObservabilityPackage

__all__ = [name for name in globals() if not name.startswith("_")]
