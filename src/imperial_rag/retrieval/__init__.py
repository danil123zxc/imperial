# ruff: noqa: F405
from __future__ import annotations

import sys
import types

from imperial_rag.retrieval import service as _service
from imperial_rag.retrieval.service import *  # noqa: F403

for _name in dir(_service):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_service, _name))


class _RetrievalPackage(types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if hasattr(_service, name):
            setattr(_service, name, value)


sys.modules[__name__].__class__ = _RetrievalPackage

__all__ = [
    "CandidateMerger",
    "FallbackRanker",
    "HybridRetriever",
    "RetrievalCandidateResult",
    "RetrievalResult",
    "RetrievalService",
    "RetrievalSettings",
    "Reranker",
    "RrfCandidateFusion",
]
