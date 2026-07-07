# ruff: noqa: F405
from __future__ import annotations

import sys
import types

from imperial_rag.retrieval import fusion as _fusion
from imperial_rag.retrieval import hybrid as _hybrid
from imperial_rag.retrieval import identity as _identity
from imperial_rag.retrieval import rerank as _rerank
from imperial_rag.retrieval import service as _service
from imperial_rag.retrieval import settings as _settings
from imperial_rag.retrieval import spans as _spans
from imperial_rag.retrieval.service import *  # noqa: F403

_MIRROR_MODULES = (_service, _settings, _identity, _hybrid, _fusion, _rerank, _spans)

for _module in _MIRROR_MODULES:
    for _name in dir(_module):
        if not _name.startswith("__"):
            globals().setdefault(_name, getattr(_module, _name))


class _RetrievalPackage(types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        for module in _MIRROR_MODULES:
            if hasattr(module, name):
                setattr(module, name, value)


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
