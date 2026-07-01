# ruff: noqa: F405
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

__all__ = [
    "ALLOWED_EVENTS",
    "APP_ALLOWED_FIELDS",
    "BASE_ALLOWED_FIELDS",
    "DEFAULT_EVENT_DATA_STREAM",
    "DEFAULT_EVAL_DATA_STREAM",
    "DEPENDENCY_ALLOWED_FIELDS",
    "ElasticsearchEventSink",
    "EventSchemaError",
    "INGEST_ALLOWED_FIELDS",
    "JsonEventFormatter",
    "LEGACY_EVENT_MAP",
    "LOGGER_NAME",
    "PlainEventFormatter",
    "QUERY_ALLOWED_FIELDS",
    "TRUE_ENV_VALUES",
    "build_event_document",
    "configure_observability",
    "log_event",
    "log_failure",
    "sanitize_log_fields",
]
