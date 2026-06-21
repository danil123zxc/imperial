from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any


LOGGER_NAME = "imperial_rag.events"
_CONFIGURED_LOGGER: logging.Logger | None = None
_HANDLER: logging.Handler | None = None
_SENSITIVE_KEYS = {
    "question",
    "answer",
    "page_content",
    "documents",
    "sources",
    "citations",
    "path",
    "file_path",
    "absolute_path",
    "relative_path",
    "file_name",
    "filename",
    "api_key",
    "dsn",
    "authorization",
    "token",
    "secret",
}


class JsonEventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = dict(getattr(record, "event_payload", {}) or {})
        payload.setdefault("timestamp", datetime.fromtimestamp(record.created, UTC).isoformat().replace("+00:00", "Z"))
        payload.setdefault("level", record.levelname.lower())
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


class PlainEventFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = dict(getattr(record, "event_payload", {}) or {})
        payload.setdefault("timestamp", datetime.fromtimestamp(record.created, UTC).isoformat().replace("+00:00", "Z"))
        payload.setdefault("level", record.levelname.lower())
        return " ".join(f"{key}={_plain_value(payload[key])}" for key in sorted(payload))


def configure_observability(settings: Any | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(_logging_level(getattr(settings, "log_level", os.environ.get("IMPERIAL_RAG_LOG_LEVEL", "INFO"))))
    logger.propagate = False

    global _CONFIGURED_LOGGER, _HANDLER
    if _HANDLER is None:
        _HANDLER = logging.StreamHandler()
        logger.handlers = [_HANDLER]
    elif _HANDLER not in logger.handlers:
        logger.handlers = [_HANDLER]
    _HANDLER.setFormatter(_formatter_for(getattr(settings, "log_format", "json")))

    _CONFIGURED_LOGGER = logger
    return logger


def log_event(event: str, *, level: str = "info", **fields: Any) -> None:
    logger = _CONFIGURED_LOGGER or configure_observability()
    payload = sanitize_log_fields({"event": event, **fields})
    logger.log(_logging_level(level), "", extra={"event_payload": payload})


def log_failure(operation: str, exc: BaseException, **fields: Any) -> None:
    log_event(
        "imperial_rag.failure",
        level="error",
        operation=operation,
        status="error",
        exception_type=type(exc).__name__,
        **fields,
    )


def sanitize_log_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in fields.items():
        if _is_sensitive_key(key):
            continue
        cleaned = _sanitize_value(value)
        if cleaned is not None:
            sanitized[str(key)] = cleaned
    return sanitized


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return sanitize_log_fields(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sanitized_items: list[Any] = []
        for item in value:
            sanitized_item = _sanitize_value(item)
            if sanitized_item is not None:
                sanitized_items.append(sanitized_item)
        return sanitized_items
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().casefold()
    return normalized in _SENSITIVE_KEYS or any(part in normalized for part in ("api_key", "authorization", "token", "secret"))


def _formatter_for(log_format: Any) -> logging.Formatter:
    return PlainEventFormatter() if str(log_format).strip().casefold() == "plain" else JsonEventFormatter()


def _plain_value(value: Any) -> str:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _logging_level(level: Any) -> int:
    if isinstance(level, int):
        return level
    name = str(level or "INFO").strip().upper()
    resolved = logging.getLevelName(name)
    return resolved if isinstance(resolved, int) else logging.INFO


def _reset_observability_for_tests() -> None:
    global _CONFIGURED_LOGGER, _HANDLER
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers = []
    logger.propagate = True
    logger.setLevel(logging.NOTSET)
    _CONFIGURED_LOGGER = None
    _HANDLER = None
