from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


EVENT_SCHEMA_VERSION = "1"
DEFAULT_EVENT_DATA_STREAM = "imperial-rag-events-v1"
DEFAULT_EVAL_DATA_STREAM = "imperial-rag-eval-summaries-v1"

TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
BASE_ALLOWED_FIELDS = {
    "app_version",
    "component",
    "duration_ms",
    "environment",
    "git_sha",
    "image_digest",
    "image_tag",
    "operation",
    "phoenix_session_id",
    "phoenix_trace_id",
    "request_id",
    "service",
    "session_id",
    "status",
    "user_hash",
}
QUERY_ALLOWED_FIELDS = {
    "citations_valid",
    "error_code",
    "error_type",
    "fallback_count",
    "final_evidence",
    "keyword_candidates",
    "keyword_search_status",
    "merged_candidates",
    "model_error_type",
    "rerank_input_candidates",
    "reranked_candidates",
    "reranker",
    "vector_candidates",
    "vector_search_status",
}
INGEST_ALLOWED_FIELDS = {
    "chunk_count",
    "enable_ocr",
    "error_code",
    "error_type",
    "failed_files",
    "index_vectors",
    "indexed_files",
    "keyword_indexed",
    "manifest_only_files",
    "no_text_files",
    "total_files",
    "unsupported_files",
    "vector_indexed",
}
EVAL_ALLOWED_FIELDS = {
    "error_code",
    "error_type",
    "example_count",
    "passed_count",
    "phoenix_mode",
    "ragas_metrics",
    "skipped_count",
    "wrote_output",
}
DEPENDENCY_ALLOWED_FIELDS = {
    "dependency",
    "dependency_status",
    "error_code",
    "error_type",
}
APP_ALLOWED_FIELDS = {
    "eventlog_elasticsearch_enabled",
    "trace_enabled",
}
ALLOWED_EVENTS = {
    "query.completed": QUERY_ALLOWED_FIELDS,
    "query.failed": QUERY_ALLOWED_FIELDS,
    "web_query.completed": QUERY_ALLOWED_FIELDS,
    "web_query.failed": QUERY_ALLOWED_FIELDS,
    "ingest.completed": INGEST_ALLOWED_FIELDS,
    "ingest.failed": INGEST_ALLOWED_FIELDS,
    "eval.completed": EVAL_ALLOWED_FIELDS,
    "dependency.unavailable": DEPENDENCY_ALLOWED_FIELDS,
    "app.started": APP_ALLOWED_FIELDS,
}
LEGACY_EVENT_MAP = {
    "imperial_rag.query": "query.completed",
    "imperial_rag.web_query": "web_query.completed",
    "imperial_rag.ingest": "ingest.completed",
    "imperial_rag.all_evals": "eval.completed",
    "imperial_rag.phoenix_eval": "eval.completed",
    "imperial_rag.evals.ragas": "eval.completed",
    "imperial_rag.vector_store_unavailable": "dependency.unavailable",
}
FAILED_OPERATION_EVENTS = {
    "query": "query.failed",
    "web_query": "web_query.failed",
    "ingest": "ingest.failed",
}
EVAL_OPERATIONS = {"all_evals", "phoenix_eval", "ragas_eval"}
FORBIDDEN_FIELDS = {
    "absolute_path",
    "answer",
    "api_response",
    "authorization",
    "citations",
    "documents",
    "dsn",
    "exception_message",
    "file_name",
    "file_path",
    "filename",
    "messages",
    "metadata",
    "page_content",
    "path",
    "prompt",
    "question",
    "raw_exception",
    "relative_path",
    "source_lists",
    "sources",
    "traceback",
}
FORBIDDEN_FIELD_PARTS = ("api_key", "authorization", "secret", "token")


class EventSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class ElasticsearchEventSink:
    enabled: bool
    elasticsearch_url: str
    event_data_stream: str = DEFAULT_EVENT_DATA_STREAM
    eval_data_stream: str = DEFAULT_EVAL_DATA_STREAM
    client: Any | None = None

    @classmethod
    def from_settings(cls, settings: Any | None = None) -> "ElasticsearchEventSink":
        enabled = _env_bool("IMPERIAL_RAG_EVENTLOG_ELASTICSEARCH_ENABLED", False)
        elasticsearch_url = str(
            getattr(settings, "elasticsearch_url", None)
            or os.environ.get("ELASTICSEARCH_URL")
            or "http://localhost:9200"
        )
        return cls(
            enabled=enabled,
            elasticsearch_url=elasticsearch_url,
            event_data_stream=_env_str("IMPERIAL_RAG_EVENTLOG_ELASTICSEARCH_DATA_STREAM", DEFAULT_EVENT_DATA_STREAM),
            eval_data_stream=_env_str("IMPERIAL_RAG_EVENTLOG_EVAL_DATA_STREAM", DEFAULT_EVAL_DATA_STREAM),
        )

    def emit(self, document: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        client = self.client or self._client()
        client.index(index=self.index_for(document), document=dict(document), op_type="create")

    def index_for(self, document: Mapping[str, Any]) -> str:
        event_name = str(document.get("event", ""))
        return self.eval_data_stream if event_name.startswith("eval.") else self.event_data_stream

    def _client(self) -> Any:
        from elasticsearch import Elasticsearch

        return Elasticsearch(self.elasticsearch_url)


def build_event_document(
    event: str,
    *,
    level: str = "info",
    fields: Mapping[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    raw_fields = dict(fields or {})
    if "exception_type" in raw_fields and "error_type" not in raw_fields:
        raw_fields["error_type"] = raw_fields.pop("exception_type")
    event_name = _resolve_event_name(event, raw_fields)
    allowed_fields = BASE_ALLOWED_FIELDS | ALLOWED_EVENTS[event_name]
    unknown_fields = sorted(set(raw_fields) - allowed_fields)
    forbidden_fields = sorted(key for key in raw_fields if _is_forbidden_field(key))
    if forbidden_fields:
        raise EventSchemaError(f"event {event_name} contains forbidden fields")
    if unknown_fields:
        raise EventSchemaError(f"event {event_name} contains unknown fields")

    document = _base_document(event_name, level=level, fields=raw_fields, timestamp=timestamp)
    for key in sorted(ALLOWED_EVENTS[event_name]):
        if key in raw_fields:
            document[key] = _safe_scalar(key, raw_fields[key])
    return document


def _resolve_event_name(event: str, fields: Mapping[str, Any]) -> str:
    if event in ALLOWED_EVENTS:
        return _adjust_for_status(event, fields)
    if event == "imperial_rag.failure":
        return _failure_event(fields)
    if event not in LEGACY_EVENT_MAP:
        raise EventSchemaError(f"unsupported event {event}")
    return _adjust_for_status(LEGACY_EVENT_MAP[event], fields)


def _adjust_for_status(event_name: str, fields: Mapping[str, Any]) -> str:
    status = str(fields.get("status", "")).strip().casefold()
    if event_name.endswith(".completed") and status and status not in {"ok", "success"}:
        failed_event = event_name.removesuffix(".completed") + ".failed"
        if failed_event in ALLOWED_EVENTS:
            return failed_event
    return event_name


def _failure_event(fields: Mapping[str, Any]) -> str:
    operation = str(fields.get("operation", "")).strip()
    if operation in FAILED_OPERATION_EVENTS:
        return FAILED_OPERATION_EVENTS[operation]
    if operation in EVAL_OPERATIONS:
        return "eval.completed"
    raise EventSchemaError("unsupported failure operation")


def _base_document(
    event_name: str,
    *,
    level: str,
    fields: Mapping[str, Any],
    timestamp: datetime | None,
) -> dict[str, Any]:
    created_at = timestamp or datetime.now(UTC)
    status = fields.get("status") or ("error" if event_name.endswith(".failed") else "success")
    phoenix_session_id = fields.get("phoenix_session_id") or fields.get("session_id") or os.environ.get(
        "IMPERIAL_RAG_TRACE_SESSION_ID", ""
    )
    document: dict[str, Any] = {
        "@timestamp": created_at.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "schema_version": EVENT_SCHEMA_VERSION,
        "event": event_name,
        "level": str(level or "info").strip().casefold() or "info",
        "service": _field_or_env(fields, "service", "IMPERIAL_RAG_SERVICE_NAME", "imperial-rag"),
        "component": str(fields.get("component") or "unknown"),
        "environment": _field_or_env(fields, "environment", "IMPERIAL_RAG_ENVIRONMENT", "local"),
        "status": str(status),
        "duration_ms": _safe_int(fields.get("duration_ms", 0)),
        "request_id": str(fields.get("request_id") or f"event_{uuid.uuid4().hex}"),
        "session_id": str(fields.get("session_id") or phoenix_session_id or ""),
        "app_version": _field_or_env(fields, "app_version", "IMPERIAL_RAG_APP_VERSION", "unavailable"),
        "git_sha": _field_or_env(fields, "git_sha", "IMPERIAL_RAG_GIT_SHA", "unavailable"),
        "image_tag": _field_or_env(fields, "image_tag", "IMPERIAL_RAG_IMAGE_TAG", "unavailable"),
        "image_digest": _field_or_env(fields, "image_digest", "IMPERIAL_RAG_IMAGE_DIGEST", "unavailable"),
    }
    for optional in ("user_hash", "phoenix_trace_id", "phoenix_session_id", "operation"):
        value = fields.get(optional)
        if value:
            document[optional] = _safe_scalar(optional, value)
    return document


def _field_or_env(fields: Mapping[str, Any], key: str, env_name: str, default: str) -> str:
    return str(fields.get(key) or os.environ.get(env_name) or default)


def _safe_scalar(key: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return value
    raise EventSchemaError(f"event field {key} must be scalar")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_forbidden_field(key: Any) -> bool:
    normalized = str(key).strip().casefold()
    return normalized in FORBIDDEN_FIELDS or any(part in normalized for part in FORBIDDEN_FIELD_PARTS)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().casefold() in TRUE_ENV_VALUES


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()
