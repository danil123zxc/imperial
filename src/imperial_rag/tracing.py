from __future__ import annotations

import json
import os
import socket
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from openinference.semconv.trace import DocumentAttributes, RerankerAttributes, SpanAttributes

from imperial_rag.config import Settings


_CONFIGURED_PROVIDER: object | None = None
_CONFIGURED_KEY: tuple[str, str] | None = None
_TRACER_NAME = "imperial_rag.retrieval"
_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
_INPUT_VALUE = SpanAttributes.INPUT_VALUE
_OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
_RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
_RERANKER_INPUT_DOCUMENTS = RerankerAttributes.RERANKER_INPUT_DOCUMENTS
_RERANKER_OUTPUT_DOCUMENTS = RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS
_DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
_DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
_DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
_DOCUMENT_SCORE = DocumentAttributes.DOCUMENT_SCORE
_RETRIEVAL_PREVIEW_LIMIT = 3


class RetrievalTraceSpan:
    def __init__(self, span: Any) -> None:
        self._span = span

    def set_attribute(self, key: str, value: Any) -> None:
        if value is None:
            return
        self._span.set_attribute(key, _attribute_value(value))

    def set_output(self, value: Any) -> None:
        self._span.set_attribute(_OUTPUT_VALUE, _json_value(value))

    def set_retrieval_documents(self, documents: Sequence[Any]) -> None:
        self._set_documents(_RETRIEVAL_DOCUMENTS, documents)

    def set_reranker_input_documents(self, documents: Sequence[Any]) -> None:
        self._set_documents(_RERANKER_INPUT_DOCUMENTS, documents)

    def set_reranker_output_documents(self, documents: Sequence[Any]) -> None:
        self._set_documents(_RERANKER_OUTPUT_DOCUMENTS, documents)

    def _set_documents(self, key_prefix: str, documents: Sequence[Any]) -> None:
        for key, value in openinference_document_attributes(key_prefix, documents).items():
            self._span.set_attribute(key, value)


@contextmanager
def trace_retrieval_step(
    name: str,
    query: str,
    *,
    kind: str = "RETRIEVER",
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[RetrievalTraceSpan]:
    """Create a compact OpenInference span for one retrieval pipeline step."""

    tracer = trace.get_tracer(_TRACER_NAME)
    span_attributes: dict[str, Any] = {
        _SPAN_KIND: kind,
        _INPUT_VALUE: query,
    }
    for key, value in (attributes or {}).items():
        if value is not None:
            span_attributes[key] = _attribute_value(value)

    with tracer.start_as_current_span(name, attributes=span_attributes) as span:
        trace_span = RetrievalTraceSpan(span)
        try:
            yield trace_span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            trace_span.set_attribute("error.type", type(exc).__name__)
            raise
        else:
            span.set_status(Status(StatusCode.OK))


def retrieval_documents_preview(
    documents: Sequence[Any],
    *,
    limit: int = _RETRIEVAL_PREVIEW_LIMIT,
    content_chars: int = 160,
) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for rank, document in enumerate(list(documents)[:limit]):
        metadata = dict(getattr(document, "metadata", {}) or {})
        preview = _compact_text(str(getattr(document, "page_content", "")), content_chars)
        previews.append(
            {
                "rank": rank,
                "citation_id": metadata.get("citation_id"),
                "chunk_id": metadata.get("chunk_id"),
                "file_name": metadata.get("file_name"),
                "source_type": metadata.get("source_type"),
                "preview": preview,
            }
        )
    return previews


def openinference_document_attributes(key_prefix: str, documents: Sequence[Any]) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    for index, document in enumerate(list(documents)):
        content = str(getattr(document, "page_content", ""))
        metadata = dict(getattr(document, "metadata", {}) or {})
        document_id = _document_id(metadata)
        score = _document_score(metadata)
        prefix = f"{key_prefix}.{index}"

        attributes[f"{prefix}.{_DOCUMENT_CONTENT}"] = content
        if document_id is not None:
            attributes[f"{prefix}.{_DOCUMENT_ID}"] = document_id
        if metadata:
            attributes[f"{prefix}.{_DOCUMENT_METADATA}"] = _json_value(metadata)
        if score is not None:
            attributes[f"{prefix}.{_DOCUMENT_SCORE}"] = score
    return attributes


def _document_id(metadata: Mapping[str, Any]) -> str | None:
    for key in ("chunk_id", "citation_id", "_id", "file_id"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _document_score(metadata: Mapping[str, Any]) -> float | None:
    for key in ("relevance_score", "_keyword_score", "_fallback_score"):
        value = metadata.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _compact_text(value: str, limit: int) -> str:
    compact = " ".join(value.split())
    if limit <= 0 or len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."


def _attribute_value(value: Any) -> str | bool | int | float | Sequence[str | bool | int | float]:
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        values: list[str | bool | int | float] = []
        for item in value:
            if isinstance(item, (str, bool, int, float)):
                values.append(item)
            else:
                return _json_value(value)
        return values
    return _json_value(value)


def _json_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def configure_phoenix_tracing(settings: Settings | None = None, enabled: bool | None = None) -> object | None:
    """Configure Phoenix OpenTelemetry tracing once for the current process."""

    env_enabled = enabled is None
    if enabled is None:
        enabled = _env_flag("PHOENIX_TRACING_ENABLED") or _env_flag("IMPERIAL_RAG_TRACING_ENABLED")
    if not enabled:
        return None

    resolved_settings = settings or Settings()
    if env_enabled and not _collector_endpoint_reachable(resolved_settings.phoenix_collector_endpoint):
        return None

    key = (resolved_settings.phoenix_project_name, resolved_settings.phoenix_collector_endpoint)
    global _CONFIGURED_PROVIDER, _CONFIGURED_KEY
    if _CONFIGURED_PROVIDER is not None:
        if _CONFIGURED_KEY == key:
            return _CONFIGURED_PROVIDER
        raise RuntimeError(
            "Phoenix tracing is already configured for "
            f"project={_CONFIGURED_KEY[0]!r}, endpoint={_CONFIGURED_KEY[1]!r}; "
            f"cannot reconfigure to project={key[0]!r}, endpoint={key[1]!r} in the same process."
        )

    try:
        from phoenix.otel import register
    except ImportError as exc:
        raise RuntimeError(
            "Phoenix tracing dependencies are missing. Install arize-phoenix-otel and OpenInference instrumentors."
        ) from exc

    _CONFIGURED_PROVIDER = register(
        project_name=resolved_settings.phoenix_project_name,
        endpoint=resolved_settings.phoenix_collector_endpoint,
        auto_instrument=True,
        verbose=False,
    )
    _CONFIGURED_KEY = key
    return _CONFIGURED_PROVIDER


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def _collector_endpoint_reachable(endpoint: str, timeout: float = 0.2) -> bool:
    parsed = urlparse(endpoint)
    if not parsed.hostname:
        return True
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def _reset_phoenix_tracing_for_tests() -> None:
    global _CONFIGURED_PROVIDER, _CONFIGURED_KEY
    _CONFIGURED_PROVIDER = None
    _CONFIGURED_KEY = None
