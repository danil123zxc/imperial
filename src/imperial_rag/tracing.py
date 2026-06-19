from __future__ import annotations

import json
import os
import hashlib
import hmac
import socket
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any
from urllib.parse import urlparse

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from openinference.semconv.trace import DocumentAttributes, RerankerAttributes
from phoenix.otel import SpanAttributes

from imperial_rag.config import Settings


_CONFIGURED_PROVIDER: object | None = None
_CONFIGURED_KEY: tuple[str, str] | None = None
_TRACER_NAME = "imperial_rag.tracing"
_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
_INPUT_VALUE = SpanAttributes.INPUT_VALUE
_OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
_INPUT_MIME_TYPE = getattr(SpanAttributes, "INPUT_MIME_TYPE", "input.mime_type")
_OUTPUT_MIME_TYPE = getattr(SpanAttributes, "OUTPUT_MIME_TYPE", "output.mime_type")
_RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
_RERANKER_INPUT_DOCUMENTS = RerankerAttributes.RERANKER_INPUT_DOCUMENTS
_RERANKER_OUTPUT_DOCUMENTS = RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS
_LLM_INPUT_MESSAGES = "llm.input_messages"
_LLM_OUTPUT_MESSAGES = "llm.output_messages"
_LLM_TOOLS = "llm.tools"
_DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
_DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
_DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
_DOCUMENT_SCORE = DocumentAttributes.DOCUMENT_SCORE
_RETRIEVAL_PREVIEW_LIMIT = 3
_TRACE_DOCUMENT_LIMIT = 10
_TRACE_DOCUMENT_CONTENT_CHARS = 800
_TRACE_SCHEMA_VERSION_VALUE = "rag-v2"
_TEXT_MIME_TYPE = "text/plain"
_JSON_MIME_TYPE = "application/json"
_IMPERIAL_PHASE = "imperial.phase"
_IMPERIAL_STEP = "imperial.step"
_IMPERIAL_TRACE_SCHEMA_VERSION = "imperial.trace_schema_version"
_TRACE_METADATA_ALLOWLIST = frozenset(
    {
        "citation_id",
        "chunk_id",
        "chunk_index",
        "file_name",
        "relative_path",
        "source_type",
        "section_heading",
        "page_number",
        "sheet_name",
        "image_index",
        "relevance_score",
        "_keyword_score",
        "_keyword_rank",
        "_vector_rank",
        "_fusion_rank",
        "_rrf_score",
        "_fallback_score",
    }
)


class OpenInferenceTraceSpan:
    def __init__(self, span: Any, kind: str | None = None) -> None:
        self._span = span
        self._kind = kind

    def set_attribute(self, key: str, value: Any) -> None:
        if value is None:
            return
        if _attribute_hidden(key, self._kind):
            return
        self._span.set_attribute(key, _attribute_value(value))

    def set_output(self, value: Any) -> None:
        if _hide_outputs():
            return
        self._span.set_attribute(_OUTPUT_VALUE, _json_value(value))
        self._span.set_attribute(_OUTPUT_MIME_TYPE, _JSON_MIME_TYPE)

    def set_retrieval_documents(self, documents: Sequence[Any]) -> None:
        self._set_documents(_RETRIEVAL_DOCUMENTS, documents)

    def set_reranker_input_documents(self, documents: Sequence[Any]) -> None:
        self._set_documents(_RERANKER_INPUT_DOCUMENTS, documents)

    def set_reranker_output_documents(self, documents: Sequence[Any]) -> None:
        self._set_documents(_RERANKER_OUTPUT_DOCUMENTS, documents)

    def set_final_evidence_documents(self, documents: Sequence[Any]) -> None:
        content_chars = 0 if _trace_full_final_evidence() else None
        self._set_documents(_RETRIEVAL_DOCUMENTS, documents, content_chars=content_chars)

    def _set_documents(
        self,
        key_prefix: str,
        documents: Sequence[Any],
        *,
        content_chars: int | None = None,
    ) -> None:
        document_attributes = openinference_document_attributes(
            key_prefix,
            documents,
            content_chars=content_chars,
        )
        for key, value in document_attributes.items():
            self._span.set_attribute(key, value)


RetrievalTraceSpan = OpenInferenceTraceSpan


@contextmanager
def trace_openinference_step(
    name: str,
    input_value: str,
    *,
    kind: str,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[OpenInferenceTraceSpan]:
    """Create a compact OpenInference span for a local RAG pipeline step."""

    tracer = trace.get_tracer(_TRACER_NAME)
    span_attributes: dict[str, Any] = {
        _SPAN_KIND: kind,
    }
    if not _hide_span_input(kind):
        span_attributes[_INPUT_VALUE] = input_value
        span_attributes[_INPUT_MIME_TYPE] = _TEXT_MIME_TYPE
    for key, value in (attributes or {}).items():
        if value is not None and not _attribute_hidden(key, kind):
            span_attributes[key] = _attribute_value(value)

    with tracer.start_as_current_span(name, attributes=span_attributes) as span:
        trace_span = OpenInferenceTraceSpan(span, kind=kind)
        try:
            yield trace_span
        except Exception as exc:
            if hasattr(span, "record_exception"):
                span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            trace_span.set_attribute("error.type", type(exc).__name__)
            raise
        else:
            span.set_status(Status(StatusCode.OK))


@contextmanager
def trace_agent_step(
    name: str,
    input_value: str,
    *,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[OpenInferenceTraceSpan]:
    with trace_openinference_step(name, input_value, kind="AGENT", attributes=attributes) as span:
        yield span


@contextmanager
def trace_answer_step(
    name: str,
    question: str,
    *,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[OpenInferenceTraceSpan]:
    with trace_openinference_step(name, question, kind="CHAIN", attributes=attributes) as span:
        yield span


@contextmanager
def trace_llm_step(
    name: str,
    input_value: str,
    *,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[OpenInferenceTraceSpan]:
    with trace_openinference_step(name, input_value, kind="LLM", attributes=attributes) as span:
        yield span


@contextmanager
def trace_pipeline_step(
    name: str,
    input_value: str,
    *,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[OpenInferenceTraceSpan]:
    with trace_openinference_step(name, input_value, kind="CHAIN", attributes=attributes) as span:
        yield span


@contextmanager
def trace_embedding_step(
    name: str,
    input_value: str,
    *,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[OpenInferenceTraceSpan]:
    with trace_openinference_step(name, input_value, kind="EMBEDDING", attributes=attributes) as span:
        yield span


@contextmanager
def trace_retrieval_step(
    name: str,
    query: str,
    *,
    kind: str = "RETRIEVER",
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[RetrievalTraceSpan]:
    """Create a compact OpenInference span for one retrieval pipeline step."""

    with trace_openinference_step(name, query, kind=kind, attributes=attributes) as span:
        yield span


def imperial_trace_attributes(
    phase: str,
    step: str,
    attributes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    trace_attributes: dict[str, Any] = {
        _IMPERIAL_PHASE: phase,
        _IMPERIAL_STEP: step,
        _IMPERIAL_TRACE_SCHEMA_VERSION: _TRACE_SCHEMA_VERSION_VALUE,
    }
    trace_attributes.update(dict(attributes or {}))
    return trace_attributes


def trace_candidate_documents_enabled() -> bool:
    return _env_flag("IMPERIAL_RAG_TRACE_CANDIDATE_DOCUMENTS")


def trace_user_id_from_email(email: str) -> str:
    """Return a pseudonymous Phoenix user ID."""

    normalized = str(email).strip().casefold()
    if not normalized:
        return ""
    secret = os.environ.get("IMPERIAL_RAG_TRACE_USER_HASH_SECRET", "").strip()
    if secret:
        digest = hmac.new(secret.encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
        return f"user_hmac_sha256:{digest}"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"user_sha256:{digest}"


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


def openinference_document_attributes(
    key_prefix: str,
    documents: Sequence[Any],
    *,
    document_limit: int | None = None,
    content_chars: int | None = None,
) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    resolved_document_limit = (
        _env_int("IMPERIAL_RAG_TRACE_DOCUMENT_LIMIT", _TRACE_DOCUMENT_LIMIT, minimum=0)
        if document_limit is None
        else max(document_limit, 0)
    )
    resolved_content_chars = (
        _env_int("IMPERIAL_RAG_TRACE_DOCUMENT_CONTENT_CHARS", _TRACE_DOCUMENT_CONTENT_CHARS)
        if content_chars is None
        else content_chars
    )
    for index, document in enumerate(list(documents)[:resolved_document_limit]):
        content = _compact_text(str(getattr(document, "page_content", "")), resolved_content_chars)
        metadata = dict(getattr(document, "metadata", {}) or {})
        document_id = _document_id(metadata)
        score = _document_score(metadata)
        prefix = f"{key_prefix}.{index}"

        if not _hide_input_text():
            attributes[f"{prefix}.{_DOCUMENT_CONTENT}"] = content
        if document_id is not None:
            attributes[f"{prefix}.{_DOCUMENT_ID}"] = document_id
        trace_metadata = _trace_document_metadata(metadata)
        if trace_metadata:
            attributes[f"{prefix}.{_DOCUMENT_METADATA}"] = _json_value(trace_metadata)
        if score is not None:
            attributes[f"{prefix}.{_DOCUMENT_SCORE}"] = score
    return attributes


def _trace_document_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if _env_flag("IMPERIAL_RAG_TRACE_FULL_METADATA"):
        return dict(metadata)
    return {
        key: value
        for key, value in metadata.items()
        if key in _TRACE_METADATA_ALLOWLIST and value is not None
    }


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


@contextmanager
def phoenix_trace_context(
    session_id: str | None = None,
    *,
    user_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
) -> Iterator[None]:
    """Propagate private Phoenix trace context to child spans when available."""

    resolved_session_id = str(session_id).strip() if session_id is not None else ""
    resolved_user_id = str(user_id).strip() if user_id is not None else ""
    resolved_metadata: dict[str, Any] = {_IMPERIAL_TRACE_SCHEMA_VERSION: _TRACE_SCHEMA_VERSION_VALUE}
    for key, value in dict(metadata or {}).items():
        if value is not None:
            resolved_metadata[str(key)] = value
    resolved_tags = _dedupe_trace_tags(tags or [])
    try:
        from phoenix.otel import using_metadata, using_session, using_tags, using_user
    except ImportError:
        yield
        return

    with ExitStack() as stack:
        if resolved_session_id:
            stack.enter_context(using_session(resolved_session_id))
        if resolved_user_id:
            stack.enter_context(using_user(resolved_user_id))
        stack.enter_context(using_metadata(resolved_metadata))
        if resolved_tags:
            stack.enter_context(using_tags(resolved_tags))
        yield


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
        auto_instrument=_env_flag("IMPERIAL_RAG_TRACE_AUTO_INSTRUMENT"),
        batch=_env_flag("IMPERIAL_RAG_TRACE_BATCH"),
        verbose=False,
    )
    _CONFIGURED_KEY = key
    return _CONFIGURED_PROVIDER


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None:
        return max(value, minimum)
    return value


def _dedupe_trace_tags(tags: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        value = str(tag).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        resolved.append(value)
    return resolved


def _hide_inputs() -> bool:
    return _env_flag("OPENINFERENCE_HIDE_INPUTS")


def _hide_span_input(kind: str | None) -> bool:
    if _hide_inputs():
        return True
    return str(kind).strip().upper() == "LLM" and _hide_input_messages()


def _hide_outputs() -> bool:
    return _env_flag("OPENINFERENCE_HIDE_OUTPUTS")


def _hide_input_messages() -> bool:
    return _hide_inputs() or _env_flag("OPENINFERENCE_HIDE_INPUT_MESSAGES") or _env_flag("OPENINFERENCE_HIDE_LLM_PROMPTS")


def _hide_output_messages() -> bool:
    return _hide_outputs() or _env_flag("OPENINFERENCE_HIDE_OUTPUT_MESSAGES")


def _hide_llm_tools() -> bool:
    return _hide_inputs() or _env_flag("OPENINFERENCE_HIDE_LLM_TOOLS")


def _hide_input_text() -> bool:
    return _hide_inputs() or _env_flag("OPENINFERENCE_HIDE_INPUT_TEXT")


def _trace_full_final_evidence() -> bool:
    return _env_flag("IMPERIAL_RAG_TRACE_FULL_FINAL_EVIDENCE")


def _attribute_hidden(key: str, kind: str | None = None) -> bool:
    if _hide_span_input(kind) and (key == _INPUT_VALUE or key == _INPUT_MIME_TYPE or key.startswith("input.")):
        return True
    if _hide_outputs() and (key == _OUTPUT_VALUE or key == _OUTPUT_MIME_TYPE or key.startswith("output.")):
        return True
    if _hide_input_messages() and key.startswith(f"{_LLM_INPUT_MESSAGES}."):
        return True
    if _hide_output_messages() and key.startswith(f"{_LLM_OUTPUT_MESSAGES}."):
        return True
    if _hide_llm_tools() and key.startswith(f"{_LLM_TOOLS}."):
        return True
    if _hide_input_text() and key.endswith(f".{_DOCUMENT_CONTENT}"):
        return True
    return False


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
