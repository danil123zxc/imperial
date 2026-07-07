from __future__ import annotations

import json
import logging
from io import StringIO
from datetime import UTC, datetime
from types import SimpleNamespace

from imperial_rag import observability
from imperial_rag.observability.eventlog import EventSchemaError, build_event_document


def setup_function() -> None:
    observability._reset_observability_for_tests()


def teardown_function() -> None:
    observability._reset_observability_for_tests()


def test_configure_observability_is_idempotent() -> None:
    settings = SimpleNamespace(log_level="INFO", log_format="json")

    first = observability.configure_observability(settings)
    second = observability.configure_observability(settings)

    assert first is second
    assert len(first.handlers) == 1
    assert first.propagate is False


def test_log_event_emits_json_with_required_fields(capsys) -> None:
    observability.configure_observability(SimpleNamespace(log_level="INFO", log_format="json"))

    observability.log_event(
        "imperial_rag.query",
        operation="query",
        status="success",
        component="cli",
        duration_ms=12,
        final_evidence=3,
    )

    payload = json.loads(capsys.readouterr().err)
    assert payload["event"] == "imperial_rag.query"
    assert payload["operation"] == "query"
    assert payload["status"] == "success"
    assert payload["component"] == "cli"
    assert payload["duration_ms"] == 12
    assert payload["final_evidence"] == 3
    assert payload["level"] == "info"
    assert payload["timestamp"].endswith("Z")


def test_plain_log_format_emits_readable_event(capsys) -> None:
    observability.configure_observability(SimpleNamespace(log_level="INFO", log_format="plain"))

    observability.log_event(
        "imperial_rag.query",
        operation="query",
        status="success",
        component="cli",
        final_evidence=3,
    )

    raw = capsys.readouterr().err
    assert "event=imperial_rag.query" in raw
    assert "operation=query" in raw
    assert "status=success" in raw
    assert "component=cli" in raw
    assert "final_evidence=3" in raw


def test_log_level_filters_info_events(capsys) -> None:
    observability.configure_observability(SimpleNamespace(log_level="ERROR", log_format="json"))

    observability.log_event("imperial_rag.query", operation="query", status="success")

    assert capsys.readouterr().err == ""


def test_log_level_honors_debug_events(capsys) -> None:
    observability.configure_observability(SimpleNamespace(log_level="DEBUG", log_format="json"))

    observability.log_event("imperial_rag.debug", level="debug", operation="debug", status="success")

    payload = json.loads(capsys.readouterr().err)
    assert payload["event"] == "imperial_rag.debug"
    assert payload["level"] == "debug"


def test_sanitize_fields_removes_sensitive_keys_recursively() -> None:
    sanitized = observability.sanitize_log_fields(
        {
            "operation": "query",
            "question": "private question",
            "answer": "private answer",
            "nested": {
                "file_name": "private.docx",
                "safe_count": 3,
                "items": [{"token": "secret-token", "failed_files": 1}],
            },
            "documents": [{"page_content": "private text"}],
        }
    )

    assert sanitized == {
        "operation": "query",
        "nested": {
            "safe_count": 3,
            "items": [{"failed_files": 1}],
        },
    }


def test_log_failure_includes_exception_type_without_private_values(capsys) -> None:
    observability.configure_observability(SimpleNamespace(log_level="INFO", log_format="json"))

    observability.log_failure(
        "query",
        RuntimeError("private answer should not appear"),
        question="private question",
        answer="private answer",
        file_path="/private/path.docx",
        final_evidence=0,
    )

    raw = capsys.readouterr().err
    payload = json.loads(raw)
    assert payload["event"] == "imperial_rag.failure"
    assert payload["operation"] == "query"
    assert payload["status"] == "error"
    assert payload["exception_type"] == "RuntimeError"
    assert payload["final_evidence"] == 0
    assert "private" not in raw


def test_json_formatter_can_be_attached_to_custom_handler() -> None:
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(observability.JsonEventFormatter())
    logger = logging.getLogger("imperial_rag.events.test")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    logger.info("", extra={"event_payload": {"event": "custom", "operation": "unit"}})

    assert json.loads(stream.getvalue())["event"] == "custom"


def test_build_event_document_maps_legacy_query_to_closed_schema(monkeypatch) -> None:
    monkeypatch.setenv("IMPERIAL_RAG_GIT_SHA", "abc123")

    document = build_event_document(
        "imperial_rag.query",
        fields={
            "operation": "query",
            "status": "success",
            "component": "cli",
            "duration_ms": 12,
            "final_evidence": 3,
            "vector_candidates": 5,
            "request_id": "request-1",
            "session_id": "session-1",
            "phoenix_session_id": "session-1",
        },
        timestamp=datetime(2026, 6, 22, tzinfo=UTC),
    )

    assert document["@timestamp"] == "2026-06-22T00:00:00Z"
    assert document["schema_version"] == "1"
    assert document["event"] == "query.completed"
    assert document["service"] == "imperial-rag"
    assert document["environment"] == "local"
    assert document["component"] == "cli"
    assert document["duration_ms"] == 12
    assert document["final_evidence"] == 3
    assert document["vector_candidates"] == 5
    assert document["git_sha"] == "abc123"
    assert "question" not in document


def test_build_event_document_rejects_unknown_fields() -> None:
    try:
        build_event_document(
            "query.completed",
            fields={
                "operation": "query",
                "status": "success",
                "component": "cli",
                "duration_ms": 1,
                "surprise": "not allowed",
            },
        )
    except EventSchemaError:
        return

    raise AssertionError("unknown fields must be rejected")


def test_build_event_document_rejects_private_canary_fields() -> None:
    private_fields = {
        "question": "private question",
        "answer": "private answer",
        "page_content": "private text",
        "path": "/private/path.docx",
        "file_path": "/private/path.docx",
        "relative_path": "private/path.docx",
        "filename": "private.docx",
        "exception_message": "private answer should not appear",
        "raw_exception": "private stack",
        "traceback": "private traceback",
        "citations": ["private citation text"],
        "sources": ["private source text"],
        "source_lists": ["private source list"],
        "documents": [{"page_content": "private"}],
        "metadata": {"source": "private.docx"},
        "api_response": {"message": "private"},
    }
    for key, value in private_fields.items():
        try:
            build_event_document(
                "query.completed",
                fields={
                    "operation": "query",
                    "status": "success",
                    "component": "cli",
                    "duration_ms": 1,
                    key: value,
                },
            )
        except EventSchemaError:
            continue
        raise AssertionError(f"{key} must be rejected")


def test_build_event_document_normalizes_failure_exception_type() -> None:
    document = build_event_document(
        "imperial_rag.failure",
        fields={
            "operation": "query",
            "status": "error",
            "component": "cli",
            "duration_ms": 4,
            "exception_type": "RuntimeError",
        },
    )

    assert document["event"] == "query.failed"
    assert document["error_type"] == "RuntimeError"
    assert "exception_type" not in document


def test_log_event_sends_closed_schema_document_to_configured_sink(capsys) -> None:
    observability.configure_observability(SimpleNamespace(log_level="INFO", log_format="json"))

    class FakeSink:
        enabled = True

        def __init__(self) -> None:
            self.documents = []

        def emit(self, document):
            self.documents.append(document)

    sink = FakeSink()
    observability._EVENT_SINK = sink

    observability.log_event(
        "imperial_rag.query",
        operation="query",
        status="success",
        component="cli",
        duration_ms=12,
        request_id="request-1",
        session_id="session-1",
        final_evidence=3,
    )

    stderr_payload = json.loads(capsys.readouterr().err)
    assert stderr_payload["event"] == "imperial_rag.query"
    assert sink.documents[0]["event"] == "query.completed"
    assert sink.documents[0]["request_id"] == "request-1"
    assert sink.documents[0]["session_id"] == "session-1"
    assert sink.documents[0]["final_evidence"] == 3


def test_log_event_delivery_failure_is_non_blocking_and_sanitized(capsys) -> None:
    observability.configure_observability(SimpleNamespace(log_level="INFO", log_format="json"))

    class BrokenSink:
        enabled = True

        def emit(self, document):
            raise RuntimeError("private path /secret.docx")

    observability._EVENT_SINK = BrokenSink()

    observability.log_event(
        "imperial_rag.query",
        operation="query",
        status="success",
        component="cli",
        duration_ms=12,
        request_id="request-1",
    )

    raw = capsys.readouterr().err
    records = [json.loads(line) for line in raw.splitlines()]
    assert records[0]["event"] == "imperial_rag.query"
    assert records[1]["event"] == "imperial_rag.eventlog_delivery_failed"
    assert records[1]["error_type"] == "RuntimeError"
    assert "private" not in raw
    assert "/secret" not in raw
