from __future__ import annotations

import json
import logging
from io import StringIO
from types import SimpleNamespace

from imperial_rag import observability


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
