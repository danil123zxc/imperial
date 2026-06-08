from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from imperial_rag.config import Settings
import imperial_rag.tracing as tracing_module
from imperial_rag.tracing import _reset_phoenix_tracing_for_tests, configure_phoenix_tracing


def test_configure_phoenix_tracing_returns_none_when_disabled(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    monkeypatch.delenv("PHOENIX_TRACING_ENABLED", raising=False)
    monkeypatch.delenv("IMPERIAL_RAG_TRACING_ENABLED", raising=False)
    settings = Settings(workspace_root=tmp_path)

    assert configure_phoenix_tracing(settings, enabled=False) is None
    assert configure_phoenix_tracing(settings, enabled=None) is None


def test_configure_phoenix_tracing_registers_once(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    calls: list[dict[str, object]] = []
    provider = object()
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel = types.ModuleType("phoenix.otel")

    def register(**kwargs):
        calls.append(kwargs)
        return provider

    fake_otel.register = register
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.otel", fake_otel)
    settings = Settings(
        workspace_root=tmp_path,
        phoenix_project_name="trace-project",
        phoenix_collector_endpoint="http://localhost:6006/v1/traces",
    )

    assert configure_phoenix_tracing(settings, enabled=True) is provider
    assert configure_phoenix_tracing(settings, enabled=True) is provider
    assert calls == [
        {
            "project_name": "trace-project",
            "endpoint": "http://localhost:6006/v1/traces",
            "auto_instrument": True,
            "verbose": False,
        }
    ]


def test_configure_phoenix_tracing_rejects_changed_key_after_configuration(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    calls: list[dict[str, object]] = []
    provider = object()
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel = types.ModuleType("phoenix.otel")

    def register(**kwargs):
        calls.append(kwargs)
        return provider

    fake_otel.register = register
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.otel", fake_otel)
    settings_a = Settings(
        workspace_root=tmp_path,
        phoenix_project_name="project-a",
        phoenix_collector_endpoint="http://localhost:6006/v1/traces",
    )
    settings_b = Settings(
        workspace_root=tmp_path,
        phoenix_project_name="project-b",
        phoenix_collector_endpoint="http://localhost:7007/v1/traces",
    )

    assert configure_phoenix_tracing(settings_a, enabled=True) is provider
    with pytest.raises(RuntimeError, match="Phoenix tracing is already configured"):
        configure_phoenix_tracing(settings_b, enabled=True)
    assert calls == [
        {
            "project_name": "project-a",
            "endpoint": "http://localhost:6006/v1/traces",
            "auto_instrument": True,
            "verbose": False,
        }
    ]


def test_configure_phoenix_tracing_can_be_enabled_by_env(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    provider = object()
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel = types.ModuleType("phoenix.otel")
    fake_otel.register = lambda **kwargs: provider
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.otel", fake_otel)
    monkeypatch.setenv("PHOENIX_TRACING_ENABLED", "true")
    monkeypatch.setattr("imperial_rag.tracing._collector_endpoint_reachable", lambda endpoint: True)

    assert configure_phoenix_tracing(Settings(workspace_root=tmp_path), enabled=None) is provider


def test_env_enabled_phoenix_tracing_skips_when_collector_is_unreachable(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel = types.ModuleType("phoenix.otel")
    fake_otel.register = lambda **kwargs: pytest.fail("Phoenix should not register when env-enabled endpoint is down")
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.otel", fake_otel)
    monkeypatch.setenv("PHOENIX_TRACING_ENABLED", "true")
    monkeypatch.setattr("imperial_rag.tracing._collector_endpoint_reachable", lambda endpoint: False)
    settings = Settings(
        workspace_root=tmp_path,
        phoenix_collector_endpoint="http://localhost:6006/v1/traces",
    )

    assert configure_phoenix_tracing(settings, enabled=None) is None


def test_configure_phoenix_tracing_errors_clearly_when_dependency_missing(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    monkeypatch.setitem(sys.modules, "phoenix.otel", None)
    settings = Settings(workspace_root=tmp_path)

    with pytest.raises(RuntimeError, match="Phoenix tracing dependencies are missing"):
        configure_phoenix_tracing(settings, enabled=True)


def test_trace_retrieval_step_sets_openinference_attributes_and_output(monkeypatch) -> None:
    records: list[dict[str, object]] = []

    class FakeSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}
            self.status = None

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def set_status(self, status):
            self.status = status

    class FakeSpanContext:
        def __init__(self, span: FakeSpan) -> None:
            self.span = span

        def __enter__(self):
            return self.span

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeTracer:
        def start_as_current_span(self, name, attributes=None):
            span = FakeSpan()
            records.append({"name": name, "attributes": dict(attributes or {}), "span": span})
            return FakeSpanContext(span)

    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: FakeTracer())

    with tracing_module.trace_retrieval_step(
        "retrieve.vector_search",
        "возврат брака",
        attributes={"retrieval.k": 8, "retrieval.options": {"fetch_k": 80}},
    ) as span:
        span.set_attribute("retrieval.status", "ok")
        span.set_output({"count": 1, "top_documents": [{"citation_id": "S1"}]})

    assert records[0]["name"] == "retrieve.vector_search"
    assert records[0]["attributes"]["openinference.span.kind"] == "RETRIEVER"
    assert records[0]["attributes"]["input.value"] == "возврат брака"
    assert records[0]["attributes"]["retrieval.k"] == 8
    assert records[0]["attributes"]["retrieval.options"] == '{"fetch_k": 80}'
    recorded_span = records[0]["span"]
    assert recorded_span.attributes["retrieval.status"] == "ok"
    assert recorded_span.attributes["output.value"] == '{"count": 1, "top_documents": [{"citation_id": "S1"}]}'


def test_trace_agent_step_sets_parent_span_attributes_output_and_status(monkeypatch) -> None:
    records: list[dict[str, object]] = []

    class FakeSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}
            self.status = None

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def set_status(self, status):
            self.status = status

    class FakeSpanContext:
        def __init__(self, span: FakeSpan) -> None:
            self.span = span

        def __enter__(self):
            return self.span

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeTracer:
        def start_as_current_span(self, name, attributes=None):
            span = FakeSpan()
            records.append({"name": name, "attributes": dict(attributes or {}), "span": span})
            return FakeSpanContext(span)

    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: FakeTracer())

    with tracing_module.trace_agent_step(
        "imperial_rag.query",
        "Что делать с браком?",
        attributes={"runtime.workspace_root": "/tmp/imperial"},
    ) as span:
        span.set_output(
            {
                "answer": "Оформить акт. [S1]",
                "citations_valid": True,
                "evidence_count": 1,
                "retrieval": {"final_evidence": 1, "reranker": "fallback:deterministic"},
            }
        )

    assert records[0]["name"] == "imperial_rag.query"
    assert records[0]["attributes"]["openinference.span.kind"] == "AGENT"
    assert records[0]["attributes"]["input.value"] == "Что делать с браком?"
    assert records[0]["attributes"]["runtime.workspace_root"] == "/tmp/imperial"
    recorded_span = records[0]["span"]
    assert json.loads(recorded_span.attributes["output.value"]) == {
        "answer": "Оформить акт. [S1]",
        "citations_valid": True,
        "evidence_count": 1,
        "retrieval": {"final_evidence": 1, "reranker": "fallback:deterministic"},
    }
    assert recorded_span.status.status_code is tracing_module.StatusCode.OK


def test_trace_agent_step_records_errors(monkeypatch) -> None:
    records: list[dict[str, object]] = []

    class FakeSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}
            self.status = None

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def set_status(self, status):
            self.status = status

    class FakeSpanContext:
        def __init__(self, span: FakeSpan) -> None:
            self.span = span

        def __enter__(self):
            return self.span

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeTracer:
        def start_as_current_span(self, name, attributes=None):
            span = FakeSpan()
            records.append({"name": name, "attributes": dict(attributes or {}), "span": span})
            return FakeSpanContext(span)

    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: FakeTracer())

    with pytest.raises(RuntimeError, match="boom"):
        with tracing_module.trace_agent_step("imperial_rag.query", "Что делать?"):
            raise RuntimeError("boom")

    recorded_span = records[0]["span"]
    assert recorded_span.attributes["error.type"] == "RuntimeError"
    assert recorded_span.status.status_code is tracing_module.StatusCode.ERROR


def test_trace_answer_step_sets_chain_span_attributes_and_output(monkeypatch) -> None:
    records: list[dict[str, object]] = []

    class FakeSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}
            self.status = None

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def set_status(self, status):
            self.status = status

    class FakeSpanContext:
        def __init__(self, span: FakeSpan) -> None:
            self.span = span

        def __enter__(self):
            return self.span

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeTracer:
        def start_as_current_span(self, name, attributes=None):
            span = FakeSpan()
            records.append({"name": name, "attributes": dict(attributes or {}), "span": span})
            return FakeSpanContext(span)

    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: FakeTracer())

    with tracing_module.trace_answer_step(
        "answer.generate",
        "Что делать с браком?",
        attributes={"answer.evidence_count": 1},
    ) as span:
        span.set_output({"answer": "Оформить акт. [S1]", "citations_valid": True, "refused": False})

    assert records[0]["name"] == "answer.generate"
    assert records[0]["attributes"]["openinference.span.kind"] == "CHAIN"
    assert records[0]["attributes"]["input.value"] == "Что делать с браком?"
    assert records[0]["attributes"]["answer.evidence_count"] == 1
    recorded_span = records[0]["span"]
    assert json.loads(recorded_span.attributes["output.value"]) == {
        "answer": "Оформить акт. [S1]",
        "citations_valid": True,
        "refused": False,
    }
    assert recorded_span.status.status_code is tracing_module.StatusCode.OK


def test_trace_span_sets_native_retrieval_documents(monkeypatch) -> None:
    records: list[dict[str, object]] = []

    class FakeSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def set_status(self, status):
            pass

    class FakeSpanContext:
        def __enter__(self):
            span = FakeSpan()
            records.append({"span": span})
            return span

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeTracer:
        def start_as_current_span(self, name, attributes=None):
            return FakeSpanContext()

    document = type(
        "Document",
        (),
        {
            "page_content": "Порядок возврата брака",
            "metadata": {
                "citation_id": "docs/return.docx#body:chunk-0",
                "chunk_id": "chunk-0",
                "file_name": "return.docx",
                "_keyword_score": -2.5,
            },
        },
    )()
    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: FakeTracer())

    with tracing_module.trace_retrieval_step("retrieve.keyword_search", "возврат") as span:
        span.set_retrieval_documents([document])

    recorded_span = records[0]["span"]
    assert recorded_span.attributes["retrieval.documents.0.document.content"] == "Порядок возврата брака"
    assert recorded_span.attributes["retrieval.documents.0.document.id"] == "chunk-0"
    assert json.loads(recorded_span.attributes["retrieval.documents.0.document.metadata"]) == {
        "citation_id": "docs/return.docx#body:chunk-0",
        "chunk_id": "chunk-0",
        "file_name": "return.docx",
        "_keyword_score": -2.5,
    }
    assert recorded_span.attributes["retrieval.documents.0.document.score"] == -2.5


def test_trace_span_sets_native_reranker_documents(monkeypatch) -> None:
    records: list[dict[str, object]] = []

    class FakeSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def set_status(self, status):
            pass

    class FakeSpanContext:
        def __enter__(self):
            span = FakeSpan()
            records.append({"span": span})
            return span

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeTracer:
        def start_as_current_span(self, name, attributes=None):
            return FakeSpanContext()

    input_document = type("Document", (), {"page_content": "candidate", "metadata": {"chunk_id": "in"}})()
    output_document = type("Document", (), {"page_content": "reranked", "metadata": {"chunk_id": "out"}})()
    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: FakeTracer())

    with tracing_module.trace_retrieval_step("retrieve.rerank", "возврат", kind="RERANKER") as span:
        span.set_reranker_input_documents([input_document])
        span.set_reranker_output_documents([output_document])

    recorded_span = records[0]["span"]
    assert recorded_span.attributes["reranker.input_documents.0.document.content"] == "candidate"
    assert recorded_span.attributes["reranker.input_documents.0.document.id"] == "in"
    assert recorded_span.attributes["reranker.output_documents.0.document.content"] == "reranked"
    assert recorded_span.attributes["reranker.output_documents.0.document.id"] == "out"


def test_retrieval_documents_preview_keeps_trace_payload_compact() -> None:
    document = type(
        "Document",
        (),
        {
            "page_content": "  Очень длинный   текст документа " * 20,
            "metadata": {
                "citation_id": "S1",
                "chunk_id": "chunk-1",
                "file_name": "policy.docx",
                "source_type": "body",
            },
        },
    )()

    preview = tracing_module.retrieval_documents_preview([document], content_chars=40)

    assert preview == [
        {
            "rank": 0,
            "citation_id": "S1",
            "chunk_id": "chunk-1",
            "file_name": "policy.docx",
            "source_type": "body",
            "preview": "Очень длинный текст документа Очень длин...",
        }
    ]
