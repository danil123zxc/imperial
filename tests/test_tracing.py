from __future__ import annotations

import json
import hashlib
import hmac
import sys
import types
from pathlib import Path
from typing import Any, TypedDict

import pytest

from imperial_rag.config import Settings
import imperial_rag.tracing as tracing_module
from imperial_rag.tracing import _reset_phoenix_tracing_for_tests, configure_phoenix_tracing


class FakeSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.status: Any = None
        self.exceptions: list[object] = []

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value

    def set_status(self, status: object) -> None:
        self.status = status

    def record_exception(self, exc: object) -> None:
        self.exceptions.append(exc)


class TraceRecord(TypedDict):
    name: str
    attributes: dict[str, object]
    span: FakeSpan


class FakeSpanContext:
    def __init__(self, span: FakeSpan) -> None:
        self.span = span

    def __enter__(self) -> FakeSpan:
        return self.span

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False


class FakeTracer:
    def __init__(self, records: list[TraceRecord]) -> None:
        self.records = records

    def start_as_current_span(self, name: str, attributes: dict[str, object] | None = None) -> FakeSpanContext:
        span = FakeSpan()
        self.records.append({"name": name, "attributes": dict(attributes or {}), "span": span})
        return FakeSpanContext(span)


def make_fake_tracer(records: list[TraceRecord]) -> object:
    return FakeTracer(records)


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
    fake_otel: Any = types.ModuleType("phoenix.otel")

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
            "auto_instrument": False,
            "batch": False,
            "verbose": False,
        }
    ]


def test_configure_phoenix_tracing_rejects_changed_key_after_configuration(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    calls: list[dict[str, object]] = []
    provider = object()
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel: Any = types.ModuleType("phoenix.otel")

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
            "auto_instrument": False,
            "batch": False,
            "verbose": False,
        }
    ]


def test_configure_phoenix_tracing_honors_trace_batch_env(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    calls: list[dict[str, object]] = []
    provider = object()
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel: Any = types.ModuleType("phoenix.otel")
    fake_otel.register = lambda **kwargs: calls.append(kwargs) or provider
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.otel", fake_otel)
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_BATCH", "true")

    assert configure_phoenix_tracing(Settings(workspace_root=tmp_path), enabled=True) is provider

    assert calls[0]["batch"] is True


def test_configure_phoenix_tracing_honors_auto_instrument_env(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    calls: list[dict[str, object]] = []
    provider = object()
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel: Any = types.ModuleType("phoenix.otel")
    fake_otel.register = lambda **kwargs: calls.append(kwargs) or provider
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.otel", fake_otel)
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_AUTO_INSTRUMENT", "true")

    assert configure_phoenix_tracing(Settings(workspace_root=tmp_path), enabled=True) is provider

    assert calls[0]["auto_instrument"] is True


def test_configure_phoenix_tracing_can_be_enabled_by_env(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    provider = object()
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel: Any = types.ModuleType("phoenix.otel")
    fake_otel.register = lambda **kwargs: provider
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.otel", fake_otel)
    monkeypatch.setenv("PHOENIX_TRACING_ENABLED", "true")
    monkeypatch.setattr("imperial_rag.tracing._collector_endpoint_reachable", lambda endpoint: True)

    assert configure_phoenix_tracing(Settings(workspace_root=tmp_path), enabled=None) is provider


def test_env_enabled_phoenix_tracing_skips_when_collector_is_unreachable(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel: Any = types.ModuleType("phoenix.otel")
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
    records: list[TraceRecord] = []

    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with tracing_module.trace_retrieval_step(
        "retrieval.vector_search",
        "возврат брака",
        attributes={"retrieval.k": 8, "retrieval.options": {"fetch_k": 80}},
    ) as span:
        span.set_attribute("retrieval.status", "ok")
        span.set_output({"count": 1, "top_documents": [{"citation_id": "S1"}]})

    assert records[0]["name"] == "retrieval.vector_search"
    assert records[0]["attributes"]["openinference.span.kind"] == "RETRIEVER"
    assert records[0]["attributes"]["input.value"] == "возврат брака"
    assert records[0]["attributes"]["input.mime_type"] == "text/plain"
    assert records[0]["attributes"]["retrieval.k"] == 8
    assert records[0]["attributes"]["retrieval.options"] == '{"fetch_k": 80}'
    recorded_span = records[0]["span"]
    assert recorded_span.attributes["retrieval.status"] == "ok"
    assert recorded_span.attributes["output.value"] == '{"count": 1, "top_documents": [{"citation_id": "S1"}]}'
    assert recorded_span.attributes["output.mime_type"] == "application/json"


def test_trace_agent_step_sets_parent_span_attributes_output_and_status(monkeypatch) -> None:
    records: list[TraceRecord] = []

    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

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
    assert json.loads(str(recorded_span.attributes["output.value"])) == {
        "answer": "Оформить акт. [S1]",
        "citations_valid": True,
        "evidence_count": 1,
        "retrieval": {"final_evidence": 1, "reranker": "fallback:deterministic"},
    }
    assert recorded_span.status.status_code is tracing_module.StatusCode.OK


def test_trace_agent_step_records_errors(monkeypatch) -> None:
    records: list[TraceRecord] = []

    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with pytest.raises(RuntimeError, match="boom"):
        with tracing_module.trace_agent_step("imperial_rag.query", "Что делать?"):
            raise RuntimeError("boom")

    recorded_span = records[0]["span"]
    assert recorded_span.attributes["error.type"] == "RuntimeError"
    assert len(recorded_span.exceptions) == 1
    assert isinstance(recorded_span.exceptions[0], RuntimeError)
    assert recorded_span.status.status_code is tracing_module.StatusCode.ERROR


def test_trace_answer_step_sets_chain_span_attributes_and_output(monkeypatch) -> None:
    records: list[TraceRecord] = []

    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

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
    assert json.loads(str(recorded_span.attributes["output.value"])) == {
        "answer": "Оформить акт. [S1]",
        "citations_valid": True,
        "refused": False,
    }
    assert recorded_span.status.status_code is tracing_module.StatusCode.OK


def test_trace_llm_step_sets_openinference_llm_attributes_and_messages(monkeypatch) -> None:
    records: list[TraceRecord] = []

    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with tracing_module.trace_llm_step(
        "answer.call_model",
        "Что делать?",
        attributes={"llm.provider": "dashscope", "llm.model_name": "qwen3.7-plus"},
    ) as span:
        span.set_attribute("llm.input_messages.0.message.role", "user")
        span.set_attribute("llm.input_messages.0.message.content", "Что делать?")
        span.set_attribute("llm.output_messages.0.message.role", "assistant")
        span.set_attribute("llm.output_messages.0.message.content", "Оформить акт. [S1]")
        span.set_output({"answer_chars": 19})

    assert records[0]["name"] == "answer.call_model"
    assert records[0]["attributes"]["openinference.span.kind"] == "LLM"
    assert records[0]["attributes"]["input.value"] == "Что делать?"
    assert records[0]["attributes"]["llm.provider"] == "dashscope"
    assert records[0]["attributes"]["llm.model_name"] == "qwen3.7-plus"
    recorded_span = records[0]["span"]
    assert recorded_span.attributes["llm.input_messages.0.message.role"] == "user"
    assert recorded_span.attributes["llm.input_messages.0.message.content"] == "Что делать?"
    assert recorded_span.attributes["llm.output_messages.0.message.role"] == "assistant"
    assert recorded_span.attributes["llm.output_messages.0.message.content"] == "Оформить акт. [S1]"
    assert recorded_span.attributes["output.value"] == '{"answer_chars": 19}'
    assert recorded_span.status.status_code is tracing_module.StatusCode.OK


def test_trace_pipeline_and_embedding_steps_set_openinference_kinds(monkeypatch) -> None:
    records: list[TraceRecord] = []

    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with tracing_module.trace_pipeline_step(
        "ingest.build_chunks",
        "corpus",
        attributes={"ingest.document_count": 2},
    ) as span:
        span.set_output({"chunk_count": 3})
    with tracing_module.trace_embedding_step(
        "embedding.dashscope.batch",
        "document",
        attributes={"embedding.batch_size": 2},
    ) as span:
        span.set_output({"vector_count": 2})

    assert records[0]["name"] == "ingest.build_chunks"
    assert records[0]["attributes"]["openinference.span.kind"] == "CHAIN"
    assert records[0]["attributes"]["input.value"] == "corpus"
    assert records[0]["attributes"]["ingest.document_count"] == 2
    assert records[0]["span"].attributes["output.value"] == '{"chunk_count": 3}'
    assert records[0]["span"].status.status_code is tracing_module.StatusCode.OK
    assert records[1]["name"] == "embedding.dashscope.batch"
    assert records[1]["attributes"]["openinference.span.kind"] == "EMBEDDING"
    assert records[1]["attributes"]["input.value"] == "document"
    assert records[1]["attributes"]["embedding.batch_size"] == 2
    assert records[1]["span"].attributes["output.value"] == '{"vector_count": 2}'
    assert records[1]["span"].status.status_code is tracing_module.StatusCode.OK


def test_trace_lineage_attributes_are_applied_to_embedding_children(monkeypatch) -> None:
    records: list[TraceRecord] = []

    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with tracing_module.trace_lineage_attributes(
        {
            "imperial.ingest_run_id": "ingest_123",
            "imperial.corpus_version": "corpus_sha256:aaa",
            "imperial.embedding_model": "text-embedding-v4:2048",
            "imperial.qdrant_collection": "imperial_chunks_qwen",
        }
    ):
        with tracing_module.trace_embedding_step("embedding.dashscope.batch", "document"):
            pass

    assert records[0]["name"] == "embedding.dashscope.batch"
    assert records[0]["attributes"]["imperial.ingest_run_id"] == "ingest_123"
    assert records[0]["attributes"]["imperial.corpus_version"] == "corpus_sha256:aaa"
    assert records[0]["attributes"]["imperial.embedding_model"] == "text-embedding-v4:2048"
    assert records[0]["attributes"]["imperial.qdrant_collection"] == "imperial_chunks_qwen"


def test_trace_span_sets_native_retrieval_documents(monkeypatch) -> None:
    records: list[TraceRecord] = []

    document = type(
        "Document",
        (),
        {
            "page_content": "Порядок возврата брака",
            "metadata": {
                "citation_id": "docs/return.docx#body:chunk-0",
                "chunk_id": "chunk-0",
                "file_name": "return.docx",
                "relative_path": "docs/return.docx",
                "page_number": 7,
                "sheet_name": "Returns",
                "_keyword_score": -2.5,
                "_keyword_match_mode": "relaxed_drop_one",
                "file_path": "/private/docs/return.docx",
                "file_hash": "secret-hash",
                "parent_folder": "/private/docs",
            },
        },
    )()
    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with tracing_module.trace_retrieval_step("retrieval.keyword_search", "возврат") as span:
        span.set_retrieval_documents([document])

    recorded_span = records[0]["span"]
    assert recorded_span.attributes["retrieval.documents.0.document.content"] == "Порядок возврата брака"
    assert recorded_span.attributes["retrieval.documents.0.document.id"] == "chunk-0"
    assert json.loads(str(recorded_span.attributes["retrieval.documents.0.document.metadata"])) == {
        "citation_id": "docs/return.docx#body:chunk-0",
        "chunk_id": "chunk-0",
        "file_name": "return.docx",
        "relative_path": "docs/return.docx",
        "page_number": 7,
        "sheet_name": "Returns",
        "_keyword_score": -2.5,
        "_keyword_match_mode": "relaxed_drop_one",
    }
    assert recorded_span.attributes["retrieval.documents.0.document.score"] == -2.5


def test_trace_span_traces_all_native_retrieval_documents_by_default(monkeypatch) -> None:
    records: list[TraceRecord] = []

    documents = [
        type(
            "Document",
            (),
            {
                "page_content": f"doc-{index} " + ("x" * 900),
                "metadata": {"chunk_id": f"chunk-{index}"},
            },
        )()
        for index in range(11)
    ]
    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with tracing_module.trace_retrieval_step("retrieval.vector_search", "возврат") as span:
        span.set_retrieval_documents(documents)

    recorded_span = records[0]["span"]
    assert recorded_span.attributes["retrieval.documents.0.document.content"] == f"doc-0 {'x' * 794}..."
    assert recorded_span.attributes["retrieval.documents.10.document.id"] == "chunk-10"
    assert recorded_span.attributes["retrieval.documents.10.document.content"] == f"doc-10 {'x' * 793}..."


def test_trace_document_limits_and_truncation_are_configurable(monkeypatch) -> None:
    records: list[TraceRecord] = []

    documents = [
        type(
            "Document",
            (),
            {
                "page_content": f"doc-{index} " + ("z" * 20),
                "metadata": {"chunk_id": f"chunk-{index}"},
            },
        )()
        for index in range(3)
    ]
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_DOCUMENT_LIMIT", "1")
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_DOCUMENT_CONTENT_CHARS", "12")
    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with tracing_module.trace_retrieval_step("retrieval.vector_search", "возврат") as span:
        span.set_retrieval_documents(documents)

    recorded_span = records[0]["span"]
    assert recorded_span.attributes["retrieval.documents.0.document.content"] == "doc-0 zzzzzz..."
    assert "retrieval.documents.1.document.content" not in recorded_span.attributes


def test_final_evidence_documents_can_store_full_content_when_enabled(monkeypatch) -> None:
    records: list[TraceRecord] = []

    content = "final evidence " + ("x" * 900)
    document = type("Document", (), {"page_content": content, "metadata": {"chunk_id": "final"}})()
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_FULL_FINAL_EVIDENCE", "true")
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_DOCUMENT_CONTENT_CHARS", "12")
    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with tracing_module.trace_retrieval_step("retrieval.final_evidence", "возврат") as span:
        span.set_final_evidence_documents([document])

    recorded_span = records[0]["span"]
    assert recorded_span.attributes["retrieval.documents.0.document.content"] == content
    assert recorded_span.attributes["retrieval.documents.0.document.id"] == "final"


def test_trace_document_metadata_can_include_full_metadata_when_enabled(monkeypatch) -> None:
    document = type(
        "Document",
        (),
        {
            "page_content": "text",
            "metadata": {
                "chunk_id": "chunk-1",
                "file_path": "/private/docs/source.docx",
                "file_hash": "secret-hash",
                "parent_folder": "/private/docs",
            },
        },
    )()
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_FULL_METADATA", "1")

    attrs = tracing_module.openinference_document_attributes("retrieval.documents", [document])

    assert json.loads(attrs["retrieval.documents.0.document.metadata"]) == {
        "chunk_id": "chunk-1",
        "file_path": "/private/docs/source.docx",
        "file_hash": "secret-hash",
        "parent_folder": "/private/docs",
    }


def test_openinference_redaction_env_hides_manual_inputs_outputs_and_document_text(monkeypatch) -> None:
    records: list[TraceRecord] = []

    document = type("Document", (), {"page_content": "private corpus text", "metadata": {"chunk_id": "chunk-1"}})()
    monkeypatch.setenv("OPENINFERENCE_HIDE_INPUTS", "true")
    monkeypatch.setenv("OPENINFERENCE_HIDE_OUTPUTS", "true")
    monkeypatch.setenv("OPENINFERENCE_HIDE_INPUT_TEXT", "true")
    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with tracing_module.trace_retrieval_step("retrieval.keyword_search", "private query") as span:
        span.set_output({"count": 1})
        span.set_retrieval_documents([document])
        span.set_final_evidence_documents([document])

    assert records[0]["attributes"] == {"openinference.span.kind": "RETRIEVER"}
    recorded_span = records[0]["span"]
    assert "output.value" not in recorded_span.attributes
    assert "retrieval.documents.0.document.content" not in recorded_span.attributes
    assert recorded_span.attributes["retrieval.documents.0.document.id"] == "chunk-1"


def test_openinference_redaction_env_hides_llm_messages(monkeypatch) -> None:
    records: list[TraceRecord] = []

    monkeypatch.setenv("OPENINFERENCE_HIDE_INPUT_MESSAGES", "true")
    monkeypatch.setenv("OPENINFERENCE_HIDE_OUTPUT_MESSAGES", "true")
    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with tracing_module.trace_llm_step("answer.call_model", "private question") as span:
        span.set_attribute("llm.input_messages.0.message.role", "user")
        span.set_attribute("llm.input_messages.0.message.content", "private question")
        span.set_attribute("llm.output_messages.0.message.role", "assistant")
        span.set_attribute("llm.output_messages.0.message.content", "private answer")
        span.set_attribute("llm.model_name", "qwen3.7-plus")

    start_attrs = records[0]["attributes"]
    recorded_span = records[0]["span"]
    assert start_attrs["openinference.span.kind"] == "LLM"
    assert "input.value" not in start_attrs
    assert "input.mime_type" not in start_attrs
    assert "llm.input_messages.0.message.role" not in recorded_span.attributes
    assert "llm.input_messages.0.message.content" not in recorded_span.attributes
    assert "llm.output_messages.0.message.role" not in recorded_span.attributes
    assert "llm.output_messages.0.message.content" not in recorded_span.attributes
    assert recorded_span.attributes["llm.model_name"] == "qwen3.7-plus"


def test_openinference_redaction_env_hides_llm_prompt_input_value(monkeypatch) -> None:
    records: list[TraceRecord] = []

    monkeypatch.setenv("OPENINFERENCE_HIDE_LLM_PROMPTS", "true")
    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with tracing_module.trace_llm_step("answer.call_model", "private question") as span:
        span.set_attribute("llm.input_messages.0.message.content", "private question")
        span.set_attribute("llm.model_name", "qwen3.7-plus")

    start_attrs = records[0]["attributes"]
    recorded_span = records[0]["span"]
    assert start_attrs == {"openinference.span.kind": "LLM"}
    assert "llm.input_messages.0.message.content" not in recorded_span.attributes
    assert recorded_span.attributes["llm.model_name"] == "qwen3.7-plus"


def test_openinference_redaction_env_hides_late_llm_prompt_input_value(monkeypatch) -> None:
    records: list[TraceRecord] = []

    monkeypatch.setenv("OPENINFERENCE_HIDE_INPUTS", "false")
    monkeypatch.setenv("OPENINFERENCE_HIDE_LLM_PROMPTS", "true")
    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with tracing_module.trace_llm_step("answer.call_model", "private question") as span:
        span.set_attribute("input.value", "secret")
        span.set_attribute("llm.model_name", "qwen3.7-plus")

    recorded_span = records[0]["span"]
    assert "input.value" not in recorded_span.attributes
    assert recorded_span.attributes["llm.model_name"] == "qwen3.7-plus"


def test_trace_span_sets_native_reranker_documents(monkeypatch) -> None:
    records: list[TraceRecord] = []

    input_document = type("Document", (), {"page_content": "candidate", "metadata": {"chunk_id": "in"}})()
    output_document = type("Document", (), {"page_content": "reranked", "metadata": {"chunk_id": "out"}})()
    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with tracing_module.trace_retrieval_step("retrieval.rerank", "возврат", kind="RERANKER") as span:
        span.set_reranker_input_documents([input_document])
        span.set_reranker_output_documents([output_document])

    recorded_span = records[0]["span"]
    assert recorded_span.attributes["reranker.input_documents.0.document.content"] == "candidate"
    assert recorded_span.attributes["reranker.input_documents.0.document.id"] == "in"
    assert recorded_span.attributes["reranker.output_documents.0.document.content"] == "reranked"
    assert recorded_span.attributes["reranker.output_documents.0.document.id"] == "out"


def test_trace_span_traces_all_native_reranker_documents_by_default(monkeypatch) -> None:
    records: list[TraceRecord] = []

    documents = [
        type(
            "Document",
            (),
            {
                "page_content": f"candidate-{index} " + ("y" * 900),
                "metadata": {"chunk_id": f"candidate-{index}"},
            },
        )()
        for index in range(11)
    ]
    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with tracing_module.trace_retrieval_step("retrieval.rerank", "возврат", kind="RERANKER") as span:
        span.set_reranker_input_documents(documents)
        span.set_reranker_output_documents(documents)

    recorded_span = records[0]["span"]
    assert recorded_span.attributes["reranker.input_documents.0.document.content"] == f"candidate-0 {'y' * 788}..."
    assert recorded_span.attributes["reranker.output_documents.10.document.id"] == "candidate-10"
    assert recorded_span.attributes["reranker.input_documents.10.document.content"] == f"candidate-10 {'y' * 787}..."
    assert recorded_span.attributes["reranker.output_documents.10.document.content"] == f"candidate-10 {'y' * 787}..."


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


def test_retrieval_documents_preview_omits_text_when_input_text_hidden(monkeypatch) -> None:
    document = type(
        "Document",
        (),
        {
            "page_content": "Private corpus text that should not be serialized into output.value",
            "metadata": {
                "citation_id": "S1",
                "chunk_id": "chunk-1",
                "file_name": "policy.docx",
                "source_type": "body",
            },
        },
    )()
    monkeypatch.setenv("OPENINFERENCE_HIDE_INPUT_TEXT", "true")

    preview = tracing_module.retrieval_documents_preview([document])

    assert preview == [
        {
            "rank": 0,
            "citation_id": "S1",
            "chunk_id": "chunk-1",
            "file_name": "policy.docx",
            "source_type": "body",
        }
    ]


def test_phoenix_trace_context_uses_session_user_metadata_and_tags(monkeypatch) -> None:
    calls = []
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel: Any = types.ModuleType("phoenix.otel")

    class FakeContext:
        def __init__(self, name, payload):
            self.name = name
            self.payload = payload

        def __enter__(self):
            calls.append(("enter", self.name, self.payload))

        def __exit__(self, exc_type, exc, traceback):
            calls.append(("exit", self.name, self.payload))
            return False

    fake_otel.using_session = lambda session_id: FakeContext("session", session_id)
    fake_otel.using_user = lambda user_id: FakeContext("user", user_id)
    fake_otel.using_metadata = lambda metadata: FakeContext("metadata", metadata)
    fake_otel.using_tags = lambda tags: FakeContext("tags", tags)
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.otel", fake_otel)

    with tracing_module.phoenix_trace_context(
        "session-123",
        user_id="user_abc",
        metadata={"entrypoint": "cli"},
        tags=["imperial-rag", "cli"],
    ):
        calls.append(("body", None, None))

    assert calls == [
        ("enter", "session", "session-123"),
        ("enter", "user", "user_abc"),
        (
            "enter",
            "metadata",
            {"imperial.trace_schema_version": "rag-v2", "entrypoint": "cli"},
        ),
        ("enter", "tags", ["imperial-rag", "cli"]),
        ("body", None, None),
        ("exit", "tags", ["imperial-rag", "cli"]),
        (
            "exit",
            "metadata",
            {"imperial.trace_schema_version": "rag-v2", "entrypoint": "cli"},
        ),
        ("exit", "user", "user_abc"),
        ("exit", "session", "session-123"),
    ]


def test_phoenix_trace_context_ignores_empty_session_but_keeps_metadata(monkeypatch) -> None:
    calls = []
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel: Any = types.ModuleType("phoenix.otel")
    fake_otel.using_session = lambda **kwargs: pytest.fail("empty session should not enter Phoenix context")
    fake_otel.using_user = lambda *args, **kwargs: pytest.fail("missing user should not enter Phoenix context")
    fake_otel.using_tags = lambda *args, **kwargs: pytest.fail("missing tags should not enter Phoenix context")

    class FakeMetadataContext:
        def __init__(self, metadata):
            self.metadata = metadata

        def __enter__(self):
            calls.append(("enter", self.metadata))

        def __exit__(self, exc_type, exc, traceback):
            calls.append(("exit", self.metadata))
            return False

    fake_otel.using_metadata = lambda metadata: FakeMetadataContext(metadata)
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.otel", fake_otel)

    with tracing_module.phoenix_trace_context(None):
        pass
    with tracing_module.phoenix_trace_context("  "):
        pass

    assert calls == [
        ("enter", {"imperial.trace_schema_version": "rag-v2"}),
        ("exit", {"imperial.trace_schema_version": "rag-v2"}),
        ("enter", {"imperial.trace_schema_version": "rag-v2"}),
        ("exit", {"imperial.trace_schema_version": "rag-v2"}),
    ]


def test_trace_user_id_from_email_is_deterministic_and_pseudonymous() -> None:
    expected_hash = hashlib.sha256("user@example.com".encode("utf-8")).hexdigest()[:16]

    user_id = tracing_module.trace_user_id_from_email(" User@Example.COM ")

    assert user_id == f"user_sha256:{expected_hash}"
    assert "example.com" not in user_id


def test_trace_user_id_from_email_returns_empty_for_missing_email() -> None:
    assert tracing_module.trace_user_id_from_email(None) == ""


def test_trace_user_id_from_email_uses_local_hmac_secret(monkeypatch) -> None:
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_USER_HASH_SECRET", "local secret")
    expected_hash = hmac.new(
        b"local secret",
        b"user@example.com",
        hashlib.sha256,
    ).hexdigest()[:16]

    user_id = tracing_module.trace_user_id_from_email(" User@Example.COM ")

    assert user_id == f"user_hmac_sha256:{expected_hash}"
    assert "example.com" not in user_id


def test_trace_candidate_documents_enabled_reads_env(monkeypatch) -> None:
    monkeypatch.delenv("IMPERIAL_RAG_TRACE_CANDIDATE_DOCUMENTS", raising=False)
    monkeypatch.delenv("IMPERIAL_RAG_TRACE_MODE", raising=False)
    assert tracing_module.trace_candidate_documents_enabled() is False

    monkeypatch.setenv("IMPERIAL_RAG_TRACE_CANDIDATE_DOCUMENTS", "true")
    assert tracing_module.trace_candidate_documents_enabled() is True

    monkeypatch.setenv("IMPERIAL_RAG_TRACE_CANDIDATE_DOCUMENTS", "false")
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_MODE", "retrieval_debug")
    assert tracing_module.trace_candidate_documents_enabled() is True


def test_trace_mode_defaults_to_compact_and_accepts_retrieval_debug(monkeypatch) -> None:
    monkeypatch.delenv("IMPERIAL_RAG_TRACE_MODE", raising=False)
    assert tracing_module.trace_mode() == "compact"

    monkeypatch.setenv("IMPERIAL_RAG_TRACE_MODE", "retrieval_debug")
    assert tracing_module.trace_mode() == "retrieval_debug"

    monkeypatch.setenv("IMPERIAL_RAG_TRACE_MODE", "unexpected")
    assert tracing_module.trace_mode() == "compact"


def test_retrieval_debug_redaction_hides_candidate_document_content(monkeypatch) -> None:
    records: list[TraceRecord] = []

    document = type("Document", (), {"page_content": "private corpus text", "metadata": {"chunk_id": "chunk-1"}})()
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_MODE", "retrieval_debug")
    monkeypatch.setenv("OPENINFERENCE_HIDE_INPUT_TEXT", "true")
    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: make_fake_tracer(records))

    with tracing_module.trace_retrieval_step("retrieval.vector_search", "private query") as span:
        span.set_retrieval_documents([document])

    recorded_span = records[0]["span"]
    assert tracing_module.trace_candidate_documents_enabled() is True
    assert "retrieval.documents.0.document.content" not in recorded_span.attributes
    assert recorded_span.attributes["retrieval.documents.0.document.id"] == "chunk-1"


def test_trace_internal_spans_suppressed_defaults_to_true(monkeypatch) -> None:
    monkeypatch.delenv("IMPERIAL_RAG_TRACE_SUPPRESS_INTERNALS", raising=False)
    assert tracing_module.trace_internal_spans_suppressed() is True

    monkeypatch.setenv("IMPERIAL_RAG_TRACE_SUPPRESS_INTERNALS", "false")
    assert tracing_module.trace_internal_spans_suppressed() is False

    monkeypatch.setenv("IMPERIAL_RAG_TRACE_SUPPRESS_INTERNALS", "true")
    assert tracing_module.trace_internal_spans_suppressed() is True


def test_trace_provenance_attributes_include_runtime_identity_and_flags(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IMPERIAL_RAG_GIT_SHA", "abc1234")
    monkeypatch.setenv("IMPERIAL_RAG_IMAGE_DIGEST", "sha256:deadbeef")
    monkeypatch.setenv("IMPERIAL_RAG_IMAGE_TAG", "imperial:test")
    monkeypatch.setenv("IMPERIAL_RAG_APP_VERSION", "2026.06.22")
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_AUTO_INSTRUMENT", "false")
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_SUPPRESS_INTERNALS", "true")
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_MODE", "compact")
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_CANDIDATE_DOCUMENTS", "false")
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_FULL_FINAL_EVIDENCE", "false")
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_FULL_METADATA", "false")
    monkeypatch.setenv("OPENINFERENCE_HIDE_INPUTS", "true")
    settings = Settings(
        workspace_root=tmp_path,
        phoenix_project_name="imperial-rag-readable",
    )

    attrs = tracing_module.trace_provenance_attributes(settings, run_id="run-123")

    assert attrs == {
        "imperial.trace_run_id": "run-123",
        "imperial.phoenix_project": "imperial-rag-readable",
        "imperial.git_sha": "abc1234",
        "imperial.image_digest": "sha256:deadbeef",
        "imperial.image_tag": "imperial:test",
        "imperial.app_version": "2026.06.22",
        "imperial.keyword_index": "imperial_keyword_chunks",
        "imperial.qdrant_collection": "imperial_chunks_qwen",
        "imperial.embedding_model": "text-embedding-v4:2048",
        "imperial.index_fresh": "unknown",
        "imperial.trace_mode": "compact",
        "imperial.trace_auto_instrument": False,
        "imperial.trace_suppress_internals": True,
        "imperial.trace_candidate_documents": False,
        "imperial.trace_full_final_evidence": False,
        "imperial.trace_full_metadata": False,
        "openinference.hide_inputs": True,
        "openinference.hide_outputs": False,
        "openinference.hide_input_text": True,
        "openinference.hide_input_messages": True,
        "openinference.hide_output_messages": False,
        "openinference.hide_llm_prompts": False,
        "openinference.hide_llm_tools": True,
    }


def test_trace_provenance_attributes_include_latest_index_lineage(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IMPERIAL_RAG_GIT_SHA", "abc1234")
    settings = Settings(
        workspace_root=tmp_path,
        elasticsearch_index="keyword_live",
        qdrant_collection="qdrant_live",
    )
    settings.extraction_root.mkdir(parents=True)
    (settings.extraction_root / "index-lineage.json").write_text(
        json.dumps(
            {
                "ingest_run_id": "ingest_123",
                "corpus_version": "corpus_sha256:aaa",
                "index_version": "index_sha256:bbb",
                "keyword_index": "keyword_live",
                "qdrant_collection": "qdrant_live",
                "embedding_model": "text-embedding-v4:2048",
            }
        ),
        encoding="utf-8",
    )

    attrs = tracing_module.trace_provenance_attributes(settings, run_id="run-123")

    assert attrs["imperial.ingest_run_id"] == "ingest_123"
    assert attrs["imperial.corpus_version"] == "corpus_sha256:aaa"
    assert attrs["imperial.index_version"] == "index_sha256:bbb"
    assert attrs["imperial.keyword_index"] == "keyword_live"
    assert attrs["imperial.qdrant_collection"] == "qdrant_live"
    assert attrs["imperial.embedding_model"] == "text-embedding-v4:2048"
    assert attrs["imperial.index_fresh"] == "fresh"


def test_trace_provenance_attributes_marks_lineage_stale(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IMPERIAL_RAG_GIT_SHA", "abc1234")
    settings = Settings(
        workspace_root=tmp_path,
        elasticsearch_index="keyword_current",
        qdrant_collection="qdrant_current",
    )
    settings.extraction_root.mkdir(parents=True)
    (settings.extraction_root / "index-lineage.json").write_text(
        json.dumps(
            {
                "ingest_run_id": "ingest_123",
                "corpus_version": "corpus_sha256:aaa",
                "index_version": "index_sha256:bbb",
                "keyword_index": "keyword_old",
                "qdrant_collection": "qdrant_current",
                "embedding_model": "text-embedding-v4:2048",
            }
        ),
        encoding="utf-8",
    )

    attrs = tracing_module.trace_provenance_attributes(settings, run_id="run-123")

    assert attrs["imperial.keyword_index"] == "keyword_old"
    assert attrs["imperial.current_keyword_index"] == "keyword_current"
    assert attrs["imperial.index_fresh"] == "stale"


def test_trace_provenance_attributes_marks_unavailable_git_sha(monkeypatch, tmp_path: Path) -> None:
    for name in ("IMPERIAL_RAG_GIT_SHA", "GIT_COMMIT", "SOURCE_VERSION"):
        monkeypatch.delenv(name, raising=False)

    def fail_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(tracing_module.subprocess, "run", fail_git)
    attrs = tracing_module.trace_provenance_attributes(Settings(workspace_root=tmp_path), run_id="run-123")

    assert attrs["imperial.git_sha"] == "unavailable"
