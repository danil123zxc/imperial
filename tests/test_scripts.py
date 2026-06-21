from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest


def test_ingest_script_imports_and_defines_main():
    module = _load_script("scripts/ingest.py", "ingest_script")

    assert hasattr(module, "main")
    assert hasattr(module, "print_summary")


def test_query_script_imports_and_defines_main():
    module = _load_script("scripts/query.py", "query_script")

    assert hasattr(module, "main")


def test_phoenix_eval_script_imports_and_defines_main():
    module = _load_script("scripts/run_phoenix_eval.py", "run_phoenix_eval_script")

    assert hasattr(module, "main")
    assert hasattr(module, "citation_behavior")


def test_phoenix_trace_validator_script_imports_and_defines_main():
    module = _load_script("scripts/validate_phoenix_trace.py", "validate_phoenix_trace_script")

    assert hasattr(module, "main")
    assert hasattr(module, "validate_span_records")


def test_entrypoint_scripts_expose_phoenix_tracing_flag():
    assert "--trace-phoenix" in Path("scripts/ingest.py").read_text(encoding="utf-8")
    assert "--trace-phoenix" in Path("scripts/query.py").read_text(encoding="utf-8")
    assert "--trace-session-id" in Path("scripts/ingest.py").read_text(encoding="utf-8")
    assert "--trace-session-id" in Path("scripts/query.py").read_text(encoding="utf-8")


def test_entrypoint_scripts_configure_observability():
    assert "configure_observability" in Path("scripts/ingest.py").read_text(encoding="utf-8")
    assert "configure_observability" in Path("scripts/query.py").read_text(encoding="utf-8")
    assert "configure_observability" in Path("scripts/run_phoenix_eval.py").read_text(encoding="utf-8")
    assert "configure_observability" in Path("scripts/run_ragas_eval.py").read_text(encoding="utf-8")
    assert "configure_observability" in Path("scripts/run_all_evals.py").read_text(encoding="utf-8")


def test_validate_phoenix_trace_accepts_compact_trace_with_provenance():
    module = _load_script("scripts/validate_phoenix_trace.py", "validate_phoenix_trace_contract")
    records = [
        {
            "name": "imperial_rag.query",
            "span_kind": "CHAIN",
            "attributes": {
                "imperial.trace_run_id": "run-123",
                "imperial.git_sha": "abc1234",
                "imperial.trace_suppress_internals": True,
                "imperial.trace_auto_instrument": False,
                "imperial.phoenix_project": "imperial-rag-readable",
            },
        },
        {"name": "retrieval", "span_kind": "CHAIN", "attributes": {}},
        {"name": "retrieval.vector_search", "span_kind": "RETRIEVER", "attributes": {}},
        {"name": "retrieval.keyword_search", "span_kind": "RETRIEVER", "attributes": {}},
        {"name": "retrieval.rerank", "span_kind": "RERANKER", "attributes": {}},
        {"name": "retrieval.final_evidence", "span_kind": "RETRIEVER", "attributes": {}},
        {"name": "answer.generate", "span_kind": "CHAIN", "attributes": {}},
        {"name": "answer.call_model", "span_kind": "LLM", "attributes": {}},
        {"name": "answer.citation_check", "span_kind": "CHAIN", "attributes": {}},
    ]

    assert module.validate_span_records(records, expected_run_id="run-123") == []


def test_validate_phoenix_trace_accepts_phoenix_dataframe_attribute_columns():
    module = _load_script("scripts/validate_phoenix_trace.py", "validate_phoenix_trace_dataframe_columns")
    records = [
        {
            "name": "imperial_rag.query",
            "span_kind": "CHAIN",
            "context.trace_id": "trace-1",
            "attributes.imperial": {
                "trace_run_id": "run-123",
                "git_sha": "abc1234",
                "trace_suppress_internals": True,
                "trace_auto_instrument": False,
                "phoenix_project": "imperial-rag-readable",
            },
        },
        {"name": "retrieval", "span_kind": "CHAIN", "context.trace_id": "trace-1"},
        {"name": "retrieval.vector_search", "span_kind": "RETRIEVER", "context.trace_id": "trace-1"},
        {"name": "retrieval.keyword_search", "span_kind": "RETRIEVER", "context.trace_id": "trace-1"},
        {"name": "retrieval.rerank", "span_kind": "RERANKER", "context.trace_id": "trace-1"},
        {"name": "retrieval.final_evidence", "span_kind": "RETRIEVER", "context.trace_id": "trace-1"},
        {"name": "answer.generate", "span_kind": "CHAIN", "context.trace_id": "trace-1"},
        {"name": "answer.call_model", "span_kind": "LLM", "context.trace_id": "trace-1"},
        {"name": "answer.citation_check", "span_kind": "CHAIN", "context.trace_id": "trace-1"},
    ]

    selected = module._select_trace_records(records, run_id="run-123")

    assert [record["name"] for record in selected] == [record["name"] for record in records]
    assert module.validate_span_records(selected, expected_run_id="run-123") == []


def test_validate_phoenix_trace_rejects_stale_noise_and_missing_provenance():
    module = _load_script("scripts/validate_phoenix_trace.py", "validate_phoenix_trace_failures")
    records = [
        {
            "name": "imperial_rag.query",
            "span_kind": "CHAIN",
            "attributes": {"imperial.trace_run_id": "run-123"},
        },
        {"name": "LangGraph", "span_kind": "CHAIN", "attributes": {}},
        {"name": "retrieval", "span_kind": "CHAIN", "attributes": {}},
    ]

    errors = module.validate_span_records(records, expected_run_id="run-123")

    assert "forbidden stale/internal span present: LangGraph" in errors
    assert "missing required span: retrieval.vector_search" in errors
    assert "root span missing provenance attribute: imperial.git_sha" in errors


def test_query_script_logs_safe_completion_fields(monkeypatch, capsys):
    module = _load_script("scripts/query.py", "query_script_logging")
    events = []

    env_module = types.ModuleType("imperial_rag.env")
    env_module.load_project_env = lambda workspace_root=None: None
    config_module = types.ModuleType("imperial_rag.config")
    config_module.Settings = lambda **kwargs: types.SimpleNamespace(log_level="INFO", log_format="json")
    tracing_module = types.ModuleType("imperial_rag.tracing")
    tracing_module.configure_phoenix_tracing = lambda *args, **kwargs: None
    tracing_module.phoenix_trace_context = _null_trace_context
    observability_module = types.ModuleType("imperial_rag.observability")
    observability_module.configure_observability = lambda settings: None
    observability_module.log_event = lambda event, **fields: events.append((event, fields))
    observability_module.log_failure = lambda *args, **kwargs: pytest.fail("query should not fail")

    class FakeRuntime:
        def query(self, question):
            return {
                "answer": "public answer",
                "sources": ["public source"],
                "retrieval": {
                    "final_evidence": 2,
                    "vector_candidates": 3,
                    "keyword_candidates": 4,
                    "fallbacks": ["vector_search_failed"],
                    "file_name": "private.docx",
                },
            }

    runtime_module = types.ModuleType("imperial_rag.runtime")
    runtime_module.create_runtime = lambda settings: FakeRuntime()

    monkeypatch.setitem(sys.modules, "imperial_rag.env", env_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.config", config_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.tracing", tracing_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability", observability_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.runtime", runtime_module)

    module.main(["private question"])

    assert capsys.readouterr().out == "public answer\npublic source\n"
    assert events[0][0] == "imperial_rag.query"
    assert events[0][1]["operation"] == "query"
    assert events[0][1]["final_evidence"] == 2
    assert events[0][1]["fallback_count"] == 1
    assert "question" not in events[0][1]
    assert "file_name" not in events[0][1]


def test_ingest_script_logs_failed_file_completion(monkeypatch, capsys):
    module = _load_script("scripts/ingest.py", "ingest_script_logging")
    events = []

    env_module = types.ModuleType("imperial_rag.env")
    env_module.load_project_env = lambda workspace_root=None: None
    config_module = types.ModuleType("imperial_rag.config")
    config_module.Settings = lambda **kwargs: types.SimpleNamespace(log_level="INFO", log_format="json")
    tracing_module = types.ModuleType("imperial_rag.tracing")
    tracing_module.configure_phoenix_tracing = lambda *args, **kwargs: None
    tracing_module.phoenix_trace_context = _null_trace_context
    observability_module = types.ModuleType("imperial_rag.observability")
    observability_module.configure_observability = lambda settings: None
    observability_module.log_event = lambda event, **fields: events.append((event, fields))
    observability_module.log_failure = lambda *args, **kwargs: pytest.fail("ingest should not raise")

    summary = types.SimpleNamespace(
        total_files=2,
        indexed_files=1,
        manifest_only_files=0,
        no_text_files=0,
        unsupported_files=0,
        failed_files=1,
        chunk_count=8,
        keyword_indexed=8,
        vector_indexed=0,
    )

    monkeypatch.setitem(sys.modules, "imperial_rag.env", env_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.config", config_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.tracing", tracing_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability", observability_module)
    monkeypatch.setattr(module, "_run", lambda **kwargs: summary)

    module.main([])

    assert "failed_files=1" in capsys.readouterr().out
    assert events[0][0] == "imperial_rag.ingest"
    assert events[0][1]["level"] == "error"
    assert events[0][1]["status"] == "failed_files"
    assert events[0][1]["total_files"] == 2
    assert events[0][1]["chunk_count"] == 8


def test_ingest_ocr_gate_uses_dashscope_key(monkeypatch):
    module = _load_script("scripts/ingest.py", "ingest_script_ocr_gate")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-legacy-key")

    assert module._ocr_appears_configured() is False

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    assert module._ocr_appears_configured() is True


def test_ingest_vector_store_requires_dashscope_key_before_importing_qdrant(monkeypatch, capsys):
    module = _load_script("scripts/ingest.py", "ingest_script_vector_gate")
    providers = types.ModuleType("imperial_rag.providers")
    providers.dashscope_configured = lambda: False
    indexing = types.ModuleType("imperial_rag.indexing")
    indexing.make_qdrant_store = lambda qdrant_url, collection_name: pytest.fail(
        "Qdrant builder should not run without key"
    )
    indexing.create_qdrant_vector_store = lambda settings: pytest.fail("Qdrant builder should not run without key")

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "imperial_rag.providers", providers)
    monkeypatch.setitem(sys.modules, "imperial_rag.indexing", indexing)

    with pytest.raises(SystemExit) as exc_info:
        settings = types.SimpleNamespace(qdrant_url="http://localhost:6333", qdrant_collection="test")
        module._build_vector_store(settings, index_vectors=True)

    assert exc_info.value.code == 2
    assert "DASHSCOPE_API_KEY is required when --index-vectors is used." in capsys.readouterr().err


def test_ingest_vector_store_builds_qdrant_after_dashscope_gate(monkeypatch):
    module = _load_script("scripts/ingest.py", "ingest_script_vector_build")
    providers = types.ModuleType("imperial_rag.providers")
    providers.dashscope_configured = lambda: True
    indexing = types.ModuleType("imperial_rag.indexing")
    vector_store = object()
    created_with = []
    indexing.make_qdrant_store = lambda qdrant_url, collection_name: pytest.fail("legacy builder should not run")
    indexing.create_qdrant_vector_store = lambda settings: created_with.append(settings) or vector_store

    monkeypatch.setitem(sys.modules, "imperial_rag.providers", providers)
    monkeypatch.setitem(sys.modules, "imperial_rag.indexing", indexing)

    settings = types.SimpleNamespace(qdrant_url="http://localhost:6333", qdrant_collection="test")

    assert module._build_vector_store(settings, index_vectors=True) is vector_store
    assert created_with == [settings]


def test_ingest_vector_store_disabled_does_not_require_dashscope_key(monkeypatch):
    module = _load_script("scripts/ingest.py", "ingest_script_vector_disabled")
    providers = types.ModuleType("imperial_rag.providers")
    providers.dashscope_configured = lambda: pytest.fail("DashScope gate should not run")
    indexing = types.ModuleType("imperial_rag.indexing")
    indexing.make_qdrant_store = lambda qdrant_url, collection_name: pytest.fail("Qdrant builder should not run")
    indexing.create_qdrant_vector_store = lambda settings: pytest.fail("Qdrant builder should not run")

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "imperial_rag.providers", providers)
    monkeypatch.setitem(sys.modules, "imperial_rag.indexing", indexing)

    assert module._build_vector_store(object(), index_vectors=False) is None


def test_query_script_uses_explicit_trace_session_id(monkeypatch, capsys):
    module = _load_script("scripts/query.py", "query_script_trace_session")
    trace_contexts = []

    env_module = types.ModuleType("imperial_rag.env")
    env_module.load_project_env = lambda workspace_root=None: None
    config_module = types.ModuleType("imperial_rag.config")
    config_module.Settings = lambda **kwargs: types.SimpleNamespace(log_level="INFO", log_format="json")
    tracing_module = types.ModuleType("imperial_rag.tracing")
    tracing_module.configure_phoenix_tracing = lambda *args, **kwargs: None

    @contextmanager
    def trace_context(session_id, **kwargs):
        trace_contexts.append({"session_id": session_id, **kwargs})
        yield

    tracing_module.phoenix_trace_context = trace_context
    observability_module = types.ModuleType("imperial_rag.observability")
    observability_module.configure_observability = lambda settings: None
    observability_module.log_event = lambda *args, **kwargs: None
    observability_module.log_failure = lambda *args, **kwargs: pytest.fail("query should not fail")

    class FakeRuntime:
        def query(self, question):
            return {"answer": "ok", "sources": []}

    runtime_module = types.ModuleType("imperial_rag.runtime")
    runtime_module.create_runtime = lambda settings: FakeRuntime()

    monkeypatch.setitem(sys.modules, "imperial_rag.env", env_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.config", config_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.tracing", tracing_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.observability", observability_module)
    monkeypatch.setitem(sys.modules, "imperial_rag.runtime", runtime_module)

    module.main(["private question", "--trace-session-id", "session-explicit"])

    assert capsys.readouterr().out == "ok\n"
    assert trace_contexts == [
        {
            "session_id": "session-explicit",
            "metadata": {"entrypoint": "cli"},
            "tags": ["imperial-rag", "cli"],
        }
    ]


def test_entrypoint_trace_session_id_prefers_env_then_generated(monkeypatch):
    query_module = _load_script("scripts/query.py", "query_script_session_id")
    ingest_module = _load_script("scripts/ingest.py", "ingest_script_session_id")

    monkeypatch.setenv("IMPERIAL_RAG_TRACE_SESSION_ID", "session-env")
    assert query_module._trace_session_id(None) == "session-env"
    assert ingest_module._trace_session_id(None) == "session-env"
    assert query_module._trace_session_id("session-cli") == "session-cli"

    monkeypatch.delenv("IMPERIAL_RAG_TRACE_SESSION_ID", raising=False)
    monkeypatch.setattr(query_module.uuid, "uuid4", lambda: "generated")
    assert query_module._trace_session_id(None) == "cli_generated"


@contextmanager
def _null_trace_context(session_id, **kwargs):
    yield


def _load_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
