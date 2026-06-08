from dataclasses import dataclass
from contextlib import contextmanager

from imperial_rag.config import Settings
from imperial_rag.runtime import Runtime, build_query_dependencies, create_runtime


def test_create_runtime_constructs_without_live_services(monkeypatch):
    created = {}

    class FakeWorkflow:
        def invoke(self, state):
            created["state"] = state
            return {"answer": "ok"}

    monkeypatch.setattr("imperial_rag.runtime.build_query_workflow", lambda **kwargs: FakeWorkflow())

    runtime = create_runtime()

    assert isinstance(runtime, Runtime)
    assert runtime.query("Что делать?") == {"answer": "ok"}
    assert created["state"] == {"question": "Что делать?"}


def test_runtime_query_wraps_workflow_in_agent_span(monkeypatch):
    trace_calls = []

    class FakeTraceSpan:
        def set_output(self, value):
            trace_calls.append({"output": value})

    @contextmanager
    def fake_trace_agent_step(name, input_value, *, attributes=None):
        trace_calls.append({"name": name, "input": input_value, "attributes": attributes})
        yield FakeTraceSpan()

    class FakeWorkflow:
        def invoke(self, state):
            return {
                "answer": "Оформить акт. [S1]",
                "citations_valid": True,
                "evidence": [object(), object()],
                "retrieval": {
                    "final_evidence": 2,
                    "reranker": "fallback:deterministic",
                    "fallbacks": ["reranker_missing_dashscope_api_key"],
                },
            }

    monkeypatch.setattr("imperial_rag.runtime.trace_agent_step", fake_trace_agent_step)
    runtime = Runtime(settings=Settings(), workflow=FakeWorkflow())

    assert runtime.query("Что делать с браком?")["answer"] == "Оформить акт. [S1]"
    assert trace_calls == [
        {
            "name": "imperial_rag.query",
            "input": "Что делать с браком?",
            "attributes": {"runtime.workspace_root": "/Users/danil/Public/imperial"},
        },
        {
            "output": {
                "answer": "Оформить акт. [S1]",
                "citations_valid": True,
                "evidence_count": 2,
                "retrieval": {
                    "final_evidence": 2,
                    "reranker": "fallback:deterministic",
                    "fallbacks": ["reranker_missing_dashscope_api_key"],
                },
            }
        },
    ]


def test_runtime_query_uses_retrieval_service(monkeypatch, tmp_path):
    calls = {}
    expected_neighbor_store = object()
    evidence = [object()]
    vector_docs = [object()]
    keyword_docs = [object()]
    diagnostics = {"final_evidence": 1, "reranker": "fake"}

    @dataclass
    class FakeRetrievalResult:
        evidence: list
        vector_docs: list
        keyword_docs: list
        diagnostics: dict

    class FakeRetrievalSettings:
        @classmethod
        def from_env(cls):
            calls["settings_from_env"] = True
            return "retrieval-settings"

    class FakeChunkNeighborStore:
        @classmethod
        def from_jsonl(cls, path):
            calls["neighbor_path"] = path
            return expected_neighbor_store

    class FakeRetrievalService:
        def __init__(self, vector_search, keyword_search, neighbor_store, settings):
            calls["service_args"] = {
                "vector_search": vector_search,
                "keyword_search": keyword_search,
                "neighbor_store": neighbor_store,
                "settings": settings,
            }

        def retrieve(self, question):
            calls["retrieved_question"] = question
            return FakeRetrievalResult(
                evidence=evidence,
                vector_docs=vector_docs,
                keyword_docs=keyword_docs,
                diagnostics=diagnostics,
            )

    class FakeWorkflow:
        def __init__(self, retrieve, generate):
            self.retrieve = retrieve
            self.generate = generate

        def invoke(self, state):
            retrieved = self.retrieve(state["question"])
            calls["workflow_retrieved"] = retrieved
            return {"answer": "ok"}

    fake_vector_search = object()
    fake_keyword_search = object()
    monkeypatch.setattr("imperial_rag.runtime.RetrievalSettings", FakeRetrievalSettings)
    monkeypatch.setattr("imperial_rag.runtime.ChunkNeighborStore", FakeChunkNeighborStore)
    monkeypatch.setattr("imperial_rag.runtime.RetrievalService", FakeRetrievalService)
    monkeypatch.setattr(
        "imperial_rag.runtime.build_query_dependencies",
        lambda settings: type(
            "Deps",
            (),
            {"vector_search": fake_vector_search, "keyword_search": fake_keyword_search},
        )(),
    )
    monkeypatch.setattr("imperial_rag.runtime.build_query_workflow", lambda **kwargs: FakeWorkflow(**kwargs))

    runtime = create_runtime(Settings(workspace_root=tmp_path))

    assert runtime.query("Что делать?") == {"answer": "ok"}
    assert calls["retrieved_question"] == "Что делать?"
    assert calls["workflow_retrieved"] == {
        "retrieved_documents": evidence,
        "vector_docs": vector_docs,
        "keyword_docs": keyword_docs,
        "retrieval": diagnostics,
    }
    assert calls["neighbor_path"] == tmp_path / ".imperial_rag" / "extracted" / "chunks.jsonl"
    assert calls["service_args"]["vector_search"] is fake_vector_search
    assert calls["service_args"]["keyword_search"] is fake_keyword_search
    assert calls["service_args"]["neighbor_store"] is expected_neighbor_store
    assert calls["service_args"]["settings"] == "retrieval-settings"
    assert calls["settings_from_env"] is True


def test_semantic_search_enabled_uses_dashscope_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    from imperial_rag.runtime import _semantic_search_enabled

    assert _semantic_search_enabled() is False

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-openai-key")
    assert _semantic_search_enabled() is False

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    assert _semantic_search_enabled() is True


def test_build_query_dependencies_defers_chat_model_without_dashscope(monkeypatch, tmp_path):
    calls = []

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("imperial_rag.runtime.KeywordIndex", lambda db_path: object())

    def fake_create_chat_model():
        calls.append("factory")
        raise RuntimeError("missing dashscope")

    monkeypatch.setattr("imperial_rag.runtime.create_chat_model", fake_create_chat_model)

    deps = build_query_dependencies(Settings(workspace_root=tmp_path))

    assert calls == []

    try:
        deps.chat_model.invoke(["message"])
    except RuntimeError as exc:
        assert str(exc) == "missing dashscope"
    else:
        raise AssertionError("expected lazy chat invocation to raise provider error")
    assert calls == ["factory"]


def test_build_query_dependencies_skips_vector_search_on_metadata_mismatch(monkeypatch, tmp_path):
    calls = {}
    fake_chat_model = object()

    class FakeKeywordIndex:
        def __init__(self, db_path):
            calls["keyword_db_path"] = db_path

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setattr("imperial_rag.runtime.KeywordIndex", FakeKeywordIndex)
    monkeypatch.setattr("imperial_rag.runtime.create_chat_model", lambda: fake_chat_model, raising=False)
    monkeypatch.setattr("imperial_rag.runtime.vector_metadata_matches_config", lambda settings: False, raising=False)

    dependencies = build_query_dependencies(Settings(workspace_root=tmp_path))

    assert getattr(dependencies.vector_search, "provider_mismatch", False) is True
    assert calls["keyword_db_path"] == tmp_path / ".imperial_rag" / "keyword.sqlite3"


def test_runtime_uses_provider_chat_model_by_default(monkeypatch, tmp_path):
    calls = []

    class FakeChatModel:
        def invoke(self, messages):
            calls.append({"messages": messages})
            return "answer"

    fake_chat_model = FakeChatModel()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setattr("imperial_rag.runtime.create_chat_model", lambda: calls.append("factory") or fake_chat_model)
    monkeypatch.setattr("imperial_rag.runtime.KeywordIndex", lambda db_path: object())
    monkeypatch.setattr("imperial_rag.runtime._semantic_search_enabled", lambda: False)

    deps = build_query_dependencies(Settings(workspace_root=tmp_path))

    assert calls == []
    assert deps.chat_model.invoke(["message"]) == "answer"
    assert deps.chat_model.invoke(["again"]) == "answer"
    assert calls == [
        "factory",
        {"messages": ["message"]},
        {"messages": ["again"]},
    ]
