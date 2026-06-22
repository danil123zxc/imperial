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


def test_runtime_query_wraps_workflow_in_chain_span(monkeypatch):
    trace_calls = []

    class FakeTraceSpan:
        def set_output(self, value):
            trace_calls.append({"output": value})

    @contextmanager
    def fake_trace_pipeline_step(name, input_value, *, attributes=None):
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

    monkeypatch.setattr("imperial_rag.runtime.trace_pipeline_step", fake_trace_pipeline_step)
    monkeypatch.setattr(
        "imperial_rag.runtime.trace_provenance_attributes",
        lambda settings, run_id=None: {
            "imperial.trace_run_id": run_id,
            "imperial.phoenix_project": settings.phoenix_project_name,
            "imperial.git_sha": "abc1234",
            "imperial.trace_auto_instrument": False,
            "imperial.trace_suppress_internals": True,
        },
    )
    monkeypatch.setattr("imperial_rag.runtime._new_trace_run_id", lambda: "run-123")
    runtime = Runtime(settings=Settings(), workflow=FakeWorkflow())

    assert runtime.query("Что делать с браком?")["answer"] == "Оформить акт. [S1]"
    assert trace_calls == [
        {
            "name": "imperial_rag.query",
            "input": "Что делать с браком?",
            "attributes": {
                "imperial.phase": "query",
                "imperial.step": "run",
                "imperial.trace_schema_version": "rag-v2",
                "runtime.workspace_root": "/Users/danil/Public/imperial",
                "imperial.trace_run_id": "run-123",
                "imperial.phoenix_project": "imperial-rag",
                "imperial.git_sha": "abc1234",
                "imperial.trace_auto_instrument": False,
                "imperial.trace_suppress_internals": True,
            },
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


def test_create_runtime_generate_returns_trace_attrs_for_success_and_model_failure(monkeypatch, tmp_path):
    calls = {}

    class FakeChatModel:
        def __init__(self):
            self.raise_error = False

        def invoke(self, messages):
            calls["messages"] = messages
            if self.raise_error:
                raise RuntimeError("provider down with secret-ish detail")
            return type("Response", (), {"content": "Ответ с цитатой. [S1]"})()

    fake_chat_model = FakeChatModel()

    class FakeWorkflow:
        def __init__(self, retrieve, generate):
            self.retrieve = retrieve
            self.generate = generate

        def invoke(self, state):
            docs = self.retrieve(state["question"])["retrieved_documents"]
            return self.generate(state["question"], docs)

    monkeypatch.setattr(
        "imperial_rag.runtime.build_query_dependencies",
        lambda settings: type(
            "Deps",
            (),
            {
                "vector_search": object(),
                "keyword_search": object(),
                "chat_model": fake_chat_model,
            },
        )(),
    )
    monkeypatch.setattr(
        "imperial_rag.runtime.RetrievalService",
        lambda vector_search, keyword_search, settings: type(
            "Service",
            (),
            {
                "retrieve": lambda self, question: type(
                    "RetrievalResult",
                    (),
                    {
                        "evidence": [],
                        "vector_docs": [],
                        "keyword_docs": [],
                        "diagnostics": {"final_evidence": 0},
                    },
                )()
            },
        )(),
    )
    monkeypatch.setattr("imperial_rag.runtime.build_query_workflow", lambda **kwargs: FakeWorkflow(**kwargs))

    runtime = create_runtime(Settings(workspace_root=tmp_path))
    generate = runtime.workflow.generate
    docs = []

    success = generate("Что делать?", docs)
    fake_chat_model.raise_error = True
    failure = generate("Что делать?", docs)

    assert success == {
        "answer": "Ответ с цитатой. [S1]",
        "trace_attributes": {
            "llm.provider": "dashscope",
            "llm.model_name": "qwen3.7-plus",
            "llm.invocation_parameters": {"temperature": 0},
            "answer.model_status": "ok",
        },
    }
    assert failure == {
        "answer": "The model provider failed while answering. Check local logs and provider credentials, then try again.",
        "error": {
            "type": "model_provider_error",
            "message": "The model provider failed while answering.",
            "model_error_type": "RuntimeError",
        },
        "trace_attributes": {
            "llm.provider": "dashscope",
            "llm.model_name": "qwen3.7-plus",
            "llm.invocation_parameters": {"temperature": 0},
            "answer.model_status": "error",
            "answer.model_error_type": "RuntimeError",
            "answer.refusal_reason": "model_exception",
            "tag.tags": ["model_fallback"],
        },
    }


def test_runtime_query_uses_retrieval_service(monkeypatch, tmp_path):
    calls = {}
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

    class FakeRetrievalService:
        def __init__(self, vector_search, keyword_search, settings):
            calls["service_args"] = {
                "vector_search": vector_search,
                "keyword_search": keyword_search,
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
    assert calls["service_args"]["vector_search"] is fake_vector_search
    assert calls["service_args"]["keyword_search"] is fake_keyword_search
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
    monkeypatch.setattr("imperial_rag.runtime.ElasticsearchKeywordIndex", lambda settings: object())

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

    class FakeElasticsearchKeywordIndex:
        def __init__(self, settings):
            calls["settings"] = settings

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setattr("imperial_rag.runtime.ElasticsearchKeywordIndex", FakeElasticsearchKeywordIndex)
    monkeypatch.setattr("imperial_rag.runtime.create_chat_model", lambda: fake_chat_model, raising=False)
    monkeypatch.setattr("imperial_rag.runtime.vector_metadata_matches_config", lambda settings: False, raising=False)

    settings = Settings(workspace_root=tmp_path)
    dependencies = build_query_dependencies(settings)

    assert getattr(dependencies.vector_search, "provider_mismatch", False) is True
    assert calls["settings"] is settings


def test_build_query_dependencies_uses_qdrant_mmr_retriever(monkeypatch, tmp_path):
    calls = {}
    fake_chat_model = object()
    fake_retriever = object()

    class FakeQdrantStore:
        def as_retriever(self, **kwargs):
            calls["as_retriever"] = kwargs
            return fake_retriever

    class FakeElasticsearchKeywordIndex:
        def __init__(self, settings):
            calls["keyword_settings"] = settings

    for name in (
        "IMPERIAL_RAG_VECTOR_FETCH_K",
        "IMPERIAL_RAG_VECTOR_K",
        "IMPERIAL_RAG_MMR_LAMBDA_MULT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setattr("imperial_rag.runtime.ElasticsearchKeywordIndex", FakeElasticsearchKeywordIndex)
    monkeypatch.setattr("imperial_rag.runtime.create_chat_model", lambda: fake_chat_model, raising=False)
    monkeypatch.setattr("imperial_rag.runtime.vector_metadata_matches_config", lambda settings: True, raising=False)
    monkeypatch.setattr(
        "imperial_rag.runtime.make_qdrant_store",
        lambda qdrant_url, collection_name: FakeQdrantStore(),
        raising=False,
    )

    settings = Settings(workspace_root=tmp_path)
    dependencies = build_query_dependencies(settings)

    assert dependencies.vector_search is fake_retriever
    assert calls["as_retriever"] == {
        "search_type": "mmr",
        "search_kwargs": {"k": 70, "fetch_k": 70, "lambda_mult": 0.4},
    }
    assert calls["keyword_settings"] is settings


def test_build_query_dependencies_marks_vector_construction_failure_unavailable(monkeypatch, tmp_path):
    events = []

    class FakeElasticsearchKeywordIndex:
        def __init__(self, settings):
            self.settings = settings

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setattr("imperial_rag.runtime.ElasticsearchKeywordIndex", FakeElasticsearchKeywordIndex)
    monkeypatch.setattr("imperial_rag.runtime.vector_metadata_matches_config", lambda settings: True, raising=False)
    monkeypatch.setattr(
        "imperial_rag.runtime.make_qdrant_store",
        lambda qdrant_url, collection_name: (_ for _ in ()).throw(RuntimeError("qdrant unavailable secret")),
        raising=False,
    )
    monkeypatch.setattr(
        "imperial_rag.runtime.log_event",
        lambda event, level="info", **fields: events.append((event, level, fields)),
    )

    dependencies = build_query_dependencies(Settings(workspace_root=tmp_path))

    assert getattr(dependencies.vector_search, "vector_unavailable", False) is True
    assert getattr(dependencies.vector_search, "error_type", "") == "RuntimeError"
    assert events == [
        (
            "imperial_rag.vector_store_unavailable",
            "warning",
            {
                "operation": "build_query_dependencies",
                "status": "warning",
                "component": "runtime",
                "dependency": "qdrant",
                "dependency_status": "unavailable",
                "error_type": "RuntimeError",
            },
        )
    ]


def test_runtime_uses_provider_chat_model_by_default(monkeypatch, tmp_path):
    calls = []

    class FakeChatModel:
        def invoke(self, messages):
            calls.append({"messages": messages})
            return "answer"

    fake_chat_model = FakeChatModel()

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setattr("imperial_rag.runtime.create_chat_model", lambda: calls.append("factory") or fake_chat_model)
    monkeypatch.setattr("imperial_rag.runtime.ElasticsearchKeywordIndex", lambda settings: object())
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
