from dataclasses import dataclass

from imperial_rag.config import Settings
from imperial_rag.runtime import Runtime, create_runtime


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
