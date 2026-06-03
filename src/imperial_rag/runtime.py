from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any

from imperial_rag.answering import build_strict_messages
from imperial_rag.config import Settings
from imperial_rag.indexing import KeywordIndex, make_qdrant_store
from imperial_rag.retrieval import ChunkNeighborStore, RetrievalService, RetrievalSettings
from imperial_rag.workflows import build_query_workflow


class _LazyChatModel:
    def __init__(self, model_name: str = "gpt-4.1-mini"):
        self.model_name = model_name
        self._model: Any | None = None

    def invoke(self, messages):
        if self._model is None:
            from langchain_openai import ChatOpenAI

            self._model = ChatOpenAI(model=self.model_name, temperature=0)
        return self._model.invoke(messages)


@dataclass(frozen=True)
class QueryDependencies:
    vector_search: object
    keyword_search: object
    chat_model: object = field(default_factory=_LazyChatModel)


class _NoopVectorSearch:
    def similarity_search(self, query: str, k: int):
        return []


def build_query_dependencies(settings: Settings) -> QueryDependencies:
    vector_search: object
    if _semantic_search_enabled():
        try:
            vector_search = make_qdrant_store(settings.qdrant_url, settings.qdrant_collection)
        except Exception:
            vector_search = _NoopVectorSearch()
    else:
        vector_search = _NoopVectorSearch()
    return QueryDependencies(
        vector_search=vector_search,
        keyword_search=KeywordIndex(settings.keyword_db_path),
    )


@dataclass
class Runtime:
    settings: Settings
    workflow: object | None = None
    dependencies: QueryDependencies | None = None

    def query(self, question: str) -> dict:
        return self.query_workflow().invoke({"question": question})

    def query_workflow(self):
        if self.workflow is None:
            runtime = create_runtime(self.settings)
            self.workflow = runtime.query_workflow()
        return self.workflow


def create_runtime(settings: Settings | None = None) -> Runtime:
    resolved_settings = settings or Settings()
    dependencies_cache: dict[str, QueryDependencies] = {}
    retrieval_service_cache: dict[str, RetrievalService] = {}

    def dependencies() -> QueryDependencies:
        if "value" not in dependencies_cache:
            dependencies_cache["value"] = build_query_dependencies(resolved_settings)
        return dependencies_cache["value"]

    def retrieval_service() -> RetrievalService:
        if "value" not in retrieval_service_cache:
            deps = dependencies()
            retrieval_service_cache["value"] = RetrievalService(
                vector_search=deps.vector_search,
                keyword_search=deps.keyword_search,
                neighbor_store=ChunkNeighborStore.from_jsonl(resolved_settings.extraction_root / "chunks.jsonl"),
                settings=RetrievalSettings.from_env(),
            )
        return retrieval_service_cache["value"]

    def retrieve(question: str):
        result = retrieval_service().retrieve(question)
        return {
            "retrieved_documents": result.evidence,
            "vector_docs": result.vector_docs,
            "keyword_docs": result.keyword_docs,
            "retrieval": result.diagnostics,
        }

    def generate(question: str, docs):
        try:
            response = dependencies().chat_model.invoke(build_strict_messages(question, docs))
        except Exception:
            from imperial_rag.answering import REFUSAL_TEXT

            return REFUSAL_TEXT
        return getattr(response, "content", response)

    workflow = build_query_workflow(retrieve=retrieve, generate=generate)
    return Runtime(settings=resolved_settings, workflow=workflow)


def build_live_query_workflow(settings: Settings | None = None):
    return create_runtime(settings).query_workflow()


def _semantic_search_enabled() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY"))
