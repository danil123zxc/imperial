from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from imperial_rag.answering import build_strict_messages
from imperial_rag.config import Settings
from imperial_rag.elasticsearch_keyword import ElasticsearchKeywordIndex
from imperial_rag.indexing import make_qdrant_store
from imperial_rag.providers import QwenProviderSettings, create_chat_model, dashscope_configured, vector_metadata_matches_config
from imperial_rag.retrieval import RetrievalService, RetrievalSettings
from imperial_rag.tracing import imperial_trace_attributes, trace_pipeline_step
from imperial_rag.workflows import build_query_workflow


@dataclass(frozen=True)
class QueryDependencies:
    vector_search: object
    keyword_search: object
    chat_model: object


class _NoopVectorSearch:
    def similarity_search(self, query: str, k: int):
        return []


class _DeferredProviderChatModel:
    def __init__(self) -> None:
        self._model: Any | None = None

    def invoke(self, messages):
        if self._model is None:
            self._model = create_chat_model()
        return self._model.invoke(messages)


class _ProviderMismatchVectorSearch:
    provider_mismatch = True

    def similarity_search(self, query: str, k: int):
        return []

    def max_marginal_relevance_search(self, query: str, k: int, fetch_k: int, lambda_mult: float):
        return []


def build_query_dependencies(settings: Settings) -> QueryDependencies:
    vector_search: object
    semantic_search_enabled = _semantic_search_enabled()
    retrieval_settings = RetrievalSettings.from_env()
    if semantic_search_enabled and vector_metadata_matches_config(settings):
        try:
            vector_search = _as_mmr_retriever(
                make_qdrant_store(settings.qdrant_url, settings.qdrant_collection),
                retrieval_settings,
            )
        except Exception:
            vector_search = _NoopVectorSearch()
    elif semantic_search_enabled:
        vector_search = _ProviderMismatchVectorSearch()
    else:
        vector_search = _NoopVectorSearch()
    return QueryDependencies(
        vector_search=vector_search,
        keyword_search=ElasticsearchKeywordIndex(settings),
        chat_model=_DeferredProviderChatModel(),
    )


def _as_mmr_retriever(vector_store: object, settings: RetrievalSettings) -> object:
    if not hasattr(vector_store, "as_retriever"):
        return vector_store
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": settings.vector_k,
            "fetch_k": settings.vector_fetch_k,
            "lambda_mult": settings.mmr_lambda_mult,
        },
    )


@dataclass
class Runtime:
    settings: Settings
    workflow: object | None = None
    dependencies: QueryDependencies | None = None

    def query(self, question: str) -> dict:
        with trace_pipeline_step(
            "imperial_rag.query",
            question,
            attributes=imperial_trace_attributes(
                "query",
                "run",
                {"runtime.workspace_root": str(self.settings.workspace_root)},
            ),
        ) as span:
            result = self.query_workflow().invoke({"question": question})
            span.set_output(_query_trace_output(result))
            return result

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
        trace_attributes = _qwen_llm_trace_attributes()
        try:
            response = dependencies().chat_model.invoke(build_strict_messages(question, docs))
        except Exception as exc:
            from imperial_rag.answering import REFUSAL_TEXT

            return {
                "answer": REFUSAL_TEXT,
                "trace_attributes": {
                    **trace_attributes,
                    "answer.model_status": "error",
                    "answer.model_error_type": type(exc).__name__,
                    "answer.refusal_reason": "model_exception",
                    "tag.tags": ["model_fallback"],
                },
            }
        return {
            "answer": getattr(response, "content", response),
            "trace_attributes": {**trace_attributes, "answer.model_status": "ok"},
        }

    workflow = build_query_workflow(retrieve=retrieve, generate=generate)
    return Runtime(settings=resolved_settings, workflow=workflow)


def build_live_query_workflow(settings: Settings | None = None):
    return create_runtime(settings).query_workflow()


def _semantic_search_enabled() -> bool:
    return dashscope_configured()


def _qwen_llm_trace_attributes() -> dict[str, Any]:
    qwen_settings = QwenProviderSettings.from_env()
    return {
        "llm.provider": "dashscope",
        "llm.model_name": qwen_settings.chat_model,
        "llm.invocation_parameters": {"temperature": 0},
    }


def _query_trace_output(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"answer": getattr(result, "answer", str(result))}
    output: dict[str, Any] = {
        "answer": result.get("answer", ""),
        "citations_valid": result.get("citations_valid"),
        "evidence_count": len(result.get("evidence") or result.get("retrieved_documents") or []),
    }
    retrieval = result.get("retrieval")
    if isinstance(retrieval, dict):
        output["retrieval"] = {
            key: retrieval[key]
            for key in (
                "final_evidence",
                "reranker",
                "fallbacks",
                "vector_search_status",
                "keyword_search_status",
            )
            if key in retrieval
        }
    return output
