from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Protocol
import uuid

from imperial_rag.answering.strict import build_strict_messages
from imperial_rag.config import Settings
from imperial_rag.retrieval.elasticsearch import ElasticsearchKeywordIndex
from imperial_rag.indexing import make_qdrant_store
from imperial_rag.observability.logging import log_event
from imperial_rag.integrations.dashscope import QwenProviderSettings, create_chat_model, dashscope_configured, vector_metadata_matches_config
from imperial_rag.retrieval.service import RetrievalService, RetrievalSettings
from imperial_rag.observability.phoenix import imperial_trace_attributes, trace_pipeline_step, trace_provenance_attributes
from imperial_rag.answering.workflow import build_query_workflow

MODEL_PROVIDER_ERROR_TEXT = "The model provider failed while answering. Check local logs and provider credentials, then try again."


class SupportsInvoke(Protocol):
    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        ...


class SupportsRetrieverFactory(Protocol):
    def as_retriever(self, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True)
class QueryDependencies:
    vector_search: Any
    keyword_search: Any
    chat_model: SupportsInvoke


class _NoopVectorSearch:
    def similarity_search(self, query: str, k: int):
        return []


class _UnavailableVectorSearch:
    vector_unavailable = True

    def __init__(self, error_type: str) -> None:
        self.error_type = error_type

    def similarity_search(self, query: str, k: int):
        return []

    def max_marginal_relevance_search(self, query: str, k: int, fetch_k: int, lambda_mult: float):
        return []


class _DeferredProviderChatModel:
    def __init__(self) -> None:
        self._model: Any | None = None

    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        model = self._model
        if model is None:
            model = create_chat_model()
            self._model = model
        return model.invoke(input, *args, **kwargs)


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
        except Exception as exc:
            vector_search = _UnavailableVectorSearch(type(exc).__name__)
            log_event(
                "imperial_rag.vector_store_unavailable",
                level="warning",
                operation="build_query_dependencies",
                status="warning",
                component="runtime",
                dependency="qdrant",
                dependency_status="unavailable",
                error_type=type(exc).__name__,
            )
    elif semantic_search_enabled:
        vector_search = _ProviderMismatchVectorSearch()
    else:
        vector_search = _NoopVectorSearch()
    return QueryDependencies(
        vector_search=vector_search,
        keyword_search=ElasticsearchKeywordIndex(settings),
        chat_model=_DeferredProviderChatModel(),
    )


def _as_mmr_retriever(vector_store: Any, settings: RetrievalSettings) -> Any:
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
    workflow: SupportsInvoke | None = None
    dependencies: QueryDependencies | None = None

    def query(self, question: str) -> dict:
        run_id = _new_trace_run_id()
        with trace_pipeline_step(
            "imperial_rag.query",
            question,
            attributes=imperial_trace_attributes(
                "query",
                "run",
                {
                    "runtime.workspace_root": str(self.settings.workspace_root),
                    **trace_provenance_attributes(self.settings, run_id=run_id),
                },
            ),
        ) as span:
            result = self.query_workflow().invoke({"question": question})
            span.set_output(_query_trace_output(result))
            return result

    def query_workflow(self) -> SupportsInvoke:
        if self.workflow is None:
            runtime = create_runtime(self.settings)
            self.workflow = runtime.query_workflow()
        return self.workflow


def create_runtime(settings: Settings | None = None) -> Runtime:
    resolved_settings = settings or Settings()
    dependencies_cache: QueryDependencies | None = None
    retrieval_service_cache: RetrievalService | None = None

    def dependencies() -> QueryDependencies:
        nonlocal dependencies_cache
        if dependencies_cache is None:
            dependencies_cache = build_query_dependencies(resolved_settings)
        return dependencies_cache

    def retrieval_service() -> RetrievalService:
        nonlocal retrieval_service_cache
        if retrieval_service_cache is None:
            deps = dependencies()
            retrieval_service_cache = RetrievalService(
                vector_search=deps.vector_search,
                keyword_search=deps.keyword_search,
                settings=RetrievalSettings.from_env(),
            )
        return retrieval_service_cache

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
            return {
                "answer": MODEL_PROVIDER_ERROR_TEXT,
                "error": {
                    "type": "model_provider_error",
                    "message": "The model provider failed while answering.",
                    "model_error_type": type(exc).__name__,
                },
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


def _new_trace_run_id() -> str:
    explicit = os.environ.get("IMPERIAL_RAG_TRACE_RUN_ID", "").strip()
    if explicit:
        return explicit
    return f"query_{uuid.uuid4().hex}"


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
