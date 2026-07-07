from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from imperial_rag.observability.phoenix import (
    imperial_trace_attributes,
    suppress_internal_tracing,
    trace_retrieval_step,
)
from imperial_rag.retrieval.identity import _annotate_retrieval_documents
from imperial_rag.retrieval.settings import RetrievalSettings
from imperial_rag.retrieval.spans import (
    _keyword_match_mode,
    _set_documents_span_output,
    _record_candidate_documents,
    _vector_error_diagnostics,
)


@dataclass(frozen=True)
class RetrievalCandidateResult:
    vector_docs: list[Document]
    keyword_docs: list[Document]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _SearchStepResult:
    documents: list[Document]
    status: str
    fallbacks: tuple[str, ...] = ()
    output_metadata: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, vector_search: Any, keyword_search: Any, settings: RetrievalSettings | None = None) -> None:
        self.vector_search = vector_search
        self.keyword_search = keyword_search
        self.settings = settings or RetrievalSettings.from_env()

    def retrieve(self, query: str) -> RetrievalCandidateResult:
        vector = self._trace_search_step(
            name="retrieval.vector_search",
            step="vector_search",
            query=query,
            attributes={
                "retrieval.vector_k": self.settings.vector_k,
                "retrieval.vector_fetch_k": self.settings.vector_fetch_k,
                "retrieval.mmr_lambda_mult": self.settings.mmr_lambda_mult,
            },
            search=lambda: self._run_vector_search(query),
        )
        fallbacks = list(vector.fallbacks)
        keyword = self._trace_search_step(
            name="retrieval.keyword_search",
            step="keyword_search",
            query=query,
            attributes={"retrieval.keyword_limit": self.settings.keyword_limit},
            search=lambda: self._run_keyword_search(query),
            prior_fallbacks=fallbacks,
        )
        fallbacks.extend(keyword.fallbacks)

        return RetrievalCandidateResult(
            vector_docs=vector.documents,
            keyword_docs=keyword.documents,
            diagnostics={
                "vector_candidates": len(vector.documents),
                "keyword_candidates": len(keyword.documents),
                "vector_search_status": vector.status,
                "keyword_search_status": keyword.status,
                **_vector_error_diagnostics(self.vector_search),
                "keyword_scores_available": bool(keyword.output_metadata["keyword_scores_available"]),
                "keyword_match_mode": keyword.output_metadata["keyword_match_mode"],
                "fallbacks": fallbacks,
            },
        )

    def _trace_search_step(
        self,
        *,
        name: str,
        step: str,
        query: str,
        attributes: dict[str, Any],
        search: Callable[[], _SearchStepResult],
        prior_fallbacks: list[str] | None = None,
    ) -> _SearchStepResult:
        with trace_retrieval_step(
            name,
            query,
            attributes=imperial_trace_attributes("retrieval", step, attributes),
        ) as span:
            result = search()
            span_fallbacks = list(prior_fallbacks or [])
            span_fallbacks.extend(result.fallbacks)
            _set_documents_span_output(
                span,
                result.documents,
                status=result.status,
                fallbacks=span_fallbacks,
                **result.output_metadata,
            )
            _record_candidate_documents(span, result.documents)
            return result

    def _run_vector_search(self, query: str) -> _SearchStepResult:
        if getattr(self.vector_search, "provider_mismatch", False):
            return _SearchStepResult([], "provider_mismatch", ("vector_provider_mismatch",))
        if getattr(self.vector_search, "vector_unavailable", False):
            return _SearchStepResult([], "unavailable", ("vector_store_unavailable",))
        try:
            documents = self._vector_docs(query)
        except Exception:
            return _SearchStepResult([], "unavailable", ("vector_search_failed",))
        return _SearchStepResult(documents, "ok" if documents else "empty")

    def _run_keyword_search(self, query: str) -> _SearchStepResult:
        try:
            documents = self._keyword_docs(query)
        except Exception:
            return _SearchStepResult(
                [],
                "unavailable",
                ("keyword_search_failed",),
                {
                    "keyword_scores_available": False,
                    "keyword_match_mode": None,
                },
            )
        return _SearchStepResult(
            documents,
            "ok" if documents else "empty",
            output_metadata={
                "keyword_scores_available": _keyword_scores_available(documents),
                "keyword_match_mode": _keyword_match_mode(documents),
            },
        )

    def _vector_docs(self, query: str) -> list[Document]:
        with suppress_internal_tracing():
            if hasattr(self.vector_search, "invoke"):
                docs = self.vector_search.invoke(
                    query,
                    k=self.settings.vector_k,
                    fetch_k=self.settings.vector_fetch_k,
                    lambda_mult=self.settings.mmr_lambda_mult,
                )
            elif hasattr(self.vector_search, "max_marginal_relevance_search"):
                docs = self.vector_search.max_marginal_relevance_search(
                    query,
                    k=self.settings.vector_k,
                    fetch_k=self.settings.vector_fetch_k,
                    lambda_mult=self.settings.mmr_lambda_mult,
                )
            elif hasattr(self.vector_search, "similarity_search"):
                docs = self.vector_search.similarity_search(query, k=self.settings.vector_k)
            else:
                docs = []
        return _annotate_retrieval_documents(docs, rank_key="_vector_rank")

    def _keyword_docs(self, query: str) -> list[Document]:
        with suppress_internal_tracing():
            if hasattr(self.keyword_search, "invoke"):
                docs = self.keyword_search.invoke(query, limit=self.settings.keyword_limit)
                return _annotate_retrieval_documents(docs, rank_key="_keyword_rank")
            if hasattr(self.keyword_search, "search_with_scores"):
                hits = self.keyword_search.search_with_scores(query, limit=self.settings.keyword_limit)
                return _annotate_retrieval_documents([hit.document for hit in hits], rank_key="_keyword_rank")
            return _annotate_retrieval_documents(
                self.keyword_search.search(query, limit=self.settings.keyword_limit),
                rank_key="_keyword_rank",
            )


def _keyword_scores_available(documents: list[Document]) -> bool:
    return bool(documents) and all("_keyword_score" in (document.metadata or {}) for document in documents)
