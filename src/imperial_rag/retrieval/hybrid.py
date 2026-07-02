from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from imperial_rag.observability.phoenix import (
    imperial_trace_attributes,
    suppress_internal_tracing,
    trace_candidate_documents_enabled,
    trace_retrieval_step,
)
from imperial_rag.retrieval.identity import _annotate_retrieval_documents
from imperial_rag.retrieval.settings import RetrievalSettings
from imperial_rag.retrieval.spans import (
    _keyword_match_mode,
    _set_candidate_documents_omitted,
    _set_documents_span_output,
    _vector_error_diagnostics,
)


@dataclass(frozen=True)
class RetrievalCandidateResult:
    vector_docs: list[Document]
    keyword_docs: list[Document]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, vector_search: Any, keyword_search: Any, settings: RetrievalSettings | None = None) -> None:
        self.vector_search = vector_search
        self.keyword_search = keyword_search
        self.settings = settings or RetrievalSettings.from_env()

    def retrieve(self, query: str) -> RetrievalCandidateResult:
        fallbacks: list[str] = []
        vector_status = "ok"
        keyword_status = "ok"
        vector_docs: list[Document] = []
        keyword_docs: list[Document] = []

        with trace_retrieval_step(
            "retrieval.vector_search",
            query,
            attributes=imperial_trace_attributes(
                "retrieval",
                "vector_search",
                {
                    "retrieval.vector_k": self.settings.vector_k,
                    "retrieval.vector_fetch_k": self.settings.vector_fetch_k,
                    "retrieval.mmr_lambda_mult": self.settings.mmr_lambda_mult,
                },
            ),
        ) as span:
            if getattr(self.vector_search, "provider_mismatch", False):
                vector_status = "provider_mismatch"
                fallbacks.append("vector_provider_mismatch")
            elif getattr(self.vector_search, "vector_unavailable", False):
                vector_status = "unavailable"
                fallbacks.append("vector_store_unavailable")
            else:
                try:
                    vector_docs = self._vector_docs(query)
                except Exception:
                    vector_status = "unavailable"
                    fallbacks.append("vector_search_failed")
                    vector_docs = []
            if not vector_docs and vector_status == "ok":
                vector_status = "empty"
            _set_documents_span_output(
                span,
                vector_docs,
                status=vector_status,
                fallbacks=fallbacks,
            )
            if trace_candidate_documents_enabled():
                span.set_retrieval_documents(vector_docs)
            elif vector_docs:
                _set_candidate_documents_omitted(span)

        with trace_retrieval_step(
            "retrieval.keyword_search",
            query,
            attributes=imperial_trace_attributes(
                "retrieval",
                "keyword_search",
                {"retrieval.keyword_limit": self.settings.keyword_limit},
            ),
        ) as span:
            try:
                keyword_docs = self._keyword_docs(query)
            except Exception:
                keyword_status = "unavailable"
                fallbacks.append("keyword_search_failed")
                keyword_docs = []
            if not keyword_docs and keyword_status == "ok":
                keyword_status = "empty"
            if keyword_docs:
                keyword_scores_available = all(
                    "_keyword_score" in dict(document.metadata or {})
                    for document in keyword_docs
                )
            else:
                keyword_scores_available = False
            keyword_match_mode = _keyword_match_mode(keyword_docs)
            _set_documents_span_output(
                span,
                keyword_docs,
                status=keyword_status,
                fallbacks=fallbacks,
                keyword_scores_available=keyword_scores_available,
                keyword_match_mode=keyword_match_mode,
            )
            if trace_candidate_documents_enabled():
                span.set_retrieval_documents(keyword_docs)
            elif keyword_docs:
                _set_candidate_documents_omitted(span)

        return RetrievalCandidateResult(
            vector_docs=vector_docs,
            keyword_docs=keyword_docs,
            diagnostics={
                "vector_candidates": len(vector_docs),
                "keyword_candidates": len(keyword_docs),
                "vector_search_status": vector_status,
                "keyword_search_status": keyword_status,
                **_vector_error_diagnostics(self.vector_search),
                "keyword_scores_available": keyword_scores_available,
                "keyword_match_mode": keyword_match_mode,
                "fallbacks": fallbacks,
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
