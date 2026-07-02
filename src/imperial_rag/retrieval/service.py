from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from imperial_rag.observability.phoenix import imperial_trace_attributes, trace_mode, trace_retrieval_step
from imperial_rag.retrieval.fusion import CandidateMerger, RrfCandidateFusion
from imperial_rag.retrieval.hybrid import HybridRetriever, RetrievalCandidateResult
from imperial_rag.retrieval.identity import _annotate_retrieval_documents, _retrieval_id
from imperial_rag.retrieval.rerank import FallbackRanker, Reranker
from imperial_rag.retrieval.settings import RetrievalSettings
from imperial_rag.retrieval.spans import (
    _compact_fusion_span_output,
    _deduped_candidate_count,
    _final_evidence_span_output,
    _merge_candidates_span_output,
    _retrieval_degraded_tags,
    _retrieval_summary_output,
    _rrf_fusion_span_output,
    _set_documents_span_output,
)


@dataclass(frozen=True)
class RetrievalResult:
    evidence: list[Document]
    vector_docs: list[Document]
    keyword_docs: list[Document]
    diagnostics: dict[str, Any]


class RetrievalService:
    def __init__(
        self,
        vector_search: Any,
        keyword_search: Any,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self.settings = settings or RetrievalSettings.from_env()
        self.hybrid = HybridRetriever(vector_search=vector_search, keyword_search=keyword_search, settings=self.settings)
        self.merger = CandidateMerger()
        self.fusion = RrfCandidateFusion()
        self.reranker = Reranker(settings=self.settings)

    def retrieve(self, query: str) -> RetrievalResult:
        with trace_retrieval_step(
            "retrieval",
            query,
            kind="CHAIN",
            attributes=imperial_trace_attributes(
                "retrieval",
                "run",
                {
                    "retrieval.vector_k": self.settings.vector_k,
                    "retrieval.keyword_limit": self.settings.keyword_limit,
                    "retrieval.rerank_top_n": self.settings.rerank_top_n,
                    "retrieval.primary_reranker": self.settings.primary_reranker,
                },
            ),
        ) as parent_span:
            candidates = self.hybrid.retrieve(query)
            diagnostics = dict(candidates.diagnostics)
            merged = self.merger.merge(candidates.vector_docs, candidates.keyword_docs)
            diagnostics["merged_candidates"] = len(merged)
            diagnostics["deduped_candidates"] = _deduped_candidate_count(candidates.vector_docs, candidates.keyword_docs, merged)
            fused = self.fusion.fuse(merged, rrf_k=self.settings.rrf_k)
            rerank_input = fused[: self.settings.rerank_input_limit]
            diagnostics["fusion"] = "rrf"
            diagnostics["fusion_rrf_k"] = self.settings.rrf_k
            diagnostics["fused_candidates"] = len(fused)
            diagnostics["rerank_input_candidates"] = len(rerank_input)
            if trace_mode() == "retrieval_debug":
                with trace_retrieval_step(
                    "retrieval.merge_candidates",
                    query,
                    kind="CHAIN",
                    attributes=imperial_trace_attributes(
                        "retrieval",
                        "merge_candidates",
                        {"retrieval.merge_strategy": "dedupe_by_document_or_content"},
                    ),
                ) as span:
                    span.set_output(
                        _merge_candidates_span_output(
                            candidates.vector_docs,
                            candidates.keyword_docs,
                            merged,
                        )
                    )
                with trace_retrieval_step(
                    "retrieval.rrf_fusion",
                    query,
                    kind="CHAIN",
                    attributes=imperial_trace_attributes(
                        "retrieval",
                        "rrf_fusion",
                        {"retrieval.fusion": "rrf", "retrieval.fusion_rrf_k": self.settings.rrf_k},
                    ),
                ) as span:
                    span.set_output(_rrf_fusion_span_output(merged, fused, self.settings.rrf_k))
            else:
                with trace_retrieval_step(
                    "retrieval.fusion",
                    query,
                    kind="CHAIN",
                    attributes=imperial_trace_attributes(
                        "retrieval",
                        "fusion",
                        {"retrieval.fusion": "rrf", "retrieval.fusion_rrf_k": self.settings.rrf_k},
                    ),
                ) as span:
                    span.set_output(
                        _compact_fusion_span_output(
                            candidates.vector_docs,
                            candidates.keyword_docs,
                            merged,
                            fused,
                            rerank_input,
                            self.settings.rrf_k,
                        )
                    )

            with trace_retrieval_step(
                "retrieval.rerank",
                query,
                kind="RERANKER",
                attributes=imperial_trace_attributes(
                    "retrieval",
                    "rerank",
                    {
                        "reranker.query": query,
                        "reranker.top_k": self.settings.rerank_top_n,
                        "retrieval.rerank_input_limit": self.settings.rerank_input_limit,
                        "retrieval.rerank_top_n": self.settings.rerank_top_n,
                        "retrieval.primary_reranker": self.settings.primary_reranker,
                    },
                ),
            ) as span:
                span.set_reranker_input_documents(rerank_input)
                reranked = self.reranker.rerank(query, rerank_input, diagnostics)
                span.set_reranker_output_documents(reranked)
                span.set_attribute("reranker.model_name", diagnostics.get("reranker"))
                _set_documents_span_output(
                    span,
                    reranked,
                    reranker=diagnostics.get("reranker"),
                    rerank_input=diagnostics.get("rerank_input"),
                    reranked_candidates=diagnostics.get("reranked_candidates"),
                    fallbacks=diagnostics.get("fallbacks", []),
                )

            evidence = reranked
            diagnostics["final_evidence"] = len(evidence)
            with trace_retrieval_step(
                "retrieval.final_evidence",
                query,
                attributes=imperial_trace_attributes(
                    "retrieval",
                    "final_evidence",
                    {"retrieval.final_evidence": len(evidence)},
                ),
            ) as span:
                span.set_final_evidence_documents(evidence)
                span.set_output(_final_evidence_span_output(evidence))

            retrieval_tags = _retrieval_degraded_tags(diagnostics)
            if retrieval_tags:
                parent_span.set_attribute("tag.tags", retrieval_tags)
            parent_span.set_output(_retrieval_summary_output(diagnostics))
            return RetrievalResult(
                evidence=evidence,
                vector_docs=candidates.vector_docs,
                keyword_docs=candidates.keyword_docs,
                diagnostics=diagnostics,
            )


__all__ = [
    "CandidateMerger",
    "FallbackRanker",
    "HybridRetriever",
    "RetrievalCandidateResult",
    "RetrievalResult",
    "RetrievalService",
    "RetrievalSettings",
    "Reranker",
    "RrfCandidateFusion",
    "_annotate_retrieval_documents",
    "_retrieval_id",
]
