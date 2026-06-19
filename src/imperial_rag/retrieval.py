from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever

from imperial_rag.config import env_float, env_int, env_str
from imperial_rag.document_ids import content_key, document_key, metadata_or_content_id
from imperial_rag.keyword import searchable_document_text
from imperial_rag.providers import QwenProviderSettings, create_reranker, dashscope_configured
from imperial_rag.tracing import (
    imperial_trace_attributes,
    retrieval_documents_preview,
    trace_candidate_documents_enabled,
    trace_retrieval_step,
)


@dataclass(frozen=True)
class RetrievalSettings:
    chunk_size: int = 400
    chunk_overlap: int = 50
    vector_fetch_k: int = 70
    vector_k: int = 70
    keyword_limit: int = 30
    rerank_input_limit: int = 100
    rerank_top_n: int = 10
    mmr_lambda_mult: float = 0.4
    rrf_k: int = 60
    primary_reranker: str = "dashscope:qwen3-rerank"
    fallback_reranker: str = "fallback:deterministic"

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        qwen_rerank_model = env_str("IMPERIAL_RAG_QWEN_RERANK_MODEL", "qwen3-rerank")
        primary_reranker = env_str("IMPERIAL_RAG_PRIMARY_RERANKER", f"dashscope:{qwen_rerank_model}")
        return cls(
            chunk_size=env_int("IMPERIAL_RAG_CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=env_int("IMPERIAL_RAG_CHUNK_OVERLAP", cls.chunk_overlap),
            vector_fetch_k=env_int("IMPERIAL_RAG_VECTOR_FETCH_K", cls.vector_fetch_k),
            vector_k=env_int("IMPERIAL_RAG_VECTOR_K", cls.vector_k),
            keyword_limit=env_int("IMPERIAL_RAG_KEYWORD_LIMIT", cls.keyword_limit),
            rerank_input_limit=env_int("IMPERIAL_RAG_RERANK_INPUT_LIMIT", cls.rerank_input_limit),
            rerank_top_n=env_int("IMPERIAL_RAG_RERANK_TOP_N", cls.rerank_top_n),
            mmr_lambda_mult=env_float("IMPERIAL_RAG_MMR_LAMBDA_MULT", cls.mmr_lambda_mult),
            rrf_k=env_int("IMPERIAL_RAG_RRF_K", cls.rrf_k),
            primary_reranker=primary_reranker,
            fallback_reranker=env_str("IMPERIAL_RAG_FALLBACK_RERANKER", cls.fallback_reranker),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalCandidateResult:
    vector_docs: list[Document]
    keyword_docs: list[Document]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    evidence: list[Document]
    vector_docs: list[Document]
    keyword_docs: list[Document]
    diagnostics: dict[str, Any]


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
            _set_documents_span_output(
                span,
                keyword_docs,
                status=keyword_status,
                fallbacks=fallbacks,
                keyword_scores_available=keyword_scores_available,
            )
            if trace_candidate_documents_enabled():
                span.set_retrieval_documents(keyword_docs)

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
                "fallbacks": fallbacks,
            },
        )

    def _vector_docs(self, query: str) -> list[Document]:
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


def _retrieval_id(document: Document) -> str:
    metadata = document.metadata or {}
    return metadata_or_content_id(
        metadata.get("_retrieval_id"),
        metadata.get("citation_id"),
        metadata.get("chunk_id"),
        content=document.page_content,
    )


def _annotate_retrieval_documents(documents: list[Document], *, rank_key: str) -> list[Document]:
    annotated: list[Document] = []
    for rank, document in enumerate(documents):
        metadata = dict(document.metadata or {})
        metadata.setdefault(rank_key, rank)
        metadata.setdefault(
            "_retrieval_id",
            metadata_or_content_id(
                metadata.get("_retrieval_id"),
                metadata.get("citation_id"),
                metadata.get("chunk_id"),
                content=document.page_content,
            ),
        )
        annotated.append(Document(page_content=document.page_content, metadata=metadata))
    return annotated


def _query_tokens(query: str) -> list[str]:
    return [token for token in query.casefold().replace("-", " ").split() if token]


def _dashscope_model_name(configured: str) -> str | None:
    prefix = "dashscope:"
    if not configured.startswith(prefix):
        return None
    model_name = configured[len(prefix):].strip()
    return model_name or None


class CandidateMerger:
    def merge(self, vector_docs: list[Document], keyword_docs: list[Document]) -> list[Document]:
        merged: list[Document] = []
        index_by_key: dict[str, int] = {}
        index_by_content: dict[str, int] = {}
        for document in [*vector_docs, *keyword_docs]:
            candidate_document_key = document_key(document)
            candidate_content_key = content_key(document)

            existing_index = index_by_key.get(candidate_document_key)
            if existing_index is None:
                existing_index = index_by_content.get(candidate_content_key)
            if existing_index is not None:
                kept = merged[existing_index]
                merged[existing_index] = Document(
                    page_content=kept.page_content,
                    metadata=self._merge_metadata(kept.metadata, document.metadata),
                )
                index_by_key.setdefault(candidate_document_key, existing_index)
                index_by_content.setdefault(candidate_content_key, existing_index)
                continue

            index = len(merged)
            index_by_key[candidate_document_key] = index
            index_by_content[candidate_content_key] = index
            merged.append(Document(page_content=document.page_content, metadata=dict(document.metadata or {})))
        return merged

    def _merge_metadata(self, kept_metadata: dict[str, Any], duplicate_metadata: dict[str, Any]) -> dict[str, Any]:
        merged = dict(kept_metadata or {})
        for key, value in (duplicate_metadata or {}).items():
            if key not in merged or merged[key] in (None, ""):
                merged[key] = value
        return merged


class _StaticDocumentRetriever(BaseRetriever):
    documents: list[Document]

    def _get_relevant_documents(self, query: str, *, run_manager, **kwargs: Any) -> list[Document]:
        return list(self.documents)


class RrfCandidateFusion:
    """Fuse vector and keyword candidates with LangChain's Ensemble RRF.

    Ordering is delegated to ``EnsembleRetriever`` (Reciprocal Rank Fusion over the
    two rank-ordered candidate lists); ``_rrf_score``/``_fusion_rank`` are derived
    from the same list positions the retriever scores on.
    """

    def fuse(self, documents: list[Document], rrf_k: int) -> list[Document]:
        normalized = [self._with_retrieval_id(document) for document in documents]
        vector_docs = [doc for doc in normalized if self._rank_value(doc, "_vector_rank") is not None]
        keyword_docs = [doc for doc in normalized if self._rank_value(doc, "_keyword_rank") is not None]
        unranked_docs = [
            doc
            for doc in normalized
            if self._rank_value(doc, "_vector_rank") is None
            and self._rank_value(doc, "_keyword_rank") is None
        ]

        merger = CandidateMerger()
        ranked = merger.merge(vector_docs, keyword_docs)
        vector_list = self._rank_ordered(ranked, "_vector_rank")
        keyword_list = self._rank_ordered(ranked, "_keyword_rank")
        positions = self._positions([vector_list, keyword_list])

        ordered = self._rrf_order(vector_list, keyword_list, rrf_k)
        ordered_ids = {_retrieval_id(document) for document in ordered}
        tail = [
            document
            for document in merger.merge([], unranked_docs)
            if _retrieval_id(document) not in ordered_ids
        ]

        fused: list[Document] = []
        for fusion_rank, document in enumerate([*ordered, *tail]):
            metadata = dict(document.metadata or {})
            metadata["_rrf_score"] = self._score(positions, _retrieval_id(document), rrf_k)
            metadata["_fusion_rank"] = fusion_rank
            fused.append(Document(page_content=document.page_content, metadata=metadata))
        return fused

    def _with_retrieval_id(self, document: Document) -> Document:
        metadata = dict(document.metadata or {})
        metadata.setdefault("_retrieval_id", _retrieval_id(document))
        return Document(page_content=document.page_content, metadata=metadata)

    def _rrf_order(self, vector_list: list[Document], keyword_list: list[Document], rrf_k: int) -> list[Document]:
        if not vector_list and not keyword_list:
            return []
        ensemble = EnsembleRetriever(
            retrievers=[
                _StaticDocumentRetriever(documents=vector_list),
                _StaticDocumentRetriever(documents=keyword_list),
            ],
            weights=[1.0, 1.0],
            c=rrf_k,
            id_key="_retrieval_id",
        )
        return list(ensemble.invoke(""))

    def _rank_ordered(self, documents: list[Document], rank_key: str) -> list[Document]:
        ranked = [
            (rank, index, document)
            for index, document in enumerate(documents)
            if (rank := self._rank_value(document, rank_key)) is not None
        ]
        return [document for _rank, _index, document in sorted(ranked, key=lambda item: (item[0], item[1]))]

    def _positions(self, ranked_lists: list[list[Document]]) -> dict[str, list[int]]:
        positions: dict[str, list[int]] = {}
        for ranked in ranked_lists:
            for index, document in enumerate(ranked, start=1):
                positions.setdefault(_retrieval_id(document), []).append(index)
        return positions

    def _rank_value(self, document: Document, rank_key: str) -> int | None:
        value = (document.metadata or {}).get(rank_key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            return None
        return int(value)

    def _score(self, positions: dict[str, list[int]], retrieval_id: str, rrf_k: int) -> float:
        return sum(1.0 / (rrf_k + position) for position in positions.get(retrieval_id, []))


class FallbackRanker:
    _SOURCE_TYPE_BOOSTS = {
        "body": 0.35,
        "table": 0.3,
        "pdf_page": 0.25,
        "sheet": 0.25,
    }

    def rank(self, query: str, documents: list[Document], top_n: int) -> list[Document]:
        scored = [
            (self._score(query, document), index, document)
            for index, document in enumerate(documents)
        ]
        ranked: list[Document] = []
        for score, _index, document in sorted(scored, key=lambda item: (-item[0], item[1]))[:top_n]:
            metadata = dict(document.metadata or {})
            metadata["_fallback_score"] = score
            ranked.append(Document(page_content=document.page_content, metadata=metadata))
        return ranked

    def _score(self, query: str, document: Document) -> float:
        metadata = document.metadata or {}
        searchable = searchable_document_text(document).casefold()
        tokens = _query_tokens(query)

        score = 0.0
        score += self._rank_boost(metadata.get("_vector_rank"), weight=1.0)
        score += self._rank_boost(metadata.get("_keyword_rank"), weight=1.6)
        score += self._higher_is_better_score_boost(metadata.get("_keyword_score"), weight=0.02)
        score += self._term_boost(tokens, searchable, per_term=0.4, all_terms=1.2)

        path_text = " ".join(
            [
                str(metadata.get("file_name", "")),
                str(metadata.get("relative_path", "")),
            ]
        ).casefold()
        score += self._term_boost(tokens, path_text, per_term=0.6, all_terms=0.9)

        score += self._SOURCE_TYPE_BOOSTS.get(str(metadata.get("source_type", "")), 0.0)
        score -= self._duplicate_penalty(metadata)
        return score

    def _rank_boost(self, value: Any, weight: float) -> float:
        numeric = self._number(value)
        if numeric is None or numeric < 0:
            return 0.0
        return weight / (numeric + 1.0)

    def _higher_is_better_score_boost(self, value: Any, weight: float) -> float:
        numeric = self._number(value)
        if numeric is None:
            return 0.0
        return self._clamp(numeric * weight, lower=-0.25, upper=0.25)

    def _term_boost(self, tokens: list[str], text: str, per_term: float, all_terms: float) -> float:
        if not tokens:
            return 0.0
        matched = sum(1 for token in tokens if token in text)
        if matched == 0:
            return 0.0
        return matched * per_term + (all_terms if matched == len(tokens) else 0.0)

    def _duplicate_penalty(self, metadata: dict[str, Any]) -> float:
        penalty = 0.15 if metadata.get("duplicate_group_id") else 0.0
        duplicate_count = self._number(metadata.get("duplicate_group_size") or metadata.get("_duplicate_group_size"))
        if duplicate_count is not None and duplicate_count > 1:
            penalty += min((duplicate_count - 1.0) * 0.05, 0.25)
        return penalty

    def _number(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return min(max(value, lower), upper)


class Reranker:
    def __init__(self, settings: RetrievalSettings | None = None, fallback: FallbackRanker | None = None) -> None:
        self.settings = settings or RetrievalSettings.from_env()
        self._fallback = fallback or FallbackRanker()

    def rerank(self, query: str, documents: list[Document], diagnostics: dict[str, Any]) -> list[Document]:
        candidates = documents[: self.settings.rerank_input_limit]
        diagnostics["rerank_input"] = len(candidates)
        if not candidates:
            diagnostics["reranker"] = "none"
            diagnostics["reranked_candidates"] = 0
            return []

        primary_model = _dashscope_model_name(self.settings.primary_reranker)
        if primary_model is None:
            diagnostics.setdefault("fallbacks", []).append(f"reranker_unsupported:{self.settings.primary_reranker}")
            return self._fallback_rerank(query, candidates, diagnostics)

        if not dashscope_configured():
            diagnostics.setdefault("fallbacks", []).append("reranker_missing_dashscope_api_key")
            return self._fallback_rerank(query, candidates, diagnostics)

        try:
            reranked = self._dashscope_rerank(query, candidates, primary_model)
        except Exception:
            diagnostics.setdefault("fallbacks", []).append(f"reranker_failed:{self.settings.primary_reranker}")
            return self._fallback_rerank(query, candidates, diagnostics)

        diagnostics["reranker"] = self.settings.primary_reranker
        backfilled = self._backfill(query, reranked, candidates)
        diagnostics["reranked_candidates"] = len(backfilled)
        return backfilled

    def _dashscope_rerank(self, query: str, documents: list[Document], model_name: str) -> list[Document]:
        provider_settings = replace(QwenProviderSettings.from_env(), rerank_model=model_name)
        compressor = create_reranker(top_n=self.settings.rerank_top_n, settings=provider_settings)
        return list(compressor.compress_documents(documents, query))

    def _fallback_rerank(self, query: str, documents: list[Document], diagnostics: dict[str, Any]) -> list[Document]:
        if self.settings.fallback_reranker != "fallback:deterministic":
            diagnostics.setdefault("fallbacks", []).append(f"reranker_unsupported:{self.settings.fallback_reranker}")
        diagnostics["reranker"] = "fallback:deterministic"
        reranked = self._fallback.rank(query, documents, top_n=self.settings.rerank_top_n)
        backfilled = self._backfill(query, reranked, documents)
        diagnostics["reranked_candidates"] = len(backfilled)
        return backfilled

    def _backfill(self, query: str, reranked: list[Document], candidates: list[Document]) -> list[Document]:
        target = min(self.settings.rerank_top_n, len(candidates))
        combined = list(reranked[: self.settings.rerank_top_n])
        if len(combined) >= target:
            return combined

        seen = {document_key(document) for document in combined}
        for document in self._fallback.rank(query, candidates, top_n=len(candidates)):
            key = document_key(document)
            if key in seen:
                continue
            combined.append(document)
            seen.add(key)
            if len(combined) >= target:
                break
        return combined


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
            fused = self.fusion.fuse(merged, rrf_k=self.settings.rrf_k)
            rerank_input = fused[: self.settings.rerank_input_limit]
            diagnostics["fusion"] = "rrf"
            diagnostics["fusion_rrf_k"] = self.settings.rrf_k
            diagnostics["fused_candidates"] = len(fused)
            diagnostics["rerank_input_candidates"] = len(rerank_input)

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


def _set_documents_span_output(span: Any, documents: list[Document], **metadata: Any) -> None:
    output = {"count": len(documents)}
    for key, value in metadata.items():
        if value is not None:
            output[key] = _trace_output_value(value)
    previews = retrieval_documents_preview(documents)
    if previews:
        output["top_documents"] = previews
    span.set_output(output)


def _final_evidence_span_output(documents: list[Document]) -> dict[str, Any]:
    citation_ids: list[str] = []
    files: list[str] = []
    seen_files: set[str] = set()
    context_chars = 0
    for document in documents:
        metadata = dict(document.metadata or {})
        context_chars += len(str(document.page_content))
        citation_id = metadata.get("citation_id")
        if citation_id is not None:
            citation_ids.append(str(citation_id))
        file_name = metadata.get("file_name")
        if file_name is not None and str(file_name) not in seen_files:
            files.append(str(file_name))
            seen_files.add(str(file_name))
    return {
        "count": len(documents),
        "citation_ids": citation_ids,
        "files": files,
        "context_chars": context_chars,
    }


def _retrieval_summary_output(diagnostics: dict[str, Any]) -> dict[str, Any]:
    fallbacks = list(diagnostics.get("fallbacks") or [])
    keys = (
        "vector_candidates",
        "keyword_candidates",
        "merged_candidates",
        "fused_candidates",
        "rerank_input_candidates",
        "fusion",
        "fusion_rrf_k",
        "reranked_candidates",
        "final_evidence",
        "vector_search_status",
        "keyword_search_status",
        "reranker",
    )
    output = {key: diagnostics[key] for key in keys if key in diagnostics}
    output["fallbacks"] = fallbacks
    output["degraded"] = bool(fallbacks)
    return output


def _retrieval_degraded_tags(diagnostics: dict[str, Any]) -> list[str]:
    fallbacks = list(diagnostics.get("fallbacks") or [])
    if not fallbacks:
        return []
    tags = ["degraded"]
    seen = set(tags)
    for fallback in fallbacks:
        reason = _bounded_fallback_tag(str(fallback))
        if not reason:
            continue
        tag = f"fallback:{reason}"
        if tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return tags


def _bounded_fallback_tag(reason: str) -> str:
    sanitized = "".join(character if character.isalnum() or character in {"_", "-", ":"} else "_" for character in reason)
    return sanitized[:80].strip("_")


def _trace_output_value(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _vector_error_diagnostics(vector_search: Any) -> dict[str, Any]:
    if not getattr(vector_search, "vector_unavailable", False):
        return {}
    error_type = getattr(vector_search, "error_type", None)
    return {"vector_search_error_type": str(error_type)} if error_type else {}
