from __future__ import annotations

from dataclasses import replace
from typing import Any

from langchain_core.documents import Document

from imperial_rag.document_ids import document_key
from imperial_rag.integrations.dashscope import QwenProviderSettings, create_reranker, dashscope_configured
from imperial_rag.observability.phoenix import suppress_internal_tracing
from imperial_rag.retrieval.identity import _dashscope_model_name, _query_tokens
from imperial_rag.retrieval.lexical import searchable_document_text
from imperial_rag.retrieval.settings import RetrievalSettings


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
        score += self._authority_boost(metadata)
        score -= self._duplicate_penalty(metadata)
        return score

    def _authority_boost(self, metadata: dict[str, Any]) -> float:
        status = str(metadata.get("authority_status") or "active").casefold()
        status_score = {"active": 0.3, "draft": -0.2, "archived": -0.5}.get(status, 0.0)
        rank = self._number(metadata.get("authoritative_rank"))
        return status_score + (0.0 if rank is None else self._clamp(rank * 0.02, lower=-0.2, upper=0.4))

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
        settings = QwenProviderSettings.from_env()
        provider_settings = (
            settings.model_copy(update={"rerank_model": model_name})
            if hasattr(settings, "model_copy")
            else replace(settings, rerank_model=model_name)
        )
        compressor = create_reranker(top_n=self.settings.rerank_top_n, settings=provider_settings)
        with suppress_internal_tracing():
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
