from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw


@dataclass(frozen=True)
class RetrievalSettings:
    chunk_size: int = 400
    chunk_overlap: int = 50
    vector_fetch_k: int = 80
    vector_k: int = 32
    keyword_limit: int = 40
    rerank_input_limit: int = 60
    rerank_top_n: int = 12
    neighbor_window: int = 1
    final_evidence_min: int = 18
    final_evidence_max: int = 24
    mmr_lambda_mult: float = 0.4
    primary_reranker: str = "cohere:rerank-v3.5"
    fallback_reranker: str = "cohere:rerank-multilingual-v3.0"

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        return cls(
            chunk_size=_env_int("IMPERIAL_RAG_CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_env_int("IMPERIAL_RAG_CHUNK_OVERLAP", cls.chunk_overlap),
            vector_fetch_k=_env_int("IMPERIAL_RAG_VECTOR_FETCH_K", cls.vector_fetch_k),
            vector_k=_env_int("IMPERIAL_RAG_VECTOR_K", cls.vector_k),
            keyword_limit=_env_int("IMPERIAL_RAG_KEYWORD_LIMIT", cls.keyword_limit),
            rerank_input_limit=_env_int("IMPERIAL_RAG_RERANK_INPUT_LIMIT", cls.rerank_input_limit),
            rerank_top_n=_env_int("IMPERIAL_RAG_RERANK_TOP_N", cls.rerank_top_n),
            neighbor_window=_env_int("IMPERIAL_RAG_NEIGHBOR_WINDOW", cls.neighbor_window),
            final_evidence_min=_env_int("IMPERIAL_RAG_FINAL_EVIDENCE_MIN", cls.final_evidence_min),
            final_evidence_max=_env_int("IMPERIAL_RAG_FINAL_EVIDENCE_MAX", cls.final_evidence_max),
            mmr_lambda_mult=_env_float("IMPERIAL_RAG_MMR_LAMBDA_MULT", cls.mmr_lambda_mult),
            primary_reranker=_env_str("IMPERIAL_RAG_PRIMARY_RERANKER", cls.primary_reranker),
            fallback_reranker=_env_str("IMPERIAL_RAG_FALLBACK_RERANKER", cls.fallback_reranker),
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

        if getattr(self.vector_search, "provider_mismatch", False):
            vector_status = "provider_mismatch"
            fallbacks.append("vector_provider_mismatch")
        else:
            try:
                vector_docs = self._vector_docs(query)
            except Exception:
                vector_status = "unavailable"
                fallbacks.append("vector_search_failed")
                vector_docs = []

        try:
            keyword_docs = self._keyword_docs(query)
        except Exception:
            keyword_status = "unavailable"
            fallbacks.append("keyword_search_failed")
            keyword_docs = []

        if not vector_docs and vector_status == "ok":
            vector_status = "empty"
        if not keyword_docs and keyword_status == "ok":
            keyword_status = "empty"

        return RetrievalCandidateResult(
            vector_docs=vector_docs,
            keyword_docs=keyword_docs,
            diagnostics={
                "vector_candidates": len(vector_docs),
                "keyword_candidates": len(keyword_docs),
                "vector_search_status": vector_status,
                "keyword_search_status": keyword_status,
                "fallbacks": fallbacks,
            },
        )

    def _vector_docs(self, query: str) -> list[Document]:
        if hasattr(self.vector_search, "max_marginal_relevance_search"):
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
        return [
            Document(page_content=doc.page_content, metadata={**dict(doc.metadata or {}), "_vector_rank": rank})
            for rank, doc in enumerate(docs)
        ]

    def _keyword_docs(self, query: str) -> list[Document]:
        if hasattr(self.keyword_search, "search_with_scores"):
            hits = self.keyword_search.search_with_scores(query, limit=self.settings.keyword_limit)
            return [hit.document for hit in hits]
        return self.keyword_search.search(query, limit=self.settings.keyword_limit)


def _document_key(document: Document) -> str:
    metadata = document.metadata or {}
    return str(metadata.get("citation_id") or metadata.get("chunk_id") or document.page_content)


def _neighbor_key(document: Document) -> tuple[Any, Any, int] | None:
    metadata = document.metadata or {}
    file_id = metadata.get("file_id")
    source_type = metadata.get("source_type")
    chunk_index = metadata.get("chunk_index")
    if file_id is None or source_type is None:
        return None
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
        return None
    return (file_id, source_type, _source_locator(metadata), chunk_index)


def _source_locator(metadata: dict[str, Any]) -> Any:
    for key in ("page_number", "sheet_name", "image_index", "section_heading"):
        value = metadata.get(key)
        if value is not None:
            return (key, value)
    return None


class ChunkNeighborStore:
    def __init__(self, chunks: list[Document] | None = None) -> None:
        self._chunks_by_key: dict[tuple[Any, Any, int], Document] = {}
        for chunk in chunks or []:
            key = _neighbor_key(chunk)
            if key is not None:
                self._chunks_by_key[key] = chunk

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "ChunkNeighborStore":
        jsonl_path = Path(path)
        if not jsonl_path.exists():
            return cls()

        chunks: list[Document] = []
        try:
            with jsonl_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    payload = json.loads(stripped)
                    if not isinstance(payload, dict):
                        return cls([])
                    if "page_content" not in payload or "metadata" not in payload:
                        return cls([])
                    page_content = payload["page_content"]
                    metadata = payload["metadata"]
                    if not isinstance(page_content, str) or not isinstance(metadata, dict):
                        return cls([])
                    chunks.append(Document(page_content=page_content, metadata=dict(metadata)))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return cls([])
        return cls(chunks)

    def neighbors(self, document: Document, window: int) -> list[Document]:
        key = _neighbor_key(document)
        if key is None or window <= 0:
            return []

        file_id, source_type, source_locator, chunk_index = key
        neighbors: list[Document] = []
        for offset in range(1, window + 1):
            previous = self._chunks_by_key.get((file_id, source_type, source_locator, chunk_index - offset))
            if previous is not None:
                neighbors.append(previous)

            next_document = self._chunks_by_key.get((file_id, source_type, source_locator, chunk_index + offset))
            if next_document is not None:
                neighbors.append(next_document)
        return neighbors


class NeighborExpander:
    def __init__(self, store: ChunkNeighborStore, settings: RetrievalSettings | None = None) -> None:
        self.store = store
        self.settings = settings or RetrievalSettings.from_env()

    def expand(self, documents: list[Document]) -> list[Document]:
        limit = max(0, self.settings.final_evidence_max)
        expanded: list[Document] = []
        seen: set[str] = set()

        for document in documents:
            if self._append_unique(expanded, seen, document) >= limit:
                return expanded[:limit]

        for document in documents:
            for neighbor in self.store.neighbors(document, self.settings.neighbor_window):
                if self._append_unique(expanded, seen, neighbor) >= limit:
                    return expanded[:limit]
        return expanded

    def _append_unique(self, documents: list[Document], seen: set[str], document: Document) -> int:
        key = _document_key(document)
        if key not in seen:
            documents.append(document)
            seen.add(key)
        return len(documents)


class EvidenceSelector:
    def __init__(self, settings: RetrievalSettings | None = None) -> None:
        self.settings = settings or RetrievalSettings.from_env()

    def select(self, documents: list[Document]) -> list[Document]:
        return list(documents[: max(0, self.settings.final_evidence_max)])


def _content_key(document: Document) -> str:
    return " ".join(document.page_content.split()).casefold()


def _query_tokens(query: str) -> list[str]:
    return [token for token in query.casefold().replace("-", " ").split() if token]


def _searchable_text(document: Document) -> str:
    metadata = document.metadata or {}
    return " ".join(
        [
            document.page_content,
            str(metadata.get("file_name", "")),
            str(metadata.get("relative_path", "")),
            str(metadata.get("section_heading", "")),
            str(metadata.get("source_type", "")),
        ]
    ).casefold()


def _cohere_model_name(configured: str) -> str:
    prefix = "cohere:"
    if configured.startswith(prefix):
        return configured[len(prefix):]
    return configured


class CandidateMerger:
    def merge(self, vector_docs: list[Document], keyword_docs: list[Document]) -> list[Document]:
        merged: list[Document] = []
        index_by_key: dict[str, int] = {}
        index_by_content: dict[str, int] = {}
        for document in [*vector_docs, *keyword_docs]:
            document_key = _document_key(document)
            content_key = _content_key(document)

            existing_index = index_by_key.get(document_key)
            if existing_index is None:
                existing_index = index_by_content.get(content_key)
            if existing_index is not None:
                kept = merged[existing_index]
                merged[existing_index] = Document(
                    page_content=kept.page_content,
                    metadata=self._merge_metadata(kept.metadata, document.metadata),
                )
                index_by_key.setdefault(document_key, existing_index)
                index_by_content.setdefault(content_key, existing_index)
                continue

            index = len(merged)
            index_by_key[document_key] = index
            index_by_content[content_key] = index
            merged.append(Document(page_content=document.page_content, metadata=dict(document.metadata or {})))
        return merged

    def _merge_metadata(self, kept_metadata: dict[str, Any], duplicate_metadata: dict[str, Any]) -> dict[str, Any]:
        merged = dict(kept_metadata or {})
        for key, value in (duplicate_metadata or {}).items():
            if key not in merged or merged[key] in (None, ""):
                merged[key] = value
        return merged


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
        searchable = _searchable_text(document)
        tokens = _query_tokens(query)

        score = 0.0
        score += self._rank_boost(metadata.get("_vector_rank"), weight=1.0)
        score += self._rank_boost(metadata.get("_keyword_rank"), weight=1.6)
        score += self._lower_is_better_score_boost(metadata.get("_keyword_score"), weight=0.02)
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

    def _lower_is_better_score_boost(self, value: Any, weight: float) -> float:
        numeric = self._number(value)
        if numeric is None:
            return 0.0
        return self._clamp(-numeric * weight, lower=-0.25, upper=0.25)

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

        if not os.environ.get("COHERE_API_KEY", "").strip():
            diagnostics.setdefault("fallbacks", []).append("reranker_missing_api_key")
            return self._fallback_rerank(query, candidates, diagnostics)

        for configured_model in (self.settings.primary_reranker, self.settings.fallback_reranker):
            try:
                reranked = self._cohere_rerank(query, candidates, _cohere_model_name(configured_model))
            except Exception:
                diagnostics.setdefault("fallbacks", []).append(f"reranker_failed:{configured_model}")
                continue
            diagnostics["reranker"] = configured_model
            backfilled = self._backfill(query, reranked, candidates)
            diagnostics["reranked_candidates"] = len(backfilled)
            return backfilled

        return self._fallback_rerank(query, candidates, diagnostics)

    def _cohere_rerank(self, query: str, documents: list[Document], model_name: str) -> list[Document]:
        from langchain_cohere import CohereRerank

        compressor = CohereRerank(model=model_name, top_n=self.settings.rerank_top_n)
        return list(compressor.compress_documents(documents=documents, query=query))

    def _fallback_rerank(self, query: str, documents: list[Document], diagnostics: dict[str, Any]) -> list[Document]:
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

        seen = {_document_key(document) for document in combined}
        for document in self._fallback.rank(query, candidates, top_n=len(candidates)):
            key = _document_key(document)
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
        neighbor_store: ChunkNeighborStore | None = None,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self.settings = settings or RetrievalSettings.from_env()
        self.hybrid = HybridRetriever(vector_search=vector_search, keyword_search=keyword_search, settings=self.settings)
        self.merger = CandidateMerger()
        self.reranker = Reranker(settings=self.settings)
        self.neighbor_store = neighbor_store or ChunkNeighborStore([])
        self.expander = NeighborExpander(store=self.neighbor_store, settings=self.settings)
        self.selector = EvidenceSelector(settings=self.settings)

    def retrieve(self, query: str) -> RetrievalResult:
        candidates = self.hybrid.retrieve(query)
        diagnostics = dict(candidates.diagnostics)
        merged = self.merger.merge(candidates.vector_docs, candidates.keyword_docs)
        diagnostics["merged_candidates"] = len(merged)
        reranked = self.reranker.rerank(query, merged, diagnostics)
        expanded = self.expander.expand(reranked)
        evidence = self.selector.select(expanded)
        diagnostics["final_evidence"] = len(evidence)
        return RetrievalResult(
            evidence=evidence,
            vector_docs=candidates.vector_docs,
            keyword_docs=candidates.keyword_docs,
            diagnostics=diagnostics,
        )
