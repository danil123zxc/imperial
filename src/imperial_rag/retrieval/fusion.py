from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever

from imperial_rag.document_ids import content_key, document_key
from imperial_rag.observability.phoenix import suppress_internal_tracing
from imperial_rag.retrieval.identity import _document_rank, _retrieval_id


@dataclass(frozen=True)
class _CandidateKeys:
    document: str
    content: str | None


class _CandidateIdentityIndex:
    def __init__(self) -> None:
        self._index_by_document: dict[str, int] = {}
        self._index_by_content: dict[str, int] = {}

    def keys(self, document: Document) -> _CandidateKeys:
        return _CandidateKeys(document_key(document), self._content_lookup_key(document))

    def existing_index(self, keys: _CandidateKeys) -> int | None:
        existing_index = self._index_by_document.get(keys.document)
        if existing_index is None and keys.content is not None:
            existing_index = self._index_by_content.get(keys.content)
        return existing_index

    def remember(self, keys: _CandidateKeys, index: int) -> None:
        self._index_by_document.setdefault(keys.document, index)
        if keys.content is not None:
            self._index_by_content.setdefault(keys.content, index)

    def _content_lookup_key(self, document: Document) -> str | None:
        key = content_key(document)
        return key or None


@dataclass
class _RetainedCandidate:
    document: Document
    candidate_id: str | None
    sources: set[str]
    group: _DuplicateGroup | None = None


@dataclass
class _DuplicateGroup:
    retained_id: str
    dropped_ids: list[str]
    sources: set[str]


@dataclass
class _CandidateMergeResult:
    documents: list[Document]
    duplicate_groups: list[dict[str, Any]]


class CandidateMerger:
    def merge(self, vector_docs: list[Document], keyword_docs: list[Document]) -> list[Document]:
        return self._merge(vector_docs, keyword_docs, collect_duplicate_groups=False).documents

    def merge_with_duplicate_groups(
        self,
        vector_docs: list[Document],
        keyword_docs: list[Document],
        *,
        limit: int = 10,
    ) -> _CandidateMergeResult:
        return self._merge(
            vector_docs,
            keyword_docs,
            collect_duplicate_groups=True,
            duplicate_group_limit=limit,
        )

    def duplicate_groups(
        self,
        vector_docs: list[Document],
        keyword_docs: list[Document],
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return self.merge_with_duplicate_groups(vector_docs, keyword_docs, limit=limit).duplicate_groups

    def _merge(
        self,
        vector_docs: list[Document],
        keyword_docs: list[Document],
        *,
        collect_duplicate_groups: bool,
        duplicate_group_limit: int = 10,
    ) -> _CandidateMergeResult:
        identity_index = _CandidateIdentityIndex()
        retained: list[_RetainedCandidate] = []
        groups: list[_DuplicateGroup] = []
        for source, document in _source_documents(vector_docs, keyword_docs):
            keys = identity_index.keys(document)
            existing_index = identity_index.existing_index(keys)
            if existing_index is not None:
                kept = retained[existing_index].document
                retained[existing_index].document = Document(
                    page_content=kept.page_content,
                    metadata=self._merge_metadata(kept.metadata, document.metadata),
                )
                if collect_duplicate_groups:
                    self._record_duplicate_group(retained[existing_index], source, document, groups)
                identity_index.remember(keys, existing_index)
                continue

            index = len(retained)
            identity_index.remember(keys, index)
            retained.append(
                _RetainedCandidate(
                    document=Document(page_content=document.page_content, metadata=dict(document.metadata or {})),
                    candidate_id=_retrieval_id(document) if collect_duplicate_groups else None,
                    sources={source} if collect_duplicate_groups else set(),
                )
            )
        return _CandidateMergeResult(
            documents=[candidate.document for candidate in retained],
            duplicate_groups=[
                {
                    "retained_id": group.retained_id,
                    "dropped_ids": group.dropped_ids[:duplicate_group_limit],
                    "sources": sorted(group.sources),
                }
                for group in groups[:duplicate_group_limit]
            ],
        )

    def _record_duplicate_group(
        self,
        retained_candidate: _RetainedCandidate,
        source: str,
        duplicate: Document,
        groups: list[_DuplicateGroup],
    ) -> None:
        group = retained_candidate.group
        if group is None:
            group = _DuplicateGroup(
                retained_id=retained_candidate.candidate_id or _retrieval_id(retained_candidate.document),
                dropped_ids=[],
                sources=set(retained_candidate.sources),
            )
            retained_candidate.group = group
            groups.append(group)
        group.dropped_ids.append(_retrieval_id(duplicate))
        group.sources.add(source)

    def _merge_metadata(self, kept_metadata: dict[str, Any], duplicate_metadata: dict[str, Any]) -> dict[str, Any]:
        merged = dict(kept_metadata or {})
        for key, value in (duplicate_metadata or {}).items():
            if key not in merged or merged[key] in (None, ""):
                merged[key] = value
        return merged


def _source_documents(vector_docs: list[Document], keyword_docs: list[Document]) -> Iterator[tuple[str, Document]]:
    for document in vector_docs:
        yield "vector", document
    for document in keyword_docs:
        yield "keyword", document


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
        normalized = [
            self._with_retrieval_id(document)
            for document in _suppress_non_active_version_candidates(documents)
        ]
        vector_docs, keyword_docs, unranked_docs = self._split_ranked_documents(normalized)

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
        for fusion_rank, document in enumerate(chain(ordered, tail)):
            metadata = dict(document.metadata or {})
            metadata["_rrf_score"] = self._score(positions, _retrieval_id(document), rrf_k)
            metadata["_fusion_rank"] = fusion_rank
            fused.append(Document(page_content=document.page_content, metadata=metadata))
        return fused

    def _with_retrieval_id(self, document: Document) -> Document:
        metadata = dict(document.metadata or {})
        metadata.setdefault("_retrieval_id", _retrieval_id(document))
        return Document(page_content=document.page_content, metadata=metadata)

    def _split_ranked_documents(self, documents: list[Document]) -> tuple[list[Document], list[Document], list[Document]]:
        vector_docs: list[Document] = []
        keyword_docs: list[Document] = []
        unranked_docs: list[Document] = []
        for document in documents:
            vector_rank = _document_rank(document, "_vector_rank")
            keyword_rank = _document_rank(document, "_keyword_rank")
            if vector_rank is not None:
                vector_docs.append(document)
            if keyword_rank is not None:
                keyword_docs.append(document)
            if vector_rank is None and keyword_rank is None:
                unranked_docs.append(document)
        return vector_docs, keyword_docs, unranked_docs

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
        with suppress_internal_tracing():
            return list(ensemble.invoke(""))

    def _rank_ordered(self, documents: list[Document], rank_key: str) -> list[Document]:
        ranked = [
            (rank, index, document)
            for index, document in enumerate(documents)
            if (rank := _document_rank(document, rank_key)) is not None
        ]
        return [document for _rank, _index, document in sorted(ranked, key=lambda item: (item[0], item[1]))]

    def _positions(self, ranked_lists: list[list[Document]]) -> dict[str, list[int]]:
        positions: dict[str, list[int]] = {}
        for ranked in ranked_lists:
            for index, document in enumerate(ranked, start=1):
                positions.setdefault(_retrieval_id(document), []).append(index)
        return positions

    def _score(self, positions: dict[str, list[int]], retrieval_id: str, rrf_k: int) -> float:
        return sum(1.0 / (rrf_k + position) for position in positions.get(retrieval_id, []))


def _suppress_non_active_version_candidates(documents: list[Document]) -> list[Document]:
    active_groups = {
        str(group)
        for document in documents
        if (group := (document.metadata or {}).get("version_group"))
        and str((document.metadata or {}).get("authority_status") or "active").casefold() == "active"
    }
    if not active_groups:
        return documents
    return [
        document
        for document in documents
        if not (
            str((document.metadata or {}).get("version_group") or "") in active_groups
            and str((document.metadata or {}).get("authority_status") or "active").casefold() != "active"
        )
    ]
