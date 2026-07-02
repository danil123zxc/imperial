from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_classic.retrievers import EnsembleRetriever

from imperial_rag.document_ids import content_key, document_key
from imperial_rag.observability.phoenix import suppress_internal_tracing
from imperial_rag.retrieval.identity import _retrieval_id


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
        with suppress_internal_tracing():
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
