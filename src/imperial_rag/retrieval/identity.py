from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from imperial_rag.document_ids import metadata_or_content_id


def _retrieval_id(document: Document, *fallback_values: Any) -> str:
    metadata = document.metadata or {}
    return metadata_or_content_id(
        metadata.get("_retrieval_id"),
        metadata.get("citation_id"),
        metadata.get("chunk_id"),
        *fallback_values,
        content=document.page_content,
    )


def _annotate_retrieval_documents(documents: list[Document], *, rank_key: str) -> list[Document]:
    annotated: list[Document] = []
    for rank, document in enumerate(documents):
        metadata = dict(document.metadata or {})
        citation_text = metadata.pop("citation_text", None)
        page_content = str(citation_text) if citation_text is not None else document.page_content
        metadata.setdefault(rank_key, rank)
        identity_document = Document(page_content=page_content, metadata=metadata)
        metadata.setdefault("_retrieval_id", _retrieval_id(identity_document))
        annotated.append(Document(page_content=page_content, metadata=metadata))
    return annotated


def _metadata_rank_value(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return int(value)


def _document_rank(document: Document, rank_key: str) -> int | None:
    return _metadata_rank_value((document.metadata or {}).get(rank_key))


def _query_tokens(query: str) -> list[str]:
    return [token for token in query.casefold().replace("-", " ").split() if token]


def _dashscope_model_name(configured: str) -> str | None:
    prefix = "dashscope:"
    if not configured.startswith(prefix):
        return None
    model_name = configured[len(prefix):].strip()
    return model_name or None
