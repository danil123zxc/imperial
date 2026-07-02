from __future__ import annotations

from langchain_core.documents import Document

from imperial_rag.document_ids import metadata_or_content_id


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
