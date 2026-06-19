from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from imperial_rag.config import Settings
from imperial_rag.providers import (
    QwenProviderSettings,
    create_embeddings,
    ensure_vector_metadata_compatible,
    write_vector_metadata,
)


_QDRANT_ID_NAMESPACE = uuid.UUID("2f931f90-f82a-4ef6-8a49-310e6c4bd8d7")
_CITATION_METADATA_KEYS = (
    "citation_id",
    "chunk_id",
    "file_id",
    "relative_path",
    "file_path",
    "file_name",
    "source_type",
    "section_heading",
    "page_number",
    "chunk_index",
    "start_index",
)


def stable_chunk_id(document: Document) -> str:
    metadata = document.metadata or {}
    citation_metadata = {
        key: metadata[key]
        for key in _CITATION_METADATA_KEYS
        if key in metadata and metadata[key] is not None
    }
    if not citation_metadata:
        citation_metadata = {"metadata_sha1": hashlib.sha1(_json_dumps(metadata).encode("utf-8")).hexdigest()}
    payload = {
        "citation": citation_metadata,
        "content_sha256": hashlib.sha256(document.page_content.encode("utf-8")).hexdigest(),
    }
    return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, _json_dumps(payload)))


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def create_qdrant_vector_store(settings: Settings, embeddings: object | None = None) -> QdrantVectorStore:
    if embeddings is None:
        ensure_vector_metadata_compatible(settings)
        embeddings = create_embeddings()
    client = QdrantClient(url=settings.qdrant_url)
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )


def embedding_model_identifier(provider_settings: QwenProviderSettings | None = None) -> str:
    resolved = provider_settings or QwenProviderSettings.from_env()
    dimensions = resolved.embedding_dimensions
    return resolved.embedding_model if dimensions is None else f"{resolved.embedding_model}:{dimensions}"


def make_qdrant_store(qdrant_url: str, collection_name: str, embeddings: object | None = None) -> QdrantVectorStore:
    return create_qdrant_vector_store(
        Settings(qdrant_url=qdrant_url, qdrant_collection=collection_name),
        embeddings=embeddings,
    )


def stable_chunk_ids(documents: Sequence[Document]) -> list[str]:
    return [stable_chunk_id(document) for document in documents]


def index_vector_documents(
    chunks: Sequence[Document],
    *,
    settings: Settings | None = None,
    vector_store: object | None = None,
    embeddings: object | None = None,
    ids: Sequence[str] | None = None,
) -> list[str]:
    documents = list(chunks)
    resolved_ids = list(ids) if ids is not None else stable_chunk_ids(documents)
    if len(resolved_ids) != len(documents):
        raise ValueError("ids length must match chunks length")
    if vector_store is None:
        if settings is None:
            raise ValueError("settings is required when vector_store is not provided")
        vector_store = create_qdrant_vector_store(settings, embeddings=embeddings)
    added_ids = list(vector_store.add_documents(documents=documents, ids=resolved_ids))
    if settings is not None:
        write_vector_metadata(settings, QwenProviderSettings.from_env().vector_metadata())
    return added_ids


def index_documents(vector_store: object, documents: list[Document], ids: list[str] | None = None) -> list[str]:
    return index_vector_documents(documents, vector_store=vector_store, ids=ids)


def qdrant_is_healthy(qdrant_url: str) -> bool:
    client = QdrantClient(url=qdrant_url)
    try:
        client.get_collections()
    except Exception:
        return False
    return True


def qdrant_health(settings: Settings) -> bool:
    return qdrant_is_healthy(settings.qdrant_url)
