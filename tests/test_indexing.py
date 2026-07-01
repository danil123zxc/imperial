from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from langchain_core.documents import Document

from imperial_rag.config import Settings
from imperial_rag.indexing import (
    create_qdrant_vector_store,
    index_documents,
    index_vector_documents,
    reset_qdrant_collection,
    stable_chunk_id,
)


def test_stable_chunk_id_uses_citation_metadata_and_content() -> None:
    doc = Document(
        page_content="Возврат брака оформляется актом.",
        metadata={
            "citation_id": "return-policy:body:0",
            "relative_path": "docs/return.docx",
            "source_type": "body",
            "chunk_index": 0,
        },
    )

    chunk_id = stable_chunk_id(doc)

    assert str(UUID(chunk_id)) == chunk_id
    assert stable_chunk_id(doc) == chunk_id
    assert stable_chunk_id(Document(page_content=doc.page_content + "!", metadata=doc.metadata)) != chunk_id


def test_index_vector_documents_passes_stable_qdrant_ids() -> None:
    class FakeVectorStore:
        def add_documents(self, documents, ids):
            self.documents = documents
            self.ids = ids
            return ids

    docs = [
        Document(page_content="one", metadata={"citation_id": "file1:body:0"}),
        Document(page_content="two", metadata={"citation_id": "file1:body:1"}),
    ]
    store = FakeVectorStore()

    ids = index_vector_documents(docs, vector_store=store)

    assert ids == [stable_chunk_id(doc) for doc in docs]
    assert store.ids == ids
    assert store.documents == docs


def test_index_vector_documents_accepts_explicit_ids() -> None:
    class FakeVectorStore:
        def add_documents(self, documents, ids):
            self.ids = ids
            return ids

    docs = [Document(page_content="one", metadata={"citation_id": "file1:body:0"})]
    store = FakeVectorStore()

    ids = index_vector_documents(docs, vector_store=store, ids=["00000000-0000-0000-0000-000000000001"])

    assert ids == ["00000000-0000-0000-0000-000000000001"]
    assert store.ids == ids


def test_legacy_index_documents_uses_vector_store_without_live_qdrant() -> None:
    class FakeVectorStore:
        def add_documents(self, documents, ids):
            self.documents = documents
            self.ids = ids
            return ids

    docs = [Document(page_content="one", metadata={"citation_id": "file1:body:0"})]
    store = FakeVectorStore()

    ids = index_documents(store, docs)

    assert ids == [stable_chunk_id(docs[0])]
    assert store.documents == docs


def test_create_qdrant_vector_store_uses_settings(monkeypatch, tmp_path: Path) -> None:
    created = {}

    class FakeClient:
        def __init__(self, url):
            created["url"] = url

    class FakeVectorStore:
        def __init__(self, client, collection_name, embedding):
            created["client"] = client
            created["collection_name"] = collection_name
            created["embedding"] = embedding

    monkeypatch.setattr("imperial_rag.indexing.QdrantClient", FakeClient)
    monkeypatch.setattr("imperial_rag.indexing.QdrantVectorStore", FakeVectorStore)

    settings = Settings(workspace_root=tmp_path, qdrant_url="http://127.0.0.1:6333", qdrant_collection="test")
    embeddings: Any = object()

    store = create_qdrant_vector_store(settings, embeddings=embeddings)

    assert isinstance(store, FakeVectorStore)
    assert created["url"] == "http://127.0.0.1:6333"
    assert created["collection_name"] == "test"
    assert created["embedding"] is embeddings


def test_create_qdrant_vector_store_uses_qwen_embeddings_by_default(monkeypatch, tmp_path: Path) -> None:
    created = {}

    class FakeClient:
        def __init__(self, url):
            created["url"] = url

    class FakeVectorStore:
        def __init__(self, client, collection_name, embedding):
            created["client"] = client
            created["collection_name"] = collection_name
            created["embedding"] = embedding

    fake_embeddings = object()
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setattr("imperial_rag.indexing.QdrantClient", FakeClient)
    monkeypatch.setattr("imperial_rag.indexing.QdrantVectorStore", FakeVectorStore)
    monkeypatch.setattr("imperial_rag.indexing.create_embeddings", lambda: fake_embeddings)

    settings = Settings(workspace_root=tmp_path, qdrant_url="http://127.0.0.1:6333", qdrant_collection="test")
    store = create_qdrant_vector_store(settings)

    assert isinstance(store, FakeVectorStore)
    assert created["embedding"] is fake_embeddings


def test_index_vector_documents_records_qwen_vector_metadata(monkeypatch, tmp_path: Path) -> None:
    from imperial_rag.integrations.dashscope import read_vector_metadata

    class FakeVectorStore:
        def add_documents(self, documents, ids):
            return ids

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    settings = Settings(workspace_root=tmp_path)
    docs = [Document(page_content="one", metadata={"citation_id": "file1:body:0"})]

    index_vector_documents(docs, settings=settings, vector_store=FakeVectorStore())

    metadata = read_vector_metadata(settings)
    assert metadata is not None
    assert metadata.provider == "dashscope"
    assert metadata.embedding_model == "text-embedding-v4"
    assert metadata.embedding_dimensions == 2048


def test_create_qdrant_vector_store_rejects_mismatched_vector_metadata(monkeypatch, tmp_path: Path) -> None:
    from imperial_rag.integrations.dashscope import VectorProviderMetadata, VectorProviderMismatchError, write_vector_metadata

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    settings = Settings(workspace_root=tmp_path)
    write_vector_metadata(
        settings,
        VectorProviderMetadata(
            provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            distance="cosine",
        ),
    )

    with pytest.raises(VectorProviderMismatchError):
        create_qdrant_vector_store(settings)


def test_reset_qdrant_collection_deletes_existing_collection(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    class FakeClient:
        def __init__(self, url):
            calls.append(("init", url))

        def collection_exists(self, collection_name):
            calls.append(("exists", collection_name))
            return True

        def delete_collection(self, collection_name):
            calls.append(("delete", collection_name))
            return True

    monkeypatch.setattr("imperial_rag.indexing.QdrantClient", FakeClient)
    settings = Settings(workspace_root=tmp_path, qdrant_url="http://127.0.0.1:6333", qdrant_collection="shadow")

    assert reset_qdrant_collection(settings) is True
    assert calls == [
        ("init", "http://127.0.0.1:6333"),
        ("exists", "shadow"),
        ("delete", "shadow"),
    ]


def test_reset_qdrant_collection_skips_missing_collection(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    class FakeClient:
        def __init__(self, url):
            calls.append(("init", url))

        def collection_exists(self, collection_name):
            calls.append(("exists", collection_name))
            return False

        def delete_collection(self, collection_name):
            pytest.fail("delete_collection should not run for missing collections")

    monkeypatch.setattr("imperial_rag.indexing.QdrantClient", FakeClient)
    settings = Settings(workspace_root=tmp_path, qdrant_url="http://127.0.0.1:6333", qdrant_collection="shadow")

    assert reset_qdrant_collection(settings) is False
    assert calls == [
        ("init", "http://127.0.0.1:6333"),
        ("exists", "shadow"),
    ]
