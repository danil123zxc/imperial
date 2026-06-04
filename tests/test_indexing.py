from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from langchain_core.documents import Document

from imperial_rag.config import Settings
from imperial_rag.indexing import (
    KeywordIndex,
    create_qdrant_vector_store,
    index_documents,
    index_vector_documents,
    stable_chunk_id,
)


def test_keyword_index_finds_exact_russian_term(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    docs = [
        Document(page_content="Регламент возврата брака из магазина", metadata={"citation_id": "a"}),
        Document(page_content="Должностная инструкция водителя", metadata={"citation_id": "b"}),
    ]

    index.index_documents(docs)
    results = index.search("возврат брака", k=5)

    assert [result.metadata["citation_id"] for result in results] == ["a"]
    assert results[0].page_content == "Регламент возврата брака из магазина"


def test_keyword_index_finds_natural_question_when_strict_terms_miss(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    docs = [
        Document(page_content="Регламент возврата брака из магазина", metadata={"citation_id": "a"}),
        Document(page_content="Должностная инструкция водителя", metadata={"citation_id": "b"}),
    ]

    index.index_documents(docs)
    results = index.search("Как оформить возврат брака из магазина?", k=5)

    assert [result.metadata["citation_id"] for result in results[:1]] == ["a"]


def test_keyword_index_prefers_more_specific_relaxed_match(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    docs = [
        Document(page_content="Регламент возврата брака поставщику", metadata={"citation_id": "supplier"}),
        Document(page_content="Регламент возврата брака из магазина", metadata={"citation_id": "store"}),
    ]

    index.index_documents(docs)
    results = index.search("Как оформить возврат брака из магазина?", k=5)

    assert [result.metadata["citation_id"] for result in results[:1]] == ["store"]


def test_keyword_index_finds_question_with_multiple_filler_terms(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    index.index_documents([Document(page_content="Должностная инструкция водителя", metadata={"citation_id": "a"})])

    results = index.search("Какую инструкцию водителя найти?", k=5)

    assert [result.metadata["citation_id"] for result in results] == ["a"]


def test_keyword_index_does_not_match_single_broad_term_for_unrelated_question(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    index.index_documents(
        [
            Document(page_content="Температурный режим склада контролируется ежедневно.", metadata={"citation_id": "a"}),
        ]
    )

    results = index.search("Какую температуру плавления имеет вольфрам?", k=5)

    assert results == []


def test_keyword_index_relaxation_is_bounded_for_long_miss(monkeypatch, tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    calls = 0
    original_search_fts = index._search_fts

    def counted_search_fts(query_tokens, limit, operator):
        nonlocal calls
        calls += 1
        return original_search_fts(query_tokens, limit, operator)

    monkeypatch.setattr(index, "_search_fts", counted_search_fts)
    query = " ".join(f"несуществующийтермин{number}" for number in range(20))

    assert index.search(query, k=5) == []
    assert calls <= 32


def test_keyword_index_bounded_relaxation_checks_late_adjacent_terms(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    index.index_documents([Document(page_content="Должностная инструкция водителя", metadata={"citation_id": "a"})])
    filler = " ".join(f"несуществующийтермин{number}" for number in range(18))

    results = index.search(f"{filler} инструкцию водителя", k=5)

    assert [result.metadata["citation_id"] for result in results] == ["a"]


def test_keyword_index_uses_settings_keyword_db_path(tmp_path: Path) -> None:
    settings = Settings(workspace_root=tmp_path)
    index = KeywordIndex(settings)

    index.index_documents([Document(page_content="Возврат брака", metadata={"citation_id": "a"})])

    assert settings.keyword_db_path.exists()
    assert [result.metadata["citation_id"] for result in index.search("возврат", k=5)] == ["a"]


def test_keyword_index_handles_case_and_hyphenated_russian_terms(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    docs = [
        Document(
            page_content="Должностная инструкция водителя-экспедитора",
            metadata={"citation_id": "driver"},
        ),
    ]

    index.index_documents(docs)
    results = index.search("ВОДИТЕЛЬ ЭКСПЕДИТОР", k=5)

    assert [result.metadata["citation_id"] for result in results] == ["driver"]


def test_keyword_index_falls_back_to_like_table(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(KeywordIndex, "_fts5_available", lambda self: False)
    index = KeywordIndex(tmp_path / "keyword.sqlite3")

    index.index_documents([Document(page_content="Регламент возврата брака", metadata={"citation_id": "a"})])

    assert [result.metadata["citation_id"] for result in index.search("возврат брака", k=5)] == ["a"]


def test_keyword_index_like_fallback_finds_natural_question_when_strict_terms_miss(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(KeywordIndex, "_fts5_available", lambda self: False)
    index = KeywordIndex(tmp_path / "keyword.sqlite3")

    index.index_documents([Document(page_content="Регламент возврата брака", metadata={"citation_id": "a"})])

    assert [result.metadata["citation_id"] for result in index.search("Как оформить возврат брака?", k=5)] == ["a"]


def test_keyword_index_like_fallback_prefers_more_specific_relaxed_match(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(KeywordIndex, "_fts5_available", lambda self: False)
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    docs = [
        Document(page_content="Регламент возврата брака поставщику", metadata={"citation_id": "supplier"}),
        Document(page_content="Регламент возврата брака из магазина", metadata={"citation_id": "store"}),
    ]

    index.index_documents(docs)
    results = index.search("Как оформить возврат брака из магазина?", k=5)

    assert [result.metadata["citation_id"] for result in results[:1]] == ["store"]


def test_keyword_index_fallback_search_with_scores_orders_by_chunk_id(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(KeywordIndex, "_fts5_available", lambda self: False)
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    docs = [
        Document(page_content="Возврат товара на склад", metadata={"citation_id": "a"}),
        Document(page_content="Возврат брака поставщику", metadata={"citation_id": "b"}),
        Document(page_content="Возврат документов в архив", metadata={"citation_id": "c"}),
    ]
    expected_docs = sorted(docs, key=stable_chunk_id)
    inserted_docs = list(reversed(expected_docs))

    index.index_documents(inserted_docs)
    hits = index.search_with_scores("возврат", limit=5)

    assert [hit.document.metadata["citation_id"] for hit in hits] == [
        doc.metadata["citation_id"] for doc in expected_docs
    ]
    assert [hit.document.metadata["_keyword_rank"] for hit in hits] == list(range(len(hits)))
    assert [hit.score for hit in hits] == [float(rank) for rank in range(len(hits))]


def test_keyword_index_clear_removes_documents(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    index.index_documents([Document(page_content="Возврат брака", metadata={"citation_id": "a"})])

    index.clear()

    assert index.search("возврат", k=5) == []


def test_keyword_index_reindexes_same_chunk_without_duplicates(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    doc = Document(page_content="Возврат брака", metadata={"citation_id": "a"})

    index.index_documents([doc])
    index.index_documents([doc])

    assert [result.metadata["citation_id"] for result in index.search("возврат", k=5)] == ["a"]


def test_keyword_index_search_with_scores_adds_rank_metadata(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    docs = [
        Document(page_content="Регламент возврата брака", metadata={"citation_id": "a"}),
        Document(page_content="Возврат товара на склад", metadata={"citation_id": "b"}),
    ]
    index.index_documents(docs)

    hits = index.search_with_scores("возврат", limit=5)

    assert [hit.document.metadata["_keyword_rank"] for hit in hits] == list(range(len(hits)))
    assert all("_keyword_score" in hit.document.metadata for hit in hits)
    assert all(isinstance(hit.score, float) for hit in hits)


def test_keyword_index_search_preserves_document_only_api(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    index.index_documents([Document(page_content="Регламент возврата брака", metadata={"citation_id": "a"})])

    results = index.search("возврат", limit=5)

    assert [result.metadata["citation_id"] for result in results] == ["a"]
    assert results[0].metadata["_keyword_rank"] == 0


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
    embeddings = object()

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
    from imperial_rag.providers import read_vector_metadata

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
    from imperial_rag.providers import VectorProviderMetadata, VectorProviderMismatchError, write_vector_metadata

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
