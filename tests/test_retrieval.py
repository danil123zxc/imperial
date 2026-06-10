from __future__ import annotations

from contextlib import contextmanager

import pytest
from langchain_core.documents import Document

import imperial_rag.retrieval as retrieval_module
from imperial_rag.retrieval import CandidateMerger, FallbackRanker, RetrievalSettings, RrfCandidateFusion
from imperial_rag.retrieval import ChunkNeighborStore, EvidenceSelector, NeighborExpander
from imperial_rag.retrieval import HybridRetriever
from imperial_rag.retrieval import RetrievalService
from imperial_rag.retrieval import Reranker


class FakeVectorSearch:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def max_marginal_relevance_search(self, query, k, fetch_k, lambda_mult):
        self.calls.append({"query": query, "k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult})
        return self.docs[:k]


class FakeKeywordSearch:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def search_with_scores(self, query, limit):
        self.calls.append({"query": query, "limit": limit})

        class Hit:
            def __init__(self, document):
                self.document = document
                self.score = 0.0

        return [Hit(document) for document in self.docs[:limit]]


def chunk(index):
    return Document(
        page_content=f"chunk {index}",
        metadata={"citation_id": f"c{index}", "file_id": "f", "source_type": "body", "chunk_index": index},
    )


def capture_retrieval_spans(monkeypatch):
    records = []

    @contextmanager
    def fake_trace_retrieval_step(name, query, *, kind="RETRIEVER", attributes=None):
        record = {
            "name": name,
            "query": query,
            "kind": kind,
            "attributes": dict(attributes or {}),
            "output": None,
            "set_attributes": {},
        }

        class FakeSpan:
            def set_attribute(self, key, value):
                record["set_attributes"][key] = value

            def set_output(self, output):
                record["output"] = output

            def set_retrieval_documents(self, documents):
                self._set_documents("retrieval.documents", documents)

            def set_reranker_input_documents(self, documents):
                self._set_documents("reranker.input_documents", documents)

            def set_reranker_output_documents(self, documents):
                self._set_documents("reranker.output_documents", documents)

            def _set_documents(self, prefix, documents):
                for index, document in enumerate(documents):
                    metadata = dict(document.metadata or {})
                    document_id = metadata.get("chunk_id") or metadata.get("citation_id")
                    record["set_attributes"][f"{prefix}.{index}.document.content"] = document.page_content
                    if document_id is not None:
                        record["set_attributes"][f"{prefix}.{index}.document.id"] = document_id

        records.append(record)
        yield FakeSpan()

    monkeypatch.setattr(retrieval_module, "trace_retrieval_step", fake_trace_retrieval_step, raising=False)
    return records


def test_retrieval_settings_defaults_match_accuracy_spec(monkeypatch):
    for name in (
        "IMPERIAL_RAG_CHUNK_SIZE",
        "IMPERIAL_RAG_CHUNK_OVERLAP",
        "IMPERIAL_RAG_VECTOR_FETCH_K",
        "IMPERIAL_RAG_VECTOR_K",
        "IMPERIAL_RAG_KEYWORD_LIMIT",
        "IMPERIAL_RAG_RERANK_INPUT_LIMIT",
        "IMPERIAL_RAG_RERANK_TOP_N",
        "IMPERIAL_RAG_NEIGHBOR_WINDOW",
        "IMPERIAL_RAG_FINAL_EVIDENCE_MIN",
        "IMPERIAL_RAG_FINAL_EVIDENCE_MAX",
        "IMPERIAL_RAG_MMR_LAMBDA_MULT",
        "IMPERIAL_RAG_RRF_K",
        "IMPERIAL_RAG_QWEN_RERANK_MODEL",
        "IMPERIAL_RAG_PRIMARY_RERANKER",
        "IMPERIAL_RAG_FALLBACK_RERANKER",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = RetrievalSettings.from_env()

    assert settings.chunk_size == 400
    assert settings.chunk_overlap == 50
    assert settings.vector_fetch_k == 70
    assert settings.vector_k == 70
    assert settings.keyword_limit == 30
    assert settings.rerank_input_limit == 100
    assert settings.rerank_top_n == 10
    assert settings.neighbor_window == 0
    assert settings.final_evidence_min == 10
    assert settings.final_evidence_max == 10
    assert settings.mmr_lambda_mult == 0.4
    assert settings.rrf_k == 60
    assert settings.primary_reranker == "dashscope:qwen3-rerank"
    assert settings.fallback_reranker == "fallback:deterministic"


def test_retrieval_settings_qwen_rerank_model_sets_default_primary(monkeypatch):
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_RERANK_MODEL", "qwen3-rerank-custom")
    monkeypatch.delenv("IMPERIAL_RAG_PRIMARY_RERANKER", raising=False)
    monkeypatch.delenv("IMPERIAL_RAG_FALLBACK_RERANKER", raising=False)

    settings = RetrievalSettings.from_env()

    assert settings.primary_reranker == "dashscope:qwen3-rerank-custom"
    assert settings.fallback_reranker == "fallback:deterministic"


def test_retrieval_settings_primary_reranker_overrides_qwen_default(monkeypatch):
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_RERANK_MODEL", "qwen3-rerank-custom")
    monkeypatch.setenv("IMPERIAL_RAG_PRIMARY_RERANKER", "dashscope:explicit")
    monkeypatch.delenv("IMPERIAL_RAG_FALLBACK_RERANKER", raising=False)

    settings = RetrievalSettings.from_env()

    assert settings.primary_reranker == "dashscope:explicit"
    assert settings.fallback_reranker == "fallback:deterministic"


def test_retrieval_settings_read_environment_overrides(monkeypatch):
    monkeypatch.setenv("IMPERIAL_RAG_CHUNK_SIZE", "500")
    monkeypatch.setenv("IMPERIAL_RAG_CHUNK_OVERLAP", "75")
    monkeypatch.setenv("IMPERIAL_RAG_VECTOR_FETCH_K", "90")
    monkeypatch.setenv("IMPERIAL_RAG_VECTOR_K", "30")
    monkeypatch.setenv("IMPERIAL_RAG_KEYWORD_LIMIT", "35")
    monkeypatch.setenv("IMPERIAL_RAG_RERANK_INPUT_LIMIT", "55")
    monkeypatch.setenv("IMPERIAL_RAG_RERANK_TOP_N", "10")
    monkeypatch.setenv("IMPERIAL_RAG_NEIGHBOR_WINDOW", "2")
    monkeypatch.setenv("IMPERIAL_RAG_FINAL_EVIDENCE_MIN", "14")
    monkeypatch.setenv("IMPERIAL_RAG_FINAL_EVIDENCE_MAX", "20")
    monkeypatch.setenv("IMPERIAL_RAG_MMR_LAMBDA_MULT", "0.65")
    monkeypatch.setenv("IMPERIAL_RAG_RRF_K", "42")
    monkeypatch.setenv("IMPERIAL_RAG_PRIMARY_RERANKER", "dashscope:custom-primary")
    monkeypatch.setenv("IMPERIAL_RAG_FALLBACK_RERANKER", "fallback:custom")

    settings = RetrievalSettings.from_env()

    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 75
    assert settings.vector_fetch_k == 90
    assert settings.vector_k == 30
    assert settings.keyword_limit == 35
    assert settings.rerank_input_limit == 55
    assert settings.rerank_top_n == 10
    assert settings.neighbor_window == 2
    assert settings.final_evidence_min == 14
    assert settings.final_evidence_max == 20
    assert settings.mmr_lambda_mult == 0.65
    assert settings.rrf_k == 42
    assert settings.primary_reranker == "dashscope:custom-primary"
    assert settings.fallback_reranker == "fallback:custom"


def test_hybrid_retriever_uses_configured_candidate_counts():
    vector_docs = [
        Document(page_content=f"vector {index}", metadata={"citation_id": f"v{index}"})
        for index in range(40)
    ]
    keyword_docs = [
        Document(page_content=f"keyword {index}", metadata={"citation_id": f"k{index}"})
        for index in range(45)
    ]
    vector = FakeVectorSearch(vector_docs)
    keyword = FakeKeywordSearch(keyword_docs)
    settings = RetrievalSettings(vector_fetch_k=80, vector_k=32, keyword_limit=40)

    result = HybridRetriever(vector_search=vector, keyword_search=keyword, settings=settings).retrieve("возврат брака")

    assert len(result.vector_docs) == 32
    assert len(result.keyword_docs) == 40
    assert result.diagnostics["vector_candidates"] == 32
    assert result.diagnostics["keyword_candidates"] == 40
    assert result.diagnostics["vector_search_status"] == "ok"
    assert result.diagnostics["keyword_search_status"] == "ok"
    assert vector.calls == [{"query": "возврат брака", "k": 32, "fetch_k": 80, "lambda_mult": 0.4}]
    assert keyword.calls == [{"query": "возврат брака", "limit": 40}]


def test_hybrid_retriever_degrades_when_vector_search_fails():
    class BrokenVector:
        def max_marginal_relevance_search(self, query, k, fetch_k, lambda_mult):
            raise RuntimeError("qdrant unavailable")

    keyword_docs = [Document(page_content="keyword", metadata={"citation_id": "k"})]

    result = HybridRetriever(
        vector_search=BrokenVector(),
        keyword_search=FakeKeywordSearch(keyword_docs),
        settings=RetrievalSettings(),
    ).retrieve("возврат")

    assert result.vector_docs == []
    assert [doc.metadata["citation_id"] for doc in result.keyword_docs] == ["k"]
    assert result.diagnostics["vector_search_status"] == "unavailable"
    assert result.diagnostics["keyword_search_status"] == "ok"
    assert "vector_search_failed" in result.diagnostics["fallbacks"]


def test_hybrid_retriever_reports_vector_provider_mismatch_without_vector_call():
    calls = []

    class ProviderMismatchVector:
        provider_mismatch = True

        def max_marginal_relevance_search(self, query, k, fetch_k, lambda_mult):
            calls.append(query)
            raise RuntimeError("vector search should not be called")

    keyword_docs = [Document(page_content="keyword", metadata={"citation_id": "k"})]

    result = HybridRetriever(
        vector_search=ProviderMismatchVector(),
        keyword_search=FakeKeywordSearch(keyword_docs),
        settings=RetrievalSettings(),
    ).retrieve("возврат")

    assert calls == []
    assert result.vector_docs == []
    assert [doc.metadata["citation_id"] for doc in result.keyword_docs] == ["k"]
    assert result.diagnostics["vector_search_status"] == "provider_mismatch"
    assert result.diagnostics["keyword_search_status"] == "ok"
    assert "vector_provider_mismatch" in result.diagnostics["fallbacks"]


def test_retrieval_service_returns_final_evidence_and_diagnostics(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    vector_docs = [
        Document(page_content="vector return", metadata={"citation_id": "v", "file_id": "f", "source_type": "body", "chunk_index": 0})
    ]
    keyword_docs = [
        Document(page_content="Порядок возврата брака", metadata={"citation_id": "k", "file_id": "f", "source_type": "body", "chunk_index": 1, "_keyword_rank": 0})
    ]
    all_chunks = [
        vector_docs[0],
        keyword_docs[0],
        Document(page_content="neighbor", metadata={"citation_id": "n", "file_id": "f", "source_type": "body", "chunk_index": 2}),
    ]
    service = RetrievalService(
        vector_search=FakeVectorSearch(vector_docs),
        keyword_search=FakeKeywordSearch(keyword_docs),
        neighbor_store=ChunkNeighborStore(all_chunks),
        settings=RetrievalSettings(rerank_top_n=1, neighbor_window=1, final_evidence_max=3),
    )

    result = service.retrieve("возврат брака")

    assert [doc.metadata["citation_id"] for doc in result.evidence] == ["k", "v", "n"]
    assert result.diagnostics["merged_candidates"] == 2
    assert result.diagnostics["fusion"] == "rrf"
    assert result.diagnostics["fusion_rrf_k"] == 60
    assert result.diagnostics["fused_candidates"] == 2
    assert result.diagnostics["rerank_input_candidates"] == 2
    assert result.diagnostics["final_evidence"] == 3
    assert result.diagnostics["reranker"] == "fallback:deterministic"


def test_retrieval_service_traces_each_retrieval_step(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    records = capture_retrieval_spans(monkeypatch)
    vector_docs = [
        Document(page_content="vector return", metadata={"citation_id": "v", "file_id": "f", "source_type": "body", "chunk_index": 0})
    ]
    keyword_docs = [
        Document(page_content="Порядок возврата брака", metadata={"citation_id": "k", "file_id": "f", "source_type": "body", "chunk_index": 1, "_keyword_rank": 0})
    ]
    all_chunks = [
        vector_docs[0],
        keyword_docs[0],
        Document(page_content="neighbor", metadata={"citation_id": "n", "file_id": "f", "source_type": "body", "chunk_index": 2}),
    ]
    service = RetrievalService(
        vector_search=FakeVectorSearch(vector_docs),
        keyword_search=FakeKeywordSearch(keyword_docs),
        neighbor_store=ChunkNeighborStore(all_chunks),
        settings=RetrievalSettings(rerank_top_n=1, neighbor_window=1, final_evidence_max=3),
    )

    result = service.retrieve("возврат брака")

    assert [record["name"] for record in records] == [
        "retrieve.vector_search",
        "retrieve.keyword_search",
        "retrieve.merge_candidates",
        "retrieve.fuse_candidates",
        "retrieve.rerank",
        "retrieve.expand_neighbors",
        "retrieve.select_evidence",
    ]
    assert [record["query"] for record in records] == ["возврат брака"] * 7
    assert records[0]["output"]["status"] == "ok"
    assert records[0]["output"]["count"] == 1
    assert records[0]["output"]["top_documents"][0]["citation_id"] == "v"
    assert records[0]["set_attributes"]["retrieval.documents.0.document.id"] == "v"
    assert records[0]["set_attributes"]["retrieval.documents.0.document.content"] == "vector return"
    assert records[1]["output"]["status"] == "ok"
    assert records[1]["output"]["count"] == 1
    assert records[1]["set_attributes"]["retrieval.documents.0.document.id"] == "k"
    assert records[1]["set_attributes"]["retrieval.documents.0.document.content"] == "Порядок возврата брака"
    assert records[2]["kind"] == "CHAIN"
    assert records[2]["output"]["count"] == 2
    assert records[3]["kind"] == "CHAIN"
    assert records[3]["output"]["fusion"] == "rrf"
    assert records[3]["output"]["count"] == 2
    assert records[4]["kind"] == "RERANKER"
    assert records[4]["output"]["reranker"] == "fallback:deterministic"
    assert "reranker_missing_dashscope_api_key" in records[4]["output"]["fallbacks"]
    assert records[4]["set_attributes"]["reranker.input_documents.0.document.id"] == "v"
    assert records[4]["set_attributes"]["reranker.input_documents.1.document.id"] == "k"
    assert records[4]["set_attributes"]["reranker.output_documents.0.document.id"] == "k"
    assert records[5]["output"]["count"] == 3
    assert records[5]["output"]["added_neighbors"] == 2
    assert records[6]["output"]["count"] == 3
    assert records[6]["output"]["final_evidence_max"] == 3
    assert [doc.metadata["citation_id"] for doc in result.evidence] == ["k", "v", "n"]


def test_retrieval_service_traces_search_fallbacks(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    records = capture_retrieval_spans(monkeypatch)

    class BrokenVector:
        def max_marginal_relevance_search(self, query, k, fetch_k, lambda_mult):
            raise RuntimeError("qdrant unavailable")

    class BrokenKeyword:
        def search_with_scores(self, query, limit):
            raise RuntimeError("keyword unavailable")

    service = RetrievalService(
        vector_search=BrokenVector(),
        keyword_search=BrokenKeyword(),
        neighbor_store=ChunkNeighborStore([]),
        settings=RetrievalSettings(),
    )

    result = service.retrieve("возврат")

    assert [record["name"] for record in records] == [
        "retrieve.vector_search",
        "retrieve.keyword_search",
        "retrieve.merge_candidates",
        "retrieve.fuse_candidates",
        "retrieve.rerank",
        "retrieve.expand_neighbors",
        "retrieve.select_evidence",
    ]
    assert records[0]["output"]["status"] == "unavailable"
    assert records[0]["output"]["fallbacks"] == ["vector_search_failed"]
    assert records[1]["output"]["status"] == "unavailable"
    assert records[1]["output"]["fallbacks"] == ["vector_search_failed", "keyword_search_failed"]
    assert records[4]["output"]["reranker"] == "none"
    assert records[6]["output"]["count"] == 0
    assert result.evidence == []


def test_retrieval_service_traces_reranker_provider_failure(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    records = capture_retrieval_spans(monkeypatch)
    docs = [
        Document(page_content="Порядок возврата брака", metadata={"citation_id": "k", "file_id": "f", "source_type": "body", "chunk_index": 0})
    ]

    class BrokenCompressor:
        def compress_documents(self, documents, query):
            raise RuntimeError("reranker down")

    monkeypatch.setattr(retrieval_module, "create_reranker", lambda top_n, settings: BrokenCompressor(), raising=False)
    service = RetrievalService(
        vector_search=FakeVectorSearch([]),
        keyword_search=FakeKeywordSearch(docs),
        neighbor_store=ChunkNeighborStore([]),
        settings=RetrievalSettings(primary_reranker="dashscope:qwen3-rerank-test", rerank_top_n=1),
    )

    result = service.retrieve("возврат")

    assert records[4]["name"] == "retrieve.rerank"
    assert records[4]["output"]["reranker"] == "fallback:deterministic"
    assert "reranker_failed:dashscope:qwen3-rerank-test" in records[4]["output"]["fallbacks"]
    assert [doc.metadata["citation_id"] for doc in result.evidence] == ["k"]


def test_neighbor_expander_adds_previous_and_next_chunks():
    chunks = [
        Document(page_content="previous", metadata={"citation_id": "c0", "file_id": "f", "source_type": "body", "chunk_index": 0}),
        Document(page_content="hit", metadata={"citation_id": "c1", "file_id": "f", "source_type": "body", "chunk_index": 1}),
        Document(page_content="next", metadata={"citation_id": "c2", "file_id": "f", "source_type": "body", "chunk_index": 2}),
    ]
    store = ChunkNeighborStore(chunks)

    expanded = NeighborExpander(store=store, settings=RetrievalSettings(neighbor_window=1, final_evidence_max=10)).expand([chunks[1]])

    assert [doc.metadata["citation_id"] for doc in expanded] == ["c1", "c0", "c2"]


def test_chunk_neighbor_store_keeps_sheet_context_separate():
    chunks = [
        Document(
            page_content="sheet one hit",
            metadata={"citation_id": "s1c0", "file_id": "f", "source_type": "sheet", "sheet_name": "Склад", "chunk_index": 0},
        ),
        Document(
            page_content="sheet one next",
            metadata={"citation_id": "s1c1", "file_id": "f", "source_type": "sheet", "sheet_name": "Склад", "chunk_index": 1},
        ),
        Document(
            page_content="sheet two hit",
            metadata={"citation_id": "s2c0", "file_id": "f", "source_type": "sheet", "sheet_name": "Продажи", "chunk_index": 0},
        ),
        Document(
            page_content="sheet two next",
            metadata={"citation_id": "s2c1", "file_id": "f", "source_type": "sheet", "sheet_name": "Продажи", "chunk_index": 1},
        ),
    ]
    store = ChunkNeighborStore(chunks)

    assert [doc.metadata["citation_id"] for doc in store.neighbors(chunks[0], window=1)] == ["s1c1"]


def test_chunk_neighbor_store_from_jsonl_malformed_row_returns_empty_store(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        '{"page_content": "previous", "metadata": {"citation_id": "c0", "file_id": "f", "source_type": "body", "chunk_index": 0}}\n'
        'not-json\n',
        encoding="utf-8",
    )

    try:
        store = ChunkNeighborStore.from_jsonl(chunks_path)
    except Exception as exc:
        pytest.fail(f"from_jsonl should degrade safely for malformed artifacts: {exc}")

    assert store.neighbors(chunk(1), window=1) == []


def test_neighbor_expander_preserves_hits_first_then_deduped_neighbor_order():
    chunks = [chunk(index) for index in range(1, 8)]
    store = ChunkNeighborStore(chunks)

    expanded = NeighborExpander(
        store=store,
        settings=RetrievalSettings(neighbor_window=2, final_evidence_max=10),
    ).expand([chunks[2], chunks[4]])

    assert [doc.metadata["citation_id"] for doc in expanded] == ["c3", "c5", "c2", "c4", "c1", "c6", "c7"]


def test_neighbor_expander_caps_final_evidence_during_expansion():
    chunks = [chunk(index) for index in range(1, 8)]
    store = ChunkNeighborStore(chunks)

    expanded = NeighborExpander(
        store=store,
        settings=RetrievalSettings(neighbor_window=2, final_evidence_max=4),
    ).expand([chunks[2], chunks[4]])

    assert [doc.metadata["citation_id"] for doc in expanded] == ["c3", "c5", "c2", "c4"]


def test_neighbor_expander_does_not_mutate_input_documents():
    chunks = [chunk(index) for index in range(1, 4)]
    hit = chunks[1]
    original_metadata = dict(hit.metadata)

    NeighborExpander(
        store=ChunkNeighborStore(chunks),
        settings=RetrievalSettings(neighbor_window=1, final_evidence_max=10),
    ).expand([hit])

    assert hit.metadata == original_metadata


def test_evidence_selector_caps_final_evidence():
    docs = [
        Document(page_content=f"doc {index}", metadata={"citation_id": f"c{index}"})
        for index in range(30)
    ]

    selected = EvidenceSelector(settings=RetrievalSettings(final_evidence_max=10)).select(docs)

    assert len(selected) == 10
    assert selected[0].metadata["citation_id"] == "c0"
    assert selected[-1].metadata["citation_id"] == "c9"


def test_candidate_merger_deduplicates_by_citation_and_content():
    same_vector = Document(page_content="Возврат брака оформляется актом.", metadata={"citation_id": "same"})
    same_keyword = Document(page_content="Возврат брака оформляется актом.", metadata={"citation_id": "same"})
    content_duplicate = Document(page_content=" Возврат брака оформляется актом. ", metadata={"citation_id": "different"})
    unique = Document(page_content="Склад принимает товар по накладной.", metadata={"citation_id": "unique"})

    merged = CandidateMerger().merge([same_vector, content_duplicate], [same_keyword, unique])

    assert [doc.metadata["citation_id"] for doc in merged] == ["same", "unique"]


def test_candidate_merger_keeps_same_file_sheet_chunks_when_citation_ids_differ():
    sheet_one = Document(
        page_content="Остатки по складу",
        metadata={"citation_id": "book.xlsx#sheet:sheet-Склад:chunk-0", "file_id": "f", "source_type": "sheet", "sheet_name": "Склад", "chunk_index": 0},
    )
    sheet_two = Document(
        page_content="Продажи по магазину",
        metadata={"citation_id": "book.xlsx#sheet:sheet-Продажи:chunk-0", "file_id": "f", "source_type": "sheet", "sheet_name": "Продажи", "chunk_index": 0},
    )

    merged = CandidateMerger().merge([sheet_one], [sheet_two])

    assert [doc.metadata["citation_id"] for doc in merged] == [
        "book.xlsx#sheet:sheet-Склад:chunk-0",
        "book.xlsx#sheet:sheet-Продажи:chunk-0",
    ]


def test_candidate_merger_reconciles_duplicate_vector_keyword_metadata():
    same_vector = Document(
        page_content="Возврат брака оформляется актом.",
        metadata={
            "citation_id": "same",
            "_vector_rank": 1,
            "file_name": "vector-return.docx",
        },
    )
    same_keyword = Document(
        page_content="Возврат брака оформляется актом из регламента.",
        metadata={
            "citation_id": "same",
            "_keyword_rank": 0,
            "_keyword_score": -1.25,
            "file_name": "keyword-return.docx",
            "section_heading": "Возврат брака",
        },
    )

    merged = CandidateMerger().merge([same_vector], [same_keyword])

    assert len(merged) == 1
    assert merged[0].page_content == same_vector.page_content
    assert merged[0].metadata["_vector_rank"] == 1
    assert merged[0].metadata["_keyword_rank"] == 0
    assert merged[0].metadata["_keyword_score"] == -1.25
    assert merged[0].metadata["file_name"] == "vector-return.docx"
    assert merged[0].metadata["section_heading"] == "Возврат брака"


def test_rrf_candidate_fusion_prioritizes_overlap_documents():
    docs = [
        Document(page_content="vector top", metadata={"citation_id": "vector", "_vector_rank": 0}),
        Document(page_content="keyword top", metadata={"citation_id": "keyword", "_keyword_rank": 0}),
        Document(page_content="both", metadata={"citation_id": "both", "_vector_rank": 10, "_keyword_rank": 10}),
    ]

    fused = RrfCandidateFusion().fuse(docs, rrf_k=60)

    assert [doc.metadata["citation_id"] for doc in fused] == ["both", "vector", "keyword"]
    assert fused[0].metadata["_rrf_score"] > fused[1].metadata["_rrf_score"]
    assert [doc.metadata["_fusion_rank"] for doc in fused] == [0, 1, 2]


def test_rrf_candidate_fusion_interleaves_equal_vector_and_keyword_ranks():
    docs = [
        Document(page_content=f"vector {index}", metadata={"citation_id": f"v{index}", "_vector_rank": index})
        for index in range(3)
    ] + [
        Document(page_content=f"keyword {index}", metadata={"citation_id": f"k{index}", "_keyword_rank": index})
        for index in range(3)
    ]

    fused = RrfCandidateFusion().fuse(docs, rrf_k=60)

    assert [doc.metadata["citation_id"] for doc in fused] == ["v0", "k0", "v1", "k1", "v2", "k2"]


def test_rrf_candidate_fusion_does_not_mutate_input_documents():
    docs = [
        Document(page_content="vector", metadata={"citation_id": "v", "_vector_rank": 0}),
        Document(page_content="keyword", metadata={"citation_id": "k", "_keyword_rank": 0}),
    ]
    original_metadata = [dict(document.metadata) for document in docs]

    fused = RrfCandidateFusion().fuse(docs, rrf_k=60)

    assert [document.metadata for document in docs] == original_metadata
    assert all("_rrf_score" in document.metadata for document in fused)
    assert all("_fusion_rank" in document.metadata for document in fused)


def test_rrf_candidate_fusion_places_unranked_documents_last():
    docs = [
        Document(page_content="unranked first", metadata={"citation_id": "unranked"}),
        Document(page_content="vector", metadata={"citation_id": "v", "_vector_rank": 2}),
        Document(page_content="keyword", metadata={"citation_id": "k", "_keyword_rank": 1}),
    ]

    fused = RrfCandidateFusion().fuse(docs, rrf_k=60)

    assert [doc.metadata["citation_id"] for doc in fused] == ["k", "v", "unranked"]
    assert fused[-1].metadata["_rrf_score"] == 0.0


def test_retrieval_service_uses_fused_top_candidates_as_reranker_input(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    records = capture_retrieval_spans(monkeypatch)
    vector_docs = [
        Document(page_content=f"vector {index}", metadata={"citation_id": f"v{index}"})
        for index in range(32)
    ]
    keyword_docs = [
        Document(page_content=f"keyword {index}", metadata={"citation_id": f"k{index}", "_keyword_rank": index})
        for index in range(40)
    ]

    class EchoCompressor:
        def compress_documents(self, documents, query):
            return list(documents[:2])

    monkeypatch.setattr(retrieval_module, "dashscope_configured", lambda: True, raising=False)
    monkeypatch.setattr(retrieval_module, "create_reranker", lambda top_n, settings: EchoCompressor(), raising=False)
    service = RetrievalService(
        vector_search=FakeVectorSearch(vector_docs),
        keyword_search=FakeKeywordSearch(keyword_docs),
        neighbor_store=ChunkNeighborStore([]),
        settings=RetrievalSettings(rerank_input_limit=6, rerank_top_n=2),
    )

    result = service.retrieve("возврат брака")

    rerank_record = records[4]
    assert rerank_record["name"] == "retrieve.rerank"
    assert [
        rerank_record["set_attributes"][f"reranker.input_documents.{index}.document.id"]
        for index in range(6)
    ] == ["v0", "k0", "v1", "k1", "v2", "k2"]
    assert rerank_record["output"]["rerank_input"] == 6
    assert result.diagnostics["rerank_input_candidates"] == 6


def test_retrieval_service_defaults_budget_candidates_and_output_top_10(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    records = capture_retrieval_spans(monkeypatch)
    vector_docs = [
        Document(
            page_content=f"vector {index}",
            metadata={"citation_id": f"v{index}", "file_id": "vf", "source_type": "body", "chunk_index": index},
        )
        for index in range(70)
    ]
    keyword_docs = [
        Document(
            page_content=f"keyword {index}",
            metadata={
                "citation_id": f"k{index}",
                "file_id": "kf",
                "source_type": "body",
                "chunk_index": index,
                "_keyword_rank": index,
            },
        )
        for index in range(30)
    ]
    factory_calls = []
    compress_calls = []

    class EchoCompressor:
        def __init__(self, top_n):
            self.top_n = top_n

        def compress_documents(self, documents, query):
            compress_calls.append({"query": query, "count": len(documents), "top_n": self.top_n})
            return list(documents[: self.top_n])

    def fake_create_reranker(top_n, settings):
        factory_calls.append({"top_n": top_n, "rerank_model": settings.rerank_model})
        return EchoCompressor(top_n)

    monkeypatch.setattr(retrieval_module, "dashscope_configured", lambda: True, raising=False)
    monkeypatch.setattr(retrieval_module, "create_reranker", fake_create_reranker, raising=False)
    vector = FakeVectorSearch(vector_docs)
    keyword = FakeKeywordSearch(keyword_docs)
    service = RetrievalService(
        vector_search=vector,
        keyword_search=keyword,
        neighbor_store=ChunkNeighborStore([*vector_docs, *keyword_docs]),
        settings=RetrievalSettings(),
    )

    result = service.retrieve("возврат брака")

    assert vector.calls == [{"query": "возврат брака", "k": 70, "fetch_k": 70, "lambda_mult": 0.4}]
    assert keyword.calls == [{"query": "возврат брака", "limit": 30}]
    assert len(result.vector_docs) == 70
    assert len(result.keyword_docs) == 30
    assert factory_calls == [{"top_n": 10, "rerank_model": "qwen3-rerank"}]
    assert compress_calls == [{"query": "возврат брака", "count": 100, "top_n": 10}]
    assert result.diagnostics["merged_candidates"] == 100
    assert result.diagnostics["rerank_input_candidates"] == 100
    assert result.diagnostics["rerank_input"] == 100
    assert result.diagnostics["reranked_candidates"] == 10
    assert result.diagnostics["final_evidence"] == 10
    assert len(result.evidence) == 10
    assert records[5]["name"] == "retrieve.expand_neighbors"
    assert records[5]["output"]["count"] == 10
    assert records[5]["output"]["added_neighbors"] == 0
    assert records[6]["name"] == "retrieve.select_evidence"
    assert records[6]["output"]["final_evidence_max"] == 10


def test_fallback_ranker_prioritizes_keyword_and_filename_matches():
    docs = [
        Document(
            page_content="Общие правила склада.",
            metadata={"citation_id": "warehouse", "_vector_rank": 0, "file_name": "warehouse.docx"},
        ),
        Document(
            page_content="Порядок возврата брака.",
            metadata={"citation_id": "return", "_keyword_rank": 0, "file_name": "Регламент возврата брака.docx"},
        ),
        Document(
            page_content="Возврат брака оформляется актом.",
            metadata={"citation_id": "body", "_vector_rank": 2, "_keyword_rank": 2, "source_type": "body"},
        ),
    ]

    ranked = FallbackRanker().rank("возврат брака", docs, top_n=3)

    assert [doc.metadata["citation_id"] for doc in ranked] == ["return", "body", "warehouse"]
    assert ranked[0].metadata["_fallback_score"] > ranked[1].metadata["_fallback_score"]


def test_fallback_ranker_ignores_ambiguous_vector_score_direction():
    docs = [
        Document(
            page_content="Общие правила склада.",
            metadata={"citation_id": "ambiguous-score", "_vector_rank": 2, "_vector_score": 999.0},
        ),
        Document(
            page_content="Общие правила отгрузки.",
            metadata={"citation_id": "better-rank", "_vector_rank": 1, "_vector_score": 0.0},
        ),
    ]

    ranked = FallbackRanker().rank("возврат брака", docs, top_n=2)

    assert [doc.metadata["citation_id"] for doc in ranked] == ["better-rank", "ambiguous-score"]


def test_reranker_uses_dashscope_provider_when_api_key_configured(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    docs = [
        Document(page_content="Порядок возврата брака.", metadata={"citation_id": "return", "_keyword_rank": 0}),
        Document(page_content="Возврат оформляется актом.", metadata={"citation_id": "act", "_keyword_rank": 1}),
        Document(page_content="Общие правила склада.", metadata={"citation_id": "warehouse"}),
    ]
    diagnostics = {"fallbacks": []}
    calls = []

    class FakeCompressor:
        def compress_documents(self, documents, query):
            calls.append({"documents": documents, "query": query})
            return [documents[1]]

    def fake_create_reranker(top_n, settings):
        calls.append({"top_n": top_n, "rerank_model": settings.rerank_model})
        return FakeCompressor()

    monkeypatch.setattr(retrieval_module, "dashscope_configured", lambda: True, raising=False)
    monkeypatch.setattr(retrieval_module, "create_reranker", fake_create_reranker, raising=False)

    reranked = Reranker(settings=RetrievalSettings(rerank_input_limit=2, rerank_top_n=2)).rerank(
        "возврат брака",
        docs,
        diagnostics,
    )

    assert [doc.metadata["citation_id"] for doc in reranked] == ["act", "return"]
    assert calls == [
        {"top_n": 2, "rerank_model": "qwen3-rerank"},
        {"documents": docs[:2], "query": "возврат брака"},
    ]
    assert diagnostics["reranker"] == "dashscope:qwen3-rerank"
    assert diagnostics["rerank_input"] == 2
    assert diagnostics["reranked_candidates"] == 2
    assert diagnostics["fallbacks"] == []


def test_reranker_passes_explicit_dashscope_primary_model_to_provider(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_RERANK_MODEL", "qwen3-rerank-env")
    docs = [
        Document(page_content="Порядок возврата брака.", metadata={"citation_id": "return", "_keyword_rank": 0}),
        Document(page_content="Возврат оформляется актом.", metadata={"citation_id": "act", "_keyword_rank": 1}),
    ]
    diagnostics = {"fallbacks": []}
    calls = []

    class FakeCompressor:
        def compress_documents(self, documents, query):
            return [documents[0]]

    def fake_create_reranker(top_n, settings):
        calls.append({"top_n": top_n, "rerank_model": settings.rerank_model})
        return FakeCompressor()

    monkeypatch.setattr(retrieval_module, "dashscope_configured", lambda: True, raising=False)
    monkeypatch.setattr(retrieval_module, "create_reranker", fake_create_reranker, raising=False)

    settings = RetrievalSettings(primary_reranker="dashscope:qwen3-rerank-explicit", rerank_top_n=1)
    reranked = Reranker(settings=settings).rerank("возврат брака", docs, diagnostics)

    assert [doc.metadata["citation_id"] for doc in reranked] == ["return"]
    assert calls == [{"top_n": 1, "rerank_model": "qwen3-rerank-explicit"}]
    assert diagnostics["reranker"] == "dashscope:qwen3-rerank-explicit"
    assert diagnostics["fallbacks"] == []


def test_reranker_falls_back_for_unsupported_primary_without_provider_call(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    docs = [
        Document(page_content="Общие правила склада.", metadata={"citation_id": "warehouse"}),
        Document(page_content="Порядок возврата брака.", metadata={"citation_id": "return", "_keyword_rank": 0}),
    ]
    diagnostics = {"fallbacks": []}
    calls = []

    def fake_create_reranker(top_n, settings):
        calls.append({"top_n": top_n, "rerank_model": settings.rerank_model})
        raise AssertionError("unsupported primary should not create a DashScope reranker")

    monkeypatch.setattr(retrieval_module, "dashscope_configured", lambda: True, raising=False)
    monkeypatch.setattr(retrieval_module, "create_reranker", fake_create_reranker, raising=False)

    settings = RetrievalSettings(primary_reranker="cohere:stale-reranker", rerank_top_n=1)
    reranked = Reranker(settings=settings).rerank("возврат брака", docs, diagnostics)

    assert [doc.metadata["citation_id"] for doc in reranked] == ["return"]
    assert calls == []
    assert diagnostics["reranker"] == "fallback:deterministic"
    assert "reranker_unsupported:cohere:stale-reranker" in diagnostics["fallbacks"]


def test_reranker_uses_deterministic_fallback_without_dashscope_api_key(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    docs = [
        Document(page_content="Общие правила склада.", metadata={"citation_id": "warehouse"}),
        Document(page_content="Порядок возврата брака.", metadata={"citation_id": "return", "_keyword_rank": 0}),
    ]
    diagnostics = {"fallbacks": []}

    monkeypatch.setattr(retrieval_module, "dashscope_configured", lambda: False, raising=False)

    reranked = Reranker(settings=RetrievalSettings(rerank_top_n=1)).rerank("возврат брака", docs, diagnostics)

    assert [doc.metadata["citation_id"] for doc in reranked] == ["return"]
    assert diagnostics["reranker"] == "fallback:deterministic"
    assert "reranker_missing_dashscope_api_key" in diagnostics["fallbacks"]


def test_reranker_reports_deterministic_fallback_when_fallback_setting_is_stale(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    docs = [
        Document(page_content="Общие правила склада.", metadata={"citation_id": "warehouse"}),
        Document(page_content="Порядок возврата брака.", metadata={"citation_id": "return", "_keyword_rank": 0}),
    ]
    diagnostics = {"fallbacks": []}

    monkeypatch.setattr(retrieval_module, "dashscope_configured", lambda: False, raising=False)

    settings = RetrievalSettings(fallback_reranker="cohere:stale", rerank_top_n=1)
    reranked = Reranker(settings=settings).rerank("возврат брака", docs, diagnostics)

    assert [doc.metadata["citation_id"] for doc in reranked] == ["return"]
    assert diagnostics["reranker"] == "fallback:deterministic"
    assert "reranker_missing_dashscope_api_key" in diagnostics["fallbacks"]
    assert "reranker_unsupported:cohere:stale" in diagnostics["fallbacks"]


def test_reranker_falls_back_when_dashscope_provider_raises(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    docs = [
        Document(page_content="Общие правила склада.", metadata={"citation_id": "warehouse"}),
        Document(page_content="Порядок возврата брака.", metadata={"citation_id": "return", "_keyword_rank": 0}),
    ]
    diagnostics = {"fallbacks": []}

    class BrokenCompressor:
        def compress_documents(self, documents, query):
            raise RuntimeError("dashscope unavailable")

    monkeypatch.setattr(retrieval_module, "dashscope_configured", lambda: True, raising=False)
    monkeypatch.setattr(retrieval_module, "create_reranker", lambda top_n, settings: BrokenCompressor(), raising=False)

    settings = RetrievalSettings(primary_reranker="dashscope:qwen3-rerank-test", rerank_top_n=1)
    reranked = Reranker(settings=settings).rerank("возврат брака", docs, diagnostics)

    assert [doc.metadata["citation_id"] for doc in reranked] == ["return"]
    assert diagnostics["reranker"] == "fallback:deterministic"
    assert "reranker_failed:dashscope:qwen3-rerank-test" in diagnostics["fallbacks"]


def test_reranker_backfills_when_primary_returns_too_few(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    docs = [
        Document(page_content="Порядок возврата брака.", metadata={"citation_id": "return", "_keyword_rank": 0}),
        Document(page_content="Возврат оформляется актом.", metadata={"citation_id": "act", "_keyword_rank": 1}),
    ]
    diagnostics = {"fallbacks": []}

    monkeypatch.setattr(retrieval_module, "dashscope_configured", lambda: False, raising=False)

    reranked = Reranker(settings=RetrievalSettings(rerank_top_n=3)).rerank("возврат брака", docs, diagnostics)

    assert [doc.metadata["citation_id"] for doc in reranked] == ["return", "act"]
    assert diagnostics["reranked_candidates"] == 2
