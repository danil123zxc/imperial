from __future__ import annotations

import hashlib
from contextlib import contextmanager

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import Field

import imperial_rag.retrieval as retrieval_module
from imperial_rag.document_ids import document_key
from imperial_rag.retrieval import CandidateMerger, FallbackRanker, RetrievalSettings, RrfCandidateFusion
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


class FakeKeywordSearchWithoutScores:
    def __init__(self, docs):
        self.docs = docs

    def search_with_scores(self, query, limit):
        class Hit:
            def __init__(self, document):
                self.document = document
                self.score = 0.0

        return [Hit(document) for document in self.docs[:limit]]


class FakeBaseRetriever(BaseRetriever):
    docs: list[Document]
    calls: list[dict] = Field(default_factory=list)

    def _get_relevant_documents(self, query, *, run_manager, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return list(self.docs)


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

            def set_final_evidence_documents(self, documents):
                self._set_documents("retrieval.documents", documents)

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


def capture_langchain_span_names(operation):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    try:
        result = operation()
        span_names = [span.name for span in exporter.get_finished_spans()]
    finally:
        instrumentor.uninstrument()
    return span_names, result


def test_retrieval_settings_defaults_match_accuracy_spec(monkeypatch):
    for name in (
        "IMPERIAL_RAG_CHUNK_SIZE",
        "IMPERIAL_RAG_CHUNK_OVERLAP",
        "IMPERIAL_RAG_VECTOR_FETCH_K",
        "IMPERIAL_RAG_VECTOR_K",
        "IMPERIAL_RAG_KEYWORD_LIMIT",
        "IMPERIAL_RAG_RERANK_INPUT_LIMIT",
        "IMPERIAL_RAG_RERANK_TOP_N",
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


def test_hybrid_retriever_invokes_langchain_retrievers_with_configured_limits():
    vector_docs = [
        Document(page_content=f"vector {index}", metadata={"citation_id": f"v{index}"})
        for index in range(3)
    ]
    keyword_docs = [
        Document(page_content=f"keyword {index}", metadata={"citation_id": f"k{index}", "_keyword_score": 4.0})
        for index in range(4)
    ]
    vector = FakeBaseRetriever(docs=vector_docs)
    keyword = FakeBaseRetriever(docs=keyword_docs)
    settings = RetrievalSettings(vector_fetch_k=8, vector_k=2, keyword_limit=3, mmr_lambda_mult=0.25)

    result = HybridRetriever(vector_search=vector, keyword_search=keyword, settings=settings).retrieve("возврат")

    assert [doc.metadata["citation_id"] for doc in result.vector_docs] == ["v0", "v1", "v2"]
    assert [doc.metadata["citation_id"] for doc in result.keyword_docs] == ["k0", "k1", "k2", "k3"]
    assert vector.calls == [{"query": "возврат", "k": 2, "fetch_k": 8, "lambda_mult": 0.25}]
    assert keyword.calls == [{"query": "возврат", "limit": 3}]
    assert result.vector_docs[0].metadata["_vector_rank"] == 0
    assert result.keyword_docs[0].metadata["_keyword_rank"] == 0
    assert result.vector_docs[0].metadata["_retrieval_id"] == "v0"
    assert result.keyword_docs[0].metadata["_retrieval_id"] == "k0"


def test_hybrid_retriever_reports_keyword_scores_available_when_scores_are_present():
    vector = FakeVectorSearch([])
    keyword = FakeKeywordSearch(
        [
            Document(
                page_content="Порядок возврата брака",
                metadata={"citation_id": "return", "_keyword_rank": 0, "_keyword_score": 7.5},
            )
        ]
    )

    result = HybridRetriever(vector_search=vector, keyword_search=keyword, settings=RetrievalSettings()).retrieve(
        "возврат брака"
    )

    assert result.diagnostics["keyword_scores_available"] is True


def test_hybrid_retriever_reports_keyword_scores_unavailable_when_scores_are_absent():
    vector = FakeVectorSearch([])
    keyword = FakeKeywordSearchWithoutScores(
        [
            Document(
                page_content="Порядок возврата брака",
                metadata={"citation_id": "return", "_keyword_rank": 0},
            )
        ]
    )

    result = HybridRetriever(vector_search=vector, keyword_search=keyword, settings=RetrievalSettings()).retrieve(
        "возврат брака"
    )

    assert result.diagnostics["keyword_scores_available"] is False


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


def test_hybrid_retriever_reports_vector_store_unavailable_without_vector_call():
    calls = []

    class UnavailableVector:
        vector_unavailable = True
        error_type = "RuntimeError"

        def max_marginal_relevance_search(self, query, k, fetch_k, lambda_mult):
            calls.append(query)
            raise RuntimeError("vector search should not be called")

    keyword_docs = [Document(page_content="keyword", metadata={"citation_id": "k"})]

    result = HybridRetriever(
        vector_search=UnavailableVector(),
        keyword_search=FakeKeywordSearch(keyword_docs),
        settings=RetrievalSettings(),
    ).retrieve("возврат")

    assert calls == []
    assert result.vector_docs == []
    assert [doc.metadata["citation_id"] for doc in result.keyword_docs] == ["k"]
    assert result.diagnostics["vector_search_status"] == "unavailable"
    assert result.diagnostics["keyword_search_status"] == "ok"
    assert result.diagnostics["vector_search_error_type"] == "RuntimeError"
    assert "vector_store_unavailable" in result.diagnostics["fallbacks"]


def test_retrieval_service_returns_final_evidence_and_diagnostics(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    vector_docs = [
        Document(page_content="vector return", metadata={"citation_id": "v", "file_id": "f", "source_type": "body", "chunk_index": 0})
    ]
    keyword_docs = [
        Document(page_content="Порядок возврата брака", metadata={"citation_id": "k", "file_id": "f", "source_type": "body", "chunk_index": 1, "_keyword_rank": 0})
    ]
    service = RetrievalService(
        vector_search=FakeVectorSearch(vector_docs),
        keyword_search=FakeKeywordSearch(keyword_docs),
        settings=RetrievalSettings(rerank_top_n=1),
    )

    result = service.retrieve("возврат брака")

    assert [doc.metadata["citation_id"] for doc in result.evidence] == ["k"]
    assert result.diagnostics["merged_candidates"] == 2
    assert result.diagnostics["fusion"] == "rrf"
    assert result.diagnostics["fusion_rrf_k"] == 60
    assert result.diagnostics["fused_candidates"] == 2
    assert result.diagnostics["rerank_input_candidates"] == 2
    assert result.diagnostics["final_evidence"] == 1
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
    service = RetrievalService(
        vector_search=FakeVectorSearch(vector_docs),
        keyword_search=FakeKeywordSearch(keyword_docs),
        settings=RetrievalSettings(rerank_top_n=1),
    )

    result = service.retrieve("возврат брака")

    assert [record["name"] for record in records] == [
        "retrieval",
        "retrieval.vector_search",
        "retrieval.keyword_search",
        "retrieval.rerank",
        "retrieval.final_evidence",
    ]
    assert [record["query"] for record in records] == ["возврат брака"] * 5
    assert records[0]["kind"] == "CHAIN"
    assert records[0]["attributes"]["imperial.phase"] == "retrieval"
    assert records[0]["attributes"]["imperial.step"] == "run"
    assert records[0]["attributes"]["imperial.trace_schema_version"] == "rag-v2"
    assert records[1]["output"]["status"] == "ok"
    assert records[1]["output"]["count"] == 1
    assert records[1]["output"]["top_documents"][0]["citation_id"] == "v"
    assert records[1]["set_attributes"] == {}
    assert records[2]["output"]["status"] == "ok"
    assert records[2]["output"]["count"] == 1
    assert records[2]["set_attributes"] == {}
    assert records[3]["kind"] == "RERANKER"
    assert records[3]["attributes"]["imperial.phase"] == "retrieval"
    assert records[3]["attributes"]["imperial.step"] == "rerank"
    assert records[3]["attributes"]["reranker.query"] == "возврат брака"
    assert records[3]["attributes"]["reranker.top_k"] == 1
    assert records[3]["set_attributes"]["reranker.model_name"] == "fallback:deterministic"
    assert records[3]["output"]["reranker"] == "fallback:deterministic"
    assert "reranker_missing_dashscope_api_key" in records[3]["output"]["fallbacks"]
    rerank_attributes = records[3]["set_attributes"]
    assert [rerank_attributes[f"reranker.input_documents.{index}.document.id"] for index in range(2)] == ["v", "k"]
    assert [rerank_attributes[f"reranker.output_documents.{index}.document.id"] for index in range(1)] == ["k"]
    assert records[4]["name"] == "retrieval.final_evidence"
    assert records[4]["kind"] == "RETRIEVER"
    assert records[4]["attributes"]["imperial.step"] == "final_evidence"
    assert records[4]["set_attributes"]["retrieval.documents.0.document.id"] == "k"
    assert records[4]["set_attributes"]["retrieval.documents.0.document.content"] == "Порядок возврата брака"
    assert records[4]["output"]["count"] == 1
    assert records[4]["output"]["citation_ids"] == ["k"]
    assert records[4]["output"]["context_chars"] == 22
    assert records[0]["output"]["merged_candidates"] == 2
    assert records[0]["output"]["fused_candidates"] == 2
    assert records[0]["output"]["rerank_input_candidates"] == 2
    assert records[0]["output"]["fusion"] == "rrf"
    assert records[0]["output"]["fusion_rrf_k"] == 60
    assert records[0]["output"]["final_evidence"] == 1
    assert records[0]["output"]["reranker"] == "fallback:deterministic"
    assert records[0]["set_attributes"]["tag.tags"] == [
        "degraded",
        "fallback:reranker_missing_dashscope_api_key",
    ]
    assert [doc.metadata["citation_id"] for doc in result.evidence] == ["k"]
    assert result.diagnostics["final_evidence"] == 1


def test_retrieval_service_suppresses_langchain_internal_spans_by_default(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("IMPERIAL_RAG_TRACE_SUPPRESS_INTERNALS", raising=False)
    records = capture_retrieval_spans(monkeypatch)

    class VectorStoreRetriever(FakeBaseRetriever):
        pass

    class ElasticsearchRetriever(FakeBaseRetriever):
        pass

    service = RetrievalService(
        vector_search=VectorStoreRetriever(
            docs=[Document(page_content="vector return", metadata={"citation_id": "v"})]
        ),
        keyword_search=ElasticsearchRetriever(
            docs=[Document(page_content="keyword return", metadata={"citation_id": "k", "_keyword_score": 2.0})]
        ),
        settings=RetrievalSettings(rerank_top_n=1),
    )

    internal_span_names, result = capture_langchain_span_names(lambda: service.retrieve("возврат брака"))

    assert [record["name"] for record in records] == [
        "retrieval",
        "retrieval.vector_search",
        "retrieval.keyword_search",
        "retrieval.rerank",
        "retrieval.final_evidence",
    ]
    assert internal_span_names == []
    assert [doc.metadata["citation_id"] for doc in result.evidence] == ["k"]


def test_retrieval_service_can_allow_langchain_internal_spans(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_SUPPRESS_INTERNALS", "false")

    class VectorStoreRetriever(FakeBaseRetriever):
        pass

    class ElasticsearchRetriever(FakeBaseRetriever):
        pass

    service = RetrievalService(
        vector_search=VectorStoreRetriever(
            docs=[Document(page_content="vector return", metadata={"citation_id": "v"})]
        ),
        keyword_search=ElasticsearchRetriever(
            docs=[Document(page_content="keyword return", metadata={"citation_id": "k", "_keyword_score": 2.0})]
        ),
        settings=RetrievalSettings(rerank_top_n=1),
    )

    internal_span_names, result = capture_langchain_span_names(lambda: service.retrieve("возврат брака"))

    assert internal_span_names == [
        "VectorStoreRetriever",
        "ElasticsearchRetriever",
        "_StaticDocumentRetriever",
        "_StaticDocumentRetriever",
        "EnsembleRetriever",
    ]
    assert [doc.metadata["citation_id"] for doc in result.evidence] == ["k"]


def test_retrieval_service_traces_candidate_documents_only_when_enabled(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_CANDIDATE_DOCUMENTS", "true")
    records = capture_retrieval_spans(monkeypatch)
    vector_docs = [
        Document(page_content="vector return", metadata={"citation_id": "v", "chunk_id": "vector-chunk"})
    ]
    keyword_docs = [
        Document(page_content="keyword return", metadata={"citation_id": "k", "chunk_id": "keyword-chunk", "_keyword_rank": 0})
    ]
    service = RetrievalService(
        vector_search=FakeVectorSearch(vector_docs),
        keyword_search=FakeKeywordSearch(keyword_docs),
        settings=RetrievalSettings(rerank_top_n=1),
    )

    service.retrieve("возврат брака")

    assert records[1]["set_attributes"]["retrieval.documents.0.document.id"] == "vector-chunk"
    assert records[1]["set_attributes"]["retrieval.documents.0.document.content"] == "vector return"
    assert records[2]["set_attributes"]["retrieval.documents.0.document.id"] == "keyword-chunk"
    assert records[2]["set_attributes"]["retrieval.documents.0.document.content"] == "keyword return"


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
        settings=RetrievalSettings(),
    )

    result = service.retrieve("возврат")

    assert [record["name"] for record in records] == [
        "retrieval",
        "retrieval.vector_search",
        "retrieval.keyword_search",
        "retrieval.rerank",
        "retrieval.final_evidence",
    ]
    assert records[1]["output"]["status"] == "unavailable"
    assert records[1]["output"]["fallbacks"] == ["vector_search_failed"]
    assert records[2]["output"]["status"] == "unavailable"
    assert records[2]["output"]["fallbacks"] == ["vector_search_failed", "keyword_search_failed"]
    assert records[3]["output"]["reranker"] == "none"
    assert records[4]["output"]["count"] == 0
    assert records[0]["output"]["degraded"] is True
    assert records[0]["output"]["fallbacks"] == ["vector_search_failed", "keyword_search_failed"]
    assert records[0]["set_attributes"]["tag.tags"] == [
        "degraded",
        "fallback:vector_search_failed",
        "fallback:keyword_search_failed",
    ]
    assert result.evidence == []
    assert result.diagnostics["final_evidence"] == 0


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
        settings=RetrievalSettings(primary_reranker="dashscope:qwen3-rerank-test", rerank_top_n=1),
    )

    result = service.retrieve("возврат")

    assert records[3]["name"] == "retrieval.rerank"
    assert records[3]["output"]["reranker"] == "fallback:deterministic"
    assert "reranker_failed:dashscope:qwen3-rerank-test" in records[3]["output"]["fallbacks"]
    assert [doc.metadata["citation_id"] for doc in result.evidence] == ["k"]


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


def test_rrf_candidate_fusion_deduplicates_by_retrieval_id_and_merges_metadata():
    docs = [
        Document(page_content="vector copy", metadata={"citation_id": "same", "_vector_rank": 0}),
        Document(page_content="keyword copy", metadata={"citation_id": "same", "_keyword_rank": 0, "_keyword_score": 7.0}),
        Document(page_content="keyword only", metadata={"citation_id": "keyword", "_keyword_rank": 1}),
    ]

    fused = RrfCandidateFusion().fuse(docs, rrf_k=60)

    assert [doc.metadata["citation_id"] for doc in fused] == ["same", "keyword"]
    assert fused[0].metadata["_retrieval_id"] == "same"
    assert fused[0].metadata["_vector_rank"] == 0
    assert fused[0].metadata["_keyword_rank"] == 0
    assert fused[0].metadata["_keyword_score"] == 7.0


def test_retrieval_id_helpers_hash_content_when_metadata_ids_are_missing() -> None:
    document = Document(page_content="private corpus text", metadata={})
    expected = f"content_sha256:{hashlib.sha256(b'private corpus text').hexdigest()[:12]}"

    assert document_key(document) == expected
    assert retrieval_module._retrieval_id(document) == expected

    annotated = retrieval_module._annotate_retrieval_documents([document], rank_key="_vector_rank")
    assert annotated[0].metadata["_retrieval_id"] == expected
    assert "private corpus text" not in annotated[0].metadata["_retrieval_id"]


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

    # Standard Reciprocal Rank Fusion scores on list position, so a lone top-of-list
    # vector doc and a lone top-of-list keyword doc tie and resolve to source order
    # (vector list first); unranked candidates always sort to the tail with score 0.
    assert [doc.metadata["citation_id"] for doc in fused] == ["v", "k", "unranked"]
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
    compressor_inputs = []

    class EchoCompressor:
        def compress_documents(self, documents, query):
            compressor_inputs.append([document.metadata["citation_id"] for document in documents])
            return list(documents[:2])

    monkeypatch.setattr(retrieval_module, "dashscope_configured", lambda: True, raising=False)
    monkeypatch.setattr(retrieval_module, "create_reranker", lambda top_n, settings: EchoCompressor(), raising=False)
    service = RetrievalService(
        vector_search=FakeVectorSearch(vector_docs),
        keyword_search=FakeKeywordSearch(keyword_docs),
        settings=RetrievalSettings(rerank_input_limit=6, rerank_top_n=2),
    )

    result = service.retrieve("возврат брака")

    rerank_record = records[3]
    assert rerank_record["name"] == "retrieval.rerank"
    assert compressor_inputs == [["v0", "k0", "v1", "k1", "v2", "k2"]]
    rerank_attributes = rerank_record["set_attributes"]
    assert [rerank_attributes[f"reranker.input_documents.{index}.document.id"] for index in range(6)] == [
        "v0",
        "k0",
        "v1",
        "k1",
        "v2",
        "k2",
    ]
    assert [rerank_attributes[f"reranker.output_documents.{index}.document.id"] for index in range(2)] == [
        "v0",
        "k0",
    ]
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
    assert [record["name"] for record in records] == [
        "retrieval",
        "retrieval.vector_search",
        "retrieval.keyword_search",
        "retrieval.rerank",
        "retrieval.final_evidence",
    ]
    assert records[4]["output"]["count"] == 10


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


def test_fallback_ranker_treats_elasticsearch_keyword_score_as_higher_is_better():
    docs = [
        Document(
            page_content="Возврат брака оформляется актом.",
            metadata={"citation_id": "low-es-score", "_keyword_score": 1.0},
        ),
        Document(
            page_content="Возврат брака оформляется актом.",
            metadata={"citation_id": "high-es-score", "_keyword_score": 10.0},
        ),
    ]

    ranked = FallbackRanker().rank("возврат брака", docs, top_n=2)

    assert [doc.metadata["citation_id"] for doc in ranked] == ["high-es-score", "low-es-score"]


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


def test_reranker_calls_compressor_compress_documents_directly(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    docs = [
        Document(page_content="Порядок возврата брака.", metadata={"citation_id": "return", "_keyword_rank": 0}),
        Document(page_content="Возврат оформляется актом.", metadata={"citation_id": "act", "_keyword_rank": 1}),
    ]
    diagnostics = {"fallbacks": []}
    calls = []

    class FakeCompressor:
        def compress_documents(self, documents, query):
            calls.append({"documents": list(documents), "query": query})
            return [documents[1]]

    monkeypatch.setattr(retrieval_module, "dashscope_configured", lambda: True, raising=False)
    monkeypatch.setattr(retrieval_module, "create_reranker", lambda top_n, settings: FakeCompressor(), raising=False)

    reranked = Reranker(settings=RetrievalSettings(rerank_input_limit=2, rerank_top_n=2)).rerank(
        "возврат брака",
        docs,
        diagnostics,
    )

    assert [doc.metadata["citation_id"] for doc in reranked] == ["act", "return"]
    assert calls == [{"documents": docs[:2], "query": "возврат брака"}]
    assert diagnostics["reranker"] == "dashscope:qwen3-rerank"


def test_reranker_suppresses_dashscope_compressor_internals(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    docs = [
        Document(page_content="Порядок возврата брака.", metadata={"citation_id": "return", "_keyword_rank": 0}),
        Document(page_content="Возврат оформляется актом.", metadata={"citation_id": "act", "_keyword_rank": 1}),
    ]
    diagnostics = {"fallbacks": []}
    suppress_state = {"active": False}
    calls = []

    @contextmanager
    def fake_suppress_internal_tracing():
        suppress_state["active"] = True
        try:
            yield
        finally:
            suppress_state["active"] = False

    class FakeCompressor:
        def compress_documents(self, documents, query):
            calls.append({"suppressed": suppress_state["active"], "query": query})
            return [documents[0]]

    monkeypatch.setattr(retrieval_module, "suppress_internal_tracing", fake_suppress_internal_tracing, raising=False)
    monkeypatch.setattr(retrieval_module, "dashscope_configured", lambda: True, raising=False)
    monkeypatch.setattr(retrieval_module, "create_reranker", lambda top_n, settings: FakeCompressor(), raising=False)

    reranked = Reranker(settings=RetrievalSettings(rerank_input_limit=2, rerank_top_n=1)).rerank(
        "возврат брака",
        docs,
        diagnostics,
    )

    assert [doc.metadata["citation_id"] for doc in reranked] == ["return"]
    assert calls == [{"suppressed": True, "query": "возврат брака"}]
    assert diagnostics["reranker"] == "dashscope:qwen3-rerank"


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
