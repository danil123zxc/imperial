import pytest
from langchain_core.documents import Document

from imperial_rag.answering import REFUSAL_TEXT
from imperial_rag.workflows import (
    build_ingestion_workflow,
    build_query_workflow,
    rank_hybrid_candidates,
)


def test_query_workflow_smoke_refuses_without_docs():
    workflow = build_query_workflow()

    result = workflow.invoke({"question": "Как оформить возврат брака?"})

    assert result["retrieved_documents"] == []
    assert result["answer"] == REFUSAL_TEXT
    assert result["citations_valid"] is True
    assert result["invalid_citations"] == []


def test_query_workflow_with_injected_retrieval_and_generator_happy_path():
    docs = [
        Document(
            page_content="Возврат брака оформляется актом.",
            metadata={"citation_id": "return-policy"},
        )
    ]

    def retrieve(question):
        return docs

    def generate(question, retrieved_docs):
        assert question == "Как оформить возврат брака?"
        assert retrieved_docs == docs
        return "Возврат брака оформляется актом. [S1]"

    workflow = build_query_workflow(retrieve=retrieve, generate=generate)

    result = workflow.invoke({"question": "Как оформить возврат брака?"})

    assert result["retrieved_documents"] == docs
    assert result["answer"] == "Возврат брака оформляется актом. [S1]"
    assert result["citations"] == ["[S1] unknown"]
    assert result["sources"] == ["[S1] unknown"]
    assert result["citations_valid"] is True
    assert result["invalid_citations"] == []


def test_query_workflow_preserves_retrieval_diagnostics():
    docs = [Document(page_content="Возврат брака оформляется актом.", metadata={"citation_id": "return"})]

    def retrieve(question):
        return {
            "retrieved_documents": docs,
            "vector_docs": [],
            "keyword_docs": docs,
            "retrieval": {
                "vector_candidates": 0,
                "keyword_candidates": 1,
                "merged_candidates": 1,
                "reranked_candidates": 1,
                "final_evidence": 1,
                "reranker": "fallback:deterministic",
                "fallbacks": ["reranker_missing_api_key"],
            },
        }

    workflow = build_query_workflow(
        retrieve=retrieve,
        generate=lambda question, retrieved_docs: "Возврат брака оформляется актом. [S1]",
    )

    result = workflow.invoke({"question": "Как оформить возврат брака?"})

    assert result["retrieval"]["final_evidence"] == 1
    assert result["retrieval"]["reranker"] == "fallback:deterministic"


def test_query_workflow_accepts_evidence_from_custom_retrieval_mapping():
    docs = [Document(page_content="Возврат брака оформляется актом.", metadata={"citation_id": "return"})]

    workflow = build_query_workflow(
        retrieve=lambda question: {"evidence": docs},
        generate=lambda question, retrieved_docs: "Возврат брака оформляется актом. [S1]",
    )

    result = workflow.invoke({"question": "Как оформить возврат брака?"})

    assert result["evidence"] == docs
    assert result["retrieved_documents"] == docs
    assert result["answer"] == "Возврат брака оформляется актом. [S1]"
    assert result["citations_valid"] is True


def test_query_workflow_preserves_explicit_empty_retrieved_documents():
    candidate_docs = [Document(page_content="Кандидат не должен стать evidence.", metadata={"citation_id": "candidate"})]

    workflow = build_query_workflow(
        retrieve=lambda question: {"retrieved_documents": [], "vector_docs": candidate_docs},
        generate=lambda question, retrieved_docs: "Should not be called. [candidate]",
    )

    result = workflow.invoke({"question": "Что найдено?"})

    assert result["evidence"] == []
    assert result["retrieved_documents"] == []
    assert result["vector_candidates"] == candidate_docs
    assert result["answer"] == REFUSAL_TEXT


def test_query_workflow_prefers_empty_retrieved_documents_over_documents_alias():
    docs = [Document(page_content="Этот документ не должен попасть в evidence.", metadata={"citation_id": "doc"})]

    workflow = build_query_workflow(
        retrieve=lambda question: {"retrieved_documents": [], "documents": docs},
        generate=lambda question, retrieved_docs: "Should not be called. [doc]",
    )

    result = workflow.invoke({"question": "Что найдено?"})

    assert result["evidence"] == []
    assert result["retrieved_documents"] == []
    assert result["answer"] == REFUSAL_TEXT


def test_query_workflow_accepts_documents_docs_and_evidence_aliases():
    aliases = ("documents", "docs", "evidence")

    for alias in aliases:
        docs = [Document(page_content=f"Документ из {alias}.", metadata={"citation_id": alias})]

        def retrieve(question, *, alias=alias, docs=docs):
            return {alias: docs}

        def generate(question, retrieved_docs, *, alias=alias):
            return "Документ из {alias}. [S1]".format(alias=alias)

        workflow = build_query_workflow(
            retrieve=retrieve,
            generate=generate,
        )

        result = workflow.invoke({"question": "Что найдено?"})

        assert result["evidence"] == docs
        assert result["retrieved_documents"] == docs
        assert result["answer"] == f"Документ из {alias}. [S1]"


def test_query_workflow_preserves_custom_retrieval_candidate_aliases():
    vector_docs = [Document(page_content="Векторный кандидат.", metadata={"citation_id": "vector"})]
    keyword_docs = [Document(page_content="Ключевой кандидат.", metadata={"citation_id": "keyword"})]

    workflow = build_query_workflow(
        retrieve=lambda question: {
            "retrieved_documents": keyword_docs,
            "vector_candidates": vector_docs,
            "keyword_candidates": keyword_docs,
        },
        generate=lambda question, retrieved_docs: "Ключевой кандидат. [S1]",
    )

    result = workflow.invoke({"question": "Что найдено?"})

    assert result["vector_candidates"] == vector_docs
    assert result["keyword_candidates"] == keyword_docs
    assert result["retrieved_documents"] == keyword_docs


def test_query_workflow_preserves_explicit_empty_vector_candidates():
    docs = [Document(page_content="Возврат брака оформляется актом.", metadata={"citation_id": "return"})]

    workflow = build_query_workflow(
        retrieve=lambda question: {"evidence": docs, "vector_docs": []},
        generate=lambda question, retrieved_docs: "Возврат брака оформляется актом. [S1]",
    )

    result = workflow.invoke({"question": "Как оформить возврат брака?"})

    assert result["evidence"] == docs
    assert result["vector_candidates"] == []


def test_query_workflow_preserves_explicit_empty_keyword_candidates():
    docs = [Document(page_content="Возврат брака оформляется актом.", metadata={"citation_id": "return"})]

    workflow = build_query_workflow(
        retrieve=lambda question: {"evidence": docs, "keyword_candidates": []},
        generate=lambda question, retrieved_docs: "Возврат брака оформляется актом. [S1]",
    )

    result = workflow.invoke({"question": "Как оформить возврат брака?"})

    assert result["evidence"] == docs
    assert result["keyword_candidates"] == []


def test_query_workflow_replaces_unsupported_generated_answer_with_refusal():
    docs = [Document(page_content="Known fact.", metadata={"citation_id": "known"})]

    workflow = build_query_workflow(
        retrieve=lambda question: docs,
        generate=lambda question, retrieved_docs: "Unsupported fact. [missing]",
    )

    result = workflow.invoke({"question": "What is known?"})

    assert result["answer"] == REFUSAL_TEXT
    assert result["citations_valid"] is False
    assert result["invalid_citations"] == ["missing"]


def test_query_workflow_preserves_cited_answer_with_uncited_structural_headings():
    docs = [Document(page_content="Known fact.", metadata={"citation_id": "known", "source_type": "body"})]
    answer = """**Обязанности:**
* Осуществляет погрузку товара. [S1]

### Ответственность
* Несет ответственность за сохранность товара. [S1]"""

    workflow = build_query_workflow(
        retrieve=lambda question: docs,
        generate=lambda question, retrieved_docs: answer,
    )

    result = workflow.invoke({"question": "Какие обязанности указаны?"})

    assert result["answer"] == answer
    assert result["citations_valid"] is True
    assert result["invalid_citations"] == []


def test_query_workflow_default_generation_requires_legacy_openai_flag(monkeypatch):
    docs = [Document(page_content="Возврат брака оформляется актом.", metadata={"citation_id": "return"})]

    class FakeModel:
        def invoke(self, messages):
            return type("Response", (), {"content": "Возврат брака оформляется актом. [S1]"})()

    monkeypatch.delenv("IMPERIAL_RAG_ALLOW_LEGACY_OPENAI", raising=False)
    monkeypatch.setattr("imperial_rag.workflows.ChatOpenAI", lambda **kwargs: FakeModel(), raising=False)

    workflow = build_query_workflow(retrieve=lambda question: docs)

    with pytest.raises(RuntimeError, match="Legacy OpenAI chat is disabled"):
        workflow.invoke({"question": "Как оформить возврат брака?"})


def test_rank_hybrid_candidates_deduplicates_and_boosts_keyword_exact_matches():
    duplicate_vector = Document(
        page_content="Возврат брака оформляется актом.",
        metadata={"citation_id": "same", "file_name": "return.docx"},
    )
    duplicate_keyword = Document(
        page_content="Возврат брака оформляется актом.",
        metadata={"citation_id": "same", "file_name": "Регламент возврата брака.docx"},
    )
    vector_only = Document(
        page_content="Общие правила склада.",
        metadata={"citation_id": "warehouse", "file_name": "warehouse.docx"},
    )
    keyword_exact = Document(
        page_content="Порядок возврата брака.",
        metadata={"citation_id": "return", "file_name": "Регламент возврата брака.docx"},
    )

    ranked = rank_hybrid_candidates(
        "возврат брака",
        vector_docs=[duplicate_vector, vector_only],
        keyword_docs=[duplicate_keyword, keyword_exact],
        k=3,
    )

    assert [doc.metadata["citation_id"] for doc in ranked] == ["same", "return", "warehouse"]


def test_ingestion_workflow_invokes_pipeline_and_returns_status_counts():
    def run_pipeline():
        return {"documents": 2, "chunks": 5}

    workflow = build_ingestion_workflow(run_pipeline=run_pipeline)

    result = workflow.invoke({})

    assert result["status"] == "completed"
    assert result["counts"] == {"documents": 2, "chunks": 5}
