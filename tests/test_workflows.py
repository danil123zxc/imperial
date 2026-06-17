from contextlib import contextmanager

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

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


def test_query_workflow_default_generation_uses_lcel_prompt_chain():
    docs = [
        Document(
            page_content="Возврат брака оформляется актом.",
            metadata={"citation_id": "return-policy"},
        )
    ]
    calls = []

    def fake_model(prompt_value):
        messages = prompt_value.to_messages()
        calls.append(messages)
        assert messages[0].type == "system"
        assert "strict-citation RAG assistant" in messages[0].content
        assert messages[1].type == "human"
        assert "Возврат брака оформляется актом." in messages[1].content
        return "Возврат брака оформляется актом. [S1]"

    workflow = build_query_workflow(
        retrieve=lambda question: docs,
        chat_model=RunnableLambda(fake_model),
    )

    result = workflow.invoke({"question": "Как оформить возврат брака?"})

    assert result["answer"] == "Возврат брака оформляется актом. [S1]"
    assert result["citations_valid"] is True
    assert len(calls) == 1


def test_query_workflow_traces_answer_generation(monkeypatch):
    from imperial_rag import workflows as workflows_module

    docs = [
        Document(
            page_content="Возврат брака оформляется актом.",
            metadata={"citation_id": "return-policy"},
        )
    ]
    trace_calls = []

    class FakeTraceSpan:
        def set_attribute(self, key, value):
            pass

        def set_output(self, value):
            trace_calls.append({"output": value})

    @contextmanager
    def fake_trace_answer_step(name, question, *, attributes=None):
        trace_calls.append({"name": name, "question": question, "attributes": attributes})
        yield FakeTraceSpan()

    monkeypatch.setattr(workflows_module, "trace_answer_step", fake_trace_answer_step)

    workflow = build_query_workflow(
        retrieve=lambda question: docs,
        generate=lambda question, retrieved_docs: "Возврат брака оформляется актом. [S1]",
    )

    result = workflow.invoke({"question": "Как оформить возврат брака?"})

    assert result["answer"] == "Возврат брака оформляется актом. [S1]"
    assert trace_calls == [
        {
            "name": "answer.generate",
            "question": "Как оформить возврат брака?",
            "attributes": {
                "imperial.phase": "answer",
                "imperial.step": "generate",
                "imperial.trace_schema_version": "rag-v2",
                "answer.evidence_count": 1,
                "answer.citation_count": 1,
                "answer.citation_ids": ["return-policy"],
                "answer.context_chars": 32,
            },
        },
        {
            "name": "answer.prepare_context",
            "question": "Как оформить возврат брака?",
            "attributes": {
                "imperial.phase": "answer",
                "imperial.step": "prepare_context",
                "imperial.trace_schema_version": "rag-v2",
                "answer.evidence_count": 1,
                "answer.citation_count": 1,
                "answer.citation_ids": ["return-policy"],
                "answer.context_chars": 32,
            },
        },
        {
            "output": {
                "evidence_count": 1,
                "citation_count": 1,
                "citation_ids": ["return-policy"],
                "source_count": 1,
                "context_chars": 32,
            }
        },
        {
            "name": "answer.call_model",
            "question": "Как оформить возврат брака?",
            "attributes": {
                "imperial.phase": "answer",
                "imperial.step": "call_model",
                "imperial.trace_schema_version": "rag-v2",
                "answer.evidence_count": 1,
                "answer.citation_count": 1,
            },
        },
        {"output": {"answer_chars": 37, "evidence_count": 1, "citation_count": 1}},
        {
            "name": "answer.validate_citations",
            "question": "Как оформить возврат брака?",
            "attributes": {
                "imperial.phase": "answer",
                "imperial.step": "validate_citations",
                "imperial.trace_schema_version": "rag-v2",
                "answer.evidence_count": 1,
                "answer.citation_count": 1,
            },
        },
        {
            "output": {
                "citations_valid": True,
                "invalid_citations": [],
                "evidence_count": 1,
                "citation_count": 1,
            }
        },
        {
            "output": {
                "answer": "Возврат брака оформляется актом. [S1]",
                "citations_valid": True,
                "invalid_citations": [],
                "refused": False,
                "evidence_count": 1,
                "citation_count": 1,
            }
        },
    ]


def test_query_workflow_sets_prompt_provenance_and_generator_trace_attributes(monkeypatch):
    from imperial_rag import workflows as workflows_module

    docs = [
        Document(
            page_content="Known private fact.",
            metadata={"citation_id": "known"},
        )
    ]
    trace_calls = []

    class FakeTraceSpan:
        def set_attribute(self, key, value):
            trace_calls.append({"attribute": key, "value": value})

        def set_output(self, value):
            trace_calls.append({"output": value})

    @contextmanager
    def fake_trace_answer_step(name, question, *, attributes=None):
        trace_calls.append({"name": name, "question": question, "attributes": attributes})
        yield FakeTraceSpan()

    monkeypatch.setattr(workflows_module, "trace_answer_step", fake_trace_answer_step)

    workflow = build_query_workflow(
        retrieve=lambda question: docs,
        generate=lambda question, retrieved_docs: {
            "answer": "Known private fact. [S1]",
            "trace_attributes": {"answer.model_status": "ok"},
        },
    )

    result = workflow.invoke({"question": "What is known?"})

    assert result["answer"] == "Known private fact. [S1]"
    model_start_index = next(
        index for index, call in enumerate(trace_calls) if call.get("name") == "answer.call_model"
    )
    model_attribute_calls = [
        call for call in trace_calls[model_start_index + 1 :] if "attribute" in call
    ]
    prompt_attrs = {call["attribute"]: call["value"] for call in model_attribute_calls}
    assert prompt_attrs["answer.prompt_version"] == "strict-rag-v1"
    assert len(prompt_attrs["answer.prompt_skeleton_hash"]) == 64
    assert prompt_attrs["answer.evidence_count"] == 1
    assert prompt_attrs["answer.citation_count"] == 1
    assert prompt_attrs["answer.citation_ids"] == ["known"]
    assert prompt_attrs["answer.context_chars"] == len("Known private fact.")
    assert prompt_attrs["answer.model_status"] == "ok"
    assert all("Known private fact" not in str(value) for value in prompt_attrs.values())


def test_query_workflow_traces_refusal_without_evidence(monkeypatch):
    from imperial_rag import workflows as workflows_module

    trace_calls = []

    class FakeTraceSpan:
        def set_attribute(self, key, value):
            pass

        def set_output(self, value):
            trace_calls.append({"output": value})

    @contextmanager
    def fake_trace_answer_step(name, question, *, attributes=None):
        trace_calls.append({"name": name, "question": question, "attributes": attributes})
        yield FakeTraceSpan()

    monkeypatch.setattr(workflows_module, "trace_answer_step", fake_trace_answer_step)

    workflow = build_query_workflow()
    result = workflow.invoke({"question": "Что не найдено?"})

    assert result["answer"] == REFUSAL_TEXT
    assert trace_calls == [
        {
            "name": "answer.generate",
            "question": "Что не найдено?",
            "attributes": {
                "imperial.phase": "answer",
                "imperial.step": "generate",
                "imperial.trace_schema_version": "rag-v2",
                "answer.evidence_count": 0,
                "answer.citation_count": 0,
                "answer.citation_ids": [],
                "answer.context_chars": 0,
            },
        },
        {
            "name": "answer.prepare_context",
            "question": "Что не найдено?",
            "attributes": {
                "imperial.phase": "answer",
                "imperial.step": "prepare_context",
                "imperial.trace_schema_version": "rag-v2",
                "answer.evidence_count": 0,
                "answer.citation_count": 0,
                "answer.citation_ids": [],
                "answer.context_chars": 0,
            },
        },
        {
            "output": {
                "evidence_count": 0,
                "citation_count": 0,
                "citation_ids": [],
                "source_count": 0,
                "context_chars": 0,
            }
        },
        {
            "output": {
                "answer": REFUSAL_TEXT,
                "citations_valid": True,
                "invalid_citations": [],
                "refused": True,
                "evidence_count": 0,
                "citation_count": 0,
            }
        },
    ]


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


def test_query_workflow_preserves_unsupported_generated_answer_with_invalid_diagnostics():
    docs = [Document(page_content="Known fact.", metadata={"citation_id": "known"})]
    generated_answer = "Unsupported fact. [missing]"

    workflow = build_query_workflow(
        retrieve=lambda question: docs,
        generate=lambda question, retrieved_docs: generated_answer,
    )

    result = workflow.invoke({"question": "What is known?"})

    assert result["answer"] == generated_answer
    assert result["citations_valid"] is False
    assert result["invalid_citations"] == ["missing"]


def test_query_workflow_preserves_uncited_generated_answer_with_invalid_diagnostics():
    docs = [Document(page_content="Known fact.", metadata={"citation_id": "known"})]
    generated_answer = "Known fact."

    workflow = build_query_workflow(
        retrieve=lambda question: docs,
        generate=lambda question, retrieved_docs: generated_answer,
    )

    result = workflow.invoke({"question": "What is known?"})

    assert result["answer"] == generated_answer
    assert result["citations_valid"] is False
    assert result["invalid_citations"] == []


def test_query_workflow_traces_invalid_generated_answer_without_refusal(monkeypatch):
    from imperial_rag import workflows as workflows_module

    docs = [Document(page_content="Known fact.", metadata={"citation_id": "known"})]
    generated_answer = "Unsupported fact. [missing]"
    trace_calls = []

    class FakeTraceSpan:
        def set_attribute(self, key, value):
            pass

        def set_output(self, value):
            trace_calls.append({"output": value})

    @contextmanager
    def fake_trace_answer_step(name, question, *, attributes=None):
        trace_calls.append({"name": name, "question": question, "attributes": attributes})
        yield FakeTraceSpan()

    monkeypatch.setattr(workflows_module, "trace_answer_step", fake_trace_answer_step)

    workflow = build_query_workflow(
        retrieve=lambda question: docs,
        generate=lambda question, retrieved_docs: generated_answer,
    )

    result = workflow.invoke({"question": "What is known?"})

    assert result["answer"] == generated_answer
    assert trace_calls == [
        {
            "name": "answer.generate",
            "question": "What is known?",
            "attributes": {
                "imperial.phase": "answer",
                "imperial.step": "generate",
                "imperial.trace_schema_version": "rag-v2",
                "answer.evidence_count": 1,
                "answer.citation_count": 1,
                "answer.citation_ids": ["known"],
                "answer.context_chars": 11,
            },
        },
        {
            "name": "answer.prepare_context",
            "question": "What is known?",
            "attributes": {
                "imperial.phase": "answer",
                "imperial.step": "prepare_context",
                "imperial.trace_schema_version": "rag-v2",
                "answer.evidence_count": 1,
                "answer.citation_count": 1,
                "answer.citation_ids": ["known"],
                "answer.context_chars": 11,
            },
        },
        {
            "output": {
                "evidence_count": 1,
                "citation_count": 1,
                "citation_ids": ["known"],
                "source_count": 1,
                "context_chars": 11,
            }
        },
        {
            "name": "answer.call_model",
            "question": "What is known?",
            "attributes": {
                "imperial.phase": "answer",
                "imperial.step": "call_model",
                "imperial.trace_schema_version": "rag-v2",
                "answer.evidence_count": 1,
                "answer.citation_count": 1,
            },
        },
        {"output": {"answer_chars": 27, "evidence_count": 1, "citation_count": 1}},
        {
            "name": "answer.validate_citations",
            "question": "What is known?",
            "attributes": {
                "imperial.phase": "answer",
                "imperial.step": "validate_citations",
                "imperial.trace_schema_version": "rag-v2",
                "answer.evidence_count": 1,
                "answer.citation_count": 1,
            },
        },
        {
            "output": {
                "citations_valid": False,
                "invalid_citations": ["missing"],
                "evidence_count": 1,
                "citation_count": 1,
            }
        },
        {
            "output": {
                "answer": generated_answer,
                "citations_valid": False,
                "invalid_citations": ["missing"],
                "refused": False,
                "evidence_count": 1,
                "citation_count": 1,
            }
        },
    ]


def test_query_workflow_preserves_generated_refusal_text_with_evidence():
    docs = [Document(page_content="Known fact.", metadata={"citation_id": "known"})]

    workflow = build_query_workflow(
        retrieve=lambda question: docs,
        generate=lambda question, retrieved_docs: REFUSAL_TEXT,
    )

    result = workflow.invoke({"question": "What is known?"})

    assert result["answer"] == REFUSAL_TEXT
    assert result["citations_valid"] is True
    assert result["invalid_citations"] == []


def test_query_workflow_preserves_cited_answer_with_form_placeholder():
    docs = [
        Document(
            page_content="Дата «_____» _______2025 г. (укажите дату)",
            metadata={"citation_id": "resignation-form", "source_type": "body"},
        )
    ]
    answer = "В заявлении нужно заполнить поле [укажите дату]. [S1]"

    workflow = build_query_workflow(
        retrieve=lambda question: docs,
        generate=lambda question, retrieved_docs: answer,
    )

    result = workflow.invoke({"question": "Как оформляется заявление на увольнение?"})

    assert result["answer"] == answer
    assert result["citations_valid"] is True
    assert result["invalid_citations"] == []


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
