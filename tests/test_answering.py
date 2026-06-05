from langchain_core.documents import Document

from imperial_rag.answering import (
    REFUSAL_TEXT,
    answer_has_required_citations,
    build_context,
    build_strict_messages,
    format_citations,
    format_sources,
    refuse_message,
    validate_citations,
)


def test_format_citations_uses_short_evidence_labels():
    docs = [
        Document(page_content="Первый факт.", metadata={"source_type": "body"}),
        Document(page_content="Второй факт.", metadata={"source_type": "table"}),
    ]

    assert format_citations(docs) == ["[S1] body", "[S2] table"]


def test_format_sources_uses_short_labels_and_preserves_paths():
    docs = [
        Document(
            page_content="Возврат оформляется актом.",
            metadata={
                "citation_id": "return-policy:body:0",
                "file_path": "/docs/return.docx",
                "source_type": "body",
            },
        )
    ]

    assert format_sources(docs) == ["[S1] /docs/return.docx body"]


def test_build_context_includes_short_source_labels_and_content():
    docs = [
        Document(
            page_content="Возврат брака оформляется актом.",
            metadata={"citation_id": "return-policy:body:0"},
        )
    ]

    context = build_context(docs)

    assert "Source: [S1]" in context
    assert "Возврат брака оформляется актом." in context


def test_build_strict_messages_forbids_unsupported_answers():
    messages = build_strict_messages("Что делать?", [])

    rendered = "\n".join(message["content"] for message in messages)
    assert "Use only the provided context" in rendered
    assert "cite every factual claim" in rendered
    assert "Do not include uncited introductions or summaries" in rendered
    assert REFUSAL_TEXT in rendered


def test_validate_citations_rejects_unknown_ids_and_allows_refusal():
    docs = [Document(page_content="Known", metadata={"citation_id": "known"})]

    assert validate_citations("No indexed evidence was enough to answer.", docs) == (True, [])
    assert validate_citations("Fact. [S1]", docs) == (True, [])
    assert validate_citations("Fact. [missing]", docs) == (False, ["missing"])


def test_validate_citations_accepts_unicode_normalized_legacy_paths():
    decomposed = "РЕГАМЕНТ Возвраты НОВЫЙ.docx#body:chunk-17"
    composed = "РЕГАМЕНТ Возвраты НОВЫЙ.docx#body:chunk-17"
    docs = [Document(page_content="Known", metadata={"citation_id": decomposed})]

    assert validate_citations(f"Fact. [{composed}]", docs) == (True, [])


def test_answer_has_required_citations_accepts_unicode_normalized_markers():
    decomposed = "[РЕГАМЕНТ Возвраты НОВЫЙ.docx#body:chunk-17] body"
    composed = "[РЕГАМЕНТ Возвраты НОВЫЙ.docx#body:chunk-17]"

    assert answer_has_required_citations(f"Fact. {composed}", [decomposed]) is True


def test_refuse_message_mentions_question_and_refusal_text():
    message = refuse_message("Что делать с браком?")

    assert message == REFUSAL_TEXT
