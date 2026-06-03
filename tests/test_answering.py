from langchain_core.documents import Document

from imperial_rag.answering import (
    REFUSAL_TEXT,
    build_context,
    build_strict_messages,
    format_sources,
    refuse_message,
    validate_citations,
)


def test_format_sources_prefers_citation_id_metadata():
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

    assert format_sources(docs) == ["[return-policy:body:0] /docs/return.docx body"]


def test_build_context_includes_source_ids_and_content():
    docs = [
        Document(
            page_content="Возврат брака оформляется актом.",
            metadata={"citation_id": "return-policy:body:0"},
        )
    ]

    context = build_context(docs)

    assert "Source: [return-policy:body:0]" in context
    assert "Возврат брака оформляется актом." in context


def test_build_strict_messages_forbids_unsupported_answers():
    messages = build_strict_messages("Что делать?", [])

    rendered = "\n".join(message["content"] for message in messages)
    assert "Use only the provided context" in rendered
    assert "cite every factual claim" in rendered
    assert REFUSAL_TEXT in rendered


def test_validate_citations_rejects_unknown_ids_and_allows_refusal():
    docs = [Document(page_content="Known", metadata={"citation_id": "known"})]

    assert validate_citations("No indexed evidence was enough to answer.", docs) == (True, [])
    assert validate_citations("Fact. [known]", docs) == (True, [])
    assert validate_citations("Fact. [missing]", docs) == (False, ["missing"])


def test_refuse_message_mentions_question_and_refusal_text():
    message = refuse_message("Что делать с браком?")

    assert message == REFUSAL_TEXT
