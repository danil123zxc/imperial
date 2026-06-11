from __future__ import annotations

from langchain_core.documents import Document

from imperial_rag.keyword import (
    build_elasticsearch_token_query,
    keyword_query_tokens,
    normalize_search_text,
    relaxed_query_token_sets,
    searchable_document_text,
)


def test_normalize_search_text_handles_case_hyphen_and_russian_suffixes() -> None:
    assert normalize_search_text("ВОДИТЕЛЬ-ЭКСПЕДИТОРА") == "водител экспедитор"


def test_keyword_query_tokens_remove_low_value_question_words() -> None:
    assert keyword_query_tokens("Как оформить возврат брака из магазина?") == [
        "оформит",
        "возврат",
        "брак",
        "магазин",
    ]


def test_relaxed_query_token_sets_are_bounded_and_include_tail_pairs() -> None:
    tokens = [f"термин{number}" for number in range(20)]

    relaxed = relaxed_query_token_sets(tokens)

    assert len(relaxed) <= 24
    assert ["термин18", "термин19"] in relaxed


def test_searchable_document_text_includes_metadata_fields() -> None:
    document = Document(
        page_content="Регламент возврата брака",
        metadata={
            "file_name": "policy.docx",
            "relative_path": "rules/policy.docx",
            "section_heading": "Возврат",
            "source_type": "body",
        },
    )

    assert searchable_document_text(document) == (
        "Регламент возврата брака policy.docx rules/policy.docx Возврат body"
    )


def test_build_elasticsearch_token_query_requires_all_tokens() -> None:
    assert build_elasticsearch_token_query(["возврат", "брак"]) == {
        "bool": {
            "must": [
                {"match": {"normalized_text": "возврат"}},
                {"match": {"normalized_text": "брак"}},
            ]
        }
    }
