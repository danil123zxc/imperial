from __future__ import annotations

from langchain_core.documents import Document

from imperial_rag.keyword import (
    ELASTICSEARCH_BOOSTED_SEARCH_FIELDS,
    ELASTICSEARCH_REQUIRED_SEARCH_FIELDS,
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


def test_keyword_query_tokens_keep_short_numeric_page_references() -> None:
    assert keyword_query_tokens("2") == ["2"]


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
                {
                    "multi_match": {
                        "query": "возврат",
                        "fields": ELASTICSEARCH_REQUIRED_SEARCH_FIELDS,
                    }
                },
                {
                    "multi_match": {
                        "query": "брак",
                        "fields": ELASTICSEARCH_REQUIRED_SEARCH_FIELDS,
                    }
                },
            ],
            "should": [
                {
                    "multi_match": {
                        "query": "возврат брак",
                        "fields": ELASTICSEARCH_BOOSTED_SEARCH_FIELDS,
                    }
                }
            ],
        }
    }


def test_boosted_elasticsearch_fields_prioritize_metadata_over_body_text() -> None:
    assert ELASTICSEARCH_BOOSTED_SEARCH_FIELDS == [
        "file_name^6",
        "section_heading^5",
        "relative_path^4",
        "sheet_name^3",
        "source_type^2",
        "content_text^1.5",
        "normalized_text^1",
    ]
    assert "page_number_text" in ELASTICSEARCH_REQUIRED_SEARCH_FIELDS
