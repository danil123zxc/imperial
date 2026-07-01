from __future__ import annotations

from langchain_core.documents import Document

from imperial_rag.retrieval.lexical import (
    ELASTICSEARCH_BOOSTED_SEARCH_FIELDS,
    ELASTICSEARCH_REQUIRED_SEARCH_FIELDS,
    build_elasticsearch_token_query,
    keyword_query_tokens,
    normalize_search_text,
    relaxed_query_token_sets,
    searchable_document_text,
)


def expected_token_clause(token: str) -> dict:
    return {
        "bool": {
            "should": [
                {
                    "multi_match": {
                        "query": token,
                        "fields": ELASTICSEARCH_REQUIRED_SEARCH_FIELDS,
                    }
                },
                {
                    "multi_match": {
                        "query": token,
                        "fields": ELASTICSEARCH_REQUIRED_SEARCH_FIELDS,
                        "fuzziness": "AUTO",
                        "prefix_length": 1,
                        "max_expansions": 25,
                        "fuzzy_transpositions": True,
                    }
                },
            ],
            "minimum_should_match": 1,
        }
    }


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


def test_relaxed_query_token_sets_keeps_one_token_queries_strict() -> None:
    assert relaxed_query_token_sets(["ценоизменен"]) == []


def test_relaxed_query_token_sets_recovers_two_token_domain_queries() -> None:
    tokens = keyword_query_tokens("Как регулируется ценоизменение?")

    assert tokens == ["регулируетс", "ценоизменен"]
    assert relaxed_query_token_sets(tokens) == [["ценоизменен"], ["регулируетс"]]


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
    query = build_elasticsearch_token_query(["возврт", "брка"])

    assert query == {
        "bool": {
            "must": [
                expected_token_clause("возврт"),
                expected_token_clause("брка"),
            ],
            "should": [
                {
                    "multi_match": {
                        "query": "возврт брка",
                        "fields": ELASTICSEARCH_BOOSTED_SEARCH_FIELDS,
                    }
                }
            ],
        }
    }


def test_elasticsearch_token_query_adds_exact_and_fuzzy_alternatives_per_token() -> None:
    query = build_elasticsearch_token_query(["return", "defect"])

    token_clauses = query["bool"]["must"]
    assert len(token_clauses) == 2
    for token, clause in zip(["return", "defect"], token_clauses, strict=True):
        alternatives = clause["bool"]["should"]
        assert clause["bool"]["minimum_should_match"] == 1
        assert alternatives[0] == {
            "multi_match": {
                "query": token,
                "fields": ELASTICSEARCH_REQUIRED_SEARCH_FIELDS,
            }
        }
        assert alternatives[1] == {
            "multi_match": {
                "query": token,
                "fields": ELASTICSEARCH_REQUIRED_SEARCH_FIELDS,
                "fuzziness": "AUTO",
                "prefix_length": 1,
                "max_expansions": 25,
                "fuzzy_transpositions": True,
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
