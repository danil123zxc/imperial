from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from imperial_rag.config import Settings
import imperial_rag.elasticsearch_keyword as elasticsearch_keyword_module
from imperial_rag.elasticsearch_keyword import (
    ElasticsearchKeywordIndex,
    ElasticsearchKeywordRetriever,
    elasticsearch_health,
)
from imperial_rag.indexing import stable_chunk_id
from imperial_rag.keyword import (
    ELASTICSEARCH_REQUIRED_SEARCH_FIELDS,
    build_elasticsearch_token_query,
    relaxed_query_token_sets,
)


@dataclass
class FakeSettings:
    workspace_root: Path
    elasticsearch_url: str = "http://127.0.0.1:9200"
    elasticsearch_index: str = "test_keyword_chunks"


class FakeIndices:
    def __init__(self, client):
        self.client = client

    def exists(self, index):
        return index in self.client.existing_indices

    def create(self, index, mappings=None, settings=None):
        self.client.created.append({"index": index, "mappings": mappings, "settings": settings})
        self.client.existing_indices.add(index)

    def delete(self, index, ignore_unavailable=False):
        self.client.deleted.append({"index": index, "ignore_unavailable": ignore_unavailable})
        self.client.existing_indices.discard(index)


class FakeClient(Elasticsearch):
    # Subclass Elasticsearch so langchain_elasticsearch.ElasticsearchRetriever's
    # ``client: Elasticsearch`` validation passes, but skip the real network init.
    def __init__(self):
        self._headers = {}
        self.indices = FakeIndices(self)
        self.existing_indices = set()
        self.created = []
        self.deleted = []
        self.bulk_actions = []
        self.search_calls = []
        self.search_responses = []
        self.ping_result = True

    def options(self, **kwargs):
        return self

    def search(self, index, body):
        self.search_calls.append({"index": index, "body": body})
        return self.search_responses.pop(0)

    def ping(self):
        return self.ping_result


def fake_bulk(client, actions, refresh=False):
    client.bulk_actions.append({"actions": list(actions), "refresh": refresh})
    return (len(client.bulk_actions[-1]["actions"]), [])


def make_index(tmp_path: Path, client: FakeClient) -> ElasticsearchKeywordIndex:
    return ElasticsearchKeywordIndex(FakeSettings(tmp_path), client=client, bulk=fake_bulk)


def test_index_mappings_use_russian_analyzer_for_searchable_text_fields() -> None:
    mappings = elasticsearch_keyword_module.INDEX_MAPPINGS["properties"]

    for field in ("content_text", "file_name", "relative_path", "section_heading", "normalized_text"):
        assert mappings[field]["analyzer"] == "russian"


def mark_index_exists(client: FakeClient) -> None:
    client.existing_indices.add("test_keyword_chunks")


def assert_fuzzy_token_query_body(body: dict, tokens: list[str], size: int) -> None:
    assert body["size"] == size
    token_clauses = body["query"]["bool"]["must"]
    assert len(token_clauses) == len(tokens)
    for token, clause in zip(tokens, token_clauses, strict=True):
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


def test_keyword_retriever_is_langchain_retriever_and_preserves_scores() -> None:
    client = FakeClient()
    response = {
        "hits": {
            "hits": [
                {
                    "_id": "hit-1",
                    "_score": 4.25,
                    "_source": {
                        "text": "Регламент возврата брака",
                        "metadata": {
                            "citation_id": "return",
                            "file_name": "Регламент возврата брака.docx",
                        },
                    },
                }
            ]
        }
    }
    client.search_responses.extend([response, response])
    retriever = ElasticsearchKeywordRetriever(client=client, index_name="test_keyword_chunks")

    scored_hits = retriever.search_tokens(["возврат", "брак"], limit=5)
    invoked_docs = retriever.invoke("возврат брака", limit=5)

    assert isinstance(retriever, BaseRetriever)
    assert len(scored_hits) == 1
    assert scored_hits[0].hit_id == "hit-1"
    assert scored_hits[0].score == 4.25
    assert scored_hits[0].document.page_content == "Регламент возврата брака"
    assert scored_hits[0].document.metadata == {
        "citation_id": "return",
        "file_name": "Регламент возврата брака.docx",
    }
    assert [doc.metadata["citation_id"] for doc in invoked_docs] == ["return"]
    assert invoked_docs[0].metadata["_keyword_rank"] == 0
    assert invoked_docs[0].metadata["_keyword_score"] == 4.25
    assert invoked_docs[0].metadata["_retrieval_id"] == "return"
    assert client.search_calls == [
        {
            "index": "test_keyword_chunks",
            "body": {"query": build_elasticsearch_token_query(["возврат", "брак"]), "size": 5},
        },
        {
            "index": "test_keyword_chunks",
            "body": {"query": build_elasticsearch_token_query(["возврат", "брак"]), "size": 5},
        },
    ]
    assert_fuzzy_token_query_body(client.search_calls[0]["body"], ["возврат", "брак"], size=5)


def test_keyword_index_facade_uses_retriever_for_query_time_search(tmp_path: Path) -> None:
    client = FakeClient()
    mark_index_exists(client)
    client.search_responses.append(
        {
            "hits": {
                "hits": [
                    {
                        "_id": "hit-1",
                        "_score": 8.0,
                        "_source": {
                            "text": "Регламент возврата брака",
                            "metadata": {"citation_id": "return"},
                        },
                    }
                ]
            }
        }
    )
    index = make_index(tmp_path, client)

    hits = index.search_with_scores("возврат брака", limit=5)

    assert isinstance(index.retriever, ElasticsearchKeywordRetriever)
    assert [hit.document.metadata["citation_id"] for hit in hits] == ["return"]
    assert hits[0].score == 8.0
    assert hits[0].document.metadata["_keyword_rank"] == 0
    assert hits[0].document.metadata["_keyword_score"] == 8.0


def test_keyword_retriever_async_invoke_accepts_limit() -> None:
    client = FakeClient()
    client.search_responses.append(
        {
            "hits": {
                "hits": [
                    {
                        "_id": "hit-1",
                        "_score": 4.25,
                        "_source": {
                            "text": "Регламент возврата брака",
                            "metadata": {"citation_id": "return"},
                        },
                    }
                ]
            }
        }
    )
    retriever = ElasticsearchKeywordRetriever(client=client, index_name="test_keyword_chunks")

    async def invoke_retriever() -> list[Document]:
        return await retriever.ainvoke("возврат брака", limit=5)

    invoked_docs = asyncio.run(invoke_retriever())

    assert [doc.metadata["citation_id"] for doc in invoked_docs] == ["return"]
    assert invoked_docs[0].metadata["_keyword_rank"] == 0
    assert invoked_docs[0].metadata["_keyword_score"] == 4.25
    assert invoked_docs[0].metadata["_retrieval_id"] == "return"
    assert client.search_calls == [
        {
            "index": "test_keyword_chunks",
            "body": {"query": build_elasticsearch_token_query(["возврат", "брак"]), "size": 5},
        }
    ]
    assert_fuzzy_token_query_body(client.search_calls[0]["body"], ["возврат", "брак"], size=5)


def test_elasticsearch_retrieval_id_hashes_content_when_ids_and_hit_id_are_missing() -> None:
    document = Document(page_content="private keyword text", metadata={})
    expected = f"content_sha256:{hashlib.sha256(b'private keyword text').hexdigest()[:12]}"

    assert elasticsearch_keyword_module._retrieval_id(document) == expected
    assert "private keyword text" not in elasticsearch_keyword_module._retrieval_id(document)


def test_keyword_retriever_async_invocation_offloads_sync_search() -> None:
    import time

    class SlowClient(FakeClient):
        def search(self, index, body):
            self.search_calls.append({"index": index, "body": body})
            time.sleep(0.2)
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": f"hit-{len(self.search_calls)}",
                            "_score": 1.0,
                            "_source": {"text": "Регламент", "metadata": {"citation_id": "return"}},
                        }
                    ]
                }
            }

    async def run_two_calls() -> tuple[float, list[list[Document]]]:
        retriever = ElasticsearchKeywordRetriever(client=SlowClient(), index_name="test_keyword_chunks")
        started = time.perf_counter()
        docs = await asyncio.gather(
            retriever.ainvoke("возврат", limit=5),
            retriever.ainvoke("возврат", limit=5),
        )
        return time.perf_counter() - started, docs

    elapsed, docs = asyncio.run(run_two_calls())

    assert elapsed < 0.35
    assert [[doc.metadata["citation_id"] for doc in call_docs] for call_docs in docs] == [["return"], ["return"]]


@pytest.mark.parametrize(
    ("hit", "expected_hit_id"),
    [
        (
            {
                "_source": {
                    "chunk_id": "source-chunk",
                    "text": "Регламент возврата брака",
                    "metadata": {"chunk_id": "metadata-chunk", "citation_id": "return"},
                }
            },
            "source-chunk",
        ),
        (
            {
                "_source": {
                    "text": "Регламент возврата брака",
                    "metadata": {"chunk_id": "metadata-chunk", "citation_id": "return"},
                }
            },
            "metadata-chunk",
        ),
        (
            {
                "_source": {
                    "text": "Регламент возврата брака",
                    "metadata": {"citation_id": "return"},
                }
            },
            "return",
        ),
        (
            {
                "_source": {
                    "text": "Регламент возврата брака",
                    "metadata": {},
                }
            },
            "Регламент возврата брака",
        ),
    ],
)
def test_keyword_retriever_hit_id_fallback_order(hit: dict, expected_hit_id: str) -> None:
    client = FakeClient()
    client.search_responses.append({"hits": {"hits": [hit]}})
    retriever = ElasticsearchKeywordRetriever(client=client, index_name="test_keyword_chunks")

    hits = retriever.search_tokens(["возврат"], limit=1)

    assert hits[0].hit_id == expected_hit_id


def test_replace_all_recreates_index_and_bulk_indexes_documents(tmp_path: Path) -> None:
    client = FakeClient()
    client.existing_indices.add("test_keyword_chunks")
    index = make_index(tmp_path, client)
    docs = [
        Document(
            page_content="Регламент возврата брака",
            metadata={
                "citation_id": "a",
                "file_name": "Регламент возврата брака.docx",
                "relative_path": "rules/Регламент возврата брака.docx",
                "section_heading": "Возврат брака",
                "source_type": "body",
                "sheet_name": "График",
                "page_number": 2,
            },
        ),
        Document(page_content="Должностная инструкция водителя", metadata={"citation_id": "b"}),
    ]

    index.replace_all(docs)

    assert client.deleted == [{"index": "test_keyword_chunks", "ignore_unavailable": True}]
    assert len(client.created) == 1
    actions = client.bulk_actions[0]["actions"]
    assert client.bulk_actions[0]["refresh"] is True
    assert [action["_id"] for action in actions] == [stable_chunk_id(doc) for doc in docs]
    assert actions[0]["_index"] == "test_keyword_chunks"
    assert actions[0]["_source"]["text"] == "Регламент возврата брака"
    assert actions[0]["_source"]["content_text"] == "Регламент возврата брака"
    assert actions[0]["_source"]["file_name"] == "Регламент возврата брака.docx"
    assert actions[0]["_source"]["relative_path"] == "rules/Регламент возврата брака.docx"
    assert actions[0]["_source"]["section_heading"] == "Возврат брака"
    assert actions[0]["_source"]["source_type"] == "body"
    assert actions[0]["_source"]["sheet_name"] == "График"
    assert actions[0]["_source"]["page_number_text"] == "2"
    assert "регламент возврат брак" in actions[0]["_source"]["normalized_text"]
    assert actions[0]["_source"]["metadata"]["citation_id"] == "a"
    assert actions[0]["_source"]["metadata"]["page_number"] == 2


def test_replace_all_with_no_documents_still_clears_stale_index(tmp_path: Path) -> None:
    client = FakeClient()
    index = make_index(tmp_path, client)

    index.replace_all([])

    assert client.deleted == [{"index": "test_keyword_chunks", "ignore_unavailable": True}]
    assert len(client.created) == 1
    assert client.bulk_actions == []


def test_search_with_scores_uses_all_tokens_query_and_maps_hits(tmp_path: Path) -> None:
    client = FakeClient()
    mark_index_exists(client)
    client.search_responses.append(
        {
            "hits": {
                "hits": [
                    {
                        "_score": 3.5,
                        "_source": {
                            "text": "Регламент возврата брака",
                            "metadata": {"citation_id": "a"},
                        },
                    }
                ]
            }
        }
    )
    index = make_index(tmp_path, client)

    hits = index.search_with_scores("возврат брака", limit=5)

    assert client.search_calls == [
        {
            "index": "test_keyword_chunks",
            "body": {"query": build_elasticsearch_token_query(["возврат", "брак"]), "size": 5},
        }
    ]
    assert [hit.document.metadata["citation_id"] for hit in hits] == ["a"]
    assert hits[0].document.metadata["_keyword_rank"] == 0
    assert hits[0].document.metadata["_keyword_score"] == 3.5
    assert hits[0].score == 3.5


def test_search_with_scores_returns_empty_when_index_is_missing(tmp_path: Path) -> None:
    client = FakeClient()
    index = make_index(tmp_path, client)

    assert index.search_with_scores("возврат брака", limit=5) == []
    assert client.search_calls == []


def test_search_uses_relaxed_queries_when_strict_search_misses(tmp_path: Path) -> None:
    client = FakeClient()
    mark_index_exists(client)
    client.search_responses.extend(
        [
            {"hits": {"hits": []}},
            {
                "hits": {
                    "hits": [
                        {
                            "_score": 2.0,
                            "_source": {
                                "text": "Регламент возврата брака из магазина",
                                "metadata": {"citation_id": "store"},
                            },
                        }
                    ]
                }
            },
        ]
    )
    index = make_index(tmp_path, client)

    results = index.search("Как оформить возврат брака из магазина?", k=1)

    assert [result.metadata["citation_id"] for result in results] == ["store"]
    assert len(client.search_calls) == 2
    assert_fuzzy_token_query_body(
        client.search_calls[0]["body"],
        ["оформит", "возврат", "брак", "магазин"],
        size=1,
    )
    assert (
        client.search_calls[1]["body"]["query"]["bool"]["must"]
        != client.search_calls[0]["body"]["query"]["bool"]["must"]
    )


def test_search_with_scores_uses_two_token_relaxed_fallback_with_match_mode(tmp_path: Path) -> None:
    client = FakeClient()
    mark_index_exists(client)
    client.search_responses.extend(
        [
            {"hits": {"hits": []}},
            {
                "hits": {
                    "hits": [
                        {
                            "_id": "price-policy",
                            "_score": 29.442,
                            "_source": {
                                "text": "Регламент по ценоизменению",
                                "metadata": {"citation_id": "price"},
                            },
                        }
                    ]
                }
            },
        ]
    )
    index = make_index(tmp_path, client)

    hits = index.search_with_scores("Как регулируется ценоизменение?", limit=1)

    assert relaxed_query_token_sets(["регулируетс", "ценоизменен"]) == [
        ["ценоизменен"],
        ["регулируетс"],
    ]
    assert [hit.document.metadata["citation_id"] for hit in hits] == ["price"]
    assert hits[0].score == 29.442
    assert hits[0].document.metadata["_keyword_rank"] == 0
    assert hits[0].document.metadata["_keyword_score"] == 29.442
    assert hits[0].document.metadata["_retrieval_id"] == "price"
    assert hits[0].document.metadata["_keyword_match_mode"] == "relaxed_drop_one"
    assert len(client.search_calls) == 2
    assert_fuzzy_token_query_body(client.search_calls[0]["body"], ["регулируетс", "ценоизменен"], size=1)
    assert_fuzzy_token_query_body(client.search_calls[1]["body"], ["ценоизменен"], size=1)


def test_search_with_scores_marks_strict_keyword_hits(tmp_path: Path) -> None:
    client = FakeClient()
    mark_index_exists(client)
    client.search_responses.append(
        {
            "hits": {
                "hits": [
                    {
                        "_score": 29.442,
                        "_source": {
                            "text": "Регламент по ценоизменению",
                            "metadata": {"citation_id": "price"},
                        },
                    }
                ]
            }
        }
    )
    index = make_index(tmp_path, client)

    hits = index.search_with_scores("ценоизменение", limit=5)

    assert [hit.document.metadata["citation_id"] for hit in hits] == ["price"]
    assert hits[0].document.metadata["_keyword_match_mode"] == "strict"


def test_search_with_scores_preserves_elasticsearch_metadata_boost_order(tmp_path: Path) -> None:
    client = FakeClient()
    mark_index_exists(client)
    client.search_responses.append(
        {
            "hits": {
                "hits": [
                    {
                        "_score": 9.0,
                        "_source": {
                            "text": "Общие условия",
                            "metadata": {
                                "citation_id": "filename",
                                "file_name": "Регламент возврата брака.docx",
                            },
                        },
                    },
                    {
                        "_score": 3.0,
                        "_source": {
                            "text": "Возврат брака описан в теле документа",
                            "metadata": {"citation_id": "body"},
                        },
                    },
                ]
            }
        }
    )
    index = make_index(tmp_path, client)

    hits = index.search_with_scores("возврат брака", limit=5)

    assert [hit.document.metadata["citation_id"] for hit in hits] == ["filename", "body"]
    assert hits[0].document.metadata["_keyword_rank"] == 0
    assert hits[0].document.metadata["_keyword_score"] == 9.0
    assert hits[1].document.metadata["_keyword_rank"] == 1


def test_search_relaxed_queries_continue_until_limit_is_filled(tmp_path: Path) -> None:
    client = FakeClient()
    mark_index_exists(client)
    client.search_responses.extend(
        [
            {"hits": {"hits": []}},
            {
                "hits": {
                    "hits": [
                        {
                            "_id": "first",
                            "_score": 2.0,
                            "_source": {
                                "text": "Регламент возврата брака",
                                "metadata": {"citation_id": "first"},
                            },
                        }
                    ]
                }
            },
            {
                "hits": {
                    "hits": [
                        {
                            "_id": "second",
                            "_score": 1.5,
                            "_source": {
                                "text": "Возврат брака из магазина",
                                "metadata": {"citation_id": "second"},
                            },
                        }
                    ]
                }
            },
        ]
    )
    index = make_index(tmp_path, client)

    results = index.search("оформить возврат брака магазин", k=2)

    assert [result.metadata["citation_id"] for result in results] == ["first", "second"]
    assert len(client.search_calls) == 3


def test_search_returns_empty_for_stopword_only_empty_query(tmp_path: Path) -> None:
    client = FakeClient()
    index = make_index(tmp_path, client)

    assert index.search("как что где", limit=5) == []
    assert client.search_calls == []


def test_elasticsearch_health_returns_false_when_ping_fails(tmp_path: Path) -> None:
    client = FakeClient()
    client.ping_result = False

    assert elasticsearch_health(Settings(workspace_root=tmp_path), client=client) is False


def test_elasticsearch_health_returns_false_when_ping_raises(tmp_path: Path) -> None:
    class BrokenClient(FakeClient):
        def ping(self):
            raise RuntimeError("offline")

    assert elasticsearch_health(Settings(workspace_root=tmp_path), client=BrokenClient()) is False
