from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from imperial_rag.config import Settings
from imperial_rag.elasticsearch_keyword import ElasticsearchKeywordIndex, elasticsearch_health
from imperial_rag.indexing import stable_chunk_id


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


class FakeClient:
    def __init__(self):
        self.indices = FakeIndices(self)
        self.existing_indices = set()
        self.created = []
        self.deleted = []
        self.bulk_actions = []
        self.search_calls = []
        self.search_responses = []
        self.ping_result = True

    def search(self, index, query, size):
        self.search_calls.append({"index": index, "query": query, "size": size})
        return self.search_responses.pop(0)

    def ping(self):
        return self.ping_result


def fake_bulk(client, actions, refresh=False):
    client.bulk_actions.append({"actions": list(actions), "refresh": refresh})
    return (len(client.bulk_actions[-1]["actions"]), [])


def make_index(tmp_path: Path, client: FakeClient) -> ElasticsearchKeywordIndex:
    return ElasticsearchKeywordIndex(FakeSettings(tmp_path), client=client, bulk=fake_bulk)


def test_replace_all_recreates_index_and_bulk_indexes_documents(tmp_path: Path) -> None:
    client = FakeClient()
    client.existing_indices.add("test_keyword_chunks")
    index = make_index(tmp_path, client)
    docs = [
        Document(page_content="Регламент возврата брака", metadata={"citation_id": "a"}),
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
    assert "регламент возврат брак" in actions[0]["_source"]["normalized_text"]
    assert actions[0]["_source"]["metadata"] == {"citation_id": "a"}


def test_replace_all_with_no_documents_still_clears_stale_index(tmp_path: Path) -> None:
    client = FakeClient()
    index = make_index(tmp_path, client)

    index.replace_all([])

    assert client.deleted == [{"index": "test_keyword_chunks", "ignore_unavailable": True}]
    assert len(client.created) == 1
    assert client.bulk_actions == []


def test_search_with_scores_uses_all_tokens_query_and_maps_hits(tmp_path: Path) -> None:
    client = FakeClient()
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
            "query": {
                "bool": {
                    "must": [
                        {"match": {"normalized_text": "возврат"}},
                        {"match": {"normalized_text": "брак"}},
                    ]
                }
            },
            "size": 5,
        }
    ]
    assert [hit.document.metadata["citation_id"] for hit in hits] == ["a"]
    assert hits[0].document.metadata["_keyword_rank"] == 0
    assert hits[0].document.metadata["_keyword_score"] == 3.5
    assert hits[0].score == 3.5


def test_search_uses_relaxed_queries_when_strict_search_misses(tmp_path: Path) -> None:
    client = FakeClient()
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
    assert client.search_calls[1]["query"]["bool"]["must"] != client.search_calls[0]["query"]["bool"]["must"]


def test_search_relaxed_queries_continue_until_limit_is_filled(tmp_path: Path) -> None:
    client = FakeClient()
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
