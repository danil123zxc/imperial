from __future__ import annotations

import os
from dataclasses import replace

import pytest
from langchain_core.documents import Document

from imperial_rag.config import Settings
from imperial_rag.elasticsearch_keyword import ElasticsearchKeywordIndex, elasticsearch_health


@pytest.mark.skipif(
    os.environ.get("IMPERIAL_RAG_LIVE_ELASTICSEARCH") != "1",
    reason="live Elasticsearch test is opt-in",
)
def test_live_elasticsearch_keyword_index_roundtrip() -> None:
    settings = replace(Settings(), elasticsearch_index="imperial_keyword_chunks_test")
    index = ElasticsearchKeywordIndex(settings)

    assert settings.elasticsearch_url.startswith(("http://localhost", "http://127.0.0.1"))
    assert elasticsearch_health(settings) is True

    index.replace_all(
        [
            Document(page_content="Регламент возврата брака из магазина", metadata={"citation_id": "store"}),
            Document(page_content="Должностная инструкция водителя", metadata={"citation_id": "driver"}),
        ]
    )

    try:
        strict_results = index.search("возврат брака", k=5)
        relaxed_results = index.search("Как оформить возврат брака из магазина?", k=5)

        assert [result.metadata["citation_id"] for result in strict_results[:1]] == ["store"]
        assert [result.metadata["citation_id"] for result in relaxed_results[:1]] == ["store"]
        assert relaxed_results[0].metadata["_keyword_rank"] == 0
        assert isinstance(relaxed_results[0].metadata["_keyword_score"], float)
    finally:
        index.client.indices.delete(index=settings.elasticsearch_index, ignore_unavailable=True)
