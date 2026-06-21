from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_elasticsearch import ElasticsearchRetriever as LangChainElasticsearchRetriever

from imperial_rag.config import Settings
from imperial_rag.document_ids import metadata_or_content_id
from imperial_rag.retrieval.lexical import (
    KeywordHit,
    build_elasticsearch_token_query,
    content_keyword_query_tokens,
    normalize_search_text,
    relaxed_query_token_sets,
    searchable_document_text,
)


INDEX_MAPPINGS = {
    "properties": {
        "chunk_id": {"type": "keyword"},
        "text": {"type": "text", "index": False},
        "content_text": {"type": "text", "analyzer": "russian"},
        "file_name": {"type": "text", "analyzer": "russian"},
        "relative_path": {"type": "text", "analyzer": "russian"},
        "section_heading": {"type": "text", "analyzer": "russian"},
        "source_type": {"type": "text"},
        "sheet_name": {"type": "text"},
        "page_number_text": {"type": "text"},
        "normalized_text": {"type": "text", "analyzer": "russian"},
        "metadata": {"type": "object", "enabled": False},
    }
}
INDEX_SETTINGS = {"number_of_shards": 1, "number_of_replicas": 0}
_STRUCTURED_METADATA_SEARCH_FIELDS = (
    "file_name",
    "relative_path",
    "section_heading",
    "source_type",
    "sheet_name",
)


@dataclass(frozen=True)
class ElasticsearchRetrieverHit:
    document: Document
    score: float
    hit_id: str


# Elasticsearch transport fields (score, hit id) are carried back from the LangChain
# retriever in document metadata under reserved keys, then stripped so the public
# ``ElasticsearchRetrieverHit.document`` only exposes original citation metadata.
_HIT_SCORE_KEY = "__keyword_hit_score__"
_HIT_ID_KEY = "__keyword_hit_id__"


def _keyword_document_mapper(hit: Mapping[Any, Any]) -> Document:
    source = dict(hit.get("_source") or {})
    metadata = dict(source.get("metadata") or {})
    content = str(source.get("text", ""))
    score = float(hit.get("_score") or 0.0)
    hit_id = str(
        hit.get("_id")
        or source.get("chunk_id")
        or metadata.get("chunk_id")
        or metadata.get("citation_id")
        or content
    )
    return Document(page_content=content, metadata={**metadata, _HIT_SCORE_KEY: score, _HIT_ID_KEY: hit_id})


class ElasticsearchKeywordRetriever(BaseRetriever):
    client: Any
    index_name: str

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        limit: int = 5,
        **_: Any,
    ) -> list[Document]:
        return self._documents_from_hits(self.search(query, limit=limit))

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        limit: int = 5,
        **_: Any,
    ) -> list[Document]:
        hits = await asyncio.to_thread(self.search, query, limit=limit)
        return self._documents_from_hits(hits)

    def _documents_from_hits(self, hits: list[ElasticsearchRetrieverHit]) -> list[Document]:
        documents: list[Document] = []
        for rank, hit in enumerate(hits):
            metadata = dict(hit.document.metadata or {})
            metadata["_keyword_rank"] = rank
            metadata["_keyword_score"] = hit.score
            metadata["_retrieval_id"] = _retrieval_id(hit.document, hit_id=hit.hit_id)
            documents.append(Document(page_content=hit.document.page_content, metadata=metadata))
        return documents

    def search(self, query: str, limit: int = 5) -> list[ElasticsearchRetrieverHit]:
        tokens = content_keyword_query_tokens(query)
        if not tokens:
            return []
        return self.search_tokens(tokens, limit=limit)

    def search_tokens(self, tokens: list[str], limit: int) -> list[ElasticsearchRetrieverHit]:
        documents = self._langchain_retriever(tokens, limit).invoke("")
        return [self._hit_from_document(document) for document in documents]

    def _langchain_retriever(self, tokens: list[str], limit: int) -> LangChainElasticsearchRetriever:
        return LangChainElasticsearchRetriever(
            client=self.client,
            index_name=self.index_name,
            body_func=lambda _query: {"query": build_elasticsearch_token_query(tokens), "size": limit},
            document_mapper=_keyword_document_mapper,
        )

    def _hit_from_document(self, document: Document) -> ElasticsearchRetrieverHit:
        metadata = dict(document.metadata or {})
        score = float(metadata.pop(_HIT_SCORE_KEY, 0.0))
        hit_id = str(metadata.pop(_HIT_ID_KEY, document.page_content))
        return ElasticsearchRetrieverHit(
            document=Document(page_content=document.page_content, metadata=metadata),
            score=score,
            hit_id=hit_id,
        )


class ElasticsearchKeywordIndex:
    def __init__(
        self,
        settings: Settings,
        *,
        client: Any | None = None,
        bulk: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self.index_name = settings.elasticsearch_index
        if client is None:
            from elasticsearch import Elasticsearch

            client = Elasticsearch(settings.elasticsearch_url)
        if bulk is None:
            from elasticsearch.helpers import bulk as elasticsearch_bulk

            bulk = elasticsearch_bulk
        self.client = client
        self._bulk = bulk
        self.retriever = ElasticsearchKeywordRetriever(client=self.client, index_name=self.index_name)

    def clear(self) -> None:
        self.client.indices.delete(index=self.index_name, ignore_unavailable=True)
        self._create_index()

    def replace_all(self, documents: list[Document]) -> None:
        self.clear()
        if documents:
            self.index_documents(documents)

    def index_documents(self, documents: list[Document]) -> None:
        self._create_index()
        actions = list(self._actions(documents))
        if actions:
            self._bulk(self.client, actions, refresh=True)

    def search(self, query: str, limit: int = 5, k: int | None = None) -> list[Document]:
        return [hit.document for hit in self.search_with_scores(query, limit=limit, k=k)]

    def search_with_scores(self, query: str, limit: int = 5, k: int | None = None) -> list[KeywordHit]:
        resolved_limit = k if k is not None else limit
        query_tokens = content_keyword_query_tokens(query)
        if not query_tokens:
            return []
        if not self._index_exists():
            return []

        hits = self._search_tokens(query_tokens, resolved_limit)
        if not hits:
            hits = self._search_relaxed(query_tokens, resolved_limit)
        return [self._keyword_hit(hit, rank) for rank, hit in enumerate(hits[:resolved_limit])]

    def _create_index(self) -> None:
        if self._index_exists():
            return
        self.client.indices.create(index=self.index_name, mappings=INDEX_MAPPINGS, settings=INDEX_SETTINGS)

    def _index_exists(self) -> bool:
        return bool(self.client.indices.exists(index=self.index_name))

    def _actions(self, documents: list[Document]) -> Iterable[dict[str, Any]]:
        from imperial_rag.indexing import stable_chunk_id

        for document in documents:
            chunk_id = stable_chunk_id(document)
            yield {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": chunk_id,
                "_source": {
                    "chunk_id": chunk_id,
                    "text": document.page_content,
                    **_structured_search_fields(document),
                    "normalized_text": normalize_search_text(searchable_document_text(document)),
                    "metadata": dict(document.metadata or {}),
                },
            }

    def _search_tokens(self, tokens: list[str], limit: int) -> list[ElasticsearchRetrieverHit]:
        return self.retriever.search_tokens(tokens, limit=limit)

    def _search_relaxed(self, tokens: list[str], limit: int) -> list[ElasticsearchRetrieverHit]:
        seen: set[str] = set()
        ordered_hits: list[ElasticsearchRetrieverHit] = []
        for relaxed_tokens in relaxed_query_token_sets(tokens):
            for hit in self._search_tokens(relaxed_tokens, limit):
                if hit.hit_id in seen:
                    continue
                seen.add(hit.hit_id)
                ordered_hits.append(hit)
                if len(ordered_hits) >= limit:
                    return ordered_hits
        return ordered_hits

    def _keyword_hit(self, hit: ElasticsearchRetrieverHit, rank: int) -> KeywordHit:
        metadata = dict(hit.document.metadata or {})
        metadata["_keyword_rank"] = rank
        metadata["_keyword_score"] = hit.score
        metadata["_retrieval_id"] = _retrieval_id(hit.document, hit_id=hit.hit_id)
        return KeywordHit(
            document=Document(page_content=hit.document.page_content, metadata=metadata),
            score=hit.score,
        )


def _structured_search_fields(document: Document) -> dict[str, str]:
    metadata = document.metadata or {}
    fields = {"content_text": document.page_content}
    for field_name in _STRUCTURED_METADATA_SEARCH_FIELDS:
        value = metadata.get(field_name)
        if value is not None:
            fields[field_name] = str(value)
    page_number = metadata.get("page_number")
    if page_number is not None:
        fields["page_number_text"] = str(page_number)
    return fields


def _retrieval_id(document: Document, *, hit_id: str | None = None) -> str:
    metadata = document.metadata or {}
    return metadata_or_content_id(metadata.get("citation_id"), metadata.get("chunk_id"), hit_id, content=document.page_content)


def elasticsearch_health(settings: Settings, *, client: Any | None = None) -> bool:
    if client is None:
        from elasticsearch import Elasticsearch

        client = Elasticsearch(settings.elasticsearch_url)
    try:
        return bool(client.ping())
    except Exception:
        return False
