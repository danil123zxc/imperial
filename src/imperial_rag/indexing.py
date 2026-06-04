from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from imperial_rag.config import Settings
from imperial_rag.providers import (
    QwenProviderSettings,
    create_embeddings,
    ensure_vector_metadata_compatible,
    write_vector_metadata,
)


_ENDING_RE = re.compile(r"(иями|ями|ами|ого|его|ому|ему|ыми|ими|ов|ев|ей|ый|ий|ой|ая|яя|ое|ее|ам|ям|ах|ях|ом|ем|а|я|ы|и|у|ю|е|о|ь)$")
_QDRANT_ID_NAMESPACE = uuid.UUID("2f931f90-f82a-4ef6-8a49-310e6c4bd8d7")
_CITATION_METADATA_KEYS = (
    "citation_id",
    "chunk_id",
    "file_id",
    "relative_path",
    "file_path",
    "file_name",
    "source_type",
    "section_heading",
    "page_number",
    "chunk_index",
    "start_index",
)
_QUERY_STOPWORDS = frozenset(
    {
        "а",
        "без",
        "в",
        "во",
        "где",
        "для",
        "до",
        "есть",
        "если",
        "имеет",
        "из",
        "или",
        "и",
        "как",
        "каки",
        "каку",
        "каков",
        "когда",
        "кто",
        "к",
        "ко",
        "ли",
        "на",
        "не",
        "но",
        "об",
        "о",
        "от",
        "по",
        "почему",
        "при",
        "про",
        "найт",
        "с",
        "со",
        "что",
    }
)
_MAX_RELAXED_QUERY_ATTEMPTS = 24
_MAX_ONE_DROP_RELAXATION_TOKENS = 8


def _stem_token(token: str) -> str:
    token = token.casefold().replace("ё", "е")
    while len(token) > 4:
        shortened = _ENDING_RE.sub("", token)
        if shortened == token:
            break
        token = shortened
    return token


def normalize_search_text(text: str) -> str:
    return " ".join(_stem_token(token) for token in re.findall(r"\w+", text.casefold().replace("-", " "), flags=re.UNICODE))


def build_fts_match_query(query: str) -> str:
    return _build_fts_match_query(_keyword_query_tokens(query), operator="AND")


def _keyword_query_tokens(query: str) -> list[str]:
    tokens = [token for token in normalize_search_text(query).split() if token]
    content_tokens = [token for token in tokens if token not in _QUERY_STOPWORDS and len(token) > 2]
    return content_tokens or tokens


def _build_fts_match_query(tokens: list[str], operator: str) -> str:
    return f" {operator} ".join(f'"{token}"' for token in tokens)


def _relaxed_query_token_sets(tokens: list[str]) -> list[list[str]]:
    if len(tokens) < 3:
        return []
    relaxed: list[list[str]] = []

    if len(tokens) <= _MAX_ONE_DROP_RELAXATION_TOKENS:
        for drop_index in range(len(tokens)):
            relaxed.append([token for index, token in enumerate(tokens) if index != drop_index])
            if len(relaxed) >= _MAX_RELAXED_QUERY_ATTEMPTS:
                return relaxed

    for pair in _bounded_adjacent_pairs(tokens, _MAX_RELAXED_QUERY_ATTEMPTS - len(relaxed)):
        if pair not in relaxed:
            relaxed.append(pair)
        if len(relaxed) >= _MAX_RELAXED_QUERY_ATTEMPTS:
            return relaxed
    return relaxed


def _bounded_adjacent_pairs(tokens: list[str], budget: int) -> list[list[str]]:
    if budget <= 0:
        return []
    pairs = [tokens[index : index + 2] for index in range(len(tokens) - 1)]
    if len(pairs) <= budget:
        return pairs
    head_count = budget // 2
    tail_count = budget - head_count
    return pairs[:head_count] + pairs[-tail_count:]


def _relaxed_candidate_key(candidate: tuple[int, int, int, Any]) -> tuple[int, int, int]:
    matched_token_count, query_order, row_order, _row = candidate
    return (-matched_token_count, query_order, row_order)


@dataclass(frozen=True)
class KeywordHit:
    document: Document
    score: float


class KeywordIndex:
    def __init__(self, db_path: Path | Settings) -> None:
        self.db_path = db_path.keyword_db_path if hasattr(db_path, "keyword_db_path") else Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._uses_fts = self._fts5_available()
        self._create_schema()

    def clear(self) -> None:
        with self._conn:
            table = "chunks_fts" if self._uses_fts else "chunks"
            self._conn.execute(f"DELETE FROM {table}")

    def replace_all(self, documents: list[Document]) -> None:
        self.clear()
        self.index_documents(documents)

    def index_documents(self, documents: list[Document]) -> None:
        table = "chunks_fts" if self._uses_fts else "chunks"
        rows = [
            (
                stable_chunk_id(document),
                document.page_content,
                normalize_search_text(_searchable_document_text(document)),
                json.dumps(document.metadata, ensure_ascii=False),
            )
            for document in documents
        ]
        with self._conn:
            if self._uses_fts:
                self._conn.executemany(f"DELETE FROM {table} WHERE chunk_id = ?", [(row[0],) for row in rows])
                insert_verb = "INSERT INTO"
            else:
                insert_verb = "REPLACE INTO"
            self._conn.executemany(
                f"{insert_verb} {table}(chunk_id, text, normalized_text, metadata) VALUES (?, ?, ?, ?)",
                rows,
            )

    def search(self, query: str, limit: int = 5, k: int | None = None) -> list[Document]:
        return [hit.document for hit in self.search_with_scores(query, limit=limit, k=k)]

    def search_with_scores(self, query: str, limit: int = 5, k: int | None = None) -> list[KeywordHit]:
        resolved_limit = k if k is not None else limit
        query_tokens = _keyword_query_tokens(query)
        if not query_tokens:
            return []
        if self._uses_fts:
            rows = self._search_fts(query_tokens, resolved_limit, operator="AND")
            if not rows:
                rows = self._search_relaxed_fts(query_tokens, resolved_limit)
            return [
                KeywordHit(
                    document=Document(
                        page_content=text,
                        metadata={**json.loads(metadata_json), "_keyword_rank": rank, "_keyword_score": float(score)},
                    ),
                    score=float(score),
                )
                for rank, (text, metadata_json, score) in enumerate(rows)
            ]

        rows = self._search_like(query_tokens, resolved_limit, operator="AND")
        if not rows:
            rows = self._search_relaxed_like(query_tokens, resolved_limit)
        return [
            KeywordHit(
                document=Document(
                    page_content=text,
                    metadata={**json.loads(metadata_json), "_keyword_rank": rank, "_keyword_score": float(rank)},
                ),
                score=float(rank),
            )
            for rank, (text, metadata_json) in enumerate(rows)
        ]

    def _search_fts(self, query_tokens: list[str], limit: int, operator: str) -> list[tuple[str, str, float]]:
        match_query = _build_fts_match_query(query_tokens, operator=operator)
        return self._conn.execute(
            """
            SELECT text, metadata, bm25(chunks_fts) AS score
            FROM chunks_fts
            WHERE normalized_text MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (match_query, limit),
        ).fetchall()

    def _search_relaxed_fts(self, query_tokens: list[str], limit: int) -> list[tuple[str, str, float]]:
        candidates: dict[tuple[str, str], tuple[int, int, int, tuple[str, str, float]]] = {}
        for query_order, relaxed_tokens in enumerate(_relaxed_query_token_sets(query_tokens)):
            for row_order, row in enumerate(self._search_fts(relaxed_tokens, limit, operator="AND")):
                key = (row[0], row[1])
                candidate = (len(relaxed_tokens), query_order, row_order, row)
                previous = candidates.get(key)
                if previous is None or _relaxed_candidate_key(candidate) < _relaxed_candidate_key(previous):
                    candidates[key] = candidate
        return [
            candidate[-1]
            for candidate in sorted(candidates.values(), key=_relaxed_candidate_key)[:limit]
        ]

    def _search_like(self, query_tokens: list[str], limit: int, operator: str) -> list[tuple[str, str]]:
        where_clause = f" {operator} ".join("normalized_text LIKE ?" for _ in query_tokens)
        return self._conn.execute(
            f"SELECT text, metadata FROM chunks WHERE {where_clause} ORDER BY chunk_id LIMIT ?",
            [f"%{token}%" for token in query_tokens] + [limit],
        ).fetchall()

    def _search_relaxed_like(self, query_tokens: list[str], limit: int) -> list[tuple[str, str]]:
        candidates: dict[tuple[str, str], tuple[int, int, int, tuple[str, str]]] = {}
        for query_order, relaxed_tokens in enumerate(_relaxed_query_token_sets(query_tokens)):
            for row_order, row in enumerate(self._search_like(relaxed_tokens, limit, operator="AND")):
                key = (row[0], row[1])
                candidate = (len(relaxed_tokens), query_order, row_order, row)
                previous = candidates.get(key)
                if previous is None or _relaxed_candidate_key(candidate) < _relaxed_candidate_key(previous):
                    candidates[key] = candidate
        return [
            candidate[-1]
            for candidate in sorted(candidates.values(), key=_relaxed_candidate_key)[:limit]
        ]

    def _create_schema(self) -> None:
        if self._uses_fts:
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(
                    chunk_id UNINDEXED,
                    text UNINDEXED,
                    normalized_text,
                    metadata UNINDEXED
                )
                """
            )
            return
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
            """
        )

    def _fts5_available(self) -> bool:
        try:
            self._conn.execute("CREATE VIRTUAL TABLE temp.fts5_probe USING fts5(text)")
            self._conn.execute("DROP TABLE temp.fts5_probe")
        except sqlite3.OperationalError:
            return False
        return True


def _searchable_document_text(document: Document) -> str:
    metadata = document.metadata
    return " ".join(
        [
            document.page_content,
            str(metadata.get("file_name", "")),
            str(metadata.get("relative_path", "")),
            str(metadata.get("section_heading", "")),
            str(metadata.get("source_type", "")),
        ]
    )


def stable_chunk_id(document: Document) -> str:
    metadata = document.metadata or {}
    citation_metadata = {
        key: metadata[key]
        for key in _CITATION_METADATA_KEYS
        if key in metadata and metadata[key] is not None
    }
    if not citation_metadata:
        citation_metadata = {"metadata_sha1": hashlib.sha1(_json_dumps(metadata).encode("utf-8")).hexdigest()}
    payload = {
        "citation": citation_metadata,
        "content_sha256": hashlib.sha256(document.page_content.encode("utf-8")).hexdigest(),
    }
    return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, _json_dumps(payload)))


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def create_qdrant_vector_store(settings: Settings, embeddings: object | None = None) -> QdrantVectorStore:
    if embeddings is None:
        ensure_vector_metadata_compatible(settings)
        embeddings = create_embeddings()
    client = QdrantClient(url=settings.qdrant_url)
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )


def embedding_model_identifier(provider_settings: QwenProviderSettings | None = None) -> str:
    resolved = provider_settings or QwenProviderSettings.from_env()
    dimensions = resolved.embedding_dimensions
    return resolved.embedding_model if dimensions is None else f"{resolved.embedding_model}:{dimensions}"


def make_qdrant_store(qdrant_url: str, collection_name: str, embeddings: object | None = None) -> QdrantVectorStore:
    settings = Settings()
    return create_qdrant_vector_store(
        Settings(
            workspace_root=settings.workspace_root,
            qdrant_url=qdrant_url,
            qdrant_collection=collection_name,
            phoenix_project_name=settings.phoenix_project_name,
            phoenix_collector_endpoint=settings.phoenix_collector_endpoint,
            phoenix_client_endpoint=settings.phoenix_client_endpoint,
        ),
        embeddings=embeddings,
    )


def stable_chunk_ids(documents: Sequence[Document]) -> list[str]:
    return [stable_chunk_id(document) for document in documents]


def index_vector_documents(
    chunks: Sequence[Document],
    *,
    settings: Settings | None = None,
    vector_store: object | None = None,
    embeddings: object | None = None,
    ids: Sequence[str] | None = None,
) -> list[str]:
    documents = list(chunks)
    resolved_ids = list(ids) if ids is not None else stable_chunk_ids(documents)
    if len(resolved_ids) != len(documents):
        raise ValueError("ids length must match chunks length")
    if vector_store is None:
        if settings is None:
            raise ValueError("settings is required when vector_store is not provided")
        vector_store = create_qdrant_vector_store(settings, embeddings=embeddings)
    added_ids = list(vector_store.add_documents(documents=documents, ids=resolved_ids))
    if settings is not None:
        write_vector_metadata(settings, QwenProviderSettings.from_env().vector_metadata())
    return added_ids


def index_documents(vector_store: object, documents: list[Document], ids: list[str] | None = None) -> list[str]:
    return index_vector_documents(documents, vector_store=vector_store, ids=ids)


def qdrant_is_healthy(qdrant_url: str) -> bool:
    client = QdrantClient(url=qdrant_url)
    try:
        client.get_collections()
    except Exception:
        return False
    return True


def qdrant_health(settings: Settings) -> bool:
    return qdrant_is_healthy(settings.qdrant_url)
