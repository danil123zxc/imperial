# Elasticsearch Keyword Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the SQLite keyword index with a required local Elasticsearch keyword index while preserving manifest storage, OCR cache storage, hybrid retrieval behavior, and default offline tests.

**Architecture:** Add a focused keyword helper module and an `ElasticsearchKeywordIndex` adapter that preserves the current `replace_all`, `index_documents`, `search`, and `search_with_scores` contract. Runtime and ingestion construct the Elasticsearch-backed keyword service from `Settings`; retrieval, RRF, reranking, answer generation, manifest storage, OCR cache storage, and Qdrant vector search remain unchanged.

**Tech Stack:** Python 3.12, LangChain `Document`, official `elasticsearch` Python client, Elasticsearch single-node Docker service, pytest, uv.

---

## File Structure

- Create `src/imperial_rag/keyword.py`: keyword hit dataclass, protocol, Russian normalization/token helpers, relaxed-token helper, and searchable text helper.
- Create `src/imperial_rag/elasticsearch_keyword.py`: Elasticsearch client adapter, index mapping, bulk indexing, query construction, hit conversion, and health helper.
- Modify `src/imperial_rag/indexing.py`: remove SQLite keyword-index implementation from active use, re-export keyword helpers for compatibility, and keep Qdrant/vector helpers plus stable chunk ids.
- Modify `src/imperial_rag/config.py`: add Elasticsearch URL and index settings.
- Modify `src/imperial_rag/pipeline.py`: use `ElasticsearchKeywordIndex` and pass full settings to keyword indexing.
- Modify `src/imperial_rag/runtime.py`: construct `ElasticsearchKeywordIndex(settings)` instead of `KeywordIndex(settings.keyword_db_path)`.
- Modify `pyproject.toml`: add the official `elasticsearch` Python client dependency.
- Modify `.env.example`: add Elasticsearch settings and the live Elasticsearch test opt-in flag.
- Modify `compose.yaml`: add the local Elasticsearch service and volume.
- Create `scripts/start_elasticsearch.sh`: local helper for running Elasticsearch consistently with other scripts.
- Modify `README.md`: document Elasticsearch as the required keyword-search service.
- Modify tests:
  - `tests/test_config.py`
  - `tests/test_keyword.py`
  - `tests/test_elasticsearch_keyword.py`
  - `tests/test_elasticsearch_live.py`
  - `tests/test_indexing.py`
  - `tests/test_pipeline.py`
  - `tests/test_runtime.py`

Do not edit `AGENTS.md` in this implementation pass because it is already dirty from outside this session.

---

### Task 1: Add Elasticsearch Settings And Dependency

**Files:**
- Modify: `tests/test_config.py`
- Modify: `src/imperial_rag/config.py`
- Modify: `pyproject.toml`
- Modify: `.env.example`

- [ ] **Step 1: Write the failing config tests**

Edit `tests/test_config.py` so the default and override tests include Elasticsearch settings:

```python
def test_settings_defaults_to_workspace_documents():
    settings = Settings()

    assert settings.workspace_root == Path("/Users/danil/Public/imperial")
    assert settings.documents_root == Path("/Users/danil/Public/imperial/documents")
    assert settings.processed_root == Path("/Users/danil/Public/imperial/.imperial_rag")
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "imperial_chunks_qwen"
    assert settings.elasticsearch_url == "http://localhost:9200"
    assert settings.elasticsearch_index == "imperial_keyword_chunks"
    assert settings.phoenix_project_name == "imperial-rag"
    assert settings.phoenix_collector_endpoint == "http://localhost:6006/v1/traces"
    assert settings.phoenix_client_endpoint == "http://localhost:6006"
    assert settings.manifest_db_path == Path("/Users/danil/Public/imperial/.imperial_rag/manifest.sqlite3")
    assert settings.extraction_root == Path("/Users/danil/Public/imperial/.imperial_rag/extracted")
```

Also extend the environment override test:

```python
def test_settings_reads_environment_overrides_including_qdrant_collection(monkeypatch, tmp_path):
    monkeypatch.setenv("IMPERIAL_RAG_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_chunks")
    monkeypatch.setenv("ELASTICSEARCH_URL", "http://127.0.0.1:9200")
    monkeypatch.setenv("ELASTICSEARCH_INDEX", "test_keyword_chunks")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "test-project")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix.internal:6006/v1/traces")
    monkeypatch.setenv("PHOENIX_CLIENT_ENDPOINT", "http://phoenix.internal:6006")

    settings = Settings()

    assert settings.workspace_root == tmp_path
    assert settings.documents_root == tmp_path / "documents"
    assert settings.processed_root == tmp_path / ".imperial_rag"
    assert settings.qdrant_url == "http://127.0.0.1:6333"
    assert settings.qdrant_collection == "test_chunks"
    assert settings.elasticsearch_url == "http://127.0.0.1:9200"
    assert settings.elasticsearch_index == "test_keyword_chunks"
    assert settings.phoenix_project_name == "test-project"
    assert settings.phoenix_collector_endpoint == "http://phoenix.internal:6006/v1/traces"
    assert settings.phoenix_client_endpoint == "http://phoenix.internal:6006"
```

- [ ] **Step 2: Run config tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/test_config.py -q
```

Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'elasticsearch_url'`.

- [ ] **Step 3: Add settings fields**

Edit `src/imperial_rag/config.py` and add these fields after the Qdrant settings:

```python
    elasticsearch_url: str = field(default_factory=lambda: os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"))
    elasticsearch_index: str = field(default_factory=lambda: os.environ.get("ELASTICSEARCH_INDEX", "imperial_keyword_chunks"))
```

Keep this deprecated compatibility property during Tasks 1 through 3 so existing runtime and pipeline tests can be migrated in Task 4:

```python
    @property
    def keyword_db_path(self) -> Path:
        return self.processed_root / "keyword.sqlite3"
```

- [ ] **Step 4: Add the Python client dependency**

Edit `pyproject.toml` and add the official Elasticsearch client to `[project].dependencies`:

```toml
  "elasticsearch>=8.19,<9",
```

Run:

```bash
uv sync --extra dev
```

Expected: dependency resolution succeeds. Stage `uv.lock` in Step 7 when `uv sync` updates it.

- [ ] **Step 5: Add env examples**

Edit `.env.example` after the Qdrant section:

```dotenv
# Elasticsearch keyword search
ELASTICSEARCH_URL=http://localhost:9200
ELASTICSEARCH_INDEX=imperial_keyword_chunks
```

Edit the live-test flag section:

```dotenv
# Set to 1 only when intentionally running live Elasticsearch tests.
IMPERIAL_RAG_LIVE_ELASTICSEARCH=0
```

- [ ] **Step 6: Run focused checks**

Run:

```bash
uv run python -m pytest tests/test_config.py -q
uv run python -m pytest tests/test_runtime.py::test_create_runtime_constructs_without_live_services -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git status --short
git add pyproject.toml uv.lock .env.example src/imperial_rag/config.py tests/test_config.py
git commit -m "feat: add elasticsearch keyword settings"
```

Expected: commit succeeds. Do not stage `AGENTS.md`.

---

### Task 2: Extract Keyword Helpers From SQLite Code

**Files:**
- Create: `src/imperial_rag/keyword.py`
- Modify: `src/imperial_rag/indexing.py`
- Create: `tests/test_keyword.py`
- Modify: `tests/test_indexing.py`

- [ ] **Step 1: Write helper tests**

Create `tests/test_keyword.py`:

```python
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
```

- [ ] **Step 2: Run helper tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/test_keyword.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'imperial_rag.keyword'`.

- [ ] **Step 3: Create the keyword helper module**

Create `src/imperial_rag/keyword.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document


_ENDING_RE = re.compile(r"(иями|ями|ами|ого|его|ому|ему|ыми|ими|ов|ев|ей|ый|ий|ой|ая|яя|ое|ее|ам|ям|ах|ях|ом|ем|а|я|ы|и|у|ю|е|о|ь)$")
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


@dataclass(frozen=True)
class KeywordHit:
    document: Document
    score: float


class KeywordSearch(Protocol):
    def replace_all(self, documents: list[Document]) -> None: ...
    def index_documents(self, documents: list[Document]) -> None: ...
    def search(self, query: str, limit: int = 5, k: int | None = None) -> list[Document]: ...
    def search_with_scores(self, query: str, limit: int = 5, k: int | None = None) -> list[KeywordHit]: ...


def stem_token(token: str) -> str:
    token = token.casefold().replace("ё", "е")
    while len(token) > 4:
        shortened = _ENDING_RE.sub("", token)
        if shortened == token:
            break
        token = shortened
    return token


def normalize_search_text(text: str) -> str:
    return " ".join(
        stem_token(token)
        for token in re.findall(r"\w+", text.casefold().replace("-", " "), flags=re.UNICODE)
    )


def keyword_query_tokens(query: str) -> list[str]:
    tokens = [token for token in normalize_search_text(query).split() if token]
    content_tokens = [token for token in tokens if token not in _QUERY_STOPWORDS and len(token) > 2]
    return content_tokens or tokens


def relaxed_query_token_sets(tokens: list[str]) -> list[list[str]]:
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


def build_elasticsearch_token_query(tokens: list[str]) -> dict:
    return {"bool": {"must": [{"match": {"normalized_text": token}} for token in tokens]}}


def searchable_document_text(document: Document) -> str:
    metadata = document.metadata or {}
    return " ".join(
        [
            document.page_content,
            str(metadata.get("file_name", "")),
            str(metadata.get("relative_path", "")),
            str(metadata.get("section_heading", "")),
            str(metadata.get("source_type", "")),
        ]
    )


def relaxed_candidate_sort_key(candidate: tuple[int, int, int, object]) -> tuple[int, int, int]:
    matched_token_count, query_order, row_order, _row = candidate
    return (-matched_token_count, query_order, row_order)


def _bounded_adjacent_pairs(tokens: list[str], budget: int) -> list[list[str]]:
    if budget <= 0:
        return []
    pairs = [tokens[index : index + 2] for index in range(len(tokens) - 1)]
    if len(pairs) <= budget:
        return pairs
    head_count = budget // 2
    tail_count = budget - head_count
    return pairs[:head_count] + pairs[-tail_count:]
```

- [ ] **Step 4: Re-export helper names from indexing**

Edit `src/imperial_rag/indexing.py` imports:

```python
from imperial_rag.keyword import (
    KeywordHit,
    keyword_query_tokens,
    normalize_search_text,
    relaxed_candidate_sort_key,
    relaxed_query_token_sets,
    searchable_document_text,
)
```

Then replace internal helper calls:

```python
query_tokens = keyword_query_tokens(query)
```

```python
normalize_search_text(searchable_document_text(document))
```

```python
for query_order, relaxed_tokens in enumerate(relaxed_query_token_sets(query_tokens)):
```

```python
if previous is None or relaxed_candidate_sort_key(candidate) < relaxed_candidate_sort_key(previous):
```

Delete duplicated helper definitions from `src/imperial_rag/indexing.py` after tests pass.

- [ ] **Step 5: Keep vector tests and remove SQLite keyword tests from test_indexing**

Edit `tests/test_indexing.py`:

```python
from __future__ import annotations

from uuid import UUID

from langchain_core.documents import Document

from imperial_rag.indexing import (
    create_qdrant_vector_store,
    index_documents,
    index_vector_documents,
    stable_chunk_id,
)
```

Delete every function whose name starts with `test_keyword_index_`. Keep `test_stable_chunk_id_uses_citation_metadata_and_content` and all vector/Qdrant tests after it.

- [ ] **Step 6: Run focused helper and indexing tests**

Run:

```bash
uv run python -m pytest tests/test_keyword.py tests/test_indexing.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git status --short
git add src/imperial_rag/keyword.py src/imperial_rag/indexing.py tests/test_keyword.py tests/test_indexing.py
git commit -m "refactor: extract keyword search helpers"
```

Expected: commit succeeds. Do not stage generated files or `AGENTS.md`.

---

### Task 3: Implement ElasticsearchKeywordIndex With Fake-Client Tests

**Files:**
- Create: `src/imperial_rag/elasticsearch_keyword.py`
- Create: `tests/test_elasticsearch_keyword.py`
- Modify: `src/imperial_rag/indexing.py`

- [ ] **Step 1: Write Elasticsearch adapter tests**

Create `tests/test_elasticsearch_keyword.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
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

    results = index.search("Как оформить возврат брака из магазина?", k=3)

    assert [result.metadata["citation_id"] for result in results] == ["store"]
    assert len(client.search_calls) == 2
    assert client.search_calls[1]["query"]["bool"]["must"] != client.search_calls[0]["query"]["bool"]["must"]


def test_search_returns_empty_for_stopword_only_empty_query(tmp_path: Path) -> None:
    client = FakeClient()
    index = make_index(tmp_path, client)

    assert index.search("и в на", limit=5) == []
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
```

- [ ] **Step 2: Run adapter tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/test_elasticsearch_keyword.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'imperial_rag.elasticsearch_keyword'`.

- [ ] **Step 3: Implement the adapter**

Create `src/imperial_rag/elasticsearch_keyword.py`:

```python
from __future__ import annotations

from typing import Any, Callable, Iterable

from langchain_core.documents import Document

from imperial_rag.config import Settings
from imperial_rag.keyword import (
    KeywordHit,
    build_elasticsearch_token_query,
    keyword_query_tokens,
    normalize_search_text,
    relaxed_query_token_sets,
    searchable_document_text,
)


INDEX_MAPPINGS = {
    "properties": {
        "chunk_id": {"type": "keyword"},
        "text": {"type": "text", "index": False},
        "normalized_text": {"type": "text"},
        "metadata": {"type": "object", "enabled": False},
    }
}
INDEX_SETTINGS = {"number_of_shards": 1, "number_of_replicas": 0}


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
        query_tokens = keyword_query_tokens(query)
        if not query_tokens:
            return []

        hits = self._search_tokens(query_tokens, resolved_limit)
        if not hits:
            hits = self._search_relaxed(query_tokens, resolved_limit)
        return [self._keyword_hit(hit, rank) for rank, hit in enumerate(hits[:resolved_limit])]

    def _create_index(self) -> None:
        if self.client.indices.exists(index=self.index_name):
            return
        self.client.indices.create(index=self.index_name, mappings=INDEX_MAPPINGS, settings=INDEX_SETTINGS)

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
                    "normalized_text": normalize_search_text(searchable_document_text(document)),
                    "metadata": dict(document.metadata or {}),
                },
            }

    def _search_tokens(self, tokens: list[str], limit: int) -> list[dict[str, Any]]:
        response = self.client.search(
            index=self.index_name,
            query=build_elasticsearch_token_query(tokens),
            size=limit,
        )
        return list(response.get("hits", {}).get("hits", []))

    def _search_relaxed(self, tokens: list[str], limit: int) -> list[dict[str, Any]]:
        seen: set[str] = set()
        ordered_hits: list[dict[str, Any]] = []
        for relaxed_tokens in relaxed_query_token_sets(tokens):
            for hit in self._search_tokens(relaxed_tokens, limit):
                hit_id = str(hit.get("_id") or hit.get("_source", {}).get("chunk_id") or len(seen))
                if hit_id in seen:
                    continue
                seen.add(hit_id)
                ordered_hits.append(hit)
                if len(ordered_hits) >= limit:
                    return ordered_hits
        return ordered_hits

    def _keyword_hit(self, hit: dict[str, Any], rank: int) -> KeywordHit:
        source = dict(hit.get("_source") or {})
        metadata = dict(source.get("metadata") or {})
        score = float(hit.get("_score") or 0.0)
        metadata["_keyword_rank"] = rank
        metadata["_keyword_score"] = score
        return KeywordHit(
            document=Document(page_content=str(source.get("text", "")), metadata=metadata),
            score=score,
        )


KeywordIndex = ElasticsearchKeywordIndex


def elasticsearch_health(settings: Settings, *, client: Any | None = None) -> bool:
    if client is None:
        from elasticsearch import Elasticsearch

        client = Elasticsearch(settings.elasticsearch_url)
    try:
        return bool(client.ping())
    except Exception:
        return False
```

- [ ] **Step 4: Re-export the new class from indexing**

At the bottom of `src/imperial_rag/indexing.py`, after vector helper definitions, add:

```python
from imperial_rag.elasticsearch_keyword import ElasticsearchKeywordIndex, KeywordIndex, elasticsearch_health  # noqa: E402
```

This re-export is safe because `ElasticsearchKeywordIndex._actions()` imports `stable_chunk_id` lazily at runtime instead of importing `imperial_rag.indexing` at module import time.

- [ ] **Step 5: Run adapter tests**

Run:

```bash
uv run python -m pytest tests/test_keyword.py tests/test_elasticsearch_keyword.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git status --short
git add src/imperial_rag/elasticsearch_keyword.py src/imperial_rag/indexing.py tests/test_elasticsearch_keyword.py
git commit -m "feat: add elasticsearch keyword index"
```

Expected: commit succeeds.

---

### Task 4: Wire Ingestion And Runtime To Elasticsearch Keyword Search

**Files:**
- Modify: `src/imperial_rag/pipeline.py`
- Modify: `src/imperial_rag/runtime.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_runtime.py`

- [ ] **Step 1: Update pipeline tests for settings-based keyword construction**

Edit `tests/test_pipeline.py`.

Change `FakeSettings` by replacing `keyword_db_path` with Elasticsearch fields:

```python
    elasticsearch_url: str = "http://127.0.0.1:9200"
    elasticsearch_index: str = "test_keyword_chunks"
```

Change `FakeKeywordIndex`:

```python
class FakeKeywordIndex:
    last_docs = None
    last_settings = None

    def __init__(self, settings) -> None:
        self.settings = settings
        FakeKeywordIndex.last_settings = settings

    def replace_all(self, documents):
        FakeKeywordIndex.last_docs = list(documents)
```

Add this assertion to `test_run_ingestion_persists_chunks_and_updates_manifest` after `FakeKeywordIndex.last_docs is not None`:

```python
    assert FakeKeywordIndex.last_settings == FakeSettings(tmp_path)
```

In `_install_fake_dependencies`, change the fake indexing module:

```python
    indexing.ElasticsearchKeywordIndex = FakeKeywordIndex
    indexing.KeywordIndex = FakeKeywordIndex
```

- [ ] **Step 2: Run pipeline test to verify it fails**

Run:

```bash
uv run python -m pytest tests/test_pipeline.py::test_run_ingestion_persists_chunks_and_updates_manifest -q
```

Expected: FAIL because `_replace_keyword_index` still passes `Path(settings.keyword_db_path)`.

- [ ] **Step 3: Update pipeline keyword dependency**

Edit `src/imperial_rag/pipeline.py`.

Change the default dependency import in `build_ingestion_dependencies()` from `KeywordIndex` to `ElasticsearchKeywordIndex`:

```python
    from imperial_rag.elasticsearch_keyword import ElasticsearchKeywordIndex
```

Change the dependency dictionary:

```python
        "KeywordIndex": ElasticsearchKeywordIndex,
```

Change `_replace_keyword_index`:

```python
def _replace_keyword_index(keyword_index_cls: Any, settings: Any, chunks: list[Any]) -> bool:
    keyword_index = keyword_index_cls(settings)
    keyword_index.replace_all(chunks)
    return True
```

- [ ] **Step 4: Update runtime tests for ElasticsearchKeywordIndex**

Edit `tests/test_runtime.py`.

In tests that monkeypatch `KeywordIndex`, monkeypatch `ElasticsearchKeywordIndex` instead:

```python
monkeypatch.setattr("imperial_rag.runtime.ElasticsearchKeywordIndex", lambda settings: object())
```

Replace `test_build_query_dependencies_skips_vector_search_on_metadata_mismatch` with:

```python
def test_build_query_dependencies_skips_vector_search_on_metadata_mismatch(monkeypatch, tmp_path):
    calls = {}
    fake_chat_model = object()

    class FakeElasticsearchKeywordIndex:
        def __init__(self, settings):
            calls["settings"] = settings

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setattr("imperial_rag.runtime.ElasticsearchKeywordIndex", FakeElasticsearchKeywordIndex)
    monkeypatch.setattr("imperial_rag.runtime.create_chat_model", lambda: fake_chat_model, raising=False)
    monkeypatch.setattr("imperial_rag.runtime.vector_metadata_matches_config", lambda settings: False, raising=False)

    settings = Settings(workspace_root=tmp_path)
    dependencies = build_query_dependencies(settings)

    assert getattr(dependencies.vector_search, "provider_mismatch", False) is True
    assert calls["settings"] is settings
```

- [ ] **Step 5: Update runtime implementation**

Edit `src/imperial_rag/runtime.py`.

Change imports:

```python
from imperial_rag.elasticsearch_keyword import ElasticsearchKeywordIndex
from imperial_rag.indexing import make_qdrant_store
```

Change `build_query_dependencies`:

```python
    return QueryDependencies(
        vector_search=vector_search,
        keyword_search=ElasticsearchKeywordIndex(settings),
        chat_model=_DeferredProviderChatModel(),
    )
```

- [ ] **Step 6: Remove deprecated keyword_db_path configuration**

Edit `src/imperial_rag/config.py` and delete:

```python
    @property
    def keyword_db_path(self) -> Path:
        return self.processed_root / "keyword.sqlite3"
```

Run:

```bash
rg -n "keyword_db_path" src tests
```

Expected: no matches.

- [ ] **Step 7: Run focused wiring tests**

Run:

```bash
uv run python -m pytest tests/test_pipeline.py tests/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git status --short
git add src/imperial_rag/pipeline.py src/imperial_rag/runtime.py tests/test_pipeline.py tests/test_runtime.py
git commit -m "feat: wire keyword search to elasticsearch"
```

Expected: commit succeeds.

---

### Task 5: Add Local Elasticsearch Service And Docs

**Files:**
- Modify: `compose.yaml`
- Create: `scripts/start_elasticsearch.sh`
- Modify: `README.md`

- [ ] **Step 1: Add Elasticsearch to compose**

Edit `compose.yaml`:

```yaml
services:
  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"
      - "4317:4317"
    environment:
      - PHOENIX_WORKING_DIR=/mnt/data
    volumes:
      - phoenix_data:/mnt/data

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.19.0
    ports:
      - "127.0.0.1:9200:9200"
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms512m -Xmx512m
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

volumes:
  phoenix_data:
    driver: local
  elasticsearch_data:
    driver: local
```

- [ ] **Step 2: Add the helper script**

Create `scripts/start_elasticsearch.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Local-only Elasticsearch for private keyword search. Do not expose this service publicly.
docker compose up elasticsearch
```

Run:

```bash
chmod +x scripts/start_elasticsearch.sh
bash -n scripts/start_elasticsearch.sh
docker compose config --services
```

Expected: `bash -n` is silent and compose services include `elasticsearch` and `phoenix`.

- [ ] **Step 3: Update README overview and architecture**

Edit `README.md`.

Replace:

```markdown
- Builds a SQLite full-text keyword index for exact Russian/company terminology.
```

with:

```markdown
- Builds a local Elasticsearch keyword index for exact Russian/company terminology.
```

Replace the architecture snippet line:

```text
  -> .imperial_rag/keyword.sqlite3 SQLite FTS
```

with:

```text
  -> local Elasticsearch keyword index
```

Replace the module description:

```markdown
- `indexing.py` owns SQLite FTS keyword indexing and Qdrant vector indexing helpers.
```

with:

```markdown
- `elasticsearch_keyword.py` owns Elasticsearch keyword indexing, and `indexing.py` owns Qdrant vector indexing helpers plus stable chunk ids.
```

- [ ] **Step 4: Add README local service docs**

In `README.md`, add this section before `### Qdrant`:

````markdown
### Elasticsearch

Elasticsearch is required for keyword search. Start it locally before running ingestion or querying the processed corpus:

```bash
./scripts/start_elasticsearch.sh
```

Defaults:

- URL: `http://localhost:9200`
- index: `imperial_keyword_chunks`
- Docker volume: `imperial_elasticsearch_data`

Ingestion rebuilds the Elasticsearch keyword index from `.imperial_rag/extracted/chunks.jsonl`. The old `.imperial_rag/keyword.sqlite3` file is obsolete generated state and is not read by the application after the Elasticsearch migration.
````

Also add these settings to the configuration list:

```markdown
- `ELASTICSEARCH_URL`: Elasticsearch endpoint, defaulting to `http://localhost:9200`.
- `ELASTICSEARCH_INDEX`: Elasticsearch keyword index, defaulting to `imperial_keyword_chunks`.
```

Add the live test command under Testing:

````markdown
Run the live Elasticsearch test only when local Elasticsearch is intentionally running:

```bash
IMPERIAL_RAG_LIVE_ELASTICSEARCH=1 uv run python -m pytest tests/test_elasticsearch_live.py -q
```
````

- [ ] **Step 5: Run docs/service checks**

Run:

```bash
bash -n scripts/start_elasticsearch.sh
docker compose config --services
rg -n "SQLite full-text|keyword.sqlite3 SQLite|ELASTICSEARCH_URL|IMPERIAL_RAG_LIVE_ELASTICSEARCH" README.md .env.example compose.yaml
```

Expected: compose includes `elasticsearch`; no README line still describes SQLite FTS as the active keyword index; env and README mention Elasticsearch settings and live-test flag.

- [ ] **Step 6: Commit**

Run:

```bash
git status --short
git add compose.yaml scripts/start_elasticsearch.sh README.md
git commit -m "docs: add local elasticsearch keyword service"
```

Expected: commit succeeds.

---

### Task 6: Add Opt-In Live Elasticsearch Test

**Files:**
- Create: `tests/test_elasticsearch_live.py`
- Modify: `src/imperial_rag/elasticsearch_keyword.py`

- [ ] **Step 1: Write the live test**

Create `tests/test_elasticsearch_live.py`:

```python
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
```

- [ ] **Step 2: Run the live test without opt-in**

Run:

```bash
uv run python -m pytest tests/test_elasticsearch_live.py -q
```

Expected: SKIPPED with reason `live Elasticsearch test is opt-in`.

- [ ] **Step 3: Run the live test with Elasticsearch if available**

Start Elasticsearch:

```bash
./scripts/start_elasticsearch.sh
```

In another terminal, run:

```bash
IMPERIAL_RAG_LIVE_ELASTICSEARCH=1 uv run python -m pytest tests/test_elasticsearch_live.py -q
```

Expected: PASS when Elasticsearch is healthy on `127.0.0.1:9200`. When Docker is unavailable in the execution environment, skip this command and record that live verification was not run.

- [ ] **Step 4: Commit**

Run:

```bash
git status --short
git add tests/test_elasticsearch_live.py src/imperial_rag/elasticsearch_keyword.py
git commit -m "test: add live elasticsearch keyword roundtrip"
```

Expected: commit succeeds. If `src/imperial_rag/elasticsearch_keyword.py` did not change, stage only the test file.

---

### Task 7: Full Offline Verification And Cleanup

**Files:**
- Modify only files required by failing tests from this verification pass.

- [ ] **Step 1: Search for active SQLite keyword references**

Run:

```bash
rg -n "KeywordIndex|keyword_db_path|keyword\\.sqlite3|SQLite FTS|chunks_fts|fts5|sqlite3" src tests README.md .env.example docs/superpowers/specs/2026-06-11-elasticsearch-keyword-search-design.md
```

Expected active references:

- `manifest.sqlite3` and OCR cache references are allowed.
- Historical design docs under `docs/superpowers/specs/` may mention SQLite.
- No active `src/` runtime or ingestion code should depend on `keyword_db_path`, `chunks_fts`, or SQLite keyword tables.

- [ ] **Step 2: Run the default test suite**

Run:

```bash
uv run python -m pytest -q
```

Expected: PASS, with live Elasticsearch tests skipped unless `IMPERIAL_RAG_LIVE_ELASTICSEARCH=1`.

- [ ] **Step 3: Run import smoke checks**

Run:

```bash
uv run python - <<'PY'
from imperial_rag.config import Settings
from imperial_rag.elasticsearch_keyword import ElasticsearchKeywordIndex
from imperial_rag.runtime import build_query_dependencies

settings = Settings()
print(settings.elasticsearch_url)
print(settings.elasticsearch_index)
print(ElasticsearchKeywordIndex.__name__)
print(callable(build_query_dependencies))
PY
```

Expected output includes:

```text
http://localhost:9200
imperial_keyword_chunks
ElasticsearchKeywordIndex
True
```

- [ ] **Step 4: Inspect status and diff**

Run:

```bash
git status --short
git diff --stat
git diff --check
```

Expected: only intentional implementation files are modified; `git diff --check` is silent.

- [ ] **Step 5: Confirm no cleanup commit is needed**

Run:

```bash
git status --short
```

Expected: no unstaged implementation changes from verification. If this command shows modified implementation files, stop and inspect those diffs before deciding on a follow-up fix.

---

## Final Verification Checklist

Run these before handing off:

```bash
uv run python -m pytest -q
bash -n scripts/start_elasticsearch.sh
docker compose config --services
rg -n "keyword_db_path|keyword\\.sqlite3|chunks_fts|SQLite FTS" src tests README.md .env.example
git status --short
```

Expected:

- default pytest passes;
- live Elasticsearch test is skipped unless opted in;
- compose services include `elasticsearch`;
- active source/test/docs no longer describe SQLite FTS as the active keyword index;
- `AGENTS.md` remains uncommitted if it was dirty before this implementation.

Live optional check when Docker is available:

```bash
./scripts/start_elasticsearch.sh
IMPERIAL_RAG_LIVE_ELASTICSEARCH=1 uv run python -m pytest tests/test_elasticsearch_live.py -q
```

Expected: live keyword indexing and search roundtrip passes.

---

## Plan Self-Review

Spec coverage:

- Required local Elasticsearch keyword index: Tasks 1, 3, 4, 5, and 6.
- Keep `manifest.sqlite3` and OCR cache unchanged: Tasks 4 and 7 avoid those modules except existing manifest status updates.
- Preserve keyword helper semantics: Task 2 extracts and tests normalization, stopword filtering, stemming, searchable metadata, and relaxed-token behavior.
- Preserve retrieval diagnostics and hybrid flow: Task 4 only swaps the keyword-search dependency; retrieval tests remain fake-based and unchanged in behavior.
- Required local service operations: Task 5 adds compose, helper script, env examples, and README docs.
- Default offline tests and opt-in live coverage: Tasks 3, 6, and 7 keep default pytest offline and gate live Elasticsearch behind `IMPERIAL_RAG_LIVE_ELASTICSEARCH=1`.

Placeholder scan:

- No open markers, no deferred implementation sections, and no broad "write tests" steps without concrete test code.

Type and naming consistency:

- The implementation uses `ElasticsearchKeywordIndex(settings)` consistently in pipeline and runtime.
- The search contract remains `replace_all`, `index_documents`, `search`, and `search_with_scores`.
- Returned keyword metadata remains `_keyword_rank` and `_keyword_score`.
