# LangChain Elasticsearch Retriever Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route query-time Elasticsearch keyword search through a LangChain-style retriever wrapper while preserving Imperial RAG's existing keyword-search contract.

**Architecture:** Add a small `BaseRetriever` subclass inside `src/imperial_rag/elasticsearch_keyword.py` that performs score-aware Elasticsearch searches and returns LangChain `Document` objects. Keep `ElasticsearchKeywordIndex` as the public facade for ingestion and runtime, delegating query-time token searches to the retriever while preserving `search()` and `search_with_scores()`.

**Tech Stack:** Python 3.12+, LangChain Core `BaseRetriever`, LangChain `Document`, official Elasticsearch Python client, pytest, existing Imperial RAG retrieval and keyword helpers.

---

## Context Notes

- The installed `langchain_community.retrievers.elastic_search_bm25.ElasticSearchBM25Retriever` only returns `Document(page_content=r["_source"]["content"])`; it drops Imperial metadata and `_score`, so do not use it for this migration.
- Use `langchain_core.retrievers.BaseRetriever` directly. It supports `invoke("query", limit=5)` when `_get_relevant_documents()` accepts a `limit` keyword argument.
- Keep index creation, mappings, bulk indexing, `replace_all()`, and `elasticsearch_health()` in `ElasticsearchKeywordIndex`.
- Keep query normalization and relaxed-token behavior in `imperial_rag.keyword` for this implementation. Analyzer migration is outside this plan.

## File Structure

- Modify `src/imperial_rag/elasticsearch_keyword.py`
  - Add `ElasticsearchRetrieverHit`.
  - Add `ElasticsearchKeywordRetriever(BaseRetriever)`.
  - Keep `ElasticsearchKeywordIndex` as the public facade.
  - Route `_search_tokens()` through the retriever.
  - Convert retriever hits to `KeywordHit`.
- Modify `src/imperial_rag/retrieval.py`
  - Add optional keyword score availability diagnostics derived from returned keyword docs.
- Modify `tests/test_elasticsearch_keyword.py`
  - Add unit tests for the retriever wrapper.
  - Keep existing facade tests passing.
- Modify `tests/test_retrieval.py`
  - Add score-availability diagnostic coverage.
- Modify `tests/test_elasticsearch_live.py`
  - Assert the live path uses the retriever-backed facade and preserves metadata/rank/score.

---

### Task 1: Add A Score-Aware LangChain Elasticsearch Retriever

**Files:**
- Modify: `src/imperial_rag/elasticsearch_keyword.py`
- Modify: `tests/test_elasticsearch_keyword.py`

- [ ] **Step 1: Write the failing retriever unit test**

Add these imports near the top of `tests/test_elasticsearch_keyword.py`:

```python
from langchain_core.retrievers import BaseRetriever
```

Change the Elasticsearch import block to include the new retriever:

```python
from imperial_rag.elasticsearch_keyword import (
    ElasticsearchKeywordIndex,
    ElasticsearchKeywordRetriever,
    elasticsearch_health,
)
```

Add this test after `mark_index_exists()`:

```python
def test_keyword_retriever_is_langchain_retriever_and_preserves_scores(tmp_path: Path) -> None:
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
    assert client.search_calls == [
        {
            "index": "test_keyword_chunks",
            "query": build_elasticsearch_token_query(["возврат", "брак"]),
            "size": 5,
        },
        {
            "index": "test_keyword_chunks",
            "query": build_elasticsearch_token_query(["возврат", "брак"]),
            "size": 5,
        },
    ]
```

- [ ] **Step 2: Run the retriever test and verify it fails**

Run:

```bash
uv run python -m pytest tests/test_elasticsearch_keyword.py::test_keyword_retriever_is_langchain_retriever_and_preserves_scores -q
```

Expected: FAIL with an import error for `ElasticsearchKeywordRetriever`.

- [ ] **Step 3: Add the retriever implementation**

In `src/imperial_rag/elasticsearch_keyword.py`, add this import:

```python
from dataclasses import dataclass
```

Add this import near the existing LangChain `Document` import:

```python
from langchain_core.retrievers import BaseRetriever
```

Insert this dataclass immediately before `class ElasticsearchKeywordIndex`:

```python
@dataclass(frozen=True)
class ElasticsearchRetrieverHit:
    document: Document
    score: float
    hit_id: str
```

Insert this retriever class immediately before `class ElasticsearchKeywordIndex`, after `ElasticsearchRetrieverHit`:

```python
class ElasticsearchKeywordRetriever(BaseRetriever):
    client: Any
    index_name: str

    def _get_relevant_documents(self, query: str, *, limit: int = 5, **_: Any) -> list[Document]:
        return [hit.document for hit in self.search(query, limit=limit)]

    def search(self, query: str, limit: int = 5) -> list[ElasticsearchRetrieverHit]:
        tokens = content_keyword_query_tokens(query)
        if not tokens:
            return []
        return self.search_tokens(tokens, limit=limit)

    def search_tokens(self, tokens: list[str], limit: int) -> list[ElasticsearchRetrieverHit]:
        response = self.client.search(
            index=self.index_name,
            query=build_elasticsearch_token_query(tokens),
            size=limit,
        )
        hits = list(response.get("hits", {}).get("hits", []))
        return [self._hit_from_elasticsearch(hit) for hit in hits]

    def _hit_from_elasticsearch(self, hit: dict[str, Any]) -> ElasticsearchRetrieverHit:
        source = dict(hit.get("_source") or {})
        metadata = dict(source.get("metadata") or {})
        document = Document(page_content=str(source.get("text", "")), metadata=metadata)
        score = float(hit.get("_score") or 0.0)
        hit_id = str(
            hit.get("_id")
            or source.get("chunk_id")
            or metadata.get("chunk_id")
            or metadata.get("citation_id")
            or document.page_content
        )
        return ElasticsearchRetrieverHit(document=document, score=score, hit_id=hit_id)
```

- [ ] **Step 4: Run the retriever test and verify it passes**

Run:

```bash
uv run python -m pytest tests/test_elasticsearch_keyword.py::test_keyword_retriever_is_langchain_retriever_and_preserves_scores -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/imperial_rag/elasticsearch_keyword.py tests/test_elasticsearch_keyword.py
git commit -m "feat: add elasticsearch keyword retriever"
```

Expected: commit succeeds with only the two listed files staged.

---

### Task 2: Route ElasticsearchKeywordIndex Query-Time Search Through The Retriever

**Files:**
- Modify: `src/imperial_rag/elasticsearch_keyword.py`
- Modify: `tests/test_elasticsearch_keyword.py`

- [ ] **Step 1: Write the failing facade wiring test**

Add this test after `test_keyword_retriever_is_langchain_retriever_and_preserves_scores`:

```python
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
```

- [ ] **Step 2: Run the facade wiring test and verify it fails**

Run:

```bash
uv run python -m pytest tests/test_elasticsearch_keyword.py::test_keyword_index_facade_uses_retriever_for_query_time_search -q
```

Expected: FAIL because `ElasticsearchKeywordIndex` does not expose `retriever`.

- [ ] **Step 3: Wire the facade through the retriever**

In `ElasticsearchKeywordIndex.__init__`, after `self._bulk = bulk`, add:

```python
        self.retriever = ElasticsearchKeywordRetriever(client=self.client, index_name=self.index_name)
```

Replace `_search_tokens()` with:

```python
    def _search_tokens(self, tokens: list[str], limit: int) -> list[ElasticsearchRetrieverHit]:
        return self.retriever.search_tokens(tokens, limit=limit)
```

Replace `_search_relaxed()` with:

```python
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
```

Replace `_keyword_hit()` with:

```python
    def _keyword_hit(self, hit: ElasticsearchRetrieverHit, rank: int) -> KeywordHit:
        metadata = dict(hit.document.metadata or {})
        metadata["_keyword_rank"] = rank
        metadata["_keyword_score"] = hit.score
        return KeywordHit(
            document=Document(page_content=hit.document.page_content, metadata=metadata),
            score=hit.score,
        )
```

- [ ] **Step 4: Run Elasticsearch keyword unit tests**

Run:

```bash
uv run python -m pytest tests/test_elasticsearch_keyword.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add src/imperial_rag/elasticsearch_keyword.py tests/test_elasticsearch_keyword.py
git commit -m "refactor: route keyword search through retriever"
```

Expected: commit succeeds with only the two listed files staged.

---

### Task 3: Add Keyword Score Availability Diagnostics

**Files:**
- Modify: `src/imperial_rag/retrieval.py`
- Modify: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing diagnostics tests**

Add this helper class after `FakeKeywordSearch` in `tests/test_retrieval.py`:

```python
class FakeKeywordSearchWithoutScores:
    def __init__(self, docs):
        self.docs = docs

    def search_with_scores(self, query, limit):
        class Hit:
            def __init__(self, document):
                self.document = document
                self.score = 0.0

        return [Hit(document) for document in self.docs[:limit]]
```

Add these tests after `test_hybrid_retriever_uses_configured_candidate_counts`:

```python
def test_hybrid_retriever_reports_keyword_scores_available_when_scores_are_present():
    vector = FakeVectorSearch([])
    keyword = FakeKeywordSearch(
        [
            Document(
                page_content="Порядок возврата брака",
                metadata={"citation_id": "return", "_keyword_rank": 0, "_keyword_score": 7.5},
            )
        ]
    )

    result = HybridRetriever(vector_search=vector, keyword_search=keyword, settings=RetrievalSettings()).retrieve(
        "возврат брака"
    )

    assert result.diagnostics["keyword_scores_available"] is True


def test_hybrid_retriever_reports_keyword_scores_unavailable_when_scores_are_absent():
    vector = FakeVectorSearch([])
    keyword = FakeKeywordSearchWithoutScores(
        [
            Document(
                page_content="Порядок возврата брака",
                metadata={"citation_id": "return", "_keyword_rank": 0},
            )
        ]
    )

    result = HybridRetriever(vector_search=vector, keyword_search=keyword, settings=RetrievalSettings()).retrieve(
        "возврат брака"
    )

    assert result.diagnostics["keyword_scores_available"] is False
```

- [ ] **Step 2: Run the diagnostics tests and verify they fail**

Run:

```bash
uv run python -m pytest \
  tests/test_retrieval.py::test_hybrid_retriever_reports_keyword_scores_available_when_scores_are_present \
  tests/test_retrieval.py::test_hybrid_retriever_reports_keyword_scores_unavailable_when_scores_are_absent \
  -q
```

Expected: FAIL with missing `keyword_scores_available`.

- [ ] **Step 3: Add diagnostics derived from returned keyword documents**

In `HybridRetriever.retrieve()` in `src/imperial_rag/retrieval.py`, add this block after keyword status is finalized and before `_set_documents_span_output(...)` for the keyword span:

```python
            if keyword_docs:
                keyword_scores_available = all(
                    "_keyword_score" in dict(document.metadata or {})
                    for document in keyword_docs
                )
            else:
                keyword_scores_available = False
```

In the same keyword span, pass the value into `_set_documents_span_output(...)`:

```python
                keyword_scores_available=keyword_scores_available,
```

In the final `RetrievalCandidateResult` diagnostics dict, add:

```python
                "keyword_scores_available": keyword_scores_available,
```

- [ ] **Step 4: Run the diagnostics tests and verify they pass**

Run:

```bash
uv run python -m pytest \
  tests/test_retrieval.py::test_hybrid_retriever_reports_keyword_scores_available_when_scores_are_present \
  tests/test_retrieval.py::test_hybrid_retriever_reports_keyword_scores_unavailable_when_scores_are_absent \
  -q
```

Expected: PASS.

- [ ] **Step 5: Run the focused retrieval suite**

Run:

```bash
uv run python -m pytest tests/test_retrieval.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

Run:

```bash
git add src/imperial_rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: report keyword score availability"
```

Expected: commit succeeds with only the two listed files staged.

---

### Task 4: Update Live Elasticsearch Coverage For The Retriever Path

**Files:**
- Modify: `tests/test_elasticsearch_live.py`

- [ ] **Step 1: Write the live retriever assertions**

Change the import block in `tests/test_elasticsearch_live.py` to include `ElasticsearchKeywordRetriever`:

```python
from imperial_rag.elasticsearch_keyword import (
    ElasticsearchKeywordIndex,
    ElasticsearchKeywordRetriever,
    elasticsearch_health,
)
```

Inside `test_live_elasticsearch_keyword_index_roundtrip()`, after `page_results = index.search("2", k=5)`, add:

```python
        retriever_results = index.retriever.invoke("возврат брака", limit=5)
```

After the existing result assertions, add:

```python
        assert isinstance(index.retriever, ElasticsearchKeywordRetriever)
        assert [result.metadata["citation_id"] for result in retriever_results[:1]] == ["store"]
```

- [ ] **Step 2: Run the live test in default mode and verify it skips**

Run:

```bash
uv run python -m pytest tests/test_elasticsearch_live.py -q
```

Expected: PASS with `1 skipped` when `IMPERIAL_RAG_LIVE_ELASTICSEARCH` is unset.

- [ ] **Step 3: Run the live test against local Elasticsearch when it is intentionally running**

Run:

```bash
IMPERIAL_RAG_LIVE_ELASTICSEARCH=1 uv run python -m pytest tests/test_elasticsearch_live.py -q
```

Expected when local Elasticsearch is running on `127.0.0.1:9200`: PASS.

Expected when local Elasticsearch is not running: FAIL at `assert elasticsearch_health(settings) is True`; start Elasticsearch with `./scripts/start_elasticsearch.sh` before rerunning.

- [ ] **Step 4: Commit Task 4**

Run:

```bash
git add tests/test_elasticsearch_live.py
git commit -m "test: cover live elasticsearch retriever path"
```

Expected: commit succeeds with only `tests/test_elasticsearch_live.py` staged.

---

### Task 5: Final Verification

**Files:**
- Inspect: `src/imperial_rag/elasticsearch_keyword.py`
- Inspect: `src/imperial_rag/retrieval.py`
- Inspect: `tests/test_elasticsearch_keyword.py`
- Inspect: `tests/test_retrieval.py`
- Inspect: `tests/test_elasticsearch_live.py`

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run python -m pytest tests/test_elasticsearch_keyword.py tests/test_retrieval.py tests/test_elasticsearch_live.py -q
```

Expected: PASS, with `tests/test_elasticsearch_live.py` skipped unless `IMPERIAL_RAG_LIVE_ELASTICSEARCH=1` is set.

- [ ] **Step 2: Run the full default test suite**

Run:

```bash
uv run python -m pytest -q
```

Expected: PASS.

- [ ] **Step 3: Inspect git status**

Run:

```bash
git status --short
```

Expected: only unrelated pre-existing files remain modified or untracked. No implementation files from this plan should be unstaged.

- [ ] **Step 4: Inspect recent commits**

Run:

```bash
git log --oneline -5
```

Expected: the task commits from this plan appear later in history than `docs: add langchain elasticsearch retriever design`.
