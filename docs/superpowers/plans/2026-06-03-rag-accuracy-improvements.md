# RAG Accuracy Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve Imperial RAG answer accuracy by adding configurable hybrid recall, reranking, neighbor expansion, and retrieval diagnostics while preserving strict citation refusal behavior.

**Architecture:** Add a focused retrieval module at `src/imperial_rag/retrieval.py`. Runtime will call this module to retrieve broad vector and keyword candidates, merge/dedupe them, rerank them, expand neighbors for small chunks, and pass final evidence into the existing strict answer workflow. Existing manifesting, extraction, Qdrant indexing, keyword indexing, LangGraph orchestration, citation validation, and Phoenix foundations remain in place.

**Tech Stack:** Python 3.12, LangChain, LangGraph, LangChain Qdrant, LangChain Cohere, Qdrant, SQLite FTS5, Phoenix, pytest, Streamlit.

---

## Scope Check

This plan implements one subsystem: query-time retrieval accuracy. It includes small supporting changes to chunk-size configuration, keyword rank metadata, eval output diagnostics, and eval questions because those are required to verify retrieval accuracy. It does not rewrite extraction, manifesting, citation validation, the Streamlit UI, or Phoenix setup.

## File Structure

- Create: `/Users/danil/Public/imperial/src/imperial_rag/retrieval.py`
  - Owns retrieval settings, diagnostics, candidate merging, fallback ranking, Cohere reranking, neighbor expansion, and final evidence selection.
- Create: `/Users/danil/Public/imperial/tests/test_retrieval.py`
  - Unit tests for retrieval settings, dedupe, fallback ranking, neighbor expansion, and hybrid retrieval fallbacks.
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/chunking.py`
  - Change default chunk size and overlap to `400/50`.
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/pipeline.py`
  - Load retrieval settings and pass chunk size/overlap into `build_chunks`.
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/indexing.py`
  - Add keyword search with rank/score metadata while preserving existing `search()` API.
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/runtime.py`
  - Replace ad hoc retrieval closure with the new retrieval module.
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/workflows.py`
  - Preserve retrieval diagnostics in LangGraph state and final query result.
- Modify: `/Users/danil/Public/imperial/scripts/run_phoenix_eval.py`
  - Include retrieval diagnostics in eval outputs.
- Modify: `/Users/danil/Public/imperial/evals/questions.jsonl`
  - Expand to 30 Russian evaluation questions.
- Modify: `/Users/danil/Public/imperial/pyproject.toml`
  - Add `langchain-cohere`.
- Modify: `/Users/danil/Public/imperial/tests/test_dependencies.py`
  - Assert Cohere reranking dependency is present.

## Preflight

- [ ] **Step 1: Confirm the approved design spec exists**

Run:

```bash
test -f docs/superpowers/specs/2026-06-03-rag-accuracy-improvements-design.md
```

Expected: command exits `0`.

- [ ] **Step 2: Capture current test baseline**

Run:

```bash
uv run --extra dev python -m pytest -q
```

Expected: `70 passed, 1 skipped` or a larger passing count if tests have already been added.

- [ ] **Step 3: Capture current local eval baseline**

Run:

```bash
uv run --extra dev python scripts/run_phoenix_eval.py
```

Expected in the current processed state: output contains `local_eval_examples=7`. Record `local_eval_passed=<N>` in the implementation notes for before/after comparison.

- [ ] **Step 4: Confirm git availability**

Run:

```bash
git status --short
```

Expected in this checkout: `fatal: not a git repository`. If git is initialized before execution, use the commit steps below normally.

---

### Task 1: Add Reranking Dependency And Retrieval Settings Tests

**Files:**
- Modify: `/Users/danil/Public/imperial/pyproject.toml`
- Modify: `/Users/danil/Public/imperial/tests/test_dependencies.py`
- Create: `/Users/danil/Public/imperial/tests/test_retrieval.py`
- Create: `/Users/danil/Public/imperial/src/imperial_rag/retrieval.py`

- [ ] **Step 1: Write failing dependency test**

Append this test to `/Users/danil/Public/imperial/tests/test_dependencies.py`:

```python
def test_project_includes_cohere_reranking_dependency():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {_normalize_dependency_name(dependency) for dependency in pyproject["project"]["dependencies"]}

    assert "langchain-cohere" in dependencies
```

- [ ] **Step 2: Write failing retrieval settings tests**

Create `/Users/danil/Public/imperial/tests/test_retrieval.py`:

```python
from __future__ import annotations

from langchain_core.documents import Document

from imperial_rag.retrieval import RetrievalSettings


def test_retrieval_settings_defaults_match_accuracy_spec(monkeypatch):
    for name in (
        "IMPERIAL_RAG_CHUNK_SIZE",
        "IMPERIAL_RAG_CHUNK_OVERLAP",
        "IMPERIAL_RAG_VECTOR_FETCH_K",
        "IMPERIAL_RAG_VECTOR_K",
        "IMPERIAL_RAG_KEYWORD_LIMIT",
        "IMPERIAL_RAG_RERANK_INPUT_LIMIT",
        "IMPERIAL_RAG_RERANK_TOP_N",
        "IMPERIAL_RAG_NEIGHBOR_WINDOW",
        "IMPERIAL_RAG_FINAL_EVIDENCE_MIN",
        "IMPERIAL_RAG_FINAL_EVIDENCE_MAX",
        "IMPERIAL_RAG_MMR_LAMBDA_MULT",
        "IMPERIAL_RAG_PRIMARY_RERANKER",
        "IMPERIAL_RAG_FALLBACK_RERANKER",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = RetrievalSettings.from_env()

    assert settings.chunk_size == 400
    assert settings.chunk_overlap == 50
    assert settings.vector_fetch_k == 80
    assert settings.vector_k == 32
    assert settings.keyword_limit == 40
    assert settings.rerank_input_limit == 60
    assert settings.rerank_top_n == 12
    assert settings.neighbor_window == 1
    assert settings.final_evidence_min == 18
    assert settings.final_evidence_max == 24
    assert settings.mmr_lambda_mult == 0.4
    assert settings.primary_reranker == "cohere:rerank-v3.5"
    assert settings.fallback_reranker == "cohere:rerank-multilingual-v3.0"


def test_retrieval_settings_read_environment_overrides(monkeypatch):
    monkeypatch.setenv("IMPERIAL_RAG_CHUNK_SIZE", "500")
    monkeypatch.setenv("IMPERIAL_RAG_CHUNK_OVERLAP", "75")
    monkeypatch.setenv("IMPERIAL_RAG_VECTOR_FETCH_K", "90")
    monkeypatch.setenv("IMPERIAL_RAG_VECTOR_K", "30")
    monkeypatch.setenv("IMPERIAL_RAG_KEYWORD_LIMIT", "35")
    monkeypatch.setenv("IMPERIAL_RAG_RERANK_INPUT_LIMIT", "55")
    monkeypatch.setenv("IMPERIAL_RAG_RERANK_TOP_N", "10")
    monkeypatch.setenv("IMPERIAL_RAG_NEIGHBOR_WINDOW", "2")
    monkeypatch.setenv("IMPERIAL_RAG_FINAL_EVIDENCE_MIN", "14")
    monkeypatch.setenv("IMPERIAL_RAG_FINAL_EVIDENCE_MAX", "20")
    monkeypatch.setenv("IMPERIAL_RAG_MMR_LAMBDA_MULT", "0.65")
    monkeypatch.setenv("IMPERIAL_RAG_PRIMARY_RERANKER", "cohere:rerank-v3.5")
    monkeypatch.setenv("IMPERIAL_RAG_FALLBACK_RERANKER", "cohere:rerank-multilingual-v3.0")

    settings = RetrievalSettings.from_env()

    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 75
    assert settings.vector_fetch_k == 90
    assert settings.vector_k == 30
    assert settings.keyword_limit == 35
    assert settings.rerank_input_limit == 55
    assert settings.rerank_top_n == 10
    assert settings.neighbor_window == 2
    assert settings.final_evidence_min == 14
    assert settings.final_evidence_max == 20
    assert settings.mmr_lambda_mult == 0.65
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
uv run --extra dev python -m pytest tests/test_dependencies.py::test_project_includes_cohere_reranking_dependency tests/test_retrieval.py -q
```

Expected: FAIL because `langchain-cohere` is not in `pyproject.toml` and `imperial_rag.retrieval` does not exist.

- [ ] **Step 4: Add dependency**

In `/Users/danil/Public/imperial/pyproject.toml`, add this dependency inside `[project].dependencies`:

```toml
  "langchain-cohere",
```

Keep alphabetical order only if the surrounding list already follows it. The existing list is grouped by feature, so place it after `"langchain-qdrant",`.

- [ ] **Step 5: Create minimal retrieval settings implementation**

Create `/Users/danil/Public/imperial/src/imperial_rag/retrieval.py`:

```python
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from langchain_core.documents import Document


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw


@dataclass(frozen=True)
class RetrievalSettings:
    chunk_size: int = 400
    chunk_overlap: int = 50
    vector_fetch_k: int = 80
    vector_k: int = 32
    keyword_limit: int = 40
    rerank_input_limit: int = 60
    rerank_top_n: int = 12
    neighbor_window: int = 1
    final_evidence_min: int = 18
    final_evidence_max: int = 24
    mmr_lambda_mult: float = 0.4
    primary_reranker: str = "cohere:rerank-v3.5"
    fallback_reranker: str = "cohere:rerank-multilingual-v3.0"

    @classmethod
    def from_env(cls) -> "RetrievalSettings":
        return cls(
            chunk_size=_env_int("IMPERIAL_RAG_CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_env_int("IMPERIAL_RAG_CHUNK_OVERLAP", cls.chunk_overlap),
            vector_fetch_k=_env_int("IMPERIAL_RAG_VECTOR_FETCH_K", cls.vector_fetch_k),
            vector_k=_env_int("IMPERIAL_RAG_VECTOR_K", cls.vector_k),
            keyword_limit=_env_int("IMPERIAL_RAG_KEYWORD_LIMIT", cls.keyword_limit),
            rerank_input_limit=_env_int("IMPERIAL_RAG_RERANK_INPUT_LIMIT", cls.rerank_input_limit),
            rerank_top_n=_env_int("IMPERIAL_RAG_RERANK_TOP_N", cls.rerank_top_n),
            neighbor_window=_env_int("IMPERIAL_RAG_NEIGHBOR_WINDOW", cls.neighbor_window),
            final_evidence_min=_env_int("IMPERIAL_RAG_FINAL_EVIDENCE_MIN", cls.final_evidence_min),
            final_evidence_max=_env_int("IMPERIAL_RAG_FINAL_EVIDENCE_MAX", cls.final_evidence_max),
            mmr_lambda_mult=_env_float("IMPERIAL_RAG_MMR_LAMBDA_MULT", cls.mmr_lambda_mult),
            primary_reranker=_env_str("IMPERIAL_RAG_PRIMARY_RERANKER", cls.primary_reranker),
            fallback_reranker=_env_str("IMPERIAL_RAG_FALLBACK_RERANKER", cls.fallback_reranker),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

- [ ] **Step 6: Sync dependencies**

Run:

```bash
uv sync --extra dev
```

Expected: command exits `0` and `uv.lock` updates.

- [ ] **Step 7: Run tests to verify they pass**

Run:

```bash
uv run --extra dev python -m pytest tests/test_dependencies.py::test_project_includes_cohere_reranking_dependency tests/test_retrieval.py::test_retrieval_settings_defaults_match_accuracy_spec tests/test_retrieval.py::test_retrieval_settings_read_environment_overrides -q
```

Expected: PASS.

- [ ] **Step 8: Commit checkpoint**

Run:

```bash
git add pyproject.toml uv.lock src/imperial_rag/retrieval.py tests/test_retrieval.py tests/test_dependencies.py
git commit -m "feat: add retrieval accuracy settings"
```

Expected in a git repo: commit succeeds. Expected in this checkout until git is initialized: `fatal: not a git repository`.

---

### Task 2: Apply 400/50 Chunk Defaults To Ingestion

**Files:**
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/chunking.py`
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/pipeline.py`
- Modify: `/Users/danil/Public/imperial/tests/test_chunking.py`
- Modify: `/Users/danil/Public/imperial/tests/test_pipeline.py`

- [ ] **Step 1: Add failing default chunk-size test**

Append this test to `/Users/danil/Public/imperial/tests/test_chunking.py`:

```python
def test_build_chunks_defaults_to_accuracy_spec_size_and_overlap():
    source = Document(
        page_content=("А" * 400) + ("Б" * 100),
        metadata={"file_id": "file123", "relative_path": "policy.docx", "source_type": "body"},
    )

    chunks = build_chunks([source])

    assert len(chunks) == 2
    assert len(chunks[0].page_content) <= 400
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["chunk_index"] == 1
```

- [ ] **Step 2: Add failing pipeline env override test**

In `/Users/danil/Public/imperial/tests/test_pipeline.py`, update `_install_fake_dependencies()` so the fake `build_chunks` records keyword args:

```python
    chunking = ModuleType("imperial_rag.chunking")

    def build_chunks(documents, chunk_size=1000, chunk_overlap=150):
        build_chunks.calls.append({"chunk_size": chunk_size, "chunk_overlap": chunk_overlap})
        return [chunk]

    build_chunks.calls = []
    chunking.build_chunks = build_chunks
```

Then append this test to `/Users/danil/Public/imperial/tests/test_pipeline.py`:

```python
def test_run_ingestion_uses_retrieval_chunk_settings(tmp_path, monkeypatch):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.txt").write_text("Регламент возврата брака.", encoding="utf-8")
    monkeypatch.setenv("IMPERIAL_RAG_CHUNK_SIZE", "400")
    monkeypatch.setenv("IMPERIAL_RAG_CHUNK_OVERLAP", "50")
    _install_fake_dependencies(monkeypatch)

    run_ingestion(settings=FakeSettings(tmp_path), enable_ocr=False, index_vectors=False)

    build_chunks = sys.modules["imperial_rag.chunking"].build_chunks
    assert build_chunks.calls == [{"chunk_size": 400, "chunk_overlap": 50}]
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
uv run --extra dev python -m pytest tests/test_chunking.py::test_build_chunks_defaults_to_accuracy_spec_size_and_overlap tests/test_pipeline.py::test_run_ingestion_uses_retrieval_chunk_settings -q
```

Expected: FAIL because defaults are still `1000/150` and pipeline does not pass retrieval settings.

- [ ] **Step 4: Change chunk defaults**

In `/Users/danil/Public/imperial/src/imperial_rag/chunking.py`, change:

```python
def build_chunks(documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Document]:
```

to:

```python
def build_chunks(documents: list[Document], chunk_size: int = 400, chunk_overlap: int = 50) -> list[Document]:
```

- [ ] **Step 5: Pass retrieval settings through pipeline**

In `/Users/danil/Public/imperial/src/imperial_rag/pipeline.py`, add `RetrievalSettings` to `_load_dependencies()`:

```python
    from imperial_rag.retrieval import RetrievalSettings
```

and add it to the returned dependency dict:

```python
        "RetrievalSettings": RetrievalSettings,
```

Then replace:

```python
    chunks = list(deps["build_chunks"](extracted_documents))
```

with:

```python
    retrieval_settings = deps["RetrievalSettings"].from_env()
    chunks = list(
        deps["build_chunks"](
            extracted_documents,
            chunk_size=retrieval_settings.chunk_size,
            chunk_overlap=retrieval_settings.chunk_overlap,
        )
    )
```

- [ ] **Step 6: Run targeted tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_chunking.py tests/test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 7: Run full test suite**

Run:

```bash
uv run --extra dev python -m pytest -q
```

Expected: PASS.

- [ ] **Step 8: Commit checkpoint**

Run:

```bash
git add src/imperial_rag/chunking.py src/imperial_rag/pipeline.py tests/test_chunking.py tests/test_pipeline.py
git commit -m "feat: apply accuracy chunk settings"
```

Expected in a git repo: commit succeeds. Expected in this checkout until git is initialized: `fatal: not a git repository`.

---

### Task 3: Add Keyword Rank Metadata

**Files:**
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/indexing.py`
- Modify: `/Users/danil/Public/imperial/tests/test_indexing.py`

- [ ] **Step 1: Write failing keyword score tests**

Append these tests to `/Users/danil/Public/imperial/tests/test_indexing.py`:

```python
def test_keyword_index_search_with_scores_adds_rank_metadata(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    docs = [
        Document(page_content="Регламент возврата брака", metadata={"citation_id": "a"}),
        Document(page_content="Возврат товара на склад", metadata={"citation_id": "b"}),
    ]
    index.index_documents(docs)

    hits = index.search_with_scores("возврат", limit=5)

    assert [hit.document.metadata["_keyword_rank"] for hit in hits] == list(range(len(hits)))
    assert all("_keyword_score" in hit.document.metadata for hit in hits)
    assert all(isinstance(hit.score, float) for hit in hits)


def test_keyword_index_search_preserves_document_only_api(tmp_path: Path) -> None:
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    index.index_documents([Document(page_content="Регламент возврата брака", metadata={"citation_id": "a"})])

    results = index.search("возврат", limit=5)

    assert [result.metadata["citation_id"] for result in results] == ["a"]
    assert results[0].metadata["_keyword_rank"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --extra dev python -m pytest tests/test_indexing.py::test_keyword_index_search_with_scores_adds_rank_metadata tests/test_indexing.py::test_keyword_index_search_preserves_document_only_api -q
```

Expected: FAIL because `search_with_scores()` does not exist.

- [ ] **Step 3: Implement `search_with_scores()`**

In `/Users/danil/Public/imperial/src/imperial_rag/indexing.py`, replace the current `search()` method in `KeywordIndex` with these two methods:

```python
    def search(self, query: str, limit: int = 5, k: int | None = None) -> list[Document]:
        return [hit.document for hit in self.search_with_scores(query, limit=limit, k=k)]

    def search_with_scores(self, query: str, limit: int = 5, k: int | None = None) -> list[KeywordHit]:
        resolved_limit = k if k is not None else limit
        query_tokens = [token for token in normalize_search_text(query).split() if token]
        if not query_tokens:
            return []
        if self._uses_fts:
            match_query = build_fts_match_query(query)
            rows = self._conn.execute(
                """
                SELECT text, metadata, bm25(chunks_fts) AS score
                FROM chunks_fts
                WHERE normalized_text MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (match_query, resolved_limit),
            ).fetchall()
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

        where_clause = " AND ".join("normalized_text LIKE ?" for _ in query_tokens)
        rows = self._conn.execute(
            f"SELECT text, metadata FROM chunks WHERE {where_clause} LIMIT ?",
            [f"%{token}%" for token in query_tokens] + [resolved_limit],
        ).fetchall()
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
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_indexing.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint**

Run:

```bash
git add src/imperial_rag/indexing.py tests/test_indexing.py
git commit -m "feat: expose keyword rank metadata"
```

Expected in a git repo: commit succeeds. Expected in this checkout until git is initialized: `fatal: not a git repository`.

---

### Task 4: Add Candidate Merge And Deterministic Fallback Ranking

**Files:**
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/retrieval.py`
- Modify: `/Users/danil/Public/imperial/tests/test_retrieval.py`

- [ ] **Step 1: Write failing merge and fallback-rank tests**

Append these imports to `/Users/danil/Public/imperial/tests/test_retrieval.py`:

```python
from imperial_rag.retrieval import CandidateMerger, FallbackRanker
```

Append these tests:

```python
def test_candidate_merger_deduplicates_by_citation_and_content():
    same_vector = Document(page_content="Возврат брака оформляется актом.", metadata={"citation_id": "same"})
    same_keyword = Document(page_content="Возврат брака оформляется актом.", metadata={"citation_id": "same"})
    content_duplicate = Document(page_content=" Возврат брака оформляется актом. ", metadata={"citation_id": "different"})
    unique = Document(page_content="Склад принимает товар по накладной.", metadata={"citation_id": "unique"})

    merged = CandidateMerger().merge([same_vector, content_duplicate], [same_keyword, unique])

    assert [doc.metadata["citation_id"] for doc in merged] == ["same", "unique"]


def test_fallback_ranker_prioritizes_keyword_and_filename_matches():
    docs = [
        Document(
            page_content="Общие правила склада.",
            metadata={"citation_id": "warehouse", "_vector_rank": 0, "file_name": "warehouse.docx"},
        ),
        Document(
            page_content="Порядок возврата брака.",
            metadata={"citation_id": "return", "_keyword_rank": 0, "file_name": "Регламент возврата брака.docx"},
        ),
        Document(
            page_content="Возврат брака оформляется актом.",
            metadata={"citation_id": "body", "_vector_rank": 2, "_keyword_rank": 2, "source_type": "body"},
        ),
    ]

    ranked = FallbackRanker().rank("возврат брака", docs, top_n=3)

    assert [doc.metadata["citation_id"] for doc in ranked] == ["return", "body", "warehouse"]
    assert ranked[0].metadata["_fallback_score"] > ranked[1].metadata["_fallback_score"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --extra dev python -m pytest tests/test_retrieval.py::test_candidate_merger_deduplicates_by_citation_and_content tests/test_retrieval.py::test_fallback_ranker_prioritizes_keyword_and_filename_matches -q
```

Expected: FAIL because `CandidateMerger` and `FallbackRanker` do not exist.

- [ ] **Step 3: Implement merge and fallback ranking**

Append this code to `/Users/danil/Public/imperial/src/imperial_rag/retrieval.py`:

```python
def _document_key(document: Document) -> str:
    return str(document.metadata.get("citation_id") or document.metadata.get("chunk_id") or document.page_content)


def _content_key(document: Document) -> str:
    return " ".join(document.page_content.split()).casefold()


def _query_tokens(query: str) -> list[str]:
    return [token for token in query.casefold().replace("-", " ").split() if token]


def _searchable_text(document: Document) -> str:
    metadata = document.metadata
    return " ".join(
        [
            document.page_content,
            str(metadata.get("file_name", "")),
            str(metadata.get("relative_path", "")),
            str(metadata.get("section_heading", "")),
            str(metadata.get("source_type", "")),
        ]
    ).casefold()


class CandidateMerger:
    def merge(self, vector_docs: list[Document], keyword_docs: list[Document]) -> list[Document]:
        merged: list[Document] = []
        seen_ids: set[str] = set()
        seen_contents: set[str] = set()
        for document in [*vector_docs, *keyword_docs]:
            key = _document_key(document)
            content_key = _content_key(document)
            if key in seen_ids or content_key in seen_contents:
                continue
            seen_ids.add(key)
            seen_contents.add(content_key)
            merged.append(document)
        return merged


class FallbackRanker:
    def rank(self, query: str, documents: list[Document], top_n: int) -> list[Document]:
        scored = [(self._score(query, document), document) for document in documents]
        ranked: list[Document] = []
        for score, document in sorted(scored, key=lambda item: item[0], reverse=True)[:top_n]:
            metadata = dict(document.metadata)
            metadata["_fallback_score"] = score
            ranked.append(Document(page_content=document.page_content, metadata=metadata))
        return ranked

    def _score(self, query: str, document: Document) -> float:
        metadata = document.metadata
        searchable = _searchable_text(document)
        tokens = _query_tokens(query)
        score = 0.0

        vector_rank = metadata.get("_vector_rank")
        if isinstance(vector_rank, int):
            score += 1.0 / (vector_rank + 1)
        vector_score = metadata.get("_vector_score")
        if isinstance(vector_score, (int, float)):
            score += float(vector_score)

        keyword_rank = metadata.get("_keyword_rank")
        if isinstance(keyword_rank, int):
            score += 1.4 / (keyword_rank + 1)
        keyword_score = metadata.get("_keyword_score")
        if isinstance(keyword_score, (int, float)):
            score += 0.05 / (abs(float(keyword_score)) + 1.0)

        if tokens and all(token in searchable for token in tokens):
            score += 2.0
        path_text = " ".join(
            [
                str(metadata.get("file_name", "")),
                str(metadata.get("relative_path", "")),
            ]
        ).casefold()
        if tokens and all(token in path_text for token in tokens):
            score += 1.5
        if metadata.get("source_type") in {"body", "table", "pdf_page", "sheet"}:
            score += 0.25
        if metadata.get("duplicate_group_id"):
            score -= 0.15
        return score
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_retrieval.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint**

Run:

```bash
git add src/imperial_rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: add deterministic retrieval ranking"
```

Expected in a git repo: commit succeeds. Expected in this checkout until git is initialized: `fatal: not a git repository`.

---

### Task 5: Add Hybrid Retriever And Diagnostics

**Files:**
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/retrieval.py`
- Modify: `/Users/danil/Public/imperial/tests/test_retrieval.py`

- [ ] **Step 1: Write failing hybrid retriever tests**

Append these imports to `/Users/danil/Public/imperial/tests/test_retrieval.py`:

```python
from imperial_rag.retrieval import HybridRetriever
```

Append these tests:

```python
class FakeVectorSearch:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def max_marginal_relevance_search(self, query, k, fetch_k, lambda_mult):
        self.calls.append({"query": query, "k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult})
        return self.docs[:k]


class FakeKeywordSearch:
    def __init__(self, docs):
        self.docs = docs
        self.calls = []

    def search_with_scores(self, query, limit):
        self.calls.append({"query": query, "limit": limit})

        class Hit:
            def __init__(self, document):
                self.document = document
                self.score = 0.0

        return [Hit(document) for document in self.docs[:limit]]


def test_hybrid_retriever_uses_configured_candidate_counts():
    vector_docs = [
        Document(page_content=f"vector {index}", metadata={"citation_id": f"v{index}"})
        for index in range(40)
    ]
    keyword_docs = [
        Document(page_content=f"keyword {index}", metadata={"citation_id": f"k{index}"})
        for index in range(45)
    ]
    vector = FakeVectorSearch(vector_docs)
    keyword = FakeKeywordSearch(keyword_docs)
    settings = RetrievalSettings(vector_fetch_k=80, vector_k=32, keyword_limit=40)

    result = HybridRetriever(vector_search=vector, keyword_search=keyword, settings=settings).retrieve("возврат брака")

    assert len(result.vector_docs) == 32
    assert len(result.keyword_docs) == 40
    assert result.diagnostics["vector_candidates"] == 32
    assert result.diagnostics["keyword_candidates"] == 40
    assert result.diagnostics["vector_search_status"] == "ok"
    assert result.diagnostics["keyword_search_status"] == "ok"
    assert vector.calls == [{"query": "возврат брака", "k": 32, "fetch_k": 80, "lambda_mult": 0.4}]
    assert keyword.calls == [{"query": "возврат брака", "limit": 40}]


def test_hybrid_retriever_degrades_when_vector_search_fails():
    class BrokenVector:
        def max_marginal_relevance_search(self, query, k, fetch_k, lambda_mult):
            raise RuntimeError("qdrant unavailable")

    keyword_docs = [Document(page_content="keyword", metadata={"citation_id": "k"})]

    result = HybridRetriever(
        vector_search=BrokenVector(),
        keyword_search=FakeKeywordSearch(keyword_docs),
        settings=RetrievalSettings(),
    ).retrieve("возврат")

    assert result.vector_docs == []
    assert [doc.metadata["citation_id"] for doc in result.keyword_docs] == ["k"]
    assert result.diagnostics["vector_search_status"] == "unavailable"
    assert result.diagnostics["keyword_search_status"] == "ok"
    assert "vector_search_failed" in result.diagnostics["fallbacks"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --extra dev python -m pytest tests/test_retrieval.py::test_hybrid_retriever_uses_configured_candidate_counts tests/test_retrieval.py::test_hybrid_retriever_degrades_when_vector_search_fails -q
```

Expected: FAIL because `HybridRetriever` does not exist.

- [ ] **Step 3: Implement hybrid retriever**

Append this code to `/Users/danil/Public/imperial/src/imperial_rag/retrieval.py`:

```python
@dataclass(frozen=True)
class RetrievalCandidateResult:
    vector_docs: list[Document]
    keyword_docs: list[Document]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, vector_search: Any, keyword_search: Any, settings: RetrievalSettings | None = None) -> None:
        self.vector_search = vector_search
        self.keyword_search = keyword_search
        self.settings = settings or RetrievalSettings.from_env()

    def retrieve(self, query: str) -> RetrievalCandidateResult:
        fallbacks: list[str] = []
        vector_status = "ok"
        keyword_status = "ok"
        vector_docs: list[Document] = []
        keyword_docs: list[Document] = []

        try:
            vector_docs = self._vector_docs(query)
        except Exception:
            vector_status = "unavailable"
            fallbacks.append("vector_search_failed")
            vector_docs = []

        try:
            keyword_docs = self._keyword_docs(query)
        except Exception:
            keyword_status = "unavailable"
            fallbacks.append("keyword_search_failed")
            keyword_docs = []

        if not vector_docs and vector_status == "ok":
            vector_status = "empty"
        if not keyword_docs and keyword_status == "ok":
            keyword_status = "empty"

        return RetrievalCandidateResult(
            vector_docs=vector_docs,
            keyword_docs=keyword_docs,
            diagnostics={
                "vector_candidates": len(vector_docs),
                "keyword_candidates": len(keyword_docs),
                "vector_search_status": vector_status,
                "keyword_search_status": keyword_status,
                "fallbacks": fallbacks,
            },
        )

    def _vector_docs(self, query: str) -> list[Document]:
        if hasattr(self.vector_search, "max_marginal_relevance_search"):
            docs = self.vector_search.max_marginal_relevance_search(
                query,
                k=self.settings.vector_k,
                fetch_k=self.settings.vector_fetch_k,
                lambda_mult=self.settings.mmr_lambda_mult,
            )
        elif hasattr(self.vector_search, "similarity_search"):
            docs = self.vector_search.similarity_search(query, k=self.settings.vector_k)
        else:
            docs = []
        return [
            Document(page_content=doc.page_content, metadata={**dict(doc.metadata), "_vector_rank": rank})
            for rank, doc in enumerate(docs)
        ]

    def _keyword_docs(self, query: str) -> list[Document]:
        if hasattr(self.keyword_search, "search_with_scores"):
            hits = self.keyword_search.search_with_scores(query, limit=self.settings.keyword_limit)
            return [hit.document for hit in hits]
        return self.keyword_search.search(query, limit=self.settings.keyword_limit)
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_retrieval.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint**

Run:

```bash
git add src/imperial_rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: add hybrid retrieval diagnostics"
```

Expected in a git repo: commit succeeds. Expected in this checkout until git is initialized: `fatal: not a git repository`.

---

### Task 6: Add Reranker With Deterministic Fallback

**Files:**
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/retrieval.py`
- Modify: `/Users/danil/Public/imperial/tests/test_retrieval.py`

- [ ] **Step 1: Write failing reranker tests**

Append this import to `/Users/danil/Public/imperial/tests/test_retrieval.py`:

```python
from imperial_rag.retrieval import Reranker
```

Append these tests:

```python
def test_reranker_uses_deterministic_fallback_without_api_key(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    docs = [
        Document(page_content="Общие правила склада.", metadata={"citation_id": "warehouse"}),
        Document(page_content="Порядок возврата брака.", metadata={"citation_id": "return", "_keyword_rank": 0}),
    ]
    diagnostics = {"fallbacks": []}

    reranked = Reranker(settings=RetrievalSettings(rerank_top_n=1)).rerank("возврат брака", docs, diagnostics)

    assert [doc.metadata["citation_id"] for doc in reranked] == ["return"]
    assert diagnostics["reranker"] == "fallback:deterministic"
    assert "reranker_missing_api_key" in diagnostics["fallbacks"]


def test_reranker_backfills_when_primary_returns_too_few(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    docs = [
        Document(page_content="Порядок возврата брака.", metadata={"citation_id": "return", "_keyword_rank": 0}),
        Document(page_content="Возврат оформляется актом.", metadata={"citation_id": "act", "_keyword_rank": 1}),
    ]
    diagnostics = {"fallbacks": []}

    reranked = Reranker(settings=RetrievalSettings(rerank_top_n=3)).rerank("возврат брака", docs, diagnostics)

    assert [doc.metadata["citation_id"] for doc in reranked] == ["return", "act"]
    assert diagnostics["reranked_candidates"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --extra dev python -m pytest tests/test_retrieval.py::test_reranker_uses_deterministic_fallback_without_api_key tests/test_retrieval.py::test_reranker_backfills_when_primary_returns_too_few -q
```

Expected: FAIL because `Reranker` does not exist.

- [ ] **Step 3: Implement reranker**

Append this code to `/Users/danil/Public/imperial/src/imperial_rag/retrieval.py`:

```python
def _cohere_model_name(configured: str) -> str:
    prefix = "cohere:"
    return configured[len(prefix):] if configured.startswith(prefix) else configured


class Reranker:
    def __init__(self, settings: RetrievalSettings | None = None) -> None:
        self.settings = settings or RetrievalSettings.from_env()
        self._fallback = FallbackRanker()

    def rerank(self, query: str, documents: list[Document], diagnostics: dict[str, Any]) -> list[Document]:
        candidates = documents[: self.settings.rerank_input_limit]
        diagnostics["rerank_input"] = len(candidates)
        if not candidates:
            diagnostics["reranker"] = "none"
            diagnostics["reranked_candidates"] = 0
            return []
        if not os.environ.get("COHERE_API_KEY"):
            diagnostics.setdefault("fallbacks", []).append("reranker_missing_api_key")
            return self._fallback_rerank(query, candidates, diagnostics)

        for configured_model in (self.settings.primary_reranker, self.settings.fallback_reranker):
            try:
                reranked = self._cohere_rerank(query, candidates, _cohere_model_name(configured_model))
            except Exception:
                diagnostics.setdefault("fallbacks", []).append(f"reranker_failed:{configured_model}")
                continue
            diagnostics["reranker"] = configured_model
            diagnostics["reranked_candidates"] = len(reranked)
            return self._backfill(query, reranked, candidates)

        return self._fallback_rerank(query, candidates, diagnostics)

    def _cohere_rerank(self, query: str, documents: list[Document], model_name: str) -> list[Document]:
        from langchain_cohere import CohereRerank

        compressor = CohereRerank(model=model_name, top_n=self.settings.rerank_top_n)
        compressed = compressor.compress_documents(documents=documents, query=query)
        return list(compressed)

    def _fallback_rerank(self, query: str, documents: list[Document], diagnostics: dict[str, Any]) -> list[Document]:
        diagnostics["reranker"] = "fallback:deterministic"
        reranked = self._fallback.rank(query, documents, top_n=self.settings.rerank_top_n)
        diagnostics["reranked_candidates"] = len(reranked)
        return self._backfill(query, reranked, documents)

    def _backfill(self, query: str, reranked: list[Document], candidates: list[Document]) -> list[Document]:
        if len(reranked) >= min(self.settings.rerank_top_n, len(candidates)):
            return reranked[: self.settings.rerank_top_n]
        seen = {_document_key(document) for document in reranked}
        fallback = self._fallback.rank(query, candidates, top_n=len(candidates))
        combined = list(reranked)
        for document in fallback:
            if _document_key(document) in seen:
                continue
            combined.append(document)
            seen.add(_document_key(document))
            if len(combined) >= self.settings.rerank_top_n:
                break
        return combined
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_retrieval.py -q
```

Expected: PASS.

- [ ] **Step 5: Run import smoke**

Run:

```bash
uv run --extra dev python - <<'PY'
from imperial_rag.retrieval import Reranker, RetrievalSettings
print(Reranker(settings=RetrievalSettings()).settings.primary_reranker)
PY
```

Expected: output contains `cohere:rerank-v3.5`.

- [ ] **Step 6: Commit checkpoint**

Run:

```bash
git add src/imperial_rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: add reranker fallback path"
```

Expected in a git repo: commit succeeds. Expected in this checkout until git is initialized: `fatal: not a git repository`.

---

### Task 7: Add Neighbor Expansion And Final Evidence Selection

**Files:**
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/retrieval.py`
- Modify: `/Users/danil/Public/imperial/tests/test_retrieval.py`

- [ ] **Step 1: Write failing neighbor expansion tests**

Append these imports to `/Users/danil/Public/imperial/tests/test_retrieval.py`:

```python
from imperial_rag.retrieval import ChunkNeighborStore, EvidenceSelector, NeighborExpander
```

Append these tests:

```python
def test_neighbor_expander_adds_previous_and_next_chunks():
    chunks = [
        Document(page_content="previous", metadata={"citation_id": "c0", "file_id": "f", "source_type": "body", "chunk_index": 0}),
        Document(page_content="hit", metadata={"citation_id": "c1", "file_id": "f", "source_type": "body", "chunk_index": 1}),
        Document(page_content="next", metadata={"citation_id": "c2", "file_id": "f", "source_type": "body", "chunk_index": 2}),
    ]
    store = ChunkNeighborStore(chunks)

    expanded = NeighborExpander(store=store, settings=RetrievalSettings(neighbor_window=1, final_evidence_max=10)).expand([chunks[1]])

    assert [doc.metadata["citation_id"] for doc in expanded] == ["c1", "c0", "c2"]


def test_evidence_selector_caps_final_evidence():
    docs = [
        Document(page_content=f"doc {index}", metadata={"citation_id": f"c{index}"})
        for index in range(30)
    ]

    selected = EvidenceSelector(settings=RetrievalSettings(final_evidence_max=24)).select(docs)

    assert len(selected) == 24
    assert selected[0].metadata["citation_id"] == "c0"
    assert selected[-1].metadata["citation_id"] == "c23"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --extra dev python -m pytest tests/test_retrieval.py::test_neighbor_expander_adds_previous_and_next_chunks tests/test_retrieval.py::test_evidence_selector_caps_final_evidence -q
```

Expected: FAIL because neighbor classes do not exist.

- [ ] **Step 3: Implement neighbor expansion**

Append this code to `/Users/danil/Public/imperial/src/imperial_rag/retrieval.py`:

```python
class ChunkNeighborStore:
    def __init__(self, chunks: list[Document]) -> None:
        self._chunks: dict[tuple[str, str, int], Document] = {}
        for chunk in chunks:
            metadata = chunk.metadata
            file_id = metadata.get("file_id")
            source_type = metadata.get("source_type")
            chunk_index = metadata.get("chunk_index")
            if file_id is None or source_type is None or not isinstance(chunk_index, int):
                continue
            self._chunks[(str(file_id), str(source_type), chunk_index)] = chunk

    @classmethod
    def from_jsonl(cls, path) -> "ChunkNeighborStore":
        import json
        from pathlib import Path

        chunk_path = Path(path)
        if not chunk_path.exists():
            return cls([])
        chunks: list[Document] = []
        for line in chunk_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            chunks.append(Document(page_content=str(row["page_content"]), metadata=dict(row["metadata"])))
        return cls(chunks)

    def neighbors(self, document: Document, window: int) -> list[Document]:
        metadata = document.metadata
        file_id = metadata.get("file_id")
        source_type = metadata.get("source_type")
        chunk_index = metadata.get("chunk_index")
        if file_id is None or source_type is None or not isinstance(chunk_index, int):
            return []
        neighbors: list[Document] = []
        for offset in range(1, window + 1):
            previous = self._chunks.get((str(file_id), str(source_type), chunk_index - offset))
            next_chunk = self._chunks.get((str(file_id), str(source_type), chunk_index + offset))
            if previous is not None:
                neighbors.append(previous)
            if next_chunk is not None:
                neighbors.append(next_chunk)
        return neighbors


class NeighborExpander:
    def __init__(self, store: ChunkNeighborStore, settings: RetrievalSettings | None = None) -> None:
        self.store = store
        self.settings = settings or RetrievalSettings.from_env()

    def expand(self, documents: list[Document]) -> list[Document]:
        expanded: list[Document] = []
        seen: set[str] = set()
        for document in documents:
            for candidate in [document, *self.store.neighbors(document, self.settings.neighbor_window)]:
                key = _document_key(candidate)
                if key in seen:
                    continue
                expanded.append(candidate)
                seen.add(key)
                if len(expanded) >= self.settings.final_evidence_max:
                    return expanded
        return expanded


class EvidenceSelector:
    def __init__(self, settings: RetrievalSettings | None = None) -> None:
        self.settings = settings or RetrievalSettings.from_env()

    def select(self, documents: list[Document]) -> list[Document]:
        return documents[: self.settings.final_evidence_max]
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_retrieval.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint**

Run:

```bash
git add src/imperial_rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: expand reranked chunk context"
```

Expected in a git repo: commit succeeds. Expected in this checkout until git is initialized: `fatal: not a git repository`.

---

### Task 8: Compose End-To-End Retrieval Pipeline

**Files:**
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/retrieval.py`
- Modify: `/Users/danil/Public/imperial/tests/test_retrieval.py`

- [ ] **Step 1: Write failing retrieval service test**

Append these imports to `/Users/danil/Public/imperial/tests/test_retrieval.py`:

```python
from imperial_rag.retrieval import RetrievalService
```

Append this test:

```python
def test_retrieval_service_returns_final_evidence_and_diagnostics(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    vector_docs = [
        Document(page_content="vector return", metadata={"citation_id": "v", "file_id": "f", "source_type": "body", "chunk_index": 0})
    ]
    keyword_docs = [
        Document(page_content="Порядок возврата брака", metadata={"citation_id": "k", "file_id": "f", "source_type": "body", "chunk_index": 1, "_keyword_rank": 0})
    ]
    all_chunks = [
        vector_docs[0],
        keyword_docs[0],
        Document(page_content="neighbor", metadata={"citation_id": "n", "file_id": "f", "source_type": "body", "chunk_index": 2}),
    ]
    service = RetrievalService(
        vector_search=FakeVectorSearch(vector_docs),
        keyword_search=FakeKeywordSearch(keyword_docs),
        neighbor_store=ChunkNeighborStore(all_chunks),
        settings=RetrievalSettings(rerank_top_n=1, final_evidence_max=3),
    )

    result = service.retrieve("возврат брака")

    assert [doc.metadata["citation_id"] for doc in result.evidence] == ["k", "v", "n"]
    assert result.diagnostics["merged_candidates"] == 2
    assert result.diagnostics["final_evidence"] == 3
    assert result.diagnostics["reranker"] == "fallback:deterministic"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --extra dev python -m pytest tests/test_retrieval.py::test_retrieval_service_returns_final_evidence_and_diagnostics -q
```

Expected: FAIL because `RetrievalService` does not exist.

- [ ] **Step 3: Implement retrieval service**

Append this code to `/Users/danil/Public/imperial/src/imperial_rag/retrieval.py`:

```python
@dataclass(frozen=True)
class RetrievalResult:
    evidence: list[Document]
    vector_docs: list[Document]
    keyword_docs: list[Document]
    diagnostics: dict[str, Any]


class RetrievalService:
    def __init__(
        self,
        vector_search: Any,
        keyword_search: Any,
        neighbor_store: ChunkNeighborStore | None = None,
        settings: RetrievalSettings | None = None,
    ) -> None:
        self.settings = settings or RetrievalSettings.from_env()
        self.hybrid = HybridRetriever(vector_search=vector_search, keyword_search=keyword_search, settings=self.settings)
        self.merger = CandidateMerger()
        self.reranker = Reranker(settings=self.settings)
        self.neighbor_store = neighbor_store or ChunkNeighborStore([])
        self.expander = NeighborExpander(store=self.neighbor_store, settings=self.settings)
        self.selector = EvidenceSelector(settings=self.settings)

    def retrieve(self, query: str) -> RetrievalResult:
        candidates = self.hybrid.retrieve(query)
        diagnostics = dict(candidates.diagnostics)
        merged = self.merger.merge(candidates.vector_docs, candidates.keyword_docs)
        diagnostics["merged_candidates"] = len(merged)
        reranked = self.reranker.rerank(query, merged, diagnostics)
        expanded = self.expander.expand(reranked)
        evidence = self.selector.select(expanded)
        diagnostics["final_evidence"] = len(evidence)
        return RetrievalResult(
            evidence=evidence,
            vector_docs=candidates.vector_docs,
            keyword_docs=candidates.keyword_docs,
            diagnostics=diagnostics,
        )
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_retrieval.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint**

Run:

```bash
git add src/imperial_rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: compose retrieval evidence service"
```

Expected in a git repo: commit succeeds. Expected in this checkout until git is initialized: `fatal: not a git repository`.

---

### Task 9: Preserve Retrieval Diagnostics In Query Workflow

**Files:**
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/workflows.py`
- Modify: `/Users/danil/Public/imperial/tests/test_workflows.py`

- [ ] **Step 1: Write failing workflow diagnostics test**

Append this test to `/Users/danil/Public/imperial/tests/test_workflows.py`:

```python
def test_query_workflow_preserves_retrieval_diagnostics():
    docs = [Document(page_content="Возврат брака оформляется актом.", metadata={"citation_id": "return"})]

    def retrieve(question):
        return {
            "retrieved_documents": docs,
            "vector_docs": [],
            "keyword_docs": docs,
            "retrieval": {
                "vector_candidates": 0,
                "keyword_candidates": 1,
                "merged_candidates": 1,
                "reranked_candidates": 1,
                "final_evidence": 1,
                "reranker": "fallback:deterministic",
                "fallbacks": ["reranker_missing_api_key"],
            },
        }

    workflow = build_query_workflow(
        retrieve=retrieve,
        generate=lambda question, retrieved_docs: "Возврат брака оформляется актом. [return]",
    )

    result = workflow.invoke({"question": "Как оформить возврат брака?"})

    assert result["retrieval"]["final_evidence"] == 1
    assert result["retrieval"]["reranker"] == "fallback:deterministic"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --extra dev python -m pytest tests/test_workflows.py::test_query_workflow_preserves_retrieval_diagnostics -q
```

Expected: FAIL because `retrieval` is not preserved in workflow state.

- [ ] **Step 3: Update workflow state and retrieve node**

In `/Users/danil/Public/imperial/src/imperial_rag/workflows.py`, add this field to `QueryState`:

```python
    retrieval: dict[str, Any]
```

Replace the first branch of `retrieve_node()`:

```python
        if retrieve is not None:
            evidence = _coerce_retrieved_documents(_call_with_supported_args(retrieve, query, state), query)
            return {"vector_candidates": evidence, "keyword_candidates": [], "evidence": evidence, "retrieved_documents": evidence}
```

with:

```python
        if retrieve is not None:
            retrieved = _call_with_supported_args(retrieve, query, state)
            evidence = _coerce_retrieved_documents(retrieved, query)
            payload: QueryState = {
                "vector_candidates": list(retrieved.get("vector_docs", [])) if isinstance(retrieved, Mapping) else evidence,
                "keyword_candidates": list(retrieved.get("keyword_docs", [])) if isinstance(retrieved, Mapping) else [],
                "evidence": evidence,
                "retrieved_documents": evidence,
            }
            if isinstance(retrieved, Mapping) and isinstance(retrieved.get("retrieval"), Mapping):
                payload["retrieval"] = dict(retrieved["retrieval"])
            return payload
```

In the default branch of `retrieve_node()`, add a retrieval diagnostic payload:

```python
            "retrieval": {
                "vector_candidates": len(vector_docs),
                "keyword_candidates": len(keyword_docs),
                "merged_candidates": len(evidence),
                "reranked_candidates": len(evidence),
                "final_evidence": len(evidence),
                "reranker": "legacy:rank_hybrid_candidates",
                "fallbacks": [],
            },
```

- [ ] **Step 4: Run workflow tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_workflows.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint**

Run:

```bash
git add src/imperial_rag/workflows.py tests/test_workflows.py
git commit -m "feat: preserve retrieval diagnostics"
```

Expected in a git repo: commit succeeds. Expected in this checkout until git is initialized: `fatal: not a git repository`.

---

### Task 10: Wire Runtime To Retrieval Service

**Files:**
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/runtime.py`
- Modify: `/Users/danil/Public/imperial/tests/test_runtime.py`

- [ ] **Step 1: Write failing runtime retrieval test**

Append this test to `/Users/danil/Public/imperial/tests/test_runtime.py`:

```python
def test_runtime_query_uses_retrieval_service(monkeypatch, tmp_path):
    from imperial_rag.config import Settings

    captured = {}

    class FakeRetrievalResult:
        evidence = []
        vector_docs = []
        keyword_docs = []
        diagnostics = {"final_evidence": 0, "reranker": "fake"}

    class FakeRetrievalService:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def retrieve(self, question):
            captured["question"] = question
            return FakeRetrievalResult()

    class FakeWorkflow:
        def invoke(self, state):
            retrieved = state["retrieve"]("Что делать?")
            return {"answer": "ok", "retrieval": retrieved["retrieval"]}

    monkeypatch.setattr("imperial_rag.runtime.RetrievalService", FakeRetrievalService)
    monkeypatch.setattr("imperial_rag.runtime.ChunkNeighborStore", type("FakeStore", (), {"from_jsonl": classmethod(lambda cls, path: "store")}))
    monkeypatch.setattr("imperial_rag.runtime.build_query_workflow", lambda **kwargs: FakeWorkflow())

    runtime = create_runtime(Settings(workspace_root=tmp_path))

    result = runtime.query("Что делать?")

    assert result["retrieval"] == {"final_evidence": 0, "reranker": "fake"}
    assert captured["question"] == "Что делать?"
    assert captured["kwargs"]["neighbor_store"] == "store"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --extra dev python -m pytest tests/test_runtime.py::test_runtime_query_uses_retrieval_service -q
```

Expected: FAIL because runtime does not import or instantiate `RetrievalService`.

- [ ] **Step 3: Update runtime imports**

In `/Users/danil/Public/imperial/src/imperial_rag/runtime.py`, update imports to include:

```python
from imperial_rag.retrieval import ChunkNeighborStore, RetrievalService, RetrievalSettings
```

- [ ] **Step 4: Update runtime retrieval closure**

In `/Users/danil/Public/imperial/src/imperial_rag/runtime.py`, inside `create_runtime()`, add a retrieval service cache:

```python
    retrieval_service_cache: dict[str, RetrievalService] = {}
```

Add this helper inside `create_runtime()` after `dependencies()`:

```python
    def retrieval_service() -> RetrievalService:
        if "value" not in retrieval_service_cache:
            deps = dependencies()
            retrieval_settings = RetrievalSettings.from_env()
            neighbor_store = ChunkNeighborStore.from_jsonl(resolved_settings.extraction_root / "chunks.jsonl")
            retrieval_service_cache["value"] = RetrievalService(
                vector_search=deps.vector_search,
                keyword_search=deps.keyword_search,
                neighbor_store=neighbor_store,
                settings=retrieval_settings,
            )
        return retrieval_service_cache["value"]
```

Replace the current `retrieve()` function body with:

```python
    def retrieve(question: str):
        result = retrieval_service().retrieve(question)
        return {
            "retrieved_documents": result.evidence,
            "vector_docs": result.vector_docs,
            "keyword_docs": result.keyword_docs,
            "retrieval": result.diagnostics,
        }
```

- [ ] **Step 5: Run runtime tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 6: Run workflow and retrieval tests together**

Run:

```bash
uv run --extra dev python -m pytest tests/test_retrieval.py tests/test_workflows.py tests/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit checkpoint**

Run:

```bash
git add src/imperial_rag/runtime.py tests/test_runtime.py
git commit -m "feat: wire runtime retrieval service"
```

Expected in a git repo: commit succeeds. Expected in this checkout until git is initialized: `fatal: not a git repository`.

---

### Task 11: Include Retrieval Diagnostics In Phoenix Eval Outputs

**Files:**
- Modify: `/Users/danil/Public/imperial/scripts/run_phoenix_eval.py`
- Modify: `/Users/danil/Public/imperial/tests/test_evals.py`

- [ ] **Step 1: Write failing eval diagnostic test**

Append this test to `/Users/danil/Public/imperial/tests/test_evals.py`:

```python
def test_eval_runner_includes_retrieval_diagnostics_in_outputs():
    module = _load_eval_runner()

    class FakeRuntime:
        def query(self, question: str) -> dict[str, object]:
            return {
                "answer": "I could not find this clearly in the indexed documents.",
                "citations": [],
                "sources": [],
                "evidence": [],
                "retrieval": {"final_evidence": 0, "reranker": "fallback:deterministic"},
            }

    output = module.run_target({"question": "Что делать?"}, runtime=FakeRuntime())

    assert output["retrieval"] == {"final_evidence": 0, "reranker": "fallback:deterministic"}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --extra dev python -m pytest tests/test_evals.py::test_eval_runner_includes_retrieval_diagnostics_in_outputs -q
```

Expected: FAIL because `run_target()` does not include `retrieval`.

- [ ] **Step 3: Update eval target output**

In `/Users/danil/Public/imperial/scripts/run_phoenix_eval.py`, in `run_target()`, replace the returned dict with:

```python
    return {
        "answer": str(result.get("answer", "")),
        "citations": list(result.get("citations") or result.get("sources") or []),
        "sources": list(result.get("sources") or result.get("citations") or []),
        "documents": [_document_payload(document) for document in evidence],
        "retrieval": dict(result.get("retrieval") or {}),
    }
```

- [ ] **Step 4: Run eval tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_evals.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint**

Run:

```bash
git add scripts/run_phoenix_eval.py tests/test_evals.py
git commit -m "feat: include retrieval diagnostics in evals"
```

Expected in a git repo: commit succeeds. Expected in this checkout until git is initialized: `fatal: not a git repository`.

---

### Task 12: Expand Russian Evaluation Questions To 30 Cases

**Files:**
- Modify: `/Users/danil/Public/imperial/evals/questions.jsonl`
- Modify: `/Users/danil/Public/imperial/tests/test_evals.py`

- [ ] **Step 1: Strengthen eval count test**

In `/Users/danil/Public/imperial/tests/test_evals.py`, replace:

```python
    assert len(lines) >= 3
```

with:

```python
    assert len(lines) == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --extra dev python -m pytest tests/test_evals.py::test_eval_questions_are_russian_jsonl_with_expected_behavior -q
```

Expected: FAIL because `evals/questions.jsonl` currently has 7 lines.

- [ ] **Step 3: Replace eval question file**

Replace `/Users/danil/Public/imperial/evals/questions.jsonl` with exactly these 30 JSONL rows:

```jsonl
{"question":"Как оформить возврат брака из магазина?","expected_behavior":"cite_answer","expected_source_hints":["РЕГЛАМЕНТ О БРАКЕ","возврат брака"]}
{"question":"Какие обязанности у водителя-экспедитора?","expected_behavior":"cite_answer","expected_source_hints":["водителя экспедитора","ДИ ЛОГИСТИКИ"]}
{"question":"Что делать при отсутствии сотрудника на рабочем месте?","expected_behavior":"cite_answer","expected_source_hints":["Акт об отсутствии","рабочем месте"]}
{"question":"Какие правила по табелям и мотивационным листам?","expected_behavior":"cite_answer","expected_source_hints":["табелях","мотивацион"]}
{"question":"Кто отвечает за приемку товара на складе?","expected_behavior":"cite_answer","expected_source_hints":["СКЛАД","прием"]}
{"question":"Какая версия регламента склада действует, если документы противоречат друг другу?","expected_behavior":"surface_conflict","expected_source_hints":["РЕГЛАМЕНТ СКЛАДА"]}
{"question":"Какую температуру плавления имеет вольфрам?","expected_behavior":"refuse_if_not_found","expected_source_hints":[]}
{"question":"Какие действия предусмотрены при возврате потерянного товара на маршруте?","expected_behavior":"cite_answer","expected_source_hints":["Возврат потерянного товара","маршрут"]}
{"question":"Какие обязанности указаны для грузчика-экспедитора?","expected_behavior":"cite_answer","expected_source_hints":["Грузчик экспедитор","ДИ ЛОГИСТИКИ"]}
{"question":"Что должен делать старший водитель-экспедитор?","expected_behavior":"cite_answer","expected_source_hints":["Старший водитель экспедитор","ЛОГ"]}
{"question":"Какие обязанности у HR по должностной инструкции?","expected_behavior":"cite_answer","expected_source_hints":["ДИ HR","HR"]}
{"question":"Как оформляется заявление на увольнение?","expected_behavior":"cite_answer","expected_source_hints":["заявление на увольниение","БЛАНКИ"]}
{"question":"Какие документы описывают заявление на оплачиваемый отпуск?","expected_behavior":"cite_answer","expected_source_hints":["заявление на оплачиваемый отпуск","БЛАНКИ"]}
{"question":"Какие правила есть для удаленной работы?","expected_behavior":"cite_answer","expected_source_hints":["удаленк","удаленку"]}
{"question":"Какие правила описаны для делового этикета?","expected_behavior":"cite_answer","expected_source_hints":["ДЕЛОВОМ ЭТИКЕТЕ","этикет"]}
{"question":"Как регулируется ценоизменение?","expected_behavior":"cite_answer","expected_source_hints":["ЦЕНОИЗМЕНЕНИ","Регламент по ценоизменению"]}
{"question":"Какие действия предусмотрены при ревизии склада и продаж?","expected_behavior":"cite_answer","expected_source_hints":["РЕГЛАМЕНТ Ревизии","СКЛАД И ПРОДАЖИ"]}
{"question":"Какие правила есть для работы с ТСД?","expected_behavior":"cite_answer","expected_source_hints":["РЕГЛАМЕНТ  ТСД","ТСД"]}
{"question":"Какие обязанности у заведующего складом?","expected_behavior":"cite_answer","expected_source_hints":["Заведующий складом","ДИ СКЛАД"]}
{"question":"Какие обязанности у кладовщика?","expected_behavior":"cite_answer","expected_source_hints":["КЛАДОВЩИКА","ДИ СКЛАД"]}
{"question":"Какие обязанности у грузчика склада?","expected_behavior":"cite_answer","expected_source_hints":["СКЛАД Грузчик","ДИ СКЛАД"]}
{"question":"Какие правила ведения чатов описаны в регламенте?","expected_behavior":"cite_answer","expected_source_hints":["ВЕДЕНИЕ ЧАТОВ","чатов"]}
{"question":"Что регламентируется для отдела снабжения?","expected_behavior":"cite_answer","expected_source_hints":["Регламент отд. СНАБЖЕНИЯ","снабжения"]}
{"question":"Какие обязанности у менеджера по развитию в отделе продаж?","expected_behavior":"cite_answer","expected_source_hints":["менеджер по развитию","ОТдел продаж"]}
{"question":"Какие обязанности у оператора отдела продаж?","expected_behavior":"cite_answer","expected_source_hints":["ОПЕРАТОР ОТДЕЛ продаж","Оператор"]}
{"question":"Какие обязанности у супервайзера?","expected_behavior":"cite_answer","expected_source_hints":["Супервайзер","ОТдел продаж"]}
{"question":"Какие обязанности у старшего менеджера ОПТ?","expected_behavior":"cite_answer","expected_source_hints":["ОПТ СТАРШИЙ МЕНЕДЖЕР","ДИ ОПТ"]}
{"question":"Какие правила описаны для экспедиторов в регламенте логистики?","expected_behavior":"cite_answer","expected_source_hints":["Регламент ЭКСПЕДИТОРОВ","ЛОГИСТИКИ"]}
{"question":"Какие правила постановки задач в Trello описаны в документах?","expected_behavior":"cite_answer","expected_source_hints":["ПРАВИЛА ПОСТАНОВКИ ЗАДАЧ В ТРЕЛЛО","Trello"]}
{"question":"Какова столица Австралии?","expected_behavior":"refuse_if_not_found","expected_source_hints":[]}
```

- [ ] **Step 4: Run eval tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_evals.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint**

Run:

```bash
git add evals/questions.jsonl tests/test_evals.py
git commit -m "test: expand rag accuracy eval questions"
```

Expected in a git repo: commit succeeds. Expected in this checkout until git is initialized: `fatal: not a git repository`.

---

### Task 13: Run Full Verification And Operational Reindex

**Files:**
- No code changes required in this task.
- Generated local state under `/Users/danil/Public/imperial/.imperial_rag/` may change during ingestion.

- [ ] **Step 1: Run full test suite**

Run:

```bash
uv run --extra dev python -m pytest -q
```

Expected: PASS.

- [ ] **Step 2: Start Qdrant**

Run:

```bash
./scripts/start_qdrant.sh
```

Expected: Qdrant starts or reports that a container/service is already available on `127.0.0.1:6333`.

- [ ] **Step 3: Confirm Qdrant health**

Run:

```bash
IMPERIAL_RAG_LIVE_QDRANT=1 uv run --extra dev python -m pytest tests/test_qdrant_health.py -q
```

Expected: PASS. If Docker is unavailable, record that live vector indexing was not verified and continue with unit-level verification.

- [ ] **Step 4: Re-ingest with vector indexing and 400/50 chunks**

Run:

```bash
IMPERIAL_RAG_CHUNK_SIZE=400 \
IMPERIAL_RAG_CHUNK_OVERLAP=50 \
uv run --extra dev python scripts/ingest.py \
  --workspace-root /Users/danil/Public/imperial \
  --index-vectors
```

Expected:

- `chunks=<N>` where `<N>` is higher than the old `970` because chunks are smaller.
- `keyword_indexed=True`.
- `vector_indexed=True` if Qdrant and OpenAI embeddings are configured.
- If `OPENAI_API_KEY` or `AZURE_OPENAI_API_KEY` is not configured, vector indexing may fail or be skipped by the environment. Record the exact output.

- [ ] **Step 5: Run local eval without Cohere key**

Run:

```bash
COHERE_API_KEY= \
uv run --extra dev python scripts/run_phoenix_eval.py
```

Expected:

- `local_eval_examples=30`.
- `local_eval_passed=<N>`.
- No stack trace.
- Retrieval diagnostics are present in target outputs when inspected through tests or Phoenix mode.

- [ ] **Step 6: Run local eval with Cohere key when configured**

Run only if `COHERE_API_KEY` is set:

```bash
uv run --extra dev python scripts/run_phoenix_eval.py
```

Expected:

- `local_eval_examples=30`.
- `local_eval_passed=<N>`.
- The pass count is equal to or greater than the fallback-only pass count.

- [ ] **Step 7: Run one query smoke test**

Run:

```bash
uv run --extra dev python scripts/query.py "Как оформить возврат брака из магазина?"
```

Expected:

- The answer either cites retrieved evidence or refuses.
- If retrieval returns evidence but generation fails citation validation, the answer is the strict refusal text.
- No stack trace.

- [ ] **Step 8: Inspect retrieval diagnostics through Python**

Run:

```bash
uv run --extra dev python - <<'PY'
from imperial_rag.runtime import create_runtime

result = create_runtime().query("Как оформить возврат брака из магазина?")
print(result.get("retrieval", {}))
PY
```

Expected when indexes contain enough matches:

```text
{'vector_candidates': 32, 'keyword_candidates': 40, ... 'reranked_candidates': 12, 'final_evidence': <value between 1 and 24>, ...}
```

If vector search is unavailable, expected diagnostics include `vector_search_status` set to `unavailable`, `empty`, or `skipped`, and keyword retrieval still runs.

- [ ] **Step 9: Final full-suite verification**

Run:

```bash
uv run --extra dev python -m pytest -q
```

Expected: PASS.

- [ ] **Step 10: Final commit checkpoint**

Run:

```bash
git add pyproject.toml uv.lock src/imperial_rag tests scripts/run_phoenix_eval.py evals/questions.jsonl docs/superpowers/plans/2026-06-03-rag-accuracy-improvements.md
git commit -m "feat: improve rag retrieval accuracy"
```

Expected in a git repo: commit succeeds. Expected in this checkout until git is initialized: `fatal: not a git repository`.

## Self-Review Checklist

- Spec coverage:
  - `chunk_size=400` and `chunk_overlap=50`: Task 2.
  - Vector recall `fetch_k=80`, `k=32`, `lambda_mult=0.4`: Tasks 1 and 5.
  - Keyword recall `limit=40`: Tasks 1, 3, and 5.
  - Rerank input `60`, top `12`: Tasks 1 and 6.
  - Cohere `rerank-v3.5` primary and multilingual fallback: Tasks 1 and 6.
  - Deterministic fallback: Tasks 4 and 6.
  - Neighbor expansion and final cap `24`: Task 7.
  - Runtime diagnostics: Tasks 5, 8, 9, 10, and 11.
  - Eval expansion to 30 Russian questions: Task 12.
  - Operational reindex and verification: Task 13.
- Placeholder scan:
  - The plan contains concrete files, commands, expected outcomes, and code snippets for each implementation step.
- Type consistency:
  - `RetrievalSettings`, `RetrievalCandidateResult`, `RetrievalResult`, `HybridRetriever`, `Reranker`, `ChunkNeighborStore`, `NeighborExpander`, `EvidenceSelector`, and `RetrievalService` are defined before runtime wiring tasks use them.
