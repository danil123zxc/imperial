# Readability Module Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Superseded:** This plan was superseded on 2026-06-22 by the lifecycle-oriented `src/imperial_rag` restructure. It predates the current Elasticsearch keyword module, Qdrant provider metadata flow, and Phoenix trace-shape changes.

**Goal:** Reorganize the Imperial RAG Python package into readable workflow-focused modules while preserving existing behavior and public import paths.

**Architecture:** Use package folders for `corpus`, `ingestion`, `indexing`, `query`, `evals`, `observability`, and `app`. Move code with minimal algorithmic edits, keep old top-level modules as compatibility wrappers, and turn `imperial_rag.indexing` into an `indexing/__init__.py` re-export package to avoid an import-name collision.

**Tech Stack:** Python 3.12+, pytest, LangChain, LangGraph, Qdrant, SQLite, Phoenix, Streamlit.

---

## Checkpoint Policy

This workspace is not currently a Git repository. Verify with:

```bash
git status --short
```

Expected output:

```text
fatal: not a git repository (or any of the parent directories): .git
```

Because commits are unavailable in this checkout, each task ends with a test checkpoint instead of a commit. If this plan is executed from a future Git checkout, commit after each task with the task-specific message included below.

## File Structure

Create or modify these files:

- Create: `src/imperial_rag/corpus/__init__.py`
- Create: `src/imperial_rag/corpus/manifest.py`
- Create: `src/imperial_rag/corpus/scanning.py`
- Create: `src/imperial_rag/corpus/extraction.py`
- Create: `src/imperial_rag/corpus/ocr.py`
- Create: `src/imperial_rag/corpus/chunking.py`
- Create: `src/imperial_rag/ingestion/__init__.py`
- Create: `src/imperial_rag/ingestion/pipeline.py`
- Create: `src/imperial_rag/ingestion/artifacts.py`
- Create: `src/imperial_rag/ingestion/status.py`
- Create: `src/imperial_rag/ingestion/workflow.py`
- Create: `src/imperial_rag/indexing/__init__.py`
- Create: `src/imperial_rag/indexing/ids.py`
- Create: `src/imperial_rag/indexing/keyword.py`
- Create: `src/imperial_rag/indexing/vector.py`
- Create: `src/imperial_rag/indexing/health.py`
- Create: `src/imperial_rag/query/__init__.py`
- Create: `src/imperial_rag/query/answering.py`
- Create: `src/imperial_rag/query/retrieval.py`
- Create: `src/imperial_rag/query/workflow.py`
- Create: `src/imperial_rag/query/runtime.py`
- Create: `src/imperial_rag/evals/__init__.py`
- Create: `src/imperial_rag/evals/phoenix.py`
- Create: `src/imperial_rag/observability/__init__.py`
- Create: `src/imperial_rag/observability/phoenix.py`
- Create: `src/imperial_rag/app/__init__.py`
- Create: `src/imperial_rag/app/web.py`
- Create: `tests/test_module_structure.py`
- Modify: `src/imperial_rag/answering.py`
- Modify: `src/imperial_rag/chunking.py`
- Modify: `src/imperial_rag/extraction.py`
- Modify: `src/imperial_rag/manifest.py`
- Modify: `src/imperial_rag/ocr.py`
- Modify: `src/imperial_rag/pipeline.py`
- Modify: `src/imperial_rag/runtime.py`
- Modify: `src/imperial_rag/tracing.py`
- Modify: `src/imperial_rag/web_app.py`
- Modify: `src/imperial_rag/workflows.py`
- Modify: `scripts/ingest.py`
- Modify: `scripts/query.py`
- Modify: `scripts/run_phoenix_eval.py`
- Modify: `tests/test_evals.py`
- Delete: `src/imperial_rag/indexing.py` after its public symbols are available from `src/imperial_rag/indexing/__init__.py`

---

### Task 1: Add Module Structure Regression Tests

**Files:**
- Create: `tests/test_module_structure.py`

- [ ] **Step 1: Write the failing import-structure test**

Create `tests/test_module_structure.py` with this complete content:

```python
from __future__ import annotations


def test_new_readability_import_paths_are_available():
    from imperial_rag.app.web import APP_TITLE
    from imperial_rag.corpus.chunking import build_chunks
    from imperial_rag.corpus.extraction import ExtractionResult, extract_file
    from imperial_rag.corpus.manifest import FileRecord, FileStatus, IndexStatus, ManifestStore
    from imperial_rag.corpus.ocr import OcrCache, OcrClient, OcrResult
    from imperial_rag.corpus.scanning import assign_duplicate_groups, hash_file, scan_files, stable_file_id
    from imperial_rag.evals.phoenix import citation_behavior, load_questions, source_hint_behavior
    from imperial_rag.indexing.health import qdrant_health, qdrant_is_healthy
    from imperial_rag.indexing.ids import stable_chunk_id, stable_chunk_ids
    from imperial_rag.indexing.keyword import KeywordIndex, build_fts_match_query, normalize_search_text
    from imperial_rag.indexing.vector import (
        create_qdrant_vector_store,
        index_documents,
        index_vector_documents,
        make_qdrant_store,
    )
    from imperial_rag.ingestion.pipeline import IngestionSummary, ingest_corpus, run_ingestion
    from imperial_rag.ingestion.workflow import build_ingestion_workflow
    from imperial_rag.observability.phoenix import configure_phoenix_tracing
    from imperial_rag.query.answering import build_strict_messages
    from imperial_rag.query.retrieval import rank_hybrid_candidates
    from imperial_rag.query.runtime import Runtime, create_runtime
    from imperial_rag.query.workflow import build_query_workflow

    assert APP_TITLE == "Imperial RAG"
    assert callable(build_chunks)
    assert callable(extract_file)
    assert ExtractionResult.__name__ == "ExtractionResult"
    assert FileRecord.__name__ == "FileRecord"
    assert FileStatus.INDEXED.value == "indexed"
    assert IndexStatus.SKIPPED.value == "skipped"
    assert ManifestStore.__name__ == "ManifestStore"
    assert callable(hash_file)
    assert callable(stable_file_id)
    assert callable(scan_files)
    assert callable(assign_duplicate_groups)
    assert OcrResult.__name__ == "OcrResult"
    assert OcrClient.__name__ == "OcrClient"
    assert OcrCache.__name__ == "OcrCache"
    assert callable(load_questions)
    assert callable(citation_behavior)
    assert callable(source_hint_behavior)
    assert callable(qdrant_health)
    assert callable(qdrant_is_healthy)
    assert callable(stable_chunk_id)
    assert callable(stable_chunk_ids)
    assert callable(build_fts_match_query)
    assert callable(normalize_search_text)
    assert KeywordIndex.__name__ == "KeywordIndex"
    assert callable(create_qdrant_vector_store)
    assert callable(make_qdrant_store)
    assert callable(index_vector_documents)
    assert callable(index_documents)
    assert IngestionSummary.__name__ == "IngestionSummary"
    assert callable(run_ingestion)
    assert callable(ingest_corpus)
    assert callable(build_ingestion_workflow)
    assert callable(configure_phoenix_tracing)
    assert callable(build_strict_messages)
    assert callable(rank_hybrid_candidates)
    assert Runtime.__name__ == "Runtime"
    assert callable(create_runtime)
    assert callable(build_query_workflow)


def test_old_public_import_paths_still_reexport_public_symbols():
    from imperial_rag.answering import build_strict_messages as old_build_strict_messages
    from imperial_rag.chunking import build_chunks as old_build_chunks
    from imperial_rag.extraction import ExtractionResult as old_extraction_result
    from imperial_rag.extraction import extract_file as old_extract_file
    from imperial_rag.indexing import KeywordIndex as old_keyword_index
    from imperial_rag.indexing import stable_chunk_id as old_stable_chunk_id
    from imperial_rag.manifest import FileRecord as old_file_record
    from imperial_rag.manifest import scan_files as old_scan_files
    from imperial_rag.ocr import OcrClient as old_ocr_client
    from imperial_rag.pipeline import run_ingestion as old_run_ingestion
    from imperial_rag.runtime import create_runtime as old_create_runtime
    from imperial_rag.tracing import configure_phoenix_tracing as old_configure_phoenix_tracing
    from imperial_rag.web_app import APP_TITLE as old_app_title
    from imperial_rag.workflows import build_query_workflow as old_build_query_workflow
    from imperial_rag.workflows import rank_hybrid_candidates as old_rank_hybrid_candidates

    from imperial_rag.app.web import APP_TITLE
    from imperial_rag.corpus.chunking import build_chunks
    from imperial_rag.corpus.extraction import ExtractionResult, extract_file
    from imperial_rag.corpus.manifest import FileRecord
    from imperial_rag.corpus.ocr import OcrClient
    from imperial_rag.corpus.scanning import scan_files
    from imperial_rag.ingestion.pipeline import run_ingestion
    from imperial_rag.indexing.ids import stable_chunk_id
    from imperial_rag.indexing.keyword import KeywordIndex
    from imperial_rag.observability.phoenix import configure_phoenix_tracing
    from imperial_rag.query.answering import build_strict_messages
    from imperial_rag.query.retrieval import rank_hybrid_candidates
    from imperial_rag.query.runtime import create_runtime
    from imperial_rag.query.workflow import build_query_workflow

    assert old_app_title == APP_TITLE
    assert old_build_chunks is build_chunks
    assert old_extraction_result is ExtractionResult
    assert old_extract_file is extract_file
    assert old_file_record is FileRecord
    assert old_scan_files is scan_files
    assert old_ocr_client is OcrClient
    assert old_keyword_index is KeywordIndex
    assert old_stable_chunk_id is stable_chunk_id
    assert old_run_ingestion is run_ingestion
    assert old_create_runtime is create_runtime
    assert old_configure_phoenix_tracing is configure_phoenix_tracing
    assert old_build_strict_messages is build_strict_messages
    assert old_rank_hybrid_candidates is rank_hybrid_candidates
    assert old_build_query_workflow is build_query_workflow
```

- [ ] **Step 2: Run the new test and verify it fails before the move**

Run:

```bash
uv run --extra dev python -m pytest tests/test_module_structure.py -q
```

Expected: FAIL with an import error such as:

```text
ModuleNotFoundError: No module named 'imperial_rag.app'
```

- [ ] **Step 3: Checkpoint**

Run:

```bash
git status --short
```

Expected in this workspace:

```text
fatal: not a git repository (or any of the parent directories): .git
```

If executing from a Git checkout, commit with:

```bash
git add tests/test_module_structure.py
git commit -m "test: cover readability module imports"
```

---

### Task 2: Move Corpus Modules And Keep Corpus Compatibility Wrappers

**Files:**
- Create: `src/imperial_rag/corpus/__init__.py`
- Create: `src/imperial_rag/corpus/manifest.py`
- Create: `src/imperial_rag/corpus/scanning.py`
- Create: `src/imperial_rag/corpus/extraction.py`
- Create: `src/imperial_rag/corpus/ocr.py`
- Create: `src/imperial_rag/corpus/chunking.py`
- Modify: `src/imperial_rag/manifest.py`
- Modify: `src/imperial_rag/extraction.py`
- Modify: `src/imperial_rag/ocr.py`
- Modify: `src/imperial_rag/chunking.py`
- Test: `tests/test_module_structure.py`
- Test: `tests/test_manifest.py`
- Test: `tests/test_manifest_store.py`
- Test: `tests/test_extraction.py`
- Test: `tests/test_chunking.py`

- [ ] **Step 1: Create the corpus package and move leaf files**

Run:

```bash
mkdir -p src/imperial_rag/corpus
touch src/imperial_rag/corpus/__init__.py
mv src/imperial_rag/manifest.py src/imperial_rag/corpus/manifest.py
mv src/imperial_rag/extraction.py src/imperial_rag/corpus/extraction.py
mv src/imperial_rag/ocr.py src/imperial_rag/corpus/ocr.py
mv src/imperial_rag/chunking.py src/imperial_rag/corpus/chunking.py
```

Expected: command exits successfully.

- [ ] **Step 2: Split scanning helpers out of corpus manifest**

Create `src/imperial_rag/corpus/scanning.py` with this complete content:

```python
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from imperial_rag.corpus.manifest import FileRecord


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_file_id(relative_path: Path) -> str:
    return hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:16]


def scan_files(documents_root: Path) -> list[FileRecord]:
    records: list[FileRecord] = []
    for path in sorted(documents_root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        relative_path = path.relative_to(documents_root)
        records.append(
            FileRecord(
                file_id=stable_file_id(relative_path),
                absolute_path=path.resolve(),
                relative_path=relative_path,
                filename=path.name,
                extension=path.suffix.lower(),
                size_bytes=stat.st_size,
                sha256=hash_file(path),
                modified_ns=stat.st_mtime_ns,
                parent_folder=relative_path.parent,
                inferred_category=relative_path.parts[0] if relative_path.parts else "",
            )
        )
    return assign_duplicate_groups(records)


def assign_duplicate_groups(records: list[FileRecord]) -> list[FileRecord]:
    records_by_hash: dict[str, list[FileRecord]] = {}
    for record in records:
        records_by_hash.setdefault(record.sha256, []).append(record)

    grouped: list[FileRecord] = []
    for record in records:
        matches = records_by_hash[record.sha256]
        group_id = f"sha256:{record.sha256}" if len(matches) > 1 else None
        grouped.append(replace(record, duplicate_group_id=group_id))
    return grouped
```

In `src/imperial_rag/corpus/manifest.py`, delete the existing definitions of `hash_file`, `stable_file_id`, `scan_files`, and `assign_duplicate_groups`. Also remove unused imports that supported only those helpers:

```python
import hashlib
from dataclasses import replace
```

Keep the manifest imports focused on the remaining manifest responsibilities:

```python
import sqlite3
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
```

- [ ] **Step 3: Update corpus-internal imports**

In `src/imperial_rag/corpus/extraction.py`, replace the old imports:

```python
from imperial_rag.manifest import FileRecord, FileStatus
from imperial_rag.ocr import OcrCache, OcrResult
```

with:

```python
from imperial_rag.corpus.manifest import FileRecord, FileStatus
from imperial_rag.corpus.ocr import OcrCache, OcrResult
```

- [ ] **Step 4: Add corpus compatibility wrappers**

Create `src/imperial_rag/manifest.py` with this complete content:

```python
from __future__ import annotations

from imperial_rag.corpus.manifest import *
from imperial_rag.corpus.scanning import *
```

Create `src/imperial_rag/extraction.py` with this complete content:

```python
from __future__ import annotations

from imperial_rag.corpus.extraction import *
```

Create `src/imperial_rag/ocr.py` with this complete content:

```python
from __future__ import annotations

from imperial_rag.corpus.ocr import *
```

Create `src/imperial_rag/chunking.py` with this complete content:

```python
from __future__ import annotations

from imperial_rag.corpus.chunking import *
```

- [ ] **Step 5: Run corpus-focused tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_module_structure.py tests/test_manifest.py tests/test_manifest_store.py tests/test_extraction.py tests/test_chunking.py -q
```

Expected: `tests/test_module_structure.py` still fails on packages not moved yet, while the manifest, extraction, and chunking tests pass.

- [ ] **Step 6: Checkpoint**

Run:

```bash
git status --short
```

Expected in this workspace:

```text
fatal: not a git repository (or any of the parent directories): .git
```

If executing from a Git checkout, commit with:

```bash
git add src/imperial_rag tests/test_module_structure.py
git commit -m "refactor: move corpus modules"
```

---

### Task 3: Replace Flat Indexing Module With Indexing Package

**Files:**
- Create: `src/imperial_rag/indexing/__init__.py`
- Create: `src/imperial_rag/indexing/ids.py`
- Create: `src/imperial_rag/indexing/keyword.py`
- Create: `src/imperial_rag/indexing/vector.py`
- Create: `src/imperial_rag/indexing/health.py`
- Delete: `src/imperial_rag/indexing.py`
- Test: `tests/test_indexing.py`
- Test: `tests/test_qdrant_health.py`
- Test: `tests/test_module_structure.py`

- [ ] **Step 1: Move the old indexing source aside and create the package**

Run:

```bash
cp src/imperial_rag/indexing.py /tmp/imperial_rag_indexing_original.py
rm src/imperial_rag/indexing.py
mkdir -p src/imperial_rag/indexing
touch src/imperial_rag/indexing/__init__.py
python - <<'PY'
from pathlib import Path
assert Path("/tmp/imperial_rag_indexing_original.py").exists()
print("saved_old_indexing_source=/tmp/imperial_rag_indexing_original.py")
PY
```

Expected output:

```text
saved_old_indexing_source=/tmp/imperial_rag_indexing_original.py
```

- [ ] **Step 2: Create stable ID module**

Create `src/imperial_rag/indexing/ids.py` by moving these items from the old indexing source:

```python
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Sequence

from langchain_core.documents import Document


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


def stable_chunk_ids(documents: Sequence[Document]) -> list[str]:
    return [stable_chunk_id(document) for document in documents]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
```

- [ ] **Step 3: Create keyword index module**

Create `src/imperial_rag/indexing/keyword.py` by moving `KeywordHit`, `KeywordIndex`, `_stem_token`, `normalize_search_text`, `build_fts_match_query`, and `_searchable_document_text` from the old indexing source.

Use these imports at the top:

```python
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from imperial_rag.config import Settings
from imperial_rag.indexing.ids import stable_chunk_id
```

Keep this constant exactly:

```python
_ENDING_RE = re.compile(r"(иями|ями|ами|ого|его|ому|ему|ыми|ими|ов|ев|ей|ый|ий|ой|ая|яя|ое|ее|ам|ям|ах|ях|ом|ем|а|я|ы|и|у|ю|е|о|ь)$")
```

- [ ] **Step 4: Create vector indexing module**

Create `src/imperial_rag/indexing/vector.py` by moving `create_qdrant_vector_store`, `make_qdrant_store`, `index_vector_documents`, and `index_documents` from the old indexing source.

Use these imports at the top:

```python
from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from imperial_rag.config import Settings
from imperial_rag.indexing.ids import stable_chunk_ids
```

Keep this validation behavior in `index_vector_documents`:

```python
if len(resolved_ids) != len(documents):
    raise ValueError("ids length must match chunks length")
```

- [ ] **Step 5: Create Qdrant health module**

Create `src/imperial_rag/indexing/health.py` with this complete content:

```python
from __future__ import annotations

from qdrant_client import QdrantClient

from imperial_rag.config import Settings


def qdrant_is_healthy(qdrant_url: str) -> bool:
    client = QdrantClient(url=qdrant_url)
    try:
        client.get_collections()
    except Exception:
        return False
    return True


def qdrant_health(settings: Settings) -> bool:
    return qdrant_is_healthy(settings.qdrant_url)
```

- [ ] **Step 6: Create indexing re-export surface**

Create `src/imperial_rag/indexing/__init__.py` with this complete content:

```python
from __future__ import annotations

from imperial_rag.indexing.health import qdrant_health, qdrant_is_healthy
from imperial_rag.indexing.ids import stable_chunk_id, stable_chunk_ids
from imperial_rag.indexing.keyword import KeywordHit, KeywordIndex, build_fts_match_query, normalize_search_text
from imperial_rag.indexing.vector import (
    create_qdrant_vector_store,
    index_documents,
    index_vector_documents,
    make_qdrant_store,
)

__all__ = [
    "KeywordHit",
    "KeywordIndex",
    "build_fts_match_query",
    "create_qdrant_vector_store",
    "index_documents",
    "index_vector_documents",
    "make_qdrant_store",
    "normalize_search_text",
    "qdrant_health",
    "qdrant_is_healthy",
    "stable_chunk_id",
    "stable_chunk_ids",
]
```

- [ ] **Step 7: Run indexing tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_indexing.py tests/test_qdrant_health.py tests/test_module_structure.py -q
```

Expected: indexing and Qdrant health tests pass. `tests/test_module_structure.py` still fails only on packages not moved yet.

- [ ] **Step 8: Checkpoint**

Run:

```bash
git status --short
```

Expected in this workspace:

```text
fatal: not a git repository (or any of the parent directories): .git
```

If executing from a Git checkout, commit with:

```bash
git add src/imperial_rag tests/test_module_structure.py
git commit -m "refactor: split indexing package"
```

---

### Task 4: Move Ingestion Pipeline And Ingestion Workflow

**Files:**
- Create: `src/imperial_rag/ingestion/__init__.py`
- Create: `src/imperial_rag/ingestion/pipeline.py`
- Create: `src/imperial_rag/ingestion/artifacts.py`
- Create: `src/imperial_rag/ingestion/status.py`
- Create: `src/imperial_rag/ingestion/workflow.py`
- Modify: `src/imperial_rag/pipeline.py`
- Modify: `scripts/ingest.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_pipeline_integration.py`
- Test: `tests/test_workflows.py`
- Test: `tests/test_scripts.py`
- Test: `tests/test_module_structure.py`

- [ ] **Step 1: Create ingestion package and move pipeline**

Run:

```bash
mkdir -p src/imperial_rag/ingestion
touch src/imperial_rag/ingestion/__init__.py
mv src/imperial_rag/pipeline.py src/imperial_rag/ingestion/pipeline.py
```

Expected: command exits successfully.

- [ ] **Step 2: Extract artifact-writing helpers**

Create `src/imperial_rag/ingestion/artifacts.py` by moving `_write_extracted_artifact`, `_write_chunks`, and `_safe_artifact_path` from `src/imperial_rag/ingestion/pipeline.py`.

Use these imports at the top:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from imperial_rag.ingestion.status import status_value
```

Rename `_status_value` calls inside this new module to `status_value`.

- [ ] **Step 3: Extract ingestion status helpers**

Create `src/imperial_rag/ingestion/status.py` by moving `_count_chunks_by_file`, `_replace_keyword_index`, `_index_with_vector_store`, `_update_index_status`, `_status_value`, and `_planned_extraction_method` from `src/imperial_rag/ingestion/pipeline.py`.

Use these imports at the top:

```python
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
```

Rename `_status_value` to `status_value` so `artifacts.py` can import it:

```python
def status_value(status: Any) -> str:
    return str(getattr(status, "value", status))
```

- [ ] **Step 4: Update ingestion pipeline imports**

In `src/imperial_rag/ingestion/pipeline.py`, add these imports:

```python
from imperial_rag.ingestion.artifacts import write_chunks, write_extracted_artifact
from imperial_rag.ingestion.status import (
    count_chunks_by_file,
    index_with_vector_store,
    planned_extraction_method,
    replace_keyword_index,
    status_value,
    update_index_status,
)
```

Rename the moved helper calls:

```python
_write_extracted_artifact( -> write_extracted_artifact(
_write_chunks( -> write_chunks(
_replace_keyword_index( -> replace_keyword_index(
_index_with_vector_store( -> index_with_vector_store(
_count_chunks_by_file( -> count_chunks_by_file(
_update_index_status( -> update_index_status(
_status_value( -> status_value(
_planned_extraction_method( -> planned_extraction_method(
```

Update `_load_dependencies()` imports in `src/imperial_rag/ingestion/pipeline.py`:

```python
from imperial_rag.corpus.chunking import build_chunks
from imperial_rag.corpus.extraction import extract_file
from imperial_rag.corpus.manifest import FileStatus, IndexStatus, ManifestStore
from imperial_rag.corpus.scanning import assign_duplicate_groups, scan_files
from imperial_rag.indexing import KeywordIndex, index_documents
```

- [ ] **Step 5: Move ingestion workflow**

Create `src/imperial_rag/ingestion/workflow.py` with this complete content copied from the ingestion half of the current `src/imperial_rag/workflows.py`:

```python
from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, TypedDict


class IngestionState(TypedDict, total=False):
    settings: object
    ocr_client: object
    vector_store: object
    summary: object
    status: str
    counts: dict[str, int]


def build_ingestion_workflow(run_pipeline=None):
    from langgraph.graph import END, START, StateGraph

    def run_ingestion(state: IngestionState) -> IngestionState:
        if run_pipeline is not None:
            summary = _call_with_supported_args(run_pipeline, state)
        else:
            from imperial_rag.ingestion.pipeline import ingest_corpus

            summary = ingest_corpus(
                settings=state["settings"],
                ocr_client=state.get("ocr_client"),
                vector_store=state.get("vector_store"),
            )
        counts = _counts_from_summary(summary)
        status = str(summary.get("status", "completed")) if isinstance(summary, Mapping) else str(getattr(summary, "status", "completed"))
        return {"summary": summary, "status": status, "counts": counts}

    graph = StateGraph(IngestionState)
    graph.add_node("run_ingestion", run_ingestion)
    graph.add_edge(START, "run_ingestion")
    graph.add_edge("run_ingestion", END)
    return graph.compile()


def _call_with_supported_args(callable_, *args):
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return callable_(*args)
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in signature.parameters.values()):
        return callable_(*args)
    positional_count = sum(
        1
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    return callable_(*args[:positional_count])


def _counts_from_summary(summary: Any) -> dict[str, int]:
    if isinstance(summary, Mapping):
        counts = summary.get("counts")
        if isinstance(counts, Mapping):
            return {str(key): int(value) for key, value in counts.items()}
        return {str(key): int(value) for key, value in summary.items() if isinstance(value, int)}
    counts: dict[str, int] = {}
    for source_name, target_name in (
        ("total_files", "files"),
        ("document_count", "documents"),
        ("documents", "documents"),
        ("chunk_count", "chunks"),
        ("chunks", "chunks"),
        ("indexed_count", "indexed"),
        ("indexed", "indexed"),
    ):
        value = getattr(summary, source_name, None)
        if isinstance(value, int):
            counts[target_name] = value
    return counts
```

- [ ] **Step 6: Add pipeline compatibility wrapper**

Create `src/imperial_rag/pipeline.py` with this complete content:

```python
from __future__ import annotations

from imperial_rag.ingestion.pipeline import *
```

- [ ] **Step 7: Update ingestion script imports**

In `scripts/ingest.py`, replace:

```python
from imperial_rag.workflows import build_ingestion_workflow
```

with:

```python
from imperial_rag.ingestion.workflow import build_ingestion_workflow
```

Replace:

```python
from imperial_rag.pipeline import run_ingestion
```

with:

```python
from imperial_rag.ingestion.pipeline import run_ingestion
```

- [ ] **Step 8: Run ingestion tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_pipeline.py tests/test_pipeline_integration.py tests/test_workflows.py tests/test_scripts.py tests/test_module_structure.py -q
```

Expected: ingestion, script, and workflow compatibility tests pass except for `tests/test_module_structure.py` imports that belong to query, app, evals, or observability packages not moved yet.

- [ ] **Step 9: Checkpoint**

Run:

```bash
git status --short
```

Expected in this workspace:

```text
fatal: not a git repository (or any of the parent directories): .git
```

If executing from a Git checkout, commit with:

```bash
git add src/imperial_rag scripts/ingest.py tests/test_module_structure.py
git commit -m "refactor: move ingestion pipeline"
```

---

### Task 5: Move Query Runtime, Retrieval, Answering, And Query Workflow

**Files:**
- Create: `src/imperial_rag/query/__init__.py`
- Create: `src/imperial_rag/query/answering.py`
- Create: `src/imperial_rag/query/retrieval.py`
- Create: `src/imperial_rag/query/workflow.py`
- Create: `src/imperial_rag/query/runtime.py`
- Modify: `src/imperial_rag/answering.py`
- Modify: `src/imperial_rag/runtime.py`
- Modify: `src/imperial_rag/workflows.py`
- Modify: `scripts/query.py`
- Test: `tests/test_answering.py`
- Test: `tests/test_runtime.py`
- Test: `tests/test_workflows.py`
- Test: `tests/test_module_structure.py`

- [ ] **Step 1: Create query package and move direct query modules**

Run:

```bash
mkdir -p src/imperial_rag/query
touch src/imperial_rag/query/__init__.py
mv src/imperial_rag/answering.py src/imperial_rag/query/answering.py
mv src/imperial_rag/runtime.py src/imperial_rag/query/runtime.py
```

Expected: command exits successfully.

- [ ] **Step 2: Create retrieval module**

Create `src/imperial_rag/query/retrieval.py` with this complete content moved from the current `workflows.py`:

```python
from __future__ import annotations

from langchain_core.documents import Document


def _document_key(document: Document) -> str:
    return str(document.metadata.get("citation_id") or document.metadata.get("chunk_id") or document.page_content)


def _content_key(document: Document) -> str:
    return " ".join(document.page_content.split()).casefold()


def _merge_documents(vector_docs: list[Document], keyword_docs: list[Document]) -> list[Document]:
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


def _contains_query_terms(query: str, text: str) -> bool:
    normalized_text = text.casefold()
    return all(term in normalized_text for term in query.casefold().split() if term)


def rank_hybrid_candidates(
    query: str,
    vector_docs: list[Document],
    keyword_docs: list[Document],
    limit: int = 12,
    k: int | None = None,
) -> list[Document]:
    if k is not None:
        limit = k
    candidates = _merge_documents(vector_docs, keyword_docs)
    keyword_keys = {_document_key(document) for document in keyword_docs}
    keyword_contents = {_content_key(document) for document in keyword_docs}

    def score(document: Document) -> tuple[int, int]:
        searchable = " ".join(
            [
                document.page_content,
                str(document.metadata.get("file_name", "")),
                str(document.metadata.get("relative_path", "")),
                str(document.metadata.get("section_heading", "")),
                str(document.metadata.get("source_type", "")),
            ]
        )
        exact_boost = 1 if _contains_query_terms(query, searchable) else 0
        keyword_boost = 1 if _document_key(document) in keyword_keys or _content_key(document) in keyword_contents else 0
        return exact_boost, keyword_boost

    return sorted(candidates, key=score, reverse=True)[:limit]
```

- [ ] **Step 3: Create query workflow module**

Create `src/imperial_rag/query/workflow.py` with this complete content:

```python
from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, Protocol, TypedDict

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from imperial_rag.query.answering import (
    REFUSAL_TEXT,
    answer_has_required_citations,
    build_strict_messages,
    format_citations,
    validate_citations,
)
from imperial_rag.query.retrieval import rank_hybrid_candidates


class VectorSearch(Protocol):
    def similarity_search(self, query: str, k: int) -> list[Document]:
        raise NotImplementedError


class KeywordSearch(Protocol):
    def search(self, query: str, limit: int = 5) -> list[Document]:
        raise NotImplementedError


class ChatModel(Protocol):
    def invoke(self, messages):
        raise NotImplementedError


class QueryState(TypedDict, total=False):
    question: str
    normalized_query: str
    vector_candidates: list[Document]
    keyword_candidates: list[Document]
    evidence: list[Document]
    retrieved_documents: list[Document]
    answer: str
    citations: list[str]
    citations_valid: bool
    invalid_citations: list[str]


def _call_with_supported_args(callable_, *args):
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return callable_(*args)
    if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in signature.parameters.values()):
        return callable_(*args)
    positional_count = sum(
        1
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    return callable_(*args[:positional_count])


def _coerce_retrieved_documents(retrieved: Any, query: str) -> list[Document]:
    if retrieved is None:
        return []
    if isinstance(retrieved, Mapping):
        direct_docs = retrieved.get("retrieved_documents") or retrieved.get("documents") or retrieved.get("docs")
        if direct_docs is not None:
            return list(direct_docs)
        vector_docs = list(retrieved.get("vector_docs") or retrieved.get("vector_documents") or [])
        keyword_docs = list(retrieved.get("keyword_docs") or retrieved.get("keyword_documents") or [])
        return rank_hybrid_candidates(query, vector_docs, keyword_docs)
    if isinstance(retrieved, tuple) and len(retrieved) == 2:
        return rank_hybrid_candidates(query, list(retrieved[0]), list(retrieved[1]))
    return list(retrieved)


def _coerce_answer(answer: Any) -> str:
    if isinstance(answer, Mapping) and "answer" in answer:
        return str(answer["answer"])
    content = getattr(answer, "content", None)
    if content is not None:
        return str(content)
    return str(answer)


def build_query_workflow(
    vector_search: VectorSearch | None = None,
    keyword_search: KeywordSearch | None = None,
    chat_model: ChatModel | None = None,
    retrieve=None,
    generate=None,
):
    model = chat_model

    def normalize_query(state: QueryState) -> QueryState:
        return {"normalized_query": state["question"].strip()}

    def retrieve_node(state: QueryState) -> QueryState:
        query = state["normalized_query"]
        if retrieve is not None:
            evidence = _coerce_retrieved_documents(_call_with_supported_args(retrieve, query, state), query)
            return {"vector_candidates": evidence, "keyword_candidates": [], "evidence": evidence, "retrieved_documents": evidence}
        vector_docs = vector_search.similarity_search(query, k=8) if vector_search is not None else []
        keyword_docs = keyword_search.search(query, limit=8) if keyword_search is not None else []
        evidence = rank_hybrid_candidates(query, vector_docs, keyword_docs)
        return {
            "vector_candidates": vector_docs,
            "keyword_candidates": keyword_docs,
            "evidence": evidence,
            "retrieved_documents": evidence,
        }

    def call_model(state: QueryState) -> QueryState:
        evidence = state.get("evidence", [])
        citations = format_citations(evidence)
        if not evidence:
            return {"answer": REFUSAL_TEXT, "citations": [], "citations_valid": True, "invalid_citations": []}
        if generate is not None:
            answer = _coerce_answer(_call_with_supported_args(generate, state["question"], evidence, build_strict_messages(state["question"], evidence), state))
        else:
            resolved_model = model or ChatOpenAI(model="gpt-4.1-mini", temperature=0)
            response = resolved_model.invoke(build_strict_messages(state["question"], evidence))
            answer = str(response.content)
        valid, invalid = validate_citations(answer, evidence)
        if not valid or not answer_has_required_citations(answer, citations):
            return {
                "answer": REFUSAL_TEXT,
                "citations": citations,
                "citations_valid": False,
                "invalid_citations": invalid,
            }
        return {"answer": answer, "citations": citations, "citations_valid": True, "invalid_citations": []}

    graph = StateGraph(QueryState)
    graph.add_node("normalize_query", normalize_query)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("call_model", call_model)
    graph.add_edge(START, "normalize_query")
    graph.add_edge("normalize_query", "retrieve")
    graph.add_edge("retrieve", "call_model")
    graph.add_edge("call_model", END)
    return graph.compile()
```

- [ ] **Step 4: Update query runtime imports**

In `src/imperial_rag/query/runtime.py`, replace:

```python
from imperial_rag.answering import build_strict_messages
from imperial_rag.workflows import build_query_workflow, rank_hybrid_candidates
```

with:

```python
from imperial_rag.query.answering import build_strict_messages
from imperial_rag.query.retrieval import rank_hybrid_candidates
from imperial_rag.query.workflow import build_query_workflow
```

Replace the fallback import inside `generate()`:

```python
from imperial_rag.answering import REFUSAL_TEXT
```

with:

```python
from imperial_rag.query.answering import REFUSAL_TEXT
```

- [ ] **Step 5: Add query compatibility wrappers**

Create `src/imperial_rag/answering.py` with this complete content:

```python
from __future__ import annotations

from imperial_rag.query.answering import *
```

Create `src/imperial_rag/runtime.py` with this complete content:

```python
from __future__ import annotations

from imperial_rag.query.runtime import *
```

Create `src/imperial_rag/workflows.py` with this complete content:

```python
from __future__ import annotations

from imperial_rag.ingestion.workflow import *
from imperial_rag.query.retrieval import *
from imperial_rag.query.workflow import *
```

- [ ] **Step 6: Update query script imports**

In `scripts/query.py`, replace:

```python
from imperial_rag.runtime import create_runtime
```

with:

```python
from imperial_rag.query.runtime import create_runtime
```

Replace:

```python
from imperial_rag.runtime import Runtime
```

with:

```python
from imperial_rag.query.runtime import Runtime
```

Replace:

```python
from imperial_rag.runtime import build_live_query_workflow
```

with:

```python
from imperial_rag.query.runtime import build_live_query_workflow
```

- [ ] **Step 7: Run query tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_answering.py tests/test_runtime.py tests/test_workflows.py tests/test_module_structure.py -q
```

Expected: answering, runtime, workflow, and query import tests pass except for `tests/test_module_structure.py` imports that belong to app, evals, or observability packages not moved yet.

- [ ] **Step 8: Checkpoint**

Run:

```bash
git status --short
```

Expected in this workspace:

```text
fatal: not a git repository (or any of the parent directories): .git
```

If executing from a Git checkout, commit with:

```bash
git add src/imperial_rag scripts/query.py tests/test_module_structure.py
git commit -m "refactor: move query modules"
```

---

### Task 6: Move Observability, App, And Phoenix Eval Helpers

**Files:**
- Create: `src/imperial_rag/observability/__init__.py`
- Create: `src/imperial_rag/observability/phoenix.py`
- Create: `src/imperial_rag/app/__init__.py`
- Create: `src/imperial_rag/app/web.py`
- Create: `src/imperial_rag/evals/__init__.py`
- Create: `src/imperial_rag/evals/phoenix.py`
- Modify: `src/imperial_rag/tracing.py`
- Modify: `src/imperial_rag/web_app.py`
- Modify: `scripts/ingest.py`
- Modify: `scripts/query.py`
- Modify: `scripts/run_phoenix_eval.py`
- Modify: `tests/test_evals.py`
- Test: `tests/test_tracing.py`
- Test: `tests/test_web_app.py`
- Test: `tests/test_evals.py`
- Test: `tests/test_scripts.py`
- Test: `tests/test_module_structure.py`

- [ ] **Step 1: Move Phoenix tracing module**

Run:

```bash
mkdir -p src/imperial_rag/observability
touch src/imperial_rag/observability/__init__.py
mv src/imperial_rag/tracing.py src/imperial_rag/observability/phoenix.py
```

Expected: command exits successfully.

Create `src/imperial_rag/tracing.py` with this complete content:

```python
from __future__ import annotations

from imperial_rag.observability.phoenix import *
```

- [ ] **Step 2: Update tracing imports in scripts and app**

In `scripts/ingest.py`, replace:

```python
from imperial_rag.tracing import configure_phoenix_tracing
```

with:

```python
from imperial_rag.observability.phoenix import configure_phoenix_tracing
```

In `scripts/query.py`, replace:

```python
from imperial_rag.tracing import configure_phoenix_tracing
```

with:

```python
from imperial_rag.observability.phoenix import configure_phoenix_tracing
```

- [ ] **Step 3: Move Streamlit app module**

Run:

```bash
mkdir -p src/imperial_rag/app
touch src/imperial_rag/app/__init__.py
mv src/imperial_rag/web_app.py src/imperial_rag/app/web.py
```

Expected: command exits successfully.

In `src/imperial_rag/app/web.py`, replace:

```python
from imperial_rag.tracing import configure_phoenix_tracing
```

with:

```python
from imperial_rag.observability.phoenix import configure_phoenix_tracing
```

Replace:

```python
from imperial_rag.runtime import create_runtime
```

with:

```python
from imperial_rag.query.runtime import create_runtime
```

Replace:

```python
from imperial_rag.runtime import Runtime
```

with:

```python
from imperial_rag.query.runtime import Runtime
```

Replace:

```python
from imperial_rag.runtime import build_live_query_workflow
```

with:

```python
from imperial_rag.query.runtime import build_live_query_workflow
```

Create `src/imperial_rag/web_app.py` with this complete content:

```python
from __future__ import annotations

from imperial_rag.app.web import *
```

- [ ] **Step 4: Move Phoenix eval helper module**

Run:

```bash
mkdir -p src/imperial_rag/evals
touch src/imperial_rag/evals/__init__.py
cp scripts/run_phoenix_eval.py src/imperial_rag/evals/phoenix.py
```

Expected: command exits successfully.

In `src/imperial_rag/evals/phoenix.py`, update runtime and tracing imports:

```python
from imperial_rag.query.runtime import build_live_query_workflow
from imperial_rag.query.runtime import Runtime
from imperial_rag.query.runtime import create_runtime
from imperial_rag.observability.phoenix import configure_phoenix_tracing
```

Keep `main`, `load_questions`, `target`, `run_target`, `build_runtime`, `citation_behavior`, `source_hint_behavior`, `phoenix_citation_behavior`, `phoenix_source_hint_behavior`, `_run_phoenix_experiment`, `_to_phoenix_dataset_rows`, and `_experiment_identifier` in `src/imperial_rag/evals/phoenix.py`.

- [ ] **Step 5: Replace the CLI script with a thin entrypoint that re-exports helpers**

Replace `scripts/run_phoenix_eval.py` with this complete content:

```python
from __future__ import annotations

import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()

from imperial_rag.evals.phoenix import (  # noqa: E402
    citation_behavior,
    load_questions,
    main,
    phoenix_citation_behavior,
    phoenix_source_hint_behavior,
    run_local_eval,
    run_target,
    source_hint_behavior,
    target,
)
from imperial_rag.evals.phoenix import _experiment_identifier, _run_phoenix_experiment, _to_phoenix_dataset_rows  # noqa: E402


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Update eval source-inspection test**

In `tests/test_evals.py`, replace:

```python
source = Path("scripts/run_phoenix_eval.py").read_text(encoding="utf-8")
```

with:

```python
source = Path("src/imperial_rag/evals/phoenix.py").read_text(encoding="utf-8")
```

Keep the existing assertions that check:

```python
assert "from phoenix.client import Client" in source
assert "client.datasets.create_dataset" in source
assert "client.experiments.run_experiment" in source
```

- [ ] **Step 7: Run observability, app, eval, and module tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_tracing.py tests/test_web_app.py tests/test_evals.py tests/test_scripts.py tests/test_module_structure.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Checkpoint**

Run:

```bash
git status --short
```

Expected in this workspace:

```text
fatal: not a git repository (or any of the parent directories): .git
```

If executing from a Git checkout, commit with:

```bash
git add src/imperial_rag scripts/run_phoenix_eval.py scripts/ingest.py scripts/query.py tests/test_evals.py tests/test_module_structure.py
git commit -m "refactor: move app observability and eval modules"
```

---

### Task 7: Clean Up Internal Imports And Verify Full Suite

**Files:**
- Modify: `src/imperial_rag/**/*.py`
- Modify: `scripts/ingest.py`
- Modify: `scripts/query.py`
- Modify: `scripts/run_phoenix_eval.py`
- Test: all tests

- [ ] **Step 1: Find stale imports from old internal paths**

Run:

```bash
rg -n "from imperial_rag\\.(answering|chunking|extraction|manifest|ocr|pipeline|runtime|tracing|workflows) import|import imperial_rag\\.(answering|chunking|extraction|manifest|ocr|pipeline|runtime|tracing|workflows)" src scripts tests
```

Expected: only compatibility wrappers and tests that explicitly assert old public imports remain. Production modules under new packages should import other new packages directly.

- [ ] **Step 2: Replace stale production imports with new paths**

Use these replacements in production modules when they appear:

```python
from imperial_rag.answering import X -> from imperial_rag.query.answering import X
from imperial_rag.chunking import X -> from imperial_rag.corpus.chunking import X
from imperial_rag.extraction import X -> from imperial_rag.corpus.extraction import X
from imperial_rag.manifest import X -> from imperial_rag.corpus.manifest import X
from imperial_rag.ocr import X -> from imperial_rag.corpus.ocr import X
from imperial_rag.pipeline import X -> from imperial_rag.ingestion.pipeline import X
from imperial_rag.runtime import X -> from imperial_rag.query.runtime import X
from imperial_rag.tracing import X -> from imperial_rag.observability.phoenix import X
from imperial_rag.workflows import build_ingestion_workflow -> from imperial_rag.ingestion.workflow import build_ingestion_workflow
from imperial_rag.workflows import build_query_workflow -> from imperial_rag.query.workflow import build_query_workflow
from imperial_rag.workflows import rank_hybrid_candidates -> from imperial_rag.query.retrieval import rank_hybrid_candidates
```

Do not replace old imports inside `tests/test_module_structure.py` because that test intentionally verifies compatibility wrappers.

- [ ] **Step 3: Run import compilation**

Run:

```bash
uv run --extra dev python -m compileall src scripts
```

Expected: compilation succeeds without `SyntaxError` or `ImportError` output.

- [ ] **Step 4: Run full test suite**

Run:

```bash
uv run --extra dev python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Inspect remaining old public wrappers**

Run:

```bash
for file in answering.py chunking.py extraction.py manifest.py ocr.py pipeline.py runtime.py tracing.py web_app.py workflows.py; do
  printf '%s\n' "src/imperial_rag/$file"
  sed -n '1,20p' "src/imperial_rag/$file"
done
```

Expected: each listed file is a thin re-export wrapper with `from __future__ import annotations` and imports from the new package location.

- [ ] **Step 6: Final checkpoint**

Run:

```bash
git status --short
```

Expected in this workspace:

```text
fatal: not a git repository (or any of the parent directories): .git
```

If executing from a Git checkout, commit with:

```bash
git add src/imperial_rag scripts tests
git commit -m "refactor: organize imperial rag modules"
```

---

## Final Verification

Run:

```bash
uv run --extra dev python -m pytest -q
```

Expected: all tests pass.

Run:

```bash
uv run --extra dev python -m compileall src scripts
```

Expected: all Python files compile successfully.

Run:

```bash
uv run --extra dev python scripts/run_phoenix_eval.py --questions-path evals/questions.jsonl
```

Expected output includes:

```text
local_eval_examples=
local_eval_passed=
```

This final eval command may report a low pass count if local indexes or services are not populated. For this readability migration, the important verification is that the command imports and runs through local deterministic mode without module import errors.
