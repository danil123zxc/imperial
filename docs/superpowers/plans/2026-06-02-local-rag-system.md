# Local RAG System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local/private RAG web chat for `/Users/danil/Public/imperial/documents` with Qdrant, LangChain integrations, LangGraph workflows, LangSmith tracing/evals, full file manifesting, OCR, strict citations, and auditability.

**Architecture:** A Python package owns ingestion, extraction, chunking, indexing, retrieval, strict answer generation, and a local web chat. LangChain integrations are preferred for loaders, documents, splitters, embeddings, Qdrant access, retrievers, model calls, and output parsing; LangGraph orchestrates ingestion/query workflows; LangSmith traces and evaluates the system. Qdrant is the local vector DB; a local keyword index handles exact Russian terms; custom code exists only for corpus-specific manifesting, citation formatting, embedded DOCX image handling, archive policy, and UI glue.

**Tech Stack:** Python 3.12, LangChain, LangGraph, LangSmith, Qdrant local Docker, LangChain Qdrant integration, LangChain OpenAI integration, SQLite, pytest, Streamlit.

---

Supersession note: `2026-06-03-phoenix-observability-design.md` supersedes the LangSmith tracing and evaluation tasks in this plan. Use the Phoenix observability implementation plan for that slice.

## Workspace Paths

All paths are under `/Users/danil/Public/imperial`.

## File Structure

- Create: `pyproject.toml` - package metadata, dependencies, pytest config.
- Create: `src/imperial_rag/__init__.py` - package marker.
- Create: `src/imperial_rag/config.py` - paths, environment config, LangSmith defaults.
- Create: `src/imperial_rag/manifest.py` - file scanning, hashing, manifest record model, SQLite persistence.
- Create: `src/imperial_rag/extraction.py` - extraction adapters returning LangChain `Document` objects.
- Create: `src/imperial_rag/ocr.py` - AI API OCR wrapper using LangChain chat model integration.
- Create: `src/imperial_rag/chunking.py` - LangChain text splitter and citation metadata preservation.
- Create: `src/imperial_rag/indexing.py` - Qdrant vector indexing through LangChain integration plus local keyword index.
- Create: `src/imperial_rag/pipeline.py` - end-to-end ingestion pipeline, artifact persistence, and indexing coordination.
- Create: `src/imperial_rag/workflows.py` - LangGraph ingestion and query workflows.
- Create: `src/imperial_rag/answering.py` - strict citation prompt, refusal behavior, source formatting.
- Create: `src/imperial_rag/runtime.py` - live query dependency wiring for CLI and web app.
- Create: `src/imperial_rag/web_app.py` - Streamlit local web chat and ingestion status panel.
- Create: `scripts/start_qdrant.sh` - local-only Qdrant startup helper.
- Create: `scripts/ingest.py` - CLI entrypoint for ingestion.
- Create: `scripts/query.py` - CLI query smoke test.
- Create: `scripts/run_langsmith_eval.py` - LangSmith evaluation runner.
- Create: `evals/questions.jsonl` - gold Russian evaluation questions.
- Create: `tests/` - focused pytest coverage for each subsystem.

Commit checkpoints are included for a future git repo. The current workspace is not a git repository, so commit steps will fail until git is initialized.

## Preflight

- [ ] **Step 0: Record the editable-install requirement for script smoke checks**

Do not run this yet; Task 1 creates `pyproject.toml`. After Task 1 succeeds, run:

```bash
python -m pip install -e ".[dev]"
```

Expected after Task 1: command exits `0`. This makes `src/imperial_rag` importable for `python scripts/...` commands. If running a script before editable install, prefix the command with `PYTHONPATH=src`.

- [ ] **Step 1: Confirm the design spec exists**

Run:

```bash
test -f docs/superpowers/specs/2026-06-02-local-rag-system-design.md
```

Expected: command exits `0`.

- [ ] **Step 2: Confirm Qdrant is planned as local-only**

Run:

```bash
rg -n "Qdrant|Do not expose Qdrant|LangChain Ecosystem Policy|LangSmith" docs/superpowers/specs/2026-06-02-local-rag-system-design.md
```

Expected: output includes the Qdrant, LangChain, LangGraph, and LangSmith requirements.

---

### Task 1: Project Skeleton And Configuration

**Files:**
- Create: `/Users/danil/Public/imperial/pyproject.toml`
- Create: `/Users/danil/Public/imperial/src/imperial_rag/__init__.py`
- Create: `/Users/danil/Public/imperial/src/imperial_rag/config.py`
- Create: `/Users/danil/Public/imperial/tests/test_config.py`

- [ ] **Step 1: Write the failing config test**

Create `tests/test_config.py`:

```python
from pathlib import Path

from imperial_rag.config import Settings


def test_settings_defaults_to_workspace_documents():
    settings = Settings()

    assert settings.workspace_root == Path("/Users/danil/Public/imperial")
    assert settings.documents_root == Path("/Users/danil/Public/imperial/documents")
    assert settings.processed_root == Path("/Users/danil/Public/imperial/.imperial_rag")
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "imperial_chunks"
    assert settings.langsmith_project == "imperial-rag"
    assert settings.keyword_db_path == Path("/Users/danil/Public/imperial/.imperial_rag/keyword.sqlite3")


def test_settings_reads_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("IMPERIAL_RAG_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_chunks")
    monkeypatch.setenv("LANGSMITH_PROJECT", "test-project")

    settings = Settings()

    assert settings.workspace_root == tmp_path
    assert settings.documents_root == tmp_path / "documents"
    assert settings.processed_root == tmp_path / ".imperial_rag"
    assert settings.qdrant_url == "http://127.0.0.1:6333"
    assert settings.qdrant_collection == "test_chunks"
    assert settings.langsmith_project == "test-project"
```

- [ ] **Step 2: Run the config test to verify it fails**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: FAIL because `imperial_rag.config` does not exist.

- [ ] **Step 3: Create package metadata**

Create `pyproject.toml`:

```toml
[project]
name = "imperial-rag"
version = "0.1.0"
description = "Local/private RAG system for the Imperial document corpus"
requires-python = ">=3.12"
dependencies = [
  "langchain",
  "langchain-community",
  "langchain-core",
  "langchain-openai",
  "langchain-qdrant",
  "langchain-text-splitters",
  "langgraph",
  "langsmith",
  "qdrant-client",
  "python-docx",
  "openpyxl",
  "pypdf",
  "pymupdf",
  "pillow",
  "striprtf",
  "streamlit",
]

[project.optional-dependencies]
dev = [
  "pytest",
]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

Create `src/imperial_rag/__init__.py`:

```python
"""Local RAG system for the Imperial document corpus."""
```

- [ ] **Step 4: Implement settings**

Create `src/imperial_rag/config.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_WORKSPACE_ROOT = Path("/Users/danil/Public/imperial")


@dataclass(frozen=True)
class Settings:
    workspace_root: Path = field(default_factory=lambda: Path(os.environ.get("IMPERIAL_RAG_WORKSPACE_ROOT", DEFAULT_WORKSPACE_ROOT)))
    qdrant_url: str = field(default_factory=lambda: os.environ.get("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(default_factory=lambda: os.environ.get("QDRANT_COLLECTION", "imperial_chunks"))
    langsmith_project: str = field(default_factory=lambda: os.environ.get("LANGSMITH_PROJECT", "imperial-rag"))

    @property
    def documents_root(self) -> Path:
        return self.workspace_root / "documents"

    @property
    def processed_root(self) -> Path:
        return self.workspace_root / ".imperial_rag"

    @property
    def manifest_db_path(self) -> Path:
        return self.processed_root / "manifest.sqlite3"

    @property
    def keyword_db_path(self) -> Path:
        return self.processed_root / "keyword.sqlite3"

    @property
    def extraction_root(self) -> Path:
        return self.processed_root / "extracted"
```

- [ ] **Step 5: Run the config test to verify it passes**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Install the package in editable mode for future script smoke checks**

Run:

```bash
python -m pip install -e ".[dev]"
```

Expected: command exits `0`.

- [ ] **Step 7: Commit checkpoint if git is initialized**

Run:

```bash
git add pyproject.toml src/imperial_rag/__init__.py src/imperial_rag/config.py tests/test_config.py
git commit -m "chore: scaffold local rag project"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 2: Full File Manifest

**Files:**
- Create: `/Users/danil/Public/imperial/src/imperial_rag/manifest.py`
- Create: `/Users/danil/Public/imperial/tests/test_manifest.py`

- [ ] **Step 1: Write failing manifest scanner tests**

Create `tests/test_manifest.py`:

```python
from pathlib import Path

from imperial_rag.manifest import FileStatus, scan_files


def test_scan_files_records_every_file_including_temp_and_archives(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "~$lock.docx").write_bytes(b"lock")
    (docs / ".~lock.file.docx#").write_bytes(b"lock2")
    (docs / "Thumbs.db").write_bytes(b"thumbs")
    (docs / "archive.rar").write_bytes(b"rar")
    (docs / "policy.docx").write_bytes(b"docx")

    records = scan_files(docs)

    assert {record.relative_path for record in records} == {
        Path("~$lock.docx"),
        Path(".~lock.file.docx#"),
        Path("Thumbs.db"),
        Path("archive.rar"),
        Path("policy.docx"),
    }


def test_scan_files_hashes_duplicates_without_removing_them(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "a.docx").write_bytes(b"same")
    (docs / "b.docx").write_bytes(b"same")

    records = scan_files(docs)

    assert len(records) == 2
    assert records[0].sha256 == records[1].sha256
    assert all(record.status == FileStatus.PENDING for record in records)
```

- [ ] **Step 2: Run manifest tests to verify they fail**

Run:

```bash
python -m pytest tests/test_manifest.py -q
```

Expected: FAIL because `imperial_rag.manifest` does not exist.

- [ ] **Step 3: Implement manifest scanner**

Create `src/imperial_rag/manifest.py`:

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path


class FileStatus(StrEnum):
    PENDING = "pending"
    INDEXED = "indexed"
    MANIFEST_ONLY = "manifest_only"
    NO_TEXT = "no_text"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class IndexStatus(StrEnum):
    PENDING = "pending"
    INDEXED = "indexed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class FileRecord:
    file_id: str
    absolute_path: Path
    relative_path: Path
    filename: str
    extension: str
    size_bytes: int
    sha256: str
    modified_ns: int
    parent_folder: Path
    inferred_category: str
    status: FileStatus = FileStatus.PENDING
    extraction_method: str | None = None
    error_message: str | None = None
    chunk_count: int = 0
    duplicate_group_id: str | None = None
    keyword_index_status: IndexStatus = IndexStatus.PENDING
    vector_index_status: IndexStatus = IndexStatus.PENDING
    embedding_model: str | None = None
    index_error_message: str | None = None
    last_indexed_ns: int = 0


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_file_id(relative_path: Path) -> str:
    normalized = relative_path.as_posix()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


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
    return records
```

- [ ] **Step 4: Run manifest tests to verify they pass**

Run:

```bash
python -m pytest tests/test_manifest.py -q
```

Expected: PASS.

- [ ] **Step 5: Add duplicate group helper test**

Append to `tests/test_manifest.py`:

```python
from imperial_rag.manifest import assign_duplicate_groups


def test_assign_duplicate_groups_marks_same_hash_records(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "a.pdf").write_bytes(b"same")
    (docs / "b.pdf").write_bytes(b"same")
    (docs / "c.pdf").write_bytes(b"different")

    records = assign_duplicate_groups(scan_files(docs))
    duplicate_groups = {record.relative_path: record.duplicate_group_id for record in records}

    assert duplicate_groups[Path("a.pdf")] is not None
    assert duplicate_groups[Path("a.pdf")] == duplicate_groups[Path("b.pdf")]
    assert duplicate_groups[Path("c.pdf")] is None
```

- [ ] **Step 6: Extend `FileRecord` and implement duplicate groups**

Modify `src/imperial_rag/manifest.py`:

```python
@dataclass(frozen=True)
class FileRecord:
    file_id: str
    absolute_path: Path
    relative_path: Path
    filename: str
    extension: str
    size_bytes: int
    sha256: str
    modified_ns: int
    parent_folder: Path
    inferred_category: str
    status: FileStatus = FileStatus.PENDING
    extraction_method: str | None = None
    error_message: str | None = None
    chunk_count: int = 0
    duplicate_group_id: str | None = None
    keyword_index_status: IndexStatus = IndexStatus.PENDING
    vector_index_status: IndexStatus = IndexStatus.PENDING
    embedding_model: str | None = None
    index_error_message: str | None = None
    last_indexed_ns: int = 0
```

Add below `scan_files`:

```python
def assign_duplicate_groups(records: list[FileRecord]) -> list[FileRecord]:
    by_hash: dict[str, list[FileRecord]] = {}
    for record in records:
        by_hash.setdefault(record.sha256, []).append(record)

    grouped: list[FileRecord] = []
    for record in records:
        matches = by_hash[record.sha256]
        group_id = f"sha256:{record.sha256}" if len(matches) > 1 else None
        grouped.append(replace(record, duplicate_group_id=group_id))
    return grouped
```

- [ ] **Step 7: Run manifest tests**

Run:

```bash
python -m pytest tests/test_manifest.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit checkpoint if git is initialized**

Run:

```bash
git add src/imperial_rag/manifest.py tests/test_manifest.py
git commit -m "feat: add full file manifest scanner"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 2A: SQLite Manifest Persistence And Audit Status

**Files:**
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/manifest.py`
- Create: `/Users/danil/Public/imperial/tests/test_manifest_store.py`

- [ ] **Step 1: Write failing manifest persistence tests**

Create `tests/test_manifest_store.py`:

```python
from pathlib import Path

from imperial_rag.manifest import FileStatus, IndexStatus, ManifestStore, assign_duplicate_groups, scan_files


def test_manifest_store_persists_every_scanned_file(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.docx").write_bytes(b"docx")
    (docs / "archive.rar").write_bytes(b"rar")
    records = assign_duplicate_groups(scan_files(docs))

    store = ManifestStore(tmp_path / "manifest.sqlite3")
    store.replace_records(records)
    loaded = store.list_records()

    assert {record.relative_path for record in loaded} == {Path("policy.docx"), Path("archive.rar")}
    assert all(record.status == FileStatus.PENDING for record in loaded)


def test_manifest_store_records_status_errors_and_chunk_counts(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "scan.pdf").write_bytes(b"%PDF")
    record = scan_files(docs)[0]
    store = ManifestStore(tmp_path / "manifest.sqlite3")
    store.replace_records([record])

    store.update_status(
        file_id=record.file_id,
        status=FileStatus.FAILED,
        extraction_method="pdf_ocr",
        error_message="render failed",
        chunk_count=0,
    )
    loaded = store.get_record(record.file_id)

    assert loaded.status == FileStatus.FAILED
    assert loaded.extraction_method == "pdf_ocr"
    assert loaded.error_message == "render failed"
    assert loaded.chunk_count == 0


def test_manifest_store_records_keyword_and_vector_index_status(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.docx").write_bytes(b"docx")
    record = scan_files(docs)[0]
    store = ManifestStore(tmp_path / "manifest.sqlite3")
    store.replace_records([record])

    store.update_index_status(
        file_id=record.file_id,
        keyword_index_status=IndexStatus.INDEXED,
        vector_index_status=IndexStatus.FAILED,
        embedding_model="text-embedding-3-large",
        index_error_message="qdrant unavailable",
    )
    loaded = store.get_record(record.file_id)

    assert loaded.keyword_index_status == IndexStatus.INDEXED
    assert loaded.vector_index_status == IndexStatus.FAILED
    assert loaded.embedding_model == "text-embedding-3-large"
    assert loaded.index_error_message == "qdrant unavailable"
    assert loaded.last_indexed_ns > 0
```

- [ ] **Step 2: Run manifest persistence tests to verify they fail**

Run:

```bash
python -m pytest tests/test_manifest_store.py -q
```

Expected: FAIL because `ManifestStore` does not exist.

- [ ] **Step 3: Implement SQLite manifest store**

Append to `src/imperial_rag/manifest.py`:

```python
import sqlite3
import time


class ManifestStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                file_id TEXT PRIMARY KEY,
                absolute_path TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                extension TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                modified_ns INTEGER NOT NULL,
                parent_folder TEXT NOT NULL,
                inferred_category TEXT NOT NULL,
                status TEXT NOT NULL,
                extraction_method TEXT,
                error_message TEXT,
                chunk_count INTEGER NOT NULL,
                duplicate_group_id TEXT,
                keyword_index_status TEXT NOT NULL,
                vector_index_status TEXT NOT NULL,
                embedding_model TEXT,
                index_error_message TEXT,
                last_updated_ns INTEGER NOT NULL DEFAULT 0,
                last_indexed_ns INTEGER NOT NULL DEFAULT 0
            )
            """
        )

    def replace_records(self, records: list[FileRecord]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM files")
            self._conn.executemany(
                """
                INSERT INTO files(
                    file_id, absolute_path, relative_path, filename, extension,
                    size_bytes, sha256, modified_ns, parent_folder, inferred_category,
                    status, extraction_method, error_message, chunk_count, duplicate_group_id,
                    keyword_index_status, vector_index_status, embedding_model, index_error_message,
                    last_indexed_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self._to_row(record) for record in records],
            )

    def update_status(
        self,
        file_id: str,
        status: FileStatus,
        extraction_method: str | None = None,
        error_message: str | None = None,
        chunk_count: int = 0,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE files
                SET status = ?, extraction_method = ?, error_message = ?, chunk_count = ?,
                    last_updated_ns = ?
                WHERE file_id = ?
                """,
                (status.value, extraction_method, error_message, chunk_count, time.time_ns(), file_id),
            )

    def update_index_status(
        self,
        file_id: str,
        keyword_index_status: IndexStatus,
        vector_index_status: IndexStatus,
        embedding_model: str | None = None,
        index_error_message: str | None = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE files
                SET keyword_index_status = ?, vector_index_status = ?, embedding_model = ?,
                    index_error_message = ?, last_indexed_ns = ?
                WHERE file_id = ?
                """,
                (
                    keyword_index_status.value,
                    vector_index_status.value,
                    embedding_model,
                    index_error_message,
                    time.time_ns(),
                    file_id,
                ),
            )

    def get_record(self, file_id: str) -> FileRecord:
        row = self._conn.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
        if row is None:
            raise KeyError(file_id)
        return self._from_row(row)

    def list_records(self) -> list[FileRecord]:
        rows = self._conn.execute("SELECT * FROM files ORDER BY relative_path").fetchall()
        return [self._from_row(row) for row in rows]

    @staticmethod
    def _to_row(record: FileRecord) -> tuple[object, ...]:
        return (
            record.file_id,
            str(record.absolute_path),
            record.relative_path.as_posix(),
            record.filename,
            record.extension,
            record.size_bytes,
            record.sha256,
            record.modified_ns,
            record.parent_folder.as_posix(),
            record.inferred_category,
            record.status.value,
            record.extraction_method,
            record.error_message,
            record.chunk_count,
            record.duplicate_group_id,
            record.keyword_index_status.value,
            record.vector_index_status.value,
            record.embedding_model,
            record.index_error_message,
            record.last_indexed_ns,
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> FileRecord:
        return FileRecord(
            file_id=row["file_id"],
            absolute_path=Path(row["absolute_path"]),
            relative_path=Path(row["relative_path"]),
            filename=row["filename"],
            extension=row["extension"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            modified_ns=row["modified_ns"],
            parent_folder=Path(row["parent_folder"]),
            inferred_category=row["inferred_category"],
            status=FileStatus(row["status"]),
            extraction_method=row["extraction_method"],
            error_message=row["error_message"],
            chunk_count=row["chunk_count"],
            duplicate_group_id=row["duplicate_group_id"],
            keyword_index_status=IndexStatus(row["keyword_index_status"]),
            vector_index_status=IndexStatus(row["vector_index_status"]),
            embedding_model=row["embedding_model"],
            index_error_message=row["index_error_message"],
            last_indexed_ns=row["last_indexed_ns"],
        )
```

- [ ] **Step 4: Run manifest tests**

Run:

```bash
python -m pytest tests/test_manifest.py tests/test_manifest_store.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint if git is initialized**

Run:

```bash
git add src/imperial_rag/manifest.py tests/test_manifest_store.py
git commit -m "feat: persist manifest audit records"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 3: Extraction To LangChain Documents

**Files:**
- Create: `/Users/danil/Public/imperial/src/imperial_rag/extraction.py`
- Create: `/Users/danil/Public/imperial/tests/test_extraction.py`

- [ ] **Step 1: Write failing extraction tests**

Create `tests/test_extraction.py`:

```python
from pathlib import Path

from docx import Document as DocxDocument

from imperial_rag.extraction import extract_file
from imperial_rag.manifest import FileStatus, scan_files


def test_archive_is_manifest_only(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "archive.rar").write_bytes(b"archive")
    record = scan_files(docs)[0]

    result = extract_file(record)

    assert result.status == FileStatus.MANIFEST_ONLY
    assert result.documents == []
    assert "archive files are recorded but not extracted" in result.message


def test_docx_text_and_table_extract_to_langchain_documents(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    path = docs / "policy.docx"
    doc = DocxDocument()
    doc.add_paragraph("Регламент возврата товара")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Ответственный"
    table.cell(0, 1).text = "Склад"
    doc.save(path)
    record = scan_files(docs)[0]

    result = extract_file(record)

    assert result.status == FileStatus.INDEXED
    assert [doc.metadata["source_type"] for doc in result.documents] == ["body", "table"]
    assert "Регламент возврата товара" in result.documents[0].page_content
    assert "Ответственный | Склад" in result.documents[1].page_content
```

- [ ] **Step 2: Run extraction tests to verify they fail**

Run:

```bash
python -m pytest tests/test_extraction.py -q
```

Expected: FAIL because `imperial_rag.extraction` does not exist.

- [ ] **Step 3: Implement extraction result and DOCX/archive extraction**

Create `src/imperial_rag/extraction.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from langchain_core.documents import Document

from imperial_rag.manifest import FileRecord, FileStatus


ARCHIVE_EXTENSIONS = {".rar", ".zip", ".7z"}


@dataclass(frozen=True)
class ExtractionResult:
    record: FileRecord
    status: FileStatus
    documents: list[Document]
    extraction_method: str | None = None
    message: str = ""


def _base_metadata(record: FileRecord, source_type: str) -> dict[str, str | int | None]:
    return {
        "file_id": record.file_id,
        "file_path": str(record.absolute_path),
        "relative_path": str(record.relative_path),
        "file_name": record.filename,
        "file_extension": record.extension,
        "file_hash": record.sha256,
        "duplicate_group_id": record.duplicate_group_id,
        "parent_folder": str(record.parent_folder),
        "inferred_category": record.inferred_category,
        "source_type": source_type,
    }


def _extract_docx(record: FileRecord) -> list[Document]:
    docx = DocxDocument(record.absolute_path)
    documents: list[Document] = []
    body_text = "\n".join(paragraph.text.strip() for paragraph in docx.paragraphs if paragraph.text.strip())
    if body_text:
        documents.append(Document(page_content=body_text, metadata=_base_metadata(record, "body")))

    table_lines: list[str] = []
    for table in docx.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                table_lines.append(" | ".join(cells))
    if table_lines:
        documents.append(Document(page_content="\n".join(table_lines), metadata=_base_metadata(record, "table")))

    return documents


def extract_file(record: FileRecord) -> ExtractionResult:
    if record.extension in ARCHIVE_EXTENSIONS:
        return ExtractionResult(
            record=record,
            status=FileStatus.MANIFEST_ONLY,
            documents=[],
            extraction_method=None,
            message="archive files are recorded but not extracted in v1",
        )
    if record.extension == ".docx":
        documents = _extract_docx(record)
        return ExtractionResult(
            record=record,
            status=FileStatus.INDEXED if documents else FileStatus.NO_TEXT,
            documents=documents,
            extraction_method="python_docx",
        )
    return ExtractionResult(record=record, status=FileStatus.UNSUPPORTED, documents=[], extraction_method=None, message=f"unsupported extension: {record.extension}")
```

- [ ] **Step 4: Run extraction tests**

Run:

```bash
python -m pytest tests/test_extraction.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint if git is initialized**

Run:

```bash
git add src/imperial_rag/extraction.py tests/test_extraction.py
git commit -m "feat: extract docx content to langchain documents"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 4: OCR Hooks And Structure-Aware Chunking

**Files:**
- Create: `/Users/danil/Public/imperial/src/imperial_rag/ocr.py`
- Create: `/Users/danil/Public/imperial/src/imperial_rag/chunking.py`
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/extraction.py`
- Create: `/Users/danil/Public/imperial/tests/test_chunking.py`

- [ ] **Step 1: Write failing chunking tests**

Create `tests/test_chunking.py`:

```python
from langchain_core.documents import Document

from imperial_rag.chunking import build_chunks


def test_build_chunks_preserves_citation_metadata():
    source = Document(
        page_content="Возврат брака оформляется актом. " * 80,
        metadata={
            "file_path": "/docs/reglament.docx",
            "relative_path": "reglament.docx",
            "source_type": "body",
            "file_id": "file123",
            "file_hash": "abc",
        },
    )

    chunks = build_chunks([source])

    assert chunks
    assert all(chunk.metadata["file_path"] == "/docs/reglament.docx" for chunk in chunks)
    assert all(chunk.metadata["source_type"] == "body" for chunk in chunks)
    assert all(chunk.metadata["chunk_id"].startswith("file123:body:") for chunk in chunks)
```

- [ ] **Step 2: Run chunking tests to verify they fail**

Run:

```bash
python -m pytest tests/test_chunking.py -q
```

Expected: FAIL because `imperial_rag.chunking` does not exist.

- [ ] **Step 3: Implement OCR interface using LangChain chat model integration**

Create `src/imperial_rag/ocr.py`:

```python
from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from langchain_openai import ChatOpenAI


@dataclass(frozen=True)
class OcrResult:
    text: str
    method: str
    cached: bool = False


class OcrClient:
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self._model = ChatOpenAI(model=model, temperature=0)

    def extract_image_text(self, image_path: Path) -> OcrResult:
        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        mime_type, _ = mimetypes.guess_type(image_path.name)
        mime_type = mime_type or "image/jpeg"
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all visible Russian and English text verbatim. Do not summarize."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
                ],
            }
        ]
        response = self._model.invoke(message)
        return OcrResult(text=str(response.content).strip(), method="langchain_openai_vision")


class OcrCache:
    def __init__(self, cache_root: Path) -> None:
        self.cache_root = cache_root
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def read(self, cache_key: str) -> OcrResult | None:
        path = self.cache_root / f"{cache_key}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return OcrResult(text=payload["text"], method=payload["method"], cached=True)

    def write(self, cache_key: str, result: OcrResult) -> None:
        path = self.cache_root / f"{cache_key}.json"
        path.write_text(
            json.dumps({"text": result.text, "method": result.method}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
```

- [ ] **Step 4: Implement LangChain text splitter chunking**

Create `src/imperial_rag/chunking.py`:

```python
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def build_chunks(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=250)
    chunks: list[Document] = []
    for document in documents:
        split_docs = splitter.split_documents([document])
        for index, chunk in enumerate(split_docs):
            metadata = dict(chunk.metadata)
            metadata["chunk_index"] = index
            metadata["chunk_id"] = f"{metadata.get('file_id')}:{metadata.get('source_type')}:{index}"
            chunks.append(Document(page_content=chunk.page_content, metadata=metadata))
    return chunks
```

- [ ] **Step 5: Run chunking tests**

Run:

```bash
python -m pytest tests/test_chunking.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit checkpoint if git is initialized**

Run:

```bash
git add src/imperial_rag/ocr.py src/imperial_rag/chunking.py tests/test_chunking.py
git commit -m "feat: add ocr interface and chunk builder"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 4A: Image, PDF, And DOCX Embedded-Image OCR Extraction

**Files:**
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/extraction.py`
- Create: `/Users/danil/Public/imperial/tests/test_ocr_extraction.py`

- [ ] **Step 1: Write failing OCR extraction tests**

Create `tests/test_ocr_extraction.py`:

```python
from pathlib import Path

import fitz
from docx import Document as DocxDocument
from PIL import Image

from imperial_rag.extraction import extract_file
from imperial_rag.ocr import OcrResult
from imperial_rag.manifest import FileStatus, scan_files


class FakeOcrClient:
    def extract_image_text(self, image_path: Path) -> OcrResult:
        return OcrResult(text=f"OCR:{image_path.name}", method="fake_ocr")


def make_image(path: Path) -> None:
    Image.new("RGB", (20, 20), "white").save(path)


def test_jpg_uses_ocr_client(tmp_path):
    docs = tmp_path / "documents"
    artifacts = tmp_path / "artifacts"
    docs.mkdir()
    make_image(docs / "scan.jpg")
    record = scan_files(docs)[0]

    result = extract_file(record, ocr_client=FakeOcrClient(), artifact_root=artifacts)

    assert result.status == FileStatus.INDEXED
    assert result.documents[0].metadata["source_type"] == "image"
    assert result.documents[0].page_content == "OCR:scan.jpg"


def test_pdf_pages_are_rendered_and_ocrd(tmp_path):
    docs = tmp_path / "documents"
    artifacts = tmp_path / "artifacts"
    docs.mkdir()
    pdf_path = docs / "scan.pdf"
    pdf = fitz.open()
    pdf.new_page()
    pdf.save(pdf_path)
    record = scan_files(docs)[0]

    result = extract_file(record, ocr_client=FakeOcrClient(), artifact_root=artifacts)

    assert result.status == FileStatus.INDEXED
    assert result.documents[0].metadata["source_type"] == "page"
    assert result.documents[0].metadata["page_number"] == 1
    assert result.documents[0].page_content.startswith("OCR:scan-page-1")


def test_docx_embedded_images_are_ocrd(tmp_path):
    docs = tmp_path / "documents"
    artifacts = tmp_path / "artifacts"
    docs.mkdir()
    image_path = tmp_path / "inside.jpg"
    make_image(image_path)
    docx_path = docs / "with-image.docx"
    docx = DocxDocument()
    docx.add_paragraph("Основной текст")
    docx.add_picture(str(image_path))
    docx.save(docx_path)
    record = scan_files(docs)[0]

    result = extract_file(record, ocr_client=FakeOcrClient(), artifact_root=artifacts)

    source_types = [document.metadata["source_type"] for document in result.documents]
    assert "body" in source_types
    assert "embedded_image" in source_types
```

- [ ] **Step 2: Run OCR extraction tests to verify they fail**

Run:

```bash
python -m pytest tests/test_ocr_extraction.py -q
```

Expected: FAIL because `extract_file` does not accept `ocr_client` and `artifact_root`.

- [ ] **Step 3: Extend extraction for standalone images, PDF pages, and DOCX embedded images**

Modify `src/imperial_rag/extraction.py` by replacing `_extract_docx` and `extract_file`, and adding the helpers below:

```python
import zipfile
from typing import Protocol

import fitz


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class SupportsOcr(Protocol):
    def extract_image_text(self, image_path: Path):
        ...


def _artifact_dir(record: FileRecord, artifact_root: Path | None) -> Path:
    root = artifact_root or record.absolute_path.parent / ".imperial_rag_artifacts"
    target = root / record.sha256
    target.mkdir(parents=True, exist_ok=True)
    return target


def _ocr_image(record: FileRecord, image_path: Path, source_type: str, ocr_client: SupportsOcr | None, metadata: dict[str, str | int | None]) -> list[Document]:
    if ocr_client is None:
        return []
    ocr_result = ocr_client.extract_image_text(image_path)
    if not ocr_result.text:
        return []
    merged_metadata = _base_metadata(record, source_type)
    merged_metadata.update(metadata)
    merged_metadata["ocr_method"] = ocr_result.method
    return [Document(page_content=ocr_result.text, metadata=merged_metadata)]


def _extract_docx(record: FileRecord, ocr_client: SupportsOcr | None = None, artifact_root: Path | None = None) -> list[Document]:
    docx = DocxDocument(record.absolute_path)
    documents: list[Document] = []
    body_text = "\n".join(paragraph.text.strip() for paragraph in docx.paragraphs if paragraph.text.strip())
    if body_text:
        documents.append(Document(page_content=body_text, metadata=_base_metadata(record, "body")))

    table_lines: list[str] = []
    for table in docx.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                table_lines.append(" | ".join(cells))
    if table_lines:
        documents.append(Document(page_content="\n".join(table_lines), metadata=_base_metadata(record, "table")))

    if ocr_client is not None:
        target_dir = _artifact_dir(record, artifact_root)
        with zipfile.ZipFile(record.absolute_path) as archive:
            media_names = [name for name in archive.namelist() if name.startswith("word/media/")]
            for index, media_name in enumerate(media_names, start=1):
                suffix = Path(media_name).suffix.lower() or ".img"
                image_path = target_dir / f"embedded-{index}{suffix}"
                image_path.write_bytes(archive.read(media_name))
                documents.extend(
                    _ocr_image(
                        record,
                        image_path,
                        "embedded_image",
                        ocr_client,
                        {"image_index": index, "embedded_media_name": media_name},
                    )
                )
    return documents


def _extract_pdf(record: FileRecord, ocr_client: SupportsOcr | None, artifact_root: Path | None) -> list[Document]:
    if ocr_client is None:
        return []
    documents: list[Document] = []
    target_dir = _artifact_dir(record, artifact_root)
    pdf = fitz.open(record.absolute_path)
    for page_index, page in enumerate(pdf, start=1):
        image_path = target_dir / f"{record.absolute_path.stem}-page-{page_index}.jpg"
        pixmap = page.get_pixmap(dpi=200)
        pixmap.save(image_path)
        documents.extend(
            _ocr_image(
                record,
                image_path,
                "page",
                ocr_client,
                {"page_number": page_index, "render_dpi": 200},
            )
        )
    return documents


def extract_file(record: FileRecord, ocr_client: SupportsOcr | None = None, artifact_root: Path | None = None) -> ExtractionResult:
    if record.extension in ARCHIVE_EXTENSIONS:
        return ExtractionResult(
            record=record,
            status=FileStatus.MANIFEST_ONLY,
            documents=[],
            extraction_method=None,
            message="archive files are recorded but not extracted in v1",
        )
    if record.extension == ".docx":
        documents = _extract_docx(record, ocr_client=ocr_client, artifact_root=artifact_root)
        return ExtractionResult(record=record, status=FileStatus.INDEXED if documents else FileStatus.NO_TEXT, documents=documents, extraction_method="python_docx")
    if record.extension == ".pdf":
        documents = _extract_pdf(record, ocr_client=ocr_client, artifact_root=artifact_root)
        return ExtractionResult(record=record, status=FileStatus.INDEXED if documents else FileStatus.NO_TEXT, documents=documents, extraction_method="pymupdf_ocr")
    if record.extension in IMAGE_EXTENSIONS:
        documents = _ocr_image(record, record.absolute_path, "image", ocr_client, {"image_hash": record.sha256})
        return ExtractionResult(record=record, status=FileStatus.INDEXED if documents else FileStatus.NO_TEXT, documents=documents, extraction_method="image_ocr")
    return ExtractionResult(record=record, status=FileStatus.UNSUPPORTED, documents=[], extraction_method=None, message=f"unsupported extension: {record.extension}")
```

- [ ] **Step 4: Run extraction and OCR tests**

Run:

```bash
python -m pytest tests/test_extraction.py tests/test_ocr_extraction.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint if git is initialized**

Run:

```bash
git add pyproject.toml src/imperial_rag/extraction.py tests/test_ocr_extraction.py
git commit -m "feat: add ocr extraction for images pdfs and docx media"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 4B: OCR Cache And Remaining Corpus Format Coverage

**Files:**
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/ocr.py`
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/extraction.py`
- Create: `/Users/danil/Public/imperial/tests/test_ocr_cache.py`
- Create: `/Users/danil/Public/imperial/tests/test_additional_formats.py`

- [ ] **Step 1: Write failing OCR cache test**

Create `tests/test_ocr_cache.py`:

```python
from imperial_rag.ocr import OcrCache, OcrResult


def test_ocr_cache_persists_text_for_reuse(tmp_path):
    cache = OcrCache(tmp_path)
    cache.write("file-page-1", OcrResult(text="Текст приказа", method="fake_ocr"))

    result = cache.read("file-page-1")

    assert result is not None
    assert result.text == "Текст приказа"
    assert result.method == "fake_ocr"
    assert result.cached is True
```

- [ ] **Step 2: Write failing format coverage tests**

Create `tests/test_additional_formats.py`:

```python
from pathlib import Path

from openpyxl import Workbook

from imperial_rag.extraction import extract_file
from imperial_rag.manifest import FileStatus, scan_files


def test_xlsx_sheets_extract_to_structured_text(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    path = docs / "schedule.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "График"
    sheet.append(["Сотрудник", "Смена"])
    sheet.append(["Иванов", "Утро"])
    workbook.save(path)
    record = scan_files(docs)[0]

    result = extract_file(record)

    assert result.status == FileStatus.INDEXED
    assert result.documents[0].metadata["source_type"] == "sheet"
    assert result.documents[0].metadata["sheet_name"] == "График"
    assert "Сотрудник | Смена" in result.documents[0].page_content


def test_rtf_extracts_text_when_parser_available(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "note.rtf").write_text(r"{\rtf1\ansi Регламент склада}", encoding="utf-8")
    record = scan_files(docs)[0]

    result = extract_file(record)

    assert result.status == FileStatus.INDEXED
    assert "Регламент склада" in result.documents[0].page_content


def test_legacy_doc_is_audited_as_unsupported_without_safe_converter(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "legacy.doc").write_bytes(b"legacy binary doc")
    record = scan_files(docs)[0]

    result = extract_file(record)

    assert result.status == FileStatus.UNSUPPORTED
    assert "legacy .doc requires a safe local converter" in result.message
```

- [ ] **Step 3: Run cache and format tests to verify they fail**

Run:

```bash
python -m pytest tests/test_ocr_cache.py tests/test_additional_formats.py -q
```

Expected: FAIL because the additional format extraction and cache behavior are not fully wired.

- [ ] **Step 4: Wire OCR cache into image OCR extraction**

Modify `src/imperial_rag/extraction.py` so `extract_file`, `_extract_docx`, `_extract_pdf`, and `_ocr_image` accept `ocr_cache: OcrCache | None = None`.

Use a deterministic OCR cache key:

```python
def _ocr_cache_key(record: FileRecord, source_type: str, metadata: dict[str, str | int | None]) -> str:
    locator = ":".join(f"{key}={value}" for key, value in sorted(metadata.items()))
    return f"{record.file_id}:{record.sha256}:{source_type}:{locator or 'root'}"
```

Inside `_ocr_image`, read from cache before calling the OCR client, then write the OCR result after a successful API call:

```python
cache_key = _ocr_cache_key(record, source_type, metadata)
ocr_result = ocr_cache.read(cache_key) if ocr_cache is not None else None
if ocr_result is None:
    if ocr_client is None:
        return []
    ocr_result = ocr_client.extract_image_text(image_path)
    if ocr_cache is not None and ocr_result.text:
        ocr_cache.write(cache_key, ocr_result)
```

Expected: every OCR result for PDF pages, standalone images, and DOCX embedded images is persisted under `.imperial_rag/extracted/ocr-cache`.

- [ ] **Step 5: Add XLSX and RTF extraction**

Add helpers in `src/imperial_rag/extraction.py`:

```python
from openpyxl import load_workbook
from striprtf.striprtf import rtf_to_text


def _extract_xlsx(record: FileRecord) -> list[Document]:
    workbook = load_workbook(record.absolute_path, data_only=True, read_only=True)
    documents: list[Document] = []
    for sheet in workbook.worksheets:
        lines: list[str] = []
        for row in sheet.iter_rows(values_only=True):
            cells = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
            if cells:
                lines.append(" | ".join(cells))
        if lines:
            metadata = _base_metadata(record, "sheet")
            metadata["sheet_name"] = sheet.title
            documents.append(Document(page_content="\n".join(lines), metadata=metadata))
    return documents


def _extract_rtf(record: FileRecord) -> list[Document]:
    text = rtf_to_text(record.absolute_path.read_text(encoding="utf-8", errors="ignore")).strip()
    if not text:
        return []
    return [Document(page_content=text, metadata=_base_metadata(record, "body"))]
```

Extend `extract_file`:

```python
    if record.extension == ".xlsx":
        documents = _extract_xlsx(record)
        return ExtractionResult(record=record, status=FileStatus.INDEXED if documents else FileStatus.NO_TEXT, documents=documents, extraction_method="openpyxl")
    if record.extension == ".rtf":
        documents = _extract_rtf(record)
        return ExtractionResult(record=record, status=FileStatus.INDEXED if documents else FileStatus.NO_TEXT, documents=documents, extraction_method="striprtf")
    if record.extension == ".doc":
        return ExtractionResult(
            record=record,
            status=FileStatus.UNSUPPORTED,
            documents=[],
            extraction_method=None,
            message="legacy .doc requires a safe local converter; recorded but not extracted in v1",
        )
```

- [ ] **Step 6: Run extraction coverage tests**

Run:

```bash
python -m pytest tests/test_ocr_cache.py tests/test_additional_formats.py tests/test_ocr_extraction.py tests/test_extraction.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit checkpoint if git is initialized**

Run:

```bash
git add pyproject.toml src/imperial_rag/ocr.py src/imperial_rag/extraction.py tests/test_ocr_cache.py tests/test_additional_formats.py
git commit -m "feat: persist ocr text and cover remaining corpus formats"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 5: Qdrant Vector Index And Keyword Search

**Files:**
- Create: `/Users/danil/Public/imperial/src/imperial_rag/indexing.py`
- Create: `/Users/danil/Public/imperial/tests/test_indexing.py`

- [ ] **Step 1: Write failing keyword index test**

Create `tests/test_indexing.py`:

```python
from langchain_core.documents import Document

from imperial_rag.indexing import KeywordIndex, index_documents


def test_keyword_index_finds_exact_russian_term(tmp_path):
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    docs = [
        Document(page_content="Регламент возврата брака из магазина", metadata={"chunk_id": "a"}),
        Document(page_content="Должностная инструкция водителя", metadata={"chunk_id": "b"}),
    ]

    index.replace_all(docs)
    results = index.search("возврат брака", limit=5)

    assert [result.metadata["chunk_id"] for result in results] == ["a"]


def test_keyword_index_handles_case_and_hyphenated_russian_terms(tmp_path):
    index = KeywordIndex(tmp_path / "keyword.sqlite3")
    docs = [
        Document(page_content="Должностная инструкция водителя-экспедитора", metadata={"chunk_id": "driver"}),
    ]

    index.replace_all(docs)
    results = index.search("ВОДИТЕЛЬ ЭКСПЕДИТОР", limit=5)

    assert [result.metadata["chunk_id"] for result in results] == ["driver"]


def test_index_documents_uses_chunk_ids_as_qdrant_ids():
    class FakeVectorStore:
        def add_documents(self, documents, ids):
            self.documents = documents
            self.ids = ids
            return ids

    docs = [
        Document(page_content="one", metadata={"chunk_id": "file1:body:0"}),
        Document(page_content="two", metadata={"chunk_id": "file1:body:1"}),
    ]
    store = FakeVectorStore()

    ids = index_documents(store, docs)

    assert ids == ["file1:body:0", "file1:body:1"]
    assert store.ids == ids
```

- [ ] **Step 2: Run indexing test to verify it fails**

Run:

```bash
python -m pytest tests/test_indexing.py -q
```

Expected: FAIL because `imperial_rag.indexing` does not exist.

- [ ] **Step 3: Implement keyword index and Qdrant vector store factory**

Create `src/imperial_rag/indexing.py`:

```python
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


def build_fts_match_query(query: str) -> str:
    terms = re.findall(r"\w+", query.casefold(), flags=re.UNICODE)
    return " AND ".join(f'"{term}"' for term in terms)


class KeywordIndex:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(chunk_id, text, metadata)")

    def replace_all(self, documents: list[Document]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM chunks")
            self._conn.executemany(
                "INSERT INTO chunks(chunk_id, text, metadata) VALUES (?, ?, ?)",
                [
                    (
                        str(document.metadata["chunk_id"]),
                        document.page_content,
                        json.dumps(document.metadata, ensure_ascii=False),
                    )
                    for document in documents
                ],
            )

    def search(self, query: str, limit: int) -> list[Document]:
        match_query = build_fts_match_query(query)
        if not match_query:
            return []
        rows = self._conn.execute(
            "SELECT text, metadata FROM chunks WHERE chunks MATCH ? LIMIT ?",
            (match_query, limit),
        ).fetchall()
        return [Document(page_content=text, metadata=json.loads(metadata)) for text, metadata in rows]


def make_qdrant_store(qdrant_url: str, collection_name: str) -> QdrantVectorStore:
    client = QdrantClient(url=qdrant_url)
    embeddings = OpenAIEmbeddings()
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


def index_documents(vector_store: QdrantVectorStore, documents: list[Document]) -> list[str]:
    ids = [str(document.metadata["chunk_id"]) for document in documents]
    return vector_store.add_documents(documents=documents, ids=ids)
```

- [ ] **Step 4: Run indexing tests**

Run:

```bash
python -m pytest tests/test_indexing.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint if git is initialized**

Run:

```bash
git add src/imperial_rag/indexing.py tests/test_indexing.py
git commit -m "feat: add qdrant and keyword indexing"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 5A: Local Qdrant Bootstrap And Live Health Check

**Files:**
- Create: `/Users/danil/Public/imperial/scripts/start_qdrant.sh`
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/indexing.py`
- Create: `/Users/danil/Public/imperial/tests/test_qdrant_health.py`

- [ ] **Step 1: Add local-only Qdrant startup script**

Create `scripts/start_qdrant.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

docker run \
  --name imperial-qdrant \
  --rm \
  -p 127.0.0.1:6333:6333 \
  -v imperial-qdrant-storage:/qdrant/storage \
  qdrant/qdrant
```

Expected: Qdrant binds only to `127.0.0.1:6333`, not all interfaces.

- [ ] **Step 2: Add Qdrant health helper**

Append to `src/imperial_rag/indexing.py`:

```python
def qdrant_is_healthy(qdrant_url: str) -> bool:
    client = QdrantClient(url=qdrant_url)
    try:
        client.get_collections()
    except Exception:
        return False
    return True
```

- [ ] **Step 3: Write live Qdrant health test gated on environment**

Create `tests/test_qdrant_health.py`:

```python
import os

import pytest

from imperial_rag.config import Settings
from imperial_rag.indexing import qdrant_is_healthy


@pytest.mark.skipif(os.environ.get("IMPERIAL_RAG_LIVE_QDRANT") != "1", reason="live Qdrant test is opt-in")
def test_qdrant_health_check_reaches_local_qdrant():
    settings = Settings()

    assert settings.qdrant_url.startswith("http://localhost") or settings.qdrant_url.startswith("http://127.0.0.1")
    assert qdrant_is_healthy(settings.qdrant_url)
```

- [ ] **Step 4: Run normal tests**

Run:

```bash
python -m pytest tests/test_qdrant_health.py -q
```

Expected: PASS with the live test skipped unless `IMPERIAL_RAG_LIVE_QDRANT=1`.

- [ ] **Step 5: Run live Qdrant smoke check**

In one terminal:

```bash
bash scripts/start_qdrant.sh
```

In another terminal:

```bash
IMPERIAL_RAG_LIVE_QDRANT=1 python -m pytest tests/test_qdrant_health.py -q
```

Expected: PASS and Qdrant reachable at `http://127.0.0.1:6333`. Stop the container after the check unless continuing with indexing.

- [ ] **Step 6: Confirm the Qdrant integration accepts stable chunk IDs**

Run after Qdrant is healthy:

```bash
IMPERIAL_RAG_LIVE_QDRANT=1 python -m pytest tests/test_indexing.py -q
```

Expected: PASS. The unit test must show `index_documents(...)` passes `chunk_id` values as explicit vector-store IDs. If adding a live round-trip test later, it should insert one document, retrieve it, and confirm the returned metadata contains the same `chunk_id`, `relative_path`, and citation payload.

- [ ] **Step 7: Commit checkpoint if git is initialized**

Run:

```bash
git add scripts/start_qdrant.sh src/imperial_rag/indexing.py tests/test_qdrant_health.py
git commit -m "chore: add local qdrant health check"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 6: LangGraph Workflows And Strict Answering

**Files:**
- Create: `/Users/danil/Public/imperial/src/imperial_rag/answering.py`
- Create: `/Users/danil/Public/imperial/src/imperial_rag/workflows.py`
- Create: `/Users/danil/Public/imperial/tests/test_answering.py`
- Create: `/Users/danil/Public/imperial/tests/test_workflows.py`

- [ ] **Step 1: Write failing strict answer tests**

Create `tests/test_answering.py`:

```python
from langchain_core.documents import Document

from imperial_rag.answering import build_evidence_prompt, format_citations


def test_format_citations_uses_source_path_and_chunk_id():
    docs = [
        Document(
            page_content="Возврат оформляется актом.",
            metadata={"file_path": "/docs/return.docx", "chunk_id": "hash:body:0", "source_type": "body"},
        )
    ]

    assert format_citations(docs) == ["[/docs/return.docx#hash:body:0] body"]


def test_build_evidence_prompt_forbids_general_knowledge():
    prompt = build_evidence_prompt("Что делать?", [])

    assert "Do not use general model knowledge" in prompt
    assert "I could not find this clearly in the indexed documents" in prompt
```

- [ ] **Step 2: Write failing workflow smoke test**

Create `tests/test_workflows.py`:

```python
from imperial_rag.workflows import build_query_workflow


def test_query_workflow_compiles():
    workflow = build_query_workflow()

    assert workflow is not None
```

- [ ] **Step 3: Run answer/workflow tests to verify they fail**

Run:

```bash
python -m pytest tests/test_answering.py tests/test_workflows.py -q
```

Expected: FAIL because `answering.py` and `workflows.py` do not exist.

- [ ] **Step 4: Implement strict answer helpers**

Create `src/imperial_rag/answering.py`:

```python
from __future__ import annotations

from langchain_core.documents import Document


REFUSAL_TEXT = "I could not find this clearly in the indexed documents."


def format_citations(documents: list[Document]) -> list[str]:
    citations: list[str] = []
    for document in documents:
        file_path = document.metadata.get("file_path", "unknown")
        chunk_id = document.metadata.get("chunk_id", "unknown")
        source_type = document.metadata.get("source_type", "unknown")
        citations.append(f"[{file_path}#{chunk_id}] {source_type}")
    return citations


def build_evidence_prompt(question: str, documents: list[Document]) -> str:
    evidence = "\n\n".join(
        f"Source: {citation}\nText:\n{document.page_content}"
        for citation, document in zip(format_citations(documents), documents, strict=False)
    )
    return f"""You are answering questions about internal company documents.
Use only the evidence below.
Do not use general model knowledge.
Every meaningful claim must cite a source.
If the evidence is insufficient, answer exactly: {REFUSAL_TEXT}

Question:
{question}

Evidence:
{evidence}
"""
```

- [ ] **Step 5: Implement LangGraph query workflow skeleton**

Create `src/imperial_rag/workflows.py`:

```python
from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from imperial_rag.answering import build_evidence_prompt


class QueryState(TypedDict, total=False):
    question: str
    normalized_query: str
    vector_candidates: list[Document]
    keyword_candidates: list[Document]
    evidence: list[Document]
    prompt: str
    answer: str
    citations: list[str]


def normalize_query(state: QueryState) -> QueryState:
    return {"normalized_query": state["question"].strip()}


def empty_retrieve(state: QueryState) -> QueryState:
    return {"vector_candidates": [], "keyword_candidates": [], "evidence": []}


def build_prompt(state: QueryState) -> QueryState:
    return {"prompt": build_evidence_prompt(state["question"], state.get("evidence", []))}


def build_query_workflow():
    graph = StateGraph(QueryState)
    graph.add_node("normalize_query", normalize_query)
    graph.add_node("retrieve", empty_retrieve)
    graph.add_node("build_prompt", build_prompt)
    graph.add_edge(START, "normalize_query")
    graph.add_edge("normalize_query", "retrieve")
    graph.add_edge("retrieve", "build_prompt")
    graph.add_edge("build_prompt", END)
    return graph.compile()
```

- [ ] **Step 6: Run answer/workflow tests**

Run:

```bash
python -m pytest tests/test_answering.py tests/test_workflows.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit checkpoint if git is initialized**

Run:

```bash
git add src/imperial_rag/answering.py src/imperial_rag/workflows.py tests/test_answering.py tests/test_workflows.py
git commit -m "feat: add langgraph query workflow"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 6A: Retrieval And Model Invocation Inside LangGraph

**Files:**
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/workflows.py`
- Modify: `/Users/danil/Public/imperial/tests/test_workflows.py`

- [ ] **Step 1: Add failing workflow retrieval test**

Append to `tests/test_workflows.py`:

```python
from langchain_core.documents import Document


class FakeVectorSearch:
    def similarity_search(self, query: str, k: int):
        return [
            Document(
                page_content="Возврат брака оформляется актом.",
                metadata={"file_path": "/docs/return.docx", "chunk_id": "hash:body:0", "source_type": "body"},
            )
        ]


class FakeKeywordSearch:
    def search(self, query: str, limit: int):
        return []


class FakeModelResponse:
    content = "Возврат брака оформляется актом. [/docs/return.docx#hash:body:0]"


class FakeChatModel:
    def invoke(self, messages):
        return FakeModelResponse()


def test_query_workflow_retrieves_evidence_and_calls_model():
    workflow = build_query_workflow(
        vector_search=FakeVectorSearch(),
        keyword_search=FakeKeywordSearch(),
        chat_model=FakeChatModel(),
    )

    result = workflow.invoke({"question": "Как оформить возврат брака?"})

    assert result["evidence"]
    assert "Возврат брака" in result["answer"]
    assert result["citations"] == ["[/docs/return.docx#hash:body:0] body"]
```

- [ ] **Step 2: Run workflow tests to verify the new test fails**

Run:

```bash
python -m pytest tests/test_workflows.py -q
```

Expected: FAIL because `build_query_workflow` does not accept retriever/model dependencies.

- [ ] **Step 3: Implement dependency-injected retrieval and model call**

Replace `src/imperial_rag/workflows.py` with:

```python
from __future__ import annotations

from typing import Protocol, TypedDict

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from imperial_rag.answering import REFUSAL_TEXT, build_evidence_prompt, format_citations


class VectorSearch(Protocol):
    def similarity_search(self, query: str, k: int):
        ...


class KeywordSearch(Protocol):
    def search(self, query: str, limit: int):
        ...


class ChatModel(Protocol):
    def invoke(self, messages):
        ...


class QueryState(TypedDict, total=False):
    question: str
    normalized_query: str
    vector_candidates: list[Document]
    keyword_candidates: list[Document]
    evidence: list[Document]
    prompt: str
    answer: str
    citations: list[str]


def _merge_documents(vector_docs: list[Document], keyword_docs: list[Document]) -> list[Document]:
    merged: list[Document] = []
    seen: set[str] = set()
    for document in [*vector_docs, *keyword_docs]:
        chunk_id = str(document.metadata.get("chunk_id", id(document)))
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        merged.append(document)
    return merged


def build_query_workflow(vector_search: VectorSearch | None = None, keyword_search: KeywordSearch | None = None, chat_model: ChatModel | None = None):
    model = chat_model or ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    def normalize_query(state: QueryState) -> QueryState:
        return {"normalized_query": state["question"].strip()}

    def retrieve(state: QueryState) -> QueryState:
        query = state["normalized_query"]
        vector_docs = vector_search.similarity_search(query, k=8) if vector_search is not None else []
        keyword_docs = keyword_search.search(query, limit=8) if keyword_search is not None else []
        evidence = _merge_documents(vector_docs, keyword_docs)
        return {"vector_candidates": vector_docs, "keyword_candidates": keyword_docs, "evidence": evidence}

    def build_prompt(state: QueryState) -> QueryState:
        return {"prompt": build_evidence_prompt(state["question"], state.get("evidence", []))}

    def call_model(state: QueryState) -> QueryState:
        evidence = state.get("evidence", [])
        if not evidence:
            return {"answer": REFUSAL_TEXT, "citations": []}
        response = model.invoke([{"role": "user", "content": state["prompt"]}])
        return {"answer": str(response.content), "citations": format_citations(evidence)}

    graph = StateGraph(QueryState)
    graph.add_node("normalize_query", normalize_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("call_model", call_model)
    graph.add_edge(START, "normalize_query")
    graph.add_edge("normalize_query", "retrieve")
    graph.add_edge("retrieve", "build_prompt")
    graph.add_edge("build_prompt", "call_model")
    graph.add_edge("call_model", END)
    return graph.compile()
```

- [ ] **Step 4: Run workflow tests**

Run:

```bash
python -m pytest tests/test_workflows.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint if git is initialized**

Run:

```bash
git add src/imperial_rag/workflows.py tests/test_workflows.py
git commit -m "feat: wire retrieval and model calls into langgraph"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 6B: Hybrid Retrieval Ranking And Citation Guardrails

**Files:**
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/workflows.py`
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/answering.py`
- Create: `/Users/danil/Public/imperial/tests/test_retrieval_ranking.py`

- [ ] **Step 1: Write failing hybrid ranking tests**

Create `tests/test_retrieval_ranking.py`:

```python
from langchain_core.documents import Document

from imperial_rag.answering import REFUSAL_TEXT, answer_has_required_citations
from imperial_rag.workflows import rank_hybrid_candidates


def test_rank_hybrid_candidates_boosts_exact_filename_and_heading_matches():
    vector_doc = Document(
        page_content="Общие правила склада",
        metadata={"chunk_id": "vector", "file_name": "warehouse.docx", "section_heading": "Общие правила"},
    )
    keyword_doc = Document(
        page_content="Возврат брака оформляется актом",
        metadata={"chunk_id": "keyword", "file_name": "Регламент возврата брака.docx", "section_heading": "Возврат брака"},
    )

    ranked = rank_hybrid_candidates("возврат брака", [vector_doc], [keyword_doc])

    assert [document.metadata["chunk_id"] for document in ranked] == ["keyword", "vector"]


def test_answer_has_required_citations_detects_missing_citation_marker():
    assert answer_has_required_citations("Ответ без ссылки.", ["[/docs/a.docx#chunk] body"]) is False
    assert answer_has_required_citations("Ответ. [/docs/a.docx#chunk]", ["[/docs/a.docx#chunk] body"]) is True


def test_answer_has_required_citations_requires_each_paragraph_to_cite_known_sources():
    citations = ["[/docs/a.docx#chunk-a] body", "[/docs/b.docx#chunk-b] body"]

    assert answer_has_required_citations(
        "Первый факт. [/docs/a.docx#chunk-a]\n\nВторой факт без ссылки.",
        citations,
    ) is False
    assert answer_has_required_citations(
        "Первый факт. [/docs/a.docx#chunk-a]\n\nВторой факт. [/docs/b.docx#chunk-b]",
        citations,
    ) is True
    assert answer_has_required_citations(
        "Факт со ссылкой на неизвестный источник. [/docs/c.docx#chunk-c]",
        citations,
    ) is False
```

- [ ] **Step 2: Run ranking tests to verify they fail**

Run:

```bash
python -m pytest tests/test_retrieval_ranking.py -q
```

Expected: FAIL because ranking and citation-validation helpers do not exist.

- [ ] **Step 3: Add citation validation helper**

Append to `src/imperial_rag/answering.py`:

```python
def citation_marker(citation: str) -> str:
    return citation.split("]", maxsplit=1)[0] + "]" if "]" in citation else citation


def answer_has_required_citations(answer: str, citations: list[str]) -> bool:
    if not citations:
        return answer.strip() == REFUSAL_TEXT
    known_markers = [citation_marker(citation) for citation in citations]
    paragraphs = [paragraph.strip() for paragraph in answer.splitlines() if paragraph.strip()]
    if not paragraphs or answer.strip() == REFUSAL_TEXT:
        return False
    for paragraph in paragraphs:
        markers_in_paragraph = [marker for marker in known_markers if marker in paragraph]
        unknown_markers = [
            part.split("]", maxsplit=1)[0] + "]"
            for part in paragraph.split("[")
            if "]" in part and not any(("[" + part.split("]", maxsplit=1)[0] + "]") == marker for marker in known_markers)
        ]
        if unknown_markers or not markers_in_paragraph:
            return False
    return True
```

- [ ] **Step 4: Add deterministic hybrid ranking**

Append to `src/imperial_rag/workflows.py`:

```python
def _contains_query_terms(query: str, text: str) -> bool:
    normalized_query = query.casefold()
    normalized_text = text.casefold()
    return all(term in normalized_text for term in normalized_query.split() if term)


def rank_hybrid_candidates(query: str, vector_docs: list[Document], keyword_docs: list[Document], limit: int = 12) -> list[Document]:
    candidates = _merge_documents(vector_docs, keyword_docs)

    def score(document: Document) -> tuple[int, int]:
        searchable = " ".join(
            str(document.metadata.get(field, ""))
            for field in ("file_name", "relative_path", "section_heading", "source_type")
        )
        exact_boost = 1 if _contains_query_terms(query, searchable + " " + document.page_content) else 0
        keyword_boost = 1 if document in keyword_docs else 0
        return (exact_boost, keyword_boost)

    return sorted(candidates, key=score, reverse=True)[:limit]
```

Modify the `retrieve` node from Task 6A to call `rank_hybrid_candidates(query, vector_docs, keyword_docs)` instead of `_merge_documents(...)`.

- [ ] **Step 5: Enforce citation guardrail after model generation**

Modify `call_model` in `src/imperial_rag/workflows.py`:

```python
from imperial_rag.answering import answer_has_required_citations


        citations = format_citations(evidence)
        response = model.invoke([{"role": "user", "content": state["prompt"]}])
        answer = str(response.content)
        if not answer_has_required_citations(answer, citations):
            return {"answer": REFUSAL_TEXT, "citations": citations}
        return {"answer": answer, "citations": citations}
```

Expected: the workflow refuses rather than returning an uncited generated answer.

- [ ] **Step 6: Run ranking and workflow tests**

Run:

```bash
python -m pytest tests/test_retrieval_ranking.py tests/test_workflows.py tests/test_answering.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit checkpoint if git is initialized**

Run:

```bash
git add src/imperial_rag/workflows.py src/imperial_rag/answering.py tests/test_retrieval_ranking.py
git commit -m "feat: add hybrid retrieval ranking guardrails"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 6C: Runtime Query Wiring For CLI And Web App

**Files:**
- Create: `/Users/danil/Public/imperial/src/imperial_rag/runtime.py`
- Create: `/Users/danil/Public/imperial/tests/test_runtime.py`

- [ ] **Step 1: Write failing runtime wiring test**

Create `tests/test_runtime.py`:

```python
from imperial_rag.runtime import build_query_dependencies


def test_build_query_dependencies_returns_keyword_and_vector_components(monkeypatch, tmp_path):
    class FakeSettings:
        qdrant_url = "http://127.0.0.1:6333"
        qdrant_collection = "test"
        keyword_db_path = tmp_path / "keyword.sqlite3"

    monkeypatch.setattr("imperial_rag.runtime.make_qdrant_store", lambda qdrant_url, collection_name: "vector")
    monkeypatch.setattr("imperial_rag.runtime.KeywordIndex", lambda db_path: "keyword")

    deps = build_query_dependencies(FakeSettings())

    assert deps.vector_search == "vector"
    assert deps.keyword_search == "keyword"
```

- [ ] **Step 2: Run runtime wiring test to verify it fails**

Run:

```bash
python -m pytest tests/test_runtime.py -q
```

Expected: FAIL because `imperial_rag.runtime` does not exist.

- [ ] **Step 3: Implement runtime dependency factory**

Create `src/imperial_rag/runtime.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from imperial_rag.config import Settings
from imperial_rag.indexing import KeywordIndex, make_qdrant_store
from imperial_rag.workflows import build_query_workflow


@dataclass(frozen=True)
class QueryDependencies:
    vector_search: object
    keyword_search: object


def build_query_dependencies(settings: Settings) -> QueryDependencies:
    return QueryDependencies(
        vector_search=make_qdrant_store(settings.qdrant_url, settings.qdrant_collection),
        keyword_search=KeywordIndex(settings.keyword_db_path),
    )


def build_live_query_workflow(settings: Settings | None = None):
    resolved_settings = settings or Settings()
    deps = build_query_dependencies(resolved_settings)
    return build_query_workflow(vector_search=deps.vector_search, keyword_search=deps.keyword_search)
```

- [ ] **Step 4: Run runtime wiring tests**

Run:

```bash
python -m pytest tests/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit checkpoint if git is initialized**

Run:

```bash
git add src/imperial_rag/runtime.py tests/test_runtime.py
git commit -m "feat: wire live query dependencies"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 7: Ingestion And Query Scripts

**Files:**
- Create: `/Users/danil/Public/imperial/scripts/ingest.py`
- Create: `/Users/danil/Public/imperial/scripts/query.py`
- Create: `/Users/danil/Public/imperial/tests/test_scripts_import.py`

- [ ] **Step 1: Write failing script import tests**

Create `tests/test_scripts_import.py`:

```python
import importlib.util
from pathlib import Path


def test_ingest_script_exists_and_imports():
    path = Path("scripts/ingest.py")
    spec = importlib.util.spec_from_file_location("ingest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "main")


def test_query_script_exists_and_imports():
    path = Path("scripts/query.py")
    spec = importlib.util.spec_from_file_location("query", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "main")
```

- [ ] **Step 2: Run script tests to verify they fail**

Run:

```bash
python -m pytest tests/test_scripts_import.py -q
```

Expected: FAIL because scripts do not exist.

- [ ] **Step 3: Create ingestion script**

Create `scripts/ingest.py`:

```python
from __future__ import annotations

from imperial_rag.config import Settings
from imperial_rag.manifest import assign_duplicate_groups, scan_files


def main() -> None:
    settings = Settings()
    records = assign_duplicate_groups(scan_files(settings.documents_root))
    print(f"scanned_files={len(records)}")
    print(f"documents_root={settings.documents_root}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create query script**

Create `scripts/query.py`:

```python
from __future__ import annotations

import argparse

from imperial_rag.runtime import build_live_query_workflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()
    workflow = build_live_query_workflow()
    result = workflow.invoke({"question": args.question})
    print(result.get("answer", ""))
    for citation in result.get("citations", []):
        print(citation)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run script tests**

Run:

```bash
python -m pytest tests/test_scripts_import.py -q
```

Expected: PASS.

- [ ] **Step 6: Run ingestion smoke command**

Run:

```bash
expected_count="$(find documents -type f | wc -l | tr -d ' ')"
actual_count="$(python scripts/ingest.py | awk -F= '/^scanned_files=/{print $2}')"
test "$actual_count" = "$expected_count"
```

Expected: command exits `0`.

- [ ] **Step 7: Commit checkpoint if git is initialized**

Run:

```bash
git add scripts/ingest.py scripts/query.py tests/test_scripts_import.py
git commit -m "feat: add rag cli smoke scripts"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 7A: End-To-End Ingestion Pipeline And Artifact Persistence

**Files:**
- Create: `/Users/danil/Public/imperial/src/imperial_rag/pipeline.py`
- Modify: `/Users/danil/Public/imperial/scripts/ingest.py`
- Create: `/Users/danil/Public/imperial/tests/test_pipeline.py`

- [ ] **Step 1: Write failing pipeline test**

Create `tests/test_pipeline.py`:

```python
import json
from pathlib import Path

from docx import Document as DocxDocument

from imperial_rag.config import Settings
from imperial_rag.pipeline import ingest_corpus


def test_ingest_corpus_extracts_chunks_and_persists_jsonl(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    docx_path = docs / "policy.docx"
    docx = DocxDocument()
    docx.add_paragraph("Регламент возврата брака из магазина. Ответственный склад.")
    docx.save(docx_path)
    settings = Settings(workspace_root=tmp_path)

    summary = ingest_corpus(settings=settings, ocr_client=None, vector_store=None)

    chunks_path = tmp_path / ".imperial_rag" / "extracted" / "chunks.jsonl"
    rows = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    assert summary.total_files == 1
    assert summary.indexed_files == 1
    assert summary.chunk_count >= 1
    assert rows[0]["metadata"]["relative_path"] == "policy.docx"
```

- [ ] **Step 2: Run pipeline test to verify it fails**

Run:

```bash
python -m pytest tests/test_pipeline.py -q
```

Expected: FAIL because `imperial_rag.pipeline` does not exist.

- [ ] **Step 3: Implement ingestion pipeline**

Create `src/imperial_rag/pipeline.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.documents import Document

from imperial_rag.chunking import build_chunks
from imperial_rag.config import Settings
from imperial_rag.extraction import extract_file
from imperial_rag.indexing import KeywordIndex, index_documents
from imperial_rag.manifest import FileStatus, IndexStatus, ManifestStore, assign_duplicate_groups, scan_files
from imperial_rag.ocr import OcrCache


@dataclass(frozen=True)
class IngestionSummary:
    total_files: int
    indexed_files: int
    manifest_only_files: int
    no_text_files: int
    unsupported_files: int
    failed_files: int
    chunk_count: int


def _write_chunks(settings: Settings, chunks: list[Document]) -> None:
    settings.extraction_root.mkdir(parents=True, exist_ok=True)
    chunks_path = settings.extraction_root / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {"page_content": chunk.page_content, "metadata": chunk.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )


def ingest_corpus(settings: Settings, ocr_client=None, vector_store=None) -> IngestionSummary:
    records = assign_duplicate_groups(scan_files(settings.documents_root))
    manifest_store = ManifestStore(settings.manifest_db_path)
    manifest_store.replace_records(records)
    ocr_cache = OcrCache(settings.extraction_root / "ocr-cache")
    extracted_documents: list[Document] = []
    indexed_files = 0
    manifest_only_files = 0
    no_text_files = 0
    unsupported_files = 0
    failed_files = 0

    for record in records:
        try:
            result = extract_file(
                record,
                ocr_client=ocr_client,
                ocr_cache=ocr_cache,
                artifact_root=settings.extraction_root / "artifacts",
            )
        except Exception as exc:
            failed_files += 1
            manifest_store.update_status(
                file_id=record.file_id,
                status=FileStatus.FAILED,
                extraction_method=None,
                error_message=str(exc),
                chunk_count=0,
            )
            continue
        if result.status == FileStatus.INDEXED:
            indexed_files += 1
            extracted_documents.extend(result.documents)
        elif result.status == FileStatus.MANIFEST_ONLY:
            manifest_only_files += 1
        elif result.status == FileStatus.NO_TEXT:
            no_text_files += 1
        elif result.status == FileStatus.UNSUPPORTED:
            unsupported_files += 1
        manifest_store.update_status(
            file_id=record.file_id,
            status=result.status,
            extraction_method=result.extraction_method,
            error_message=None if result.status != FileStatus.FAILED else result.message,
            chunk_count=len(result.documents),
        )

    chunks = build_chunks(extracted_documents)
    _write_chunks(settings, chunks)
    keyword_index = KeywordIndex(settings.keyword_db_path)
    keyword_index.replace_all(chunks)
    indexed_file_ids = {str(chunk.metadata["file_id"]) for chunk in chunks}
    if vector_store is not None and chunks:
        index_documents(vector_store, chunks)
    for record in records:
        if record.file_id not in indexed_file_ids:
            continue
        manifest_store.update_index_status(
            file_id=record.file_id,
            keyword_index_status=IndexStatus.INDEXED,
            vector_index_status=IndexStatus.INDEXED if vector_store is not None else IndexStatus.SKIPPED,
        )

    return IngestionSummary(
        total_files=len(records),
        indexed_files=indexed_files,
        manifest_only_files=manifest_only_files,
        no_text_files=no_text_files,
        unsupported_files=unsupported_files,
        failed_files=failed_files,
        chunk_count=len(chunks),
    )
```

- [ ] **Step 4: Modify ingestion script to use the pipeline**

Replace `scripts/ingest.py` with:

```python
from __future__ import annotations

import argparse

from imperial_rag.config import Settings
from imperial_rag.indexing import make_qdrant_store
from imperial_rag.ocr import OcrClient
from imperial_rag.pipeline import ingest_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-ocr", action="store_true", help="Use the configured AI vision model for PDF/image OCR")
    parser.add_argument("--with-qdrant", action="store_true", help="Index chunks into the local Qdrant collection")
    args = parser.parse_args()

    settings = Settings()
    ocr_client = OcrClient() if args.with_ocr else None
    vector_store = make_qdrant_store(settings.qdrant_url, settings.qdrant_collection) if args.with_qdrant else None
    summary = ingest_corpus(settings=settings, ocr_client=ocr_client, vector_store=vector_store)
    print(f"scanned_files={summary.total_files}")
    print(f"indexed_files={summary.indexed_files}")
    print(f"manifest_only_files={summary.manifest_only_files}")
    print(f"no_text_files={summary.no_text_files}")
    print(f"unsupported_files={summary.unsupported_files}")
    print(f"failed_files={summary.failed_files}")
    print(f"chunks={summary.chunk_count}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run pipeline and script tests**

Run:

```bash
python -m pytest tests/test_pipeline.py tests/test_scripts_import.py -q
```

Expected: PASS.

- [ ] **Step 6: Run ingestion smoke command**

Run:

```bash
python scripts/ingest.py
```

Expected: output includes `scanned_files=...` matching `find documents -type f | wc -l` and a `chunks=` line.

- [ ] **Step 7: Commit checkpoint if git is initialized**

Run:

```bash
git add src/imperial_rag/pipeline.py scripts/ingest.py tests/test_pipeline.py
git commit -m "feat: add end-to-end ingestion pipeline"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 7B: LangGraph Ingestion Workflow

**Files:**
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/workflows.py`
- Modify: `/Users/danil/Public/imperial/scripts/ingest.py`
- Create: `/Users/danil/Public/imperial/tests/test_ingestion_workflow.py`

- [ ] **Step 1: Write failing ingestion workflow test**

Create `tests/test_ingestion_workflow.py`:

```python
from docx import Document as DocxDocument

from imperial_rag.config import Settings
from imperial_rag.workflows import build_ingestion_workflow


def test_ingestion_workflow_runs_pipeline_and_returns_summary(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    docx = DocxDocument()
    docx.add_paragraph("Регламент возврата брака.")
    docx.save(docs / "policy.docx")
    settings = Settings(workspace_root=tmp_path)

    workflow = build_ingestion_workflow()
    result = workflow.invoke({"settings": settings, "ocr_client": None, "vector_store": None})

    assert result["summary"].total_files == 1
    assert result["summary"].chunk_count >= 1
```

- [ ] **Step 2: Run ingestion workflow test to verify it fails**

Run:

```bash
python -m pytest tests/test_ingestion_workflow.py -q
```

Expected: FAIL because `build_ingestion_workflow` does not exist.

- [ ] **Step 3: Add LangGraph ingestion workflow**

Append to `src/imperial_rag/workflows.py`:

```python
from imperial_rag.config import Settings
from imperial_rag.pipeline import IngestionSummary, ingest_corpus


class IngestionState(TypedDict, total=False):
    settings: Settings
    ocr_client: object
    vector_store: object
    summary: IngestionSummary


def build_ingestion_workflow():
    def run_ingestion(state: IngestionState) -> IngestionState:
        summary = ingest_corpus(
            settings=state["settings"],
            ocr_client=state.get("ocr_client"),
            vector_store=state.get("vector_store"),
        )
        return {"summary": summary}

    graph = StateGraph(IngestionState)
    graph.add_node("run_ingestion", run_ingestion)
    graph.add_edge(START, "run_ingestion")
    graph.add_edge("run_ingestion", END)
    return graph.compile()
```

Expected: LangGraph owns the ingestion entrypoint, while `pipeline.py` still owns the concrete extraction, chunking, manifest, keyword, and Qdrant work.

- [ ] **Step 4: Modify ingestion script to use the ingestion workflow**

In `scripts/ingest.py`, replace:

```python
from imperial_rag.pipeline import ingest_corpus
```

with:

```python
from imperial_rag.workflows import build_ingestion_workflow
```

Replace:

```python
summary = ingest_corpus(settings=settings, ocr_client=ocr_client, vector_store=vector_store)
```

with:

```python
workflow = build_ingestion_workflow()
result = workflow.invoke({"settings": settings, "ocr_client": ocr_client, "vector_store": vector_store})
summary = result["summary"]
```

- [ ] **Step 5: Run ingestion workflow and script tests**

Run:

```bash
python -m pytest tests/test_ingestion_workflow.py tests/test_pipeline.py tests/test_scripts_import.py -q
```

Expected: PASS.

- [ ] **Step 6: Run ingestion smoke command through LangGraph**

Run:

```bash
python scripts/ingest.py
```

Expected: output includes `scanned_files=...`, `chunks=...`, and no uncaught exceptions.

- [ ] **Step 7: Commit checkpoint if git is initialized**

Run:

```bash
git add src/imperial_rag/workflows.py scripts/ingest.py tests/test_ingestion_workflow.py
git commit -m "feat: orchestrate ingestion with langgraph"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 8: Local Web Chat

**Files:**
- Create: `/Users/danil/Public/imperial/src/imperial_rag/web_app.py`
- Create: `/Users/danil/Public/imperial/tests/test_web_app.py`

- [ ] **Step 1: Write failing web app import test**

Create `tests/test_web_app.py`:

```python
from imperial_rag.web_app import APP_TITLE, build_status_summary


def test_status_summary_displays_manifest_counts():
    summary = build_status_summary(total_files=162, indexed_files=100, failed_files=3)

    assert APP_TITLE == "Imperial RAG"
    assert "Total files: 162" in summary
    assert "Indexed files: 100" in summary
    assert "Failed files: 3" in summary
```

- [ ] **Step 2: Run web app test to verify it fails**

Run:

```bash
python -m pytest tests/test_web_app.py -q
```

Expected: FAIL because `imperial_rag.web_app` does not exist.

- [ ] **Step 3: Implement Streamlit web app shell**

Create `src/imperial_rag/web_app.py`:

```python
from __future__ import annotations

import streamlit as st

from imperial_rag.config import Settings
from imperial_rag.manifest import FileStatus, ManifestStore
from imperial_rag.runtime import build_live_query_workflow


APP_TITLE = "Imperial RAG"


def build_status_summary(total_files: int, indexed_files: int, failed_files: int) -> str:
    return "\n".join(
        [
            f"Total files: {total_files}",
            f"Indexed files: {indexed_files}",
            f"Failed files: {failed_files}",
        ]
    )


def load_status_summary(settings: Settings) -> str:
    records = ManifestStore(settings.manifest_db_path).list_records()
    indexed = sum(1 for record in records if record.status == FileStatus.INDEXED)
    failed = sum(1 for record in records if record.status == FileStatus.FAILED)
    return build_status_summary(total_files=len(records), indexed_files=indexed, failed_files=failed)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Strict-citation local RAG over the Imperial document corpus")
    settings = Settings()

    question = st.chat_input("Ask a question about the indexed documents")
    if question:
        workflow = build_live_query_workflow(settings)
        result = workflow.invoke({"question": question})
        st.chat_message("user").write(question)
        st.chat_message("assistant").write(result.get("answer", ""))
        for citation in result.get("citations", []):
            st.caption(citation)

    with st.sidebar:
        st.header("Ingestion status")
        st.text(load_status_summary(settings))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run web app test**

Run:

```bash
python -m pytest tests/test_web_app.py -q
```

Expected: PASS.

- [ ] **Step 5: Run Streamlit app locally**

Run:

```bash
python -m streamlit run src/imperial_rag/web_app.py --server.address 127.0.0.1 --server.port 8501
```

Expected: local web chat starts at `http://127.0.0.1:8501`.

- [ ] **Step 6: Commit checkpoint if git is initialized**

Run:

```bash
git add src/imperial_rag/web_app.py tests/test_web_app.py
git commit -m "feat: add local rag web chat"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 9: LangSmith Evaluation And Tracing

**Files:**
- Create: `/Users/danil/Public/imperial/evals/questions.jsonl`
- Create: `/Users/danil/Public/imperial/scripts/run_langsmith_eval.py`
- Create: `/Users/danil/Public/imperial/tests/test_eval_assets.py`

- [ ] **Step 1: Write failing eval asset tests**

Create `tests/test_eval_assets.py`:

```python
import json
from pathlib import Path


def test_eval_questions_are_jsonl_with_questions():
    path = Path("evals/questions.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) >= 5
    for line in lines:
        payload = json.loads(line)
        assert payload["question"]
        assert payload["expected_behavior"] in {"cite_answer", "refuse_if_not_found", "surface_conflict"}
        assert isinstance(payload.get("expected_source_hints", []), list)


def test_eval_runner_defines_behavior_evaluators():
    source = Path("scripts/run_langsmith_eval.py").read_text(encoding="utf-8")

    assert "def citation_behavior" in source
    assert "client.evaluate" in source
```

- [ ] **Step 2: Run eval test to verify it fails**

Run:

```bash
python -m pytest tests/test_eval_assets.py -q
```

Expected: FAIL because eval assets do not exist.

- [ ] **Step 3: Create Russian evaluation set**

Create `evals/questions.jsonl`:

```jsonl
{"question":"Как оформить возврат брака из магазина?","expected_behavior":"cite_answer","expected_source_hints":["РЕГЛАМЕНТ О БРАКЕ","возврат брака"]}
{"question":"Какие обязанности у водителя-экспедитора?","expected_behavior":"cite_answer","expected_source_hints":["водителя экспедитора","ДИ ЛОГИСТИКИ"]}
{"question":"Что делать при отсутствии сотрудника на рабочем месте?","expected_behavior":"cite_answer","expected_source_hints":["Акт об отсутствии","рабочем месте"]}
{"question":"Какие правила по табелям и мотивационным листам?","expected_behavior":"cite_answer","expected_source_hints":["табелях","мотивацион"]}
{"question":"Кто отвечает за приемку товара на складе?","expected_behavior":"cite_answer","expected_source_hints":["СКЛАД","прием"]}
{"question":"Какая версия регламента склада действует, если документы противоречат друг другу?","expected_behavior":"surface_conflict","expected_source_hints":["РЕГЛАМЕНТ СКЛАДА"]}
{"question":"Какую температуру плавления имеет вольфрам?","expected_behavior":"refuse_if_not_found","expected_source_hints":[]}
```

- [ ] **Step 4: Create LangSmith eval runner**

Create `scripts/run_langsmith_eval.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from langsmith import Client, traceable

from imperial_rag.answering import REFUSAL_TEXT
from imperial_rag.config import Settings
from imperial_rag.runtime import build_live_query_workflow


@traceable(name="imperial_rag_answer")
def answer(inputs: dict) -> dict:
    workflow = build_live_query_workflow()
    result = workflow.invoke({"question": inputs["question"]})
    documents = [
        {"page_content": document.page_content, "metadata": dict(document.metadata)}
        for document in result.get("evidence", [])
    ]
    return {
        "answer": str(result.get("answer", "")),
        "citations": list(result.get("citations", [])),
        "documents": documents,
    }


def citation_behavior(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    expected_behavior = reference_outputs["expected_behavior"]
    answer_text = outputs.get("answer", "")
    citations = outputs.get("citations", [])

    if expected_behavior == "refuse_if_not_found":
        score = REFUSAL_TEXT in answer_text
    elif expected_behavior == "cite_answer":
        score = bool(citations) and REFUSAL_TEXT not in answer_text
    elif expected_behavior == "surface_conflict":
        score = bool(citations) and ("disagree" in answer_text.lower() or "conflict" in answer_text.lower() or "противореч" in answer_text.lower())
    else:
        score = False
    return {"key": "citation_behavior", "score": score}


def retrieval_returned_sources(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    expected_behavior = reference_outputs["expected_behavior"]
    documents = outputs.get("documents", [])
    if expected_behavior == "refuse_if_not_found":
        return {"key": "retrieval_returned_sources", "score": True}
    return {"key": "retrieval_returned_sources", "score": bool(documents)}


def retrieval_matches_expected_source_hints(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    hints = [str(hint).casefold() for hint in reference_outputs.get("expected_source_hints", [])]
    if not hints:
        return {"key": "retrieval_matches_expected_source_hints", "score": True}
    documents = outputs.get("documents", [])
    searchable_sources = "\n".join(
        " ".join(
            str(document.get("metadata", {}).get(field, ""))
            for field in ("relative_path", "file_name", "parent_folder", "section_heading")
        )
        for document in documents
    ).casefold()
    score = any(hint in searchable_sources for hint in hints)
    return {"key": "retrieval_matches_expected_source_hints", "score": score}


def main() -> None:
    settings = Settings()
    client = Client()
    dataset_name = f"{settings.langsmith_project}-gold-questions"
    questions_path = Path("evals/questions.jsonl")
    examples = [json.loads(line) for line in questions_path.read_text(encoding="utf-8").splitlines()]
    if not client.has_dataset(dataset_name=dataset_name):
        dataset = client.create_dataset(dataset_name=dataset_name)
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": {"question": example["question"]},
                    "outputs": {
                        "expected_behavior": example["expected_behavior"],
                        "expected_source_hints": example.get("expected_source_hints", []),
                    },
                }
                for example in examples
            ],
        )
    print(f"langsmith_dataset={dataset_name}")
    client.evaluate(
        answer,
        data=dataset_name,
        evaluators=[citation_behavior, retrieval_returned_sources, retrieval_matches_expected_source_hints],
        experiment_prefix="imperial-rag-citation-grounding",
        metadata={"workspace": str(settings.workspace_root), "version": "v1-plan"},
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run eval asset tests**

Run:

```bash
python -m pytest tests/test_eval_assets.py -q
```

Expected: PASS.

- [ ] **Step 6: Run eval dataset creation when LangSmith credentials are configured**

Run:

```bash
LANGSMITH_TRACING=true LANGSMITH_PROJECT=imperial-rag python scripts/run_langsmith_eval.py
```

Expected with valid LangSmith credentials: output includes `langsmith_dataset=imperial-rag-gold-questions` and LangSmith records an `imperial-rag-citation-grounding` experiment. Expected without credentials: LangSmith authentication error.

- [ ] **Step 7: Commit checkpoint if git is initialized**

Run:

```bash
git add evals/questions.jsonl scripts/run_langsmith_eval.py tests/test_eval_assets.py
git commit -m "feat: add langsmith rag evaluation assets"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

### Task 10: End-To-End Smoke Checks

**Files:**
- Modify: `/Users/danil/Public/imperial/docs/superpowers/plans/2026-06-02-local-rag-system.md`

- [ ] **Step 1: Run full test suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS.

- [ ] **Step 2: Confirm manifest sees all current files**

Run:

```bash
expected_count="$(find documents -type f | wc -l | tr -d ' ')"
actual_count="$(python scripts/ingest.py | awk -F= '/^scanned_files=/{print $2}')"
test "$actual_count" = "$expected_count"
```

Expected: command exits `0`. This avoids hard-coding the corpus count, which may change when temp/system files appear or disappear.

- [ ] **Step 3: Confirm local Qdrant health before live vector indexing**

Run:

```bash
IMPERIAL_RAG_LIVE_QDRANT=1 python -m pytest tests/test_qdrant_health.py -q
```

Expected: PASS when `scripts/start_qdrant.sh` is running locally.

- [ ] **Step 4: Run full local ingestion without paid OCR**

Run:

```bash
python scripts/ingest.py --with-qdrant
```

Expected: output includes `scanned_files=...`, `chunks=...`, and no uncaught exceptions. PDF/image files without OCR should be audited as `no_text` or remain non-indexed rather than failing silently.

- [ ] **Step 5: Confirm query workflow refuses unsupported answers**

Run:

```bash
python scripts/query.py "Какую температуру плавления имеет вольфрам?"
```

Expected: output includes `I could not find this clearly in the indexed documents.`

- [ ] **Step 6: Run full OCR ingestion only when API credentials are intentionally available**

Run:

```bash
python scripts/ingest.py --with-ocr --with-qdrant
```

Expected with valid AI API credentials: output includes `scanned_files=...`, `chunks=...`, and manifest rows show OCR-backed files indexed. Expected without credentials: an authentication/configuration error is recorded or surfaced clearly, not hidden.

- [ ] **Step 7: Commit checkpoint if git is initialized**

Run:

```bash
git add docs/superpowers/plans/2026-06-02-local-rag-system.md
git commit -m "docs: add local rag implementation plan"
```

Expected in a git repo: commit succeeds. Expected in current workspace: `fatal: not a git repository`.

---

## Spec Coverage Checklist

- Full scan of every file: Task 2, Task 2A, and Task 10.
- Durable manifest audit trail: Task 2A and Task 7A.
- No archive extraction: Task 3.
- DOCX text and table extraction: Task 3.
- PDF/JPG OCR and DOCX embedded-image OCR: Task 4A.
- OCR persistence path: Task 4B writes OCR cache artifacts; Task 7A writes chunk artifacts; Task 4A writes rendered/embedded image artifacts.
- XLSX/RTF extraction and legacy DOC audit status: Task 4B.
- Qdrant local vector DB: Task 5, Task 5A, Task 7A, and Task 10.
- LangChain integrations first: Task 3, Task 4, Task 5, Task 6.
- LangGraph orchestration: Task 6 and Task 6A for query workflow; Task 7B for ingestion workflow.
- LangSmith tracing/evals: Task 9 creates the dataset and runs behavior evaluators.
- Strict citations/refusals: Task 6 and Task 10.
- Retrieval and model invocation: Task 6A.
- Hybrid retrieval ranking and citation guardrails: Task 6B.
- Runtime query dependency wiring: Task 6C.
- Local web chat: Task 8.
- Audit/status UI: Task 8.
- Importable CLI scripts after editable install: Preflight Step 0, Task 1 Step 6, and Task 10.

## Execution Notes

Before implementation, verify any library API not shown in this plan with Context7 according to `/Users/danil/Public/imperial/AGENTS.md` instructions. The APIs already checked during design/planning/review were Qdrant local Docker, LangChain ecosystem RAG/LangGraph/LangSmith guidance, LangSmith `client.evaluate` usage, and LangChain reference docs.
