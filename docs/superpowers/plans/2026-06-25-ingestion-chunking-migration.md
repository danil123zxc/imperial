# Ingestion Chunking Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Imperial RAG ingestion complete, addressable, reproducible, and safely migratable before replacing the current chunk and index artifacts.

**Architecture:** Use the existing ingestion pipeline as the control plane, adding a private corpus ledger, stable extraction locators, structure-first token-budget chunks, exact duplicate filtering, shadow index targets, and promotion gates. Keep canonical Elasticsearch and Qdrant indexes unchanged until the shadow migration passes coverage, citation, eval, privacy, and rollback checks.

**Tech Stack:** Python 3.12+, LangChain `Document` and `RecursiveCharacterTextSplitter`, python-docx, PyMuPDF, openpyxl, SQLite manifest store, Elasticsearch, Qdrant, pytest, existing Imperial RAG Phoenix/eval tooling.

---

## Context Notes

- Source transcript: `docs/superpowers/reports/council-transcript-20260624-214532.md`.
- Current repo reality:
  - `src/imperial_rag/ingestion/extraction.py` is the real extraction implementation; root `src/imperial_rag/extraction.py` is a compatibility wrapper.
  - `src/imperial_rag/ingestion/chunking.py` is the real chunking implementation; root `src/imperial_rag/chunking.py` is a compatibility wrapper.
  - `src/imperial_rag/ingestion/pipeline.py` already writes `.imperial_rag/extracted/chunks.jsonl` and `.imperial_rag/extracted/index-lineage.json`.
  - `scripts/ingest.py` owns CLI flags for OCR, vectors, Phoenix tracing, and workspace-root.
  - `evals/questions.jsonl` already contains gold `reference_context_ids` for a subset of cite-answer rows.
- Context7 check performed while creating this plan:
  - `/websites/langchain_oss_python_langchain` documents `RecursiveCharacterTextSplitter(chunk_size=..., chunk_overlap=..., add_start_index=True)` and says `start_index` is stored in split metadata.
  - The same Context7 pass did not return a current Python `ParentDocumentRetriever` snippet. Parent-child retrieval is therefore outside this plan and needs a separate API check after this migration passes.

## Scope Boundary

This plan implements the ingestion, extraction, chunking, dedupe, shadow-index, and promotion-gate work needed before any parent-child retrieval migration.

ParentDocumentRetriever-style retrieval is intentionally not implemented here. A separate retrieval plan should start only after Task 7 reports that the new evidence model preserves gold `reference_context_ids`, citation IDs, and latency/cost limits.

## Migration Safety Invariants

These invariants are mandatory and resolve the review findings against the first draft of this plan:

- Baseline artifacts are captured before any shadow ingestion writes happen. A baseline copied after a shadow run is invalid.
- Shadow ingestion must write extracted artifacts to a separate root such as `.imperial_rag/extracted-shadow-v2`; `--index-suffix` isolates Elasticsearch/Qdrant targets only and does not protect `.imperial_rag/extracted`.
- Promotion gates compare immutable baseline artifacts against isolated shadow artifacts. They must fail if baseline and shadow roots are the same path or have matching artifact-root fingerprints that indicate self-comparison.
- Old-to-new ID-map coverage is all-or-reviewed: every old chunk ID in the baseline `chunks.jsonl` must either map to at least one new chunk ID or appear in an explicit reviewed drop record with reason, reviewer, timestamp, and rollback impact. A non-empty map is not enough.
- Gold `reference_context_ids` must still be present in the shadow ledger when they are file IDs; when they match old chunk or citation IDs, they must be mapped or explicitly reviewed as drops.
- Corpus smoke checks derive expected scan counts from `scan_files()` and never from raw `find documents -type f` counts.

## File Structure

- Create `src/imperial_rag/ingestion/ledger.py`
  - Builds and writes the private corpus reconciliation ledger.
  - Keeps ledger rows JSON-serializable and privacy-safe: metadata, counts, IDs, hashes, failure taxonomy, and decisions only.
- Create `src/imperial_rag/ingestion/dedupe.py`
  - Selects exact duplicate canonical files and marks index-inclusion decisions.
  - Does not delete files or extracted artifacts.
- Create `src/imperial_rag/ingestion/promotion.py`
  - Compares baseline and shadow artifacts.
  - Enforces promotion gates over coverage, locators, duplicate decisions, ID maps, eval gold IDs, and chunk budgets.
- Create `scripts/check_ingestion_promotion.py`
  - CLI wrapper for promotion-gate checks.
- Modify `src/imperial_rag/ingestion/extraction.py`
  - Add stable source document IDs, element IDs, parent IDs, source locators, section headings, element hashes, and extraction schema version.
  - Preserve current deterministic extractors first.
- Modify `src/imperial_rag/ingestion/chunking.py`
  - Replace flat character splitting with structure-first token-budget splitting, contextual headers, `add_start_index=True`, and noise filtering.
- Modify `src/imperial_rag/ingestion/pipeline.py`
  - Wire ledger generation, dedupe decisions, ID map writing, shadow index lineage, and promotion artifacts into the existing pipeline.
- Modify `src/imperial_rag/config.py`
  - Add an optional extraction-root override for isolated shadow artifacts.
- Modify `scripts/ingest.py`
  - Add shadow index suffix and artifact-root flags without changing default canonical ingestion behavior.
- Modify `src/imperial_rag/retrieval/service.py`
  - Change `RetrievalSettings` default chunk budget and overlap only after Task 4 tests define the new units.
- Modify `tests/test_extraction.py`, `tests/test_chunking.py`, `tests/test_pipeline.py`, `tests/test_manifest.py`, `tests/test_scripts.py`, `tests/test_retrieval.py`, `tests/test_config.py`
  - Focused regression coverage beside existing subsystem tests.
- Create `tests/test_ingestion_ledger.py`, `tests/test_ingestion_dedupe.py`, `tests/test_ingestion_promotion.py`
  - New unit tests for new migration helpers.
- Modify `README.md` and `.env.example`
  - Document artifacts, flags, rollback, and promotion workflow.

---

### Task 0: Baseline Artifact Preflight

**Files:**
- No committed files. This task creates private local artifacts under `.imperial_rag/`.

- [ ] **Step 1: Verify canonical artifacts exist before code changes**

Run this before Task 1 and before any shadow ingestion:

```bash
test -s .imperial_rag/extracted/chunks.jsonl
test -s .imperial_rag/extracted/index-lineage.json
```

Expected: both commands pass. If either file is missing, run the current pre-migration ingestion first, then restart this preflight. Do not create `.imperial_rag/extracted-baseline` from artifacts produced after Task 4 or Task 6 changes.

- [ ] **Step 2: Capture immutable baseline artifacts**

Run:

```bash
rm -rf .imperial_rag/extracted-baseline
cp -R .imperial_rag/extracted .imperial_rag/extracted-baseline
python - <<'PY'
from pathlib import Path
import hashlib, json

root = Path(".imperial_rag/extracted-baseline")
files = sorted(path for path in root.rglob("*") if path.is_file())
digest = hashlib.sha256()
for path in files:
    digest.update(path.relative_to(root).as_posix().encode("utf-8"))
    digest.update(path.read_bytes())
(root / "baseline-fingerprint.json").write_text(
    json.dumps(
        {
            "schema_version": "baseline-fingerprint-v1",
            "file_count": len(files),
            "sha256": digest.hexdigest(),
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)
PY
```

Expected: `.imperial_rag/extracted-baseline/chunks.jsonl`, `.imperial_rag/extracted-baseline/index-lineage.json`, and `.imperial_rag/extracted-baseline/baseline-fingerprint.json` exist. This directory is the only valid baseline for Task 8 promotion checks.

- [ ] **Step 3: Do not commit baseline artifacts**

Run:

```bash
git status --short .imperial_rag/extracted-baseline
```

Expected: no tracked files are staged or committed. These artifacts are private local state.

---

### Task 1: Corpus Ledger Artifact

**Files:**
- Create: `src/imperial_rag/ingestion/ledger.py`
- Create: `tests/test_ingestion_ledger.py`

- [ ] **Step 1: Write the failing ledger tests**

Create `tests/test_ingestion_ledger.py` with this content:

```python
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from imperial_rag.ingestion.ledger import (
    LEDGER_SCHEMA_VERSION,
    build_corpus_ledger,
    write_corpus_ledger,
)
from imperial_rag.manifest import FileStatus


def _record(file_id: str, relative_path: str, extension: str, sha256: str = "a" * 64):
    return SimpleNamespace(
        file_id=file_id,
        relative_path=Path(relative_path),
        absolute_path=Path("/private") / relative_path,
        filename=Path(relative_path).name,
        extension=extension,
        size_bytes=123,
        sha256=sha256,
        duplicate_group_id=None,
        inferred_category=Path(relative_path).parts[0] if len(Path(relative_path).parts) > 1 else "",
    )


def test_build_corpus_ledger_classifies_indexed_and_unsupported_files():
    indexed = _record("file-a", "hr/policy.docx", ".docx")
    unsupported = _record("file-b", "legacy/form.doc", ".doc")
    chunks = [
        Document(
            page_content="Section text",
            metadata={
                "file_id": "file-a",
                "source_doc_id": "file-a",
                "source_locator": "section:1",
                "citation_id": "hr/policy.docx#body:section-1:start-0:chunk-0",
                "chunk_id": "chunk-a",
                "start_index": 0,
            },
        )
    ]

    rows = build_corpus_ledger(
        [indexed, unsupported],
        extracted_documents=[
            Document(page_content="Section text", metadata={"file_id": "file-a", "source_locator": "section:1"})
        ],
        chunks=chunks,
        status_by_file={"file-a": FileStatus.INDEXED, "file-b": FileStatus.UNSUPPORTED},
        method_by_file={"file-a": "python_docx", "file-b": None},
        error_by_file={"file-b": "legacy .doc requires a safe local converter"},
        duplicate_action_by_file={},
        ocr_enabled=False,
    )

    assert [row.schema_version for row in rows] == [LEDGER_SCHEMA_VERSION, LEDGER_SCHEMA_VERSION]
    assert rows[0].file_id == "file-a"
    assert rows[0].status == "indexed"
    assert rows[0].failure_taxonomy == "indexed"
    assert rows[0].document_count == 1
    assert rows[0].chunk_count == 1
    assert rows[0].locator_coverage == 1.0
    assert rows[0].index_inclusion_reason == "indexable"
    assert rows[1].file_id == "file-b"
    assert rows[1].status == "unsupported"
    assert rows[1].failure_taxonomy == "legacy_doc_unsupported"
    assert rows[1].proposed_next_action == "convert_legacy_doc_with_local_tool_or_keep_manifest_only"


def test_write_corpus_ledger_writes_jsonl_and_summary(tmp_path):
    record = _record("file-a", "policy.docx", ".docx")

    rows = write_corpus_ledger(
        tmp_path,
        [record],
        extracted_documents=[],
        chunks=[],
        status_by_file={"file-a": FileStatus.NO_TEXT},
        method_by_file={"file-a": "python_docx"},
        error_by_file={"file-a": ""},
        duplicate_action_by_file={},
        ocr_enabled=False,
    )

    ledger_path = tmp_path / "corpus-ledger.jsonl"
    summary_path = tmp_path / "corpus-ledger-summary.json"
    ledger_rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert rows[0].file_id == "file-a"
    assert ledger_rows[0]["schema_version"] == LEDGER_SCHEMA_VERSION
    assert ledger_rows[0]["failure_taxonomy"] == "empty_extraction"
    assert summary["schema_version"] == LEDGER_SCHEMA_VERSION
    assert summary["total_files"] == 1
    assert summary["status_counts"] == {"no_text": 1}
    assert summary["failure_taxonomy_counts"] == {"empty_extraction": 1}
```

- [ ] **Step 2: Run the ledger tests and verify they fail**

Run:

```bash
uv run python -m pytest tests/test_ingestion_ledger.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'imperial_rag.ingestion.ledger'`.

- [ ] **Step 3: Add the ledger implementation**

Create `src/imperial_rag/ingestion/ledger.py` with this content:

```python
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document


LEDGER_SCHEMA_VERSION = "corpus-ledger-v1"


@dataclass(frozen=True)
class CorpusLedgerRow:
    schema_version: str
    file_id: str
    relative_path: str
    extension: str
    size_bytes: int
    sha256: str
    inferred_category: str
    duplicate_group_id: str | None
    duplicate_action: str
    status: str
    extraction_method: str | None
    failure_taxonomy: str
    error_message: str | None
    text_length: int
    document_count: int
    chunk_count: int
    locator_count: int
    locator_coverage: float
    start_index_coverage: float
    ocr_decision: str
    index_inclusion_reason: str
    proposed_next_action: str


def build_corpus_ledger(
    records: Iterable[Any],
    *,
    extracted_documents: Iterable[Document],
    chunks: Iterable[Document],
    status_by_file: dict[str, Any],
    method_by_file: dict[str, str | None],
    error_by_file: dict[str, str | None],
    duplicate_action_by_file: dict[str, str],
    ocr_enabled: bool,
) -> list[CorpusLedgerRow]:
    documents_by_file = _documents_by_file(extracted_documents)
    chunks_by_file = _documents_by_file(chunks)
    rows: list[CorpusLedgerRow] = []
    for record in records:
        file_id = str(record.file_id)
        documents = documents_by_file.get(file_id, [])
        file_chunks = chunks_by_file.get(file_id, [])
        status = _status_value(status_by_file.get(file_id, getattr(record, "status", "pending")))
        duplicate_action = duplicate_action_by_file.get(file_id, "canonical")
        failure_taxonomy = _failure_taxonomy(record, status, error_by_file.get(file_id))
        rows.append(
            CorpusLedgerRow(
                schema_version=LEDGER_SCHEMA_VERSION,
                file_id=file_id,
                relative_path=Path(record.relative_path).as_posix(),
                extension=str(getattr(record, "extension", "") or "").casefold(),
                size_bytes=int(getattr(record, "size_bytes", 0) or 0),
                sha256=str(getattr(record, "sha256", "") or ""),
                inferred_category=str(getattr(record, "inferred_category", "") or ""),
                duplicate_group_id=getattr(record, "duplicate_group_id", None),
                duplicate_action=duplicate_action,
                status=status,
                extraction_method=method_by_file.get(file_id),
                failure_taxonomy=failure_taxonomy,
                error_message=error_by_file.get(file_id) or None,
                text_length=sum(len(document.page_content) for document in documents),
                document_count=len(documents),
                chunk_count=len(file_chunks),
                locator_count=sum(1 for chunk in file_chunks if chunk.metadata.get("source_locator")),
                locator_coverage=_coverage(file_chunks, "source_locator"),
                start_index_coverage=_coverage(file_chunks, "start_index"),
                ocr_decision=_ocr_decision(record, ocr_enabled),
                index_inclusion_reason=_index_inclusion_reason(status, duplicate_action, len(file_chunks)),
                proposed_next_action=_proposed_next_action(record, status, failure_taxonomy, duplicate_action),
            )
        )
    return rows


def write_corpus_ledger(
    extraction_root: Path,
    records: Iterable[Any],
    *,
    extracted_documents: Iterable[Document],
    chunks: Iterable[Document],
    status_by_file: dict[str, Any],
    method_by_file: dict[str, str | None],
    error_by_file: dict[str, str | None],
    duplicate_action_by_file: dict[str, str],
    ocr_enabled: bool,
) -> list[CorpusLedgerRow]:
    rows = build_corpus_ledger(
        records,
        extracted_documents=extracted_documents,
        chunks=chunks,
        status_by_file=status_by_file,
        method_by_file=method_by_file,
        error_by_file=error_by_file,
        duplicate_action_by_file=duplicate_action_by_file,
        ocr_enabled=ocr_enabled,
    )
    extraction_root.mkdir(parents=True, exist_ok=True)
    ledger_path = extraction_root / "corpus-ledger.jsonl"
    with ledger_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "total_files": len(rows),
        "status_counts": dict(sorted(Counter(row.status for row in rows).items())),
        "failure_taxonomy_counts": dict(sorted(Counter(row.failure_taxonomy for row in rows).items())),
        "indexed_chunk_count": sum(row.chunk_count for row in rows if row.index_inclusion_reason == "indexable"),
        "locator_coverage": _mean(row.locator_coverage for row in rows if row.chunk_count),
        "start_index_coverage": _mean(row.start_index_coverage for row in rows if row.chunk_count),
    }
    (extraction_root / "corpus-ledger-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return rows


def _documents_by_file(documents: Iterable[Document]) -> dict[str, list[Document]]:
    grouped: dict[str, list[Document]] = {}
    for document in documents:
        file_id = document.metadata.get("file_id")
        if file_id is not None:
            grouped.setdefault(str(file_id), []).append(document)
    return grouped


def _coverage(documents: list[Document], metadata_key: str) -> float:
    if not documents:
        return 0.0
    return round(sum(1 for document in documents if document.metadata.get(metadata_key) is not None) / len(documents), 4)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(sum(materialized) / len(materialized), 4)


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status))


def _failure_taxonomy(record: Any, status: str, error_message: str | None) -> str:
    extension = str(getattr(record, "extension", "") or "").casefold()
    if status == "indexed":
        return "indexed"
    if status == "manifest_only" and extension in {".rar", ".zip", ".7z"}:
        return "archive_manifest_only"
    if extension == ".doc":
        return "legacy_doc_unsupported"
    if status == "unsupported":
        return "unsupported_extension"
    if status == "no_text":
        return "empty_extraction"
    if status == "failed":
        text = (error_message or "").casefold()
        if "ocr" in text:
            return "ocr_failed"
        return "extract_failed"
    return status or "unknown"


def _ocr_decision(record: Any, ocr_enabled: bool) -> str:
    extension = str(getattr(record, "extension", "") or "").casefold()
    if extension in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        return "ocr_enabled" if ocr_enabled else "ocr_disabled"
    if extension == ".pdf":
        return "ocr_if_native_text_missing" if ocr_enabled else "native_text_only"
    if extension == ".docx":
        return "embedded_image_ocr_enabled" if ocr_enabled else "embedded_image_ocr_disabled"
    return "not_ocr_candidate"


def _index_inclusion_reason(status: str, duplicate_action: str, chunk_count: int) -> str:
    if duplicate_action == "skip_exact_duplicate":
        return "exact_duplicate_skipped"
    if status == "indexed" and chunk_count > 0:
        return "indexable"
    if status == "indexed":
        return "indexed_without_chunks"
    return f"{status}_not_indexed"


def _proposed_next_action(record: Any, status: str, failure_taxonomy: str, duplicate_action: str) -> str:
    if duplicate_action == "skip_exact_duplicate":
        return "keep_canonical_duplicate_only_in_indexes"
    if failure_taxonomy == "indexed":
        return "keep_indexed"
    if failure_taxonomy == "archive_manifest_only":
        return "inspect_archive_contents_and_extract_supported_children"
    if failure_taxonomy == "legacy_doc_unsupported":
        return "convert_legacy_doc_with_local_tool_or_keep_manifest_only"
    if failure_taxonomy == "empty_extraction":
        return "inspect_for_scanned_text_or_junk_filter"
    if failure_taxonomy == "ocr_failed":
        return "rerun_targeted_ocr_after_provider_check"
    if failure_taxonomy == "unsupported_extension":
        return "classify_extension_or_add_extractor"
    return f"review_{status}"
```

- [ ] **Step 4: Run the ledger tests and verify they pass**

Run:

```bash
uv run python -m pytest tests/test_ingestion_ledger.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add src/imperial_rag/ingestion/ledger.py tests/test_ingestion_ledger.py
git commit -m "feat: add ingestion corpus ledger"
```

Expected: commit succeeds with only these two files staged.

---

### Task 2: Wire Ledger Into The Existing Pipeline

**Files:**
- Modify: `src/imperial_rag/ingestion/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing pipeline artifact test**

Add this test after `test_run_ingestion_persists_chunks_and_updates_manifest` in `tests/test_pipeline.py`:

```python
def test_run_ingestion_writes_corpus_ledger_artifacts(tmp_path, monkeypatch):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.txt").write_text("Регламент возврата брака.", encoding="utf-8")
    _install_fake_dependencies(monkeypatch)

    summary = run_ingestion(settings=FakeSettings(tmp_path), enable_ocr=False, index_vectors=False)

    ledger_path = tmp_path / ".imperial_rag" / "extracted" / "corpus-ledger.jsonl"
    summary_path = tmp_path / ".imperial_rag" / "extracted" / "corpus-ledger-summary.json"
    assert summary.total_files == 1
    assert ledger_path.exists()
    assert summary_path.exists()
    ledger_rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    ledger_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert ledger_rows[0]["file_id"] == "file1"
    assert ledger_rows[0]["status"] == "indexed"
    assert ledger_rows[0]["chunk_count"] == 1
    assert ledger_rows[0]["index_inclusion_reason"] == "indexable"
    assert ledger_summary["total_files"] == 1
```

Add this fake ledger module inside `_install_fake_dependencies()` before the `for module in ...` loop:

```python
    ledger = ModuleType("imperial_rag.ingestion.ledger")

    def write_corpus_ledger(
        extraction_root,
        records,
        *,
        extracted_documents,
        chunks,
        status_by_file,
        method_by_file,
        error_by_file,
        duplicate_action_by_file,
        ocr_enabled,
    ):
        extraction_root.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "schema_version": "corpus-ledger-v1",
                "file_id": str(record.file_id),
                "status": str(getattr(status_by_file[str(record.file_id)], "value", status_by_file[str(record.file_id)])),
                "chunk_count": sum(1 for chunk in chunks if chunk.metadata.get("file_id") == record.file_id),
                "index_inclusion_reason": "indexable",
            }
            for record in records
        ]
        (extraction_root / "corpus-ledger.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        (extraction_root / "corpus-ledger-summary.json").write_text(
            json.dumps({"schema_version": "corpus-ledger-v1", "total_files": len(rows)}, ensure_ascii=False),
            encoding="utf-8",
        )
        return rows

    ledger.write_corpus_ledger = write_corpus_ledger
```

Change the module registration loop to include `ledger`:

```python
    for module in (config, retrieval, manifest, extraction, chunking, elasticsearch_keyword, indexing, ledger):
        monkeypatch.setitem(sys.modules, module.__name__, module)
```

- [ ] **Step 2: Run the pipeline artifact test and verify it fails**

Run:

```bash
uv run python -m pytest tests/test_pipeline.py::test_run_ingestion_writes_corpus_ledger_artifacts -q
```

Expected: FAIL because `corpus-ledger.jsonl` is not written.

- [ ] **Step 3: Import the ledger writer through `_load_dependencies()`**

In `src/imperial_rag/ingestion/pipeline.py`, add this import inside `_load_dependencies()`:

```python
    from imperial_rag.ingestion.ledger import write_corpus_ledger
```

Add this key to the returned dependency dictionary:

```python
        "write_corpus_ledger": write_corpus_ledger,
```

- [ ] **Step 4: Write the ledger after chunks are built**

In `_run()`, immediately after `chunk_count_by_file = _count_chunks_by_file(chunks)`, add:

```python
            deps["write_corpus_ledger"](
                extraction_root,
                records,
                extracted_documents=extracted_documents,
                chunks=chunks,
                status_by_file=status_by_file,
                method_by_file=method_by_file,
                error_by_file=error_by_file,
                duplicate_action_by_file={},
                ocr_enabled=ocr_client is not None,
            )
```

- [ ] **Step 5: Run the focused pipeline tests**

Run:

```bash
uv run python -m pytest tests/test_pipeline.py::test_run_ingestion_writes_corpus_ledger_artifacts tests/test_pipeline.py::test_run_ingestion_persists_chunks_and_updates_manifest -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

Run:

```bash
git add src/imperial_rag/ingestion/pipeline.py tests/test_pipeline.py
git commit -m "feat: write corpus ledger during ingestion"
```

Expected: commit succeeds with only these two files staged.

---

### Task 3: Stable Extraction Locators And Element IDs

**Files:**
- Modify: `src/imperial_rag/ingestion/extraction.py`
- Modify: `tests/test_extraction.py`

- [ ] **Step 1: Write failing extraction locator tests**

Add these tests to `tests/test_extraction.py` after `test_docx_text_and_table_extract_to_langchain_documents_with_citation_metadata`:

```python
def test_docx_headings_create_section_documents_with_stable_locators(tmp_path):
    path = tmp_path / "policy.docx"
    docx = DocxDocument()
    docx.add_heading("Возврат брака", level=1)
    docx.add_paragraph("Оформляется актом.")
    docx.add_heading("Сроки", level=2)
    docx.add_paragraph("Передать документы в тот же день.")
    docx.save(path)
    record = _record_for(path)

    result = extract_file(record)

    body_docs = [document for document in result.documents if document.metadata["source_type"] == "body"]
    assert [document.metadata["section_heading"] for document in body_docs] == ["Возврат брака", "Сроки"]
    assert [document.metadata["source_locator"] for document in body_docs] == ["section:1", "section:2"]
    for document in body_docs:
        assert document.metadata["source_doc_id"] == record.file_id
        assert document.metadata["element_id"].startswith(f"{record.file_id}:")
        assert len(document.metadata["element_hash"]) == 64
        assert document.metadata["extraction_schema_version"] == "extraction-v2"


def test_pdf_pages_receive_source_doc_id_and_page_locators(tmp_path):
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "policy.pdf"
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Native PDF text")
    pdf.save(path)
    record = _record_for(path)

    result = extract_file(record)

    assert result.documents[0].metadata["source_doc_id"] == record.file_id
    assert result.documents[0].metadata["source_locator"] == "page:1"
    assert result.documents[0].metadata["element_id"].startswith(f"{record.file_id}:pdf_page:page:1:")
    assert len(result.documents[0].metadata["element_hash"]) == 64
```

- [ ] **Step 2: Run the locator tests and verify they fail**

Run:

```bash
uv run python -m pytest tests/test_extraction.py::test_docx_headings_create_section_documents_with_stable_locators tests/test_extraction.py::test_pdf_pages_receive_source_doc_id_and_page_locators -q
```

Expected: FAIL because the new metadata keys are missing.

- [ ] **Step 3: Add element metadata helpers**

In `src/imperial_rag/ingestion/extraction.py`, add `hashlib` to the imports:

```python
import hashlib
```

Replace `_base_metadata()` with:

```python
EXTRACTION_SCHEMA_VERSION = "extraction-v2"


def _base_metadata(record: FileRecord, source_type: str) -> dict[str, str | int | None]:
    return {
        "file_id": record.file_id,
        "source_doc_id": record.file_id,
        "file_path": str(record.absolute_path),
        "relative_path": str(record.relative_path),
        "file_name": record.filename,
        "file_extension": record.extension,
        "file_hash": record.sha256,
        "duplicate_group_id": record.duplicate_group_id,
        "parent_folder": str(record.parent_folder),
        "inferred_category": record.inferred_category,
        "source_type": source_type,
        "extraction_schema_version": EXTRACTION_SCHEMA_VERSION,
    }


def _element_metadata(
    record: FileRecord,
    source_type: str,
    *,
    source_locator: str,
    element_index: int,
    text: str,
    section_heading: str | None = None,
    parent_id: str | None = None,
) -> dict[str, str | int | None]:
    metadata = _base_metadata(record, source_type)
    element_id = f"{record.file_id}:{source_type}:{source_locator}:{element_index}"
    metadata.update(
        {
            "source_locator": source_locator,
            "element_index": element_index,
            "element_id": element_id,
            "parent_id": parent_id,
            "section_heading": section_heading,
            "element_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    )
    return metadata
```

- [ ] **Step 4: Replace DOCX body extraction with heading-aware sections**

In `src/imperial_rag/ingestion/extraction.py`, add these helpers before `_extract_docx()`:

```python
def _is_docx_heading(paragraph) -> bool:
    style_name = str(getattr(getattr(paragraph, "style", None), "name", "") or "")
    return style_name.casefold().startswith("heading")


def _docx_section_documents(record: FileRecord, docx: DocxDocument) -> list[Document]:
    documents: list[Document] = []
    current_heading: str | None = None
    buffer: list[str] = []
    section_index = 0

    def flush() -> None:
        nonlocal section_index, buffer
        text = "\n".join(line for line in buffer if line.strip()).strip()
        if not text:
            buffer = []
            return
        section_index += 1
        locator = f"section:{section_index}"
        documents.append(
            Document(
                page_content=text,
                metadata=_element_metadata(
                    record,
                    "body",
                    source_locator=locator,
                    element_index=section_index,
                    text=text,
                    section_heading=current_heading,
                ),
            )
        )
        buffer = []

    for paragraph in docx.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if _is_docx_heading(paragraph):
            flush()
            current_heading = text
            continue
        buffer.append(text)
    flush()
    return documents
```

In `_extract_docx()`, replace:

```python
    body_text = "\n".join(paragraph.text.strip() for paragraph in docx.paragraphs if paragraph.text.strip())
    if body_text:
        documents.append(Document(page_content=body_text, metadata=_base_metadata(record, "body")))
```

with:

```python
    documents.extend(_docx_section_documents(record, docx))
```

- [ ] **Step 5: Add locators to tables, PDFs, sheets, RTF, and OCR documents**

Apply these replacements:

```python
# DOCX table append
table_text = "\n".join(table_lines)
documents.append(
    Document(
        page_content=table_text,
        metadata=_element_metadata(
            record,
            "table",
            source_locator="table:1",
            element_index=1,
            text=table_text,
        ),
    )
)
```

```python
# PDF native text metadata
metadata = _element_metadata(
    record,
    "pdf_page",
    source_locator=f"page:{page_index}",
    element_index=page_index,
    text=text,
)
metadata["page_number"] = page_index
```

```python
# XLSX sheet metadata
sheet_text = "\n".join(lines)
metadata = _element_metadata(
    record,
    "sheet",
    source_locator=f"sheet:{sheet.title}",
    element_index=len(documents) + 1,
    text=sheet_text,
)
metadata["sheet_name"] = sheet.title
```

```python
# RTF return
return [
    Document(
        page_content=text,
        metadata=_element_metadata(
            record,
            "body",
            source_locator="body:1",
            element_index=1,
            text=text,
        ),
    )
]
```

In `_ocr_image()`, after `merged_metadata.update(metadata)`, add:

```python
    source_locator = str(metadata.get("source_locator") or image_id)
    merged_metadata.update(
        _element_metadata(
            record,
            source_type,
            source_locator=source_locator,
            element_index=int(metadata.get("image_index") or metadata.get("page_number") or 1),
            text=ocr_result.text,
        )
    )
```

When calling `_ocr_image()` for PDF image pages, include `"source_locator": f"page:{page_index}"` in the metadata dict.

- [ ] **Step 6: Run extraction tests**

Run:

```bash
uv run python -m pytest tests/test_extraction.py -q
```

Expected: PASS after updating any assertions that expected one DOCX body document for multiple heading sections.

- [ ] **Step 7: Commit Task 3**

Run:

```bash
git add src/imperial_rag/ingestion/extraction.py tests/test_extraction.py
git commit -m "feat: add stable extraction locators"
```

Expected: commit succeeds with only these two files staged.

---

### Task 4: Structure-First Token-Budget Chunking

**Files:**
- Modify: `src/imperial_rag/ingestion/chunking.py`
- Modify: `src/imperial_rag/retrieval/service.py`
- Modify: `tests/test_chunking.py`
- Modify: `tests/test_retrieval.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing chunking tests**

Add these tests to `tests/test_chunking.py`:

```python
def test_build_chunks_adds_contextual_headers_and_start_indexes():
    source = Document(
        page_content="Первый абзац про возврат.\n\nВторой абзац про оформление акта.",
        metadata={
            "file_id": "file123",
            "source_doc_id": "file123",
            "relative_path": "hr/policy.docx",
            "file_name": "policy.docx",
            "source_type": "body",
            "source_locator": "section:1",
            "section_heading": "Возврат брака",
            "element_id": "file123:body:section:1:1",
        },
    )

    chunks = build_chunks([source], chunk_size=18, chunk_overlap=4)

    assert chunks
    for chunk in chunks:
        assert chunk.page_content.startswith("Source: hr/policy.docx\nSection: Возврат брака\nLocator: section:1\n\n")
        assert chunk.metadata["source_doc_id"] == "file123"
        assert chunk.metadata["source_locator"] == "section:1"
        assert isinstance(chunk.metadata["start_index"], int)
        assert isinstance(chunk.metadata["body_start_index"], int)
        assert chunk.metadata["citation_id"].startswith("hr/policy.docx#body:section-1:start-")
        assert chunk.metadata["chunk_id"].startswith("file123:body:section:1:")


def test_build_chunks_drops_noise_chunks():
    source = Document(
        page_content="--\n\nОформить возврат брака по акту и передать документы ответственному сотруднику.",
        metadata={
            "file_id": "file123",
            "relative_path": "policy.docx",
            "source_type": "body",
            "source_locator": "section:1",
        },
    )

    chunks = build_chunks([source], chunk_size=8, chunk_overlap=0)

    assert all(chunk.page_content.strip() != "--" for chunk in chunks)
    assert any("возврат брака" in chunk.page_content for chunk in chunks)
```

- [ ] **Step 2: Run the new chunking tests and verify they fail**

Run:

```bash
uv run python -m pytest tests/test_chunking.py::test_build_chunks_adds_contextual_headers_and_start_indexes tests/test_chunking.py::test_build_chunks_drops_noise_chunks -q
```

Expected: FAIL because contextual headers, `add_start_index`, and noise filtering are not implemented.

- [ ] **Step 3: Replace chunking helpers**

In `src/imperial_rag/ingestion/chunking.py`, replace the file content with:

```python
from __future__ import annotations

import hashlib
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


RUSSIAN_STRUCTURE_SEPARATORS = ["\n\n", "\n", ". ", "; ", ": ", " - ", " ", ""]
NOISE_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def estimated_token_count(text: str) -> int:
    tokens = re.findall(r"[\w]+|[^\w\s]", text, flags=re.UNICODE)
    return max(1, len(tokens))


def _source_locator(metadata: dict) -> str:
    locator = metadata.get("source_locator")
    if locator is not None:
        return str(locator)
    page = metadata.get("page_number")
    if page is not None:
        return f"page:{page}"
    sheet = metadata.get("sheet_name")
    if sheet is not None:
        return f"sheet:{sheet}"
    image = metadata.get("image_index")
    if image is not None:
        return f"image:{image}"
    heading = metadata.get("section_heading")
    if heading is not None:
        return f"section:{heading}"
    return "body:1"


def _locator_for_id(locator: str) -> str:
    return locator.replace(":", "-").replace("/", "-").replace(" ", "-")


def _contextual_header(metadata: dict) -> str:
    lines = [f"Source: {metadata.get('relative_path', 'unknown')}"]
    if metadata.get("section_heading"):
        lines.append(f"Section: {metadata['section_heading']}")
    lines.append(f"Locator: {_source_locator(metadata)}")
    return "\n".join(lines) + "\n\n"


def _citation_id(metadata: dict, chunk_index: int) -> str:
    relative_path = metadata.get("relative_path", "unknown")
    source_type = metadata.get("source_type", "unknown")
    locator = _locator_for_id(_source_locator(metadata))
    start = metadata.get("body_start_index", metadata.get("start_index", 0))
    return f"{relative_path}#{source_type}:{locator}:start-{start}:chunk-{chunk_index}"


def _is_noise(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) <= 3 and NOISE_RE.match(stripped):
        return True
    return estimated_token_count(stripped) <= 2 and not any(char.isalnum() for char in stripped)


def build_chunks(documents: list[Document], chunk_size: int = 650, chunk_overlap: int = 80) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=estimated_token_count,
        separators=RUSSIAN_STRUCTURE_SEPARATORS,
        add_start_index=True,
    )
    chunks: list[Document] = []
    for document in documents:
        split_docs = splitter.split_documents([document])
        visible_index = 0
        for split_doc in split_docs:
            if _is_noise(split_doc.page_content):
                continue
            metadata = dict(split_doc.metadata)
            source_locator = _source_locator(metadata)
            metadata["source_locator"] = source_locator
            metadata["chunk_index"] = visible_index
            metadata["body_start_index"] = int(metadata.get("start_index") or 0)
            metadata["body_token_count"] = estimated_token_count(split_doc.page_content)
            metadata["contextual_header"] = _contextual_header(metadata).strip()
            metadata["citation_id"] = _citation_id(metadata, visible_index)
            base = (
                f"{metadata.get('source_doc_id') or metadata.get('file_id')}:"
                f"{metadata.get('source_type')}:{source_locator}:{visible_index}:"
                f"{metadata.get('body_start_index')}"
            )
            digest = hashlib.sha1(f"{base}:{split_doc.page_content}".encode("utf-8")).hexdigest()[:10]
            metadata["chunk_id"] = f"{base}:{digest}"
            chunks.append(
                Document(
                    page_content=f"{metadata['contextual_header']}\n\n{split_doc.page_content}",
                    metadata=metadata,
                )
            )
            visible_index += 1
    return chunks
```

- [ ] **Step 4: Change retrieval chunk defaults to token-budget units**

In `src/imperial_rag/retrieval/service.py`, change the first two defaults in `RetrievalSettings`:

```python
    chunk_size: int = 650
    chunk_overlap: int = 80
```

- [ ] **Step 5: Update existing default and citation assertions**

In `tests/test_chunking.py`, update the default-size test so it checks token budget semantics instead of 400-character semantics:

```python
def test_build_chunks_defaults_to_structure_token_budget_and_overlap():
    source = Document(
        page_content=" ".join(f"токен{i}" for i in range(900)),
        metadata={"file_id": "file123", "relative_path": "policy.docx", "source_type": "body", "source_locator": "body:1"},
    )

    chunks = build_chunks([source])

    assert len(chunks) == 2
    assert all(chunk.metadata["body_token_count"] <= 650 for chunk in chunks)
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["chunk_index"] == 1
```

Also update the existing citation-format assertions in `tests/test_chunking.py`:

- `test_build_chunks_preserves_citation_metadata_and_adds_citation_id`
  - Expected citation format becomes `reglament.pdf#pdf_page:page-3:start-<body_start_index>:chunk-<index>`.
  - Assert `body_start_index` and `body_token_count` are present on every chunk.
- `test_build_chunks_uses_sheet_name_in_citation_identity`
  - Expected citation format becomes `book.xlsx#sheet:sheet-Склад:start-0:chunk-0` and `book.xlsx#sheet:sheet-Продажи:start-0:chunk-0`.
  - Keep the assertion that sheet chunks have different `chunk_id` values.

Update all retrieval-setting default assertions affected by changing `RetrievalSettings` from character units to token-budget units:

- In `tests/test_retrieval.py::test_retrieval_settings_defaults_match_accuracy_spec`, change expected `chunk_size` from `400` to `650` and `chunk_overlap` from `50` to `80`.
- In `tests/test_pipeline.py`, update the fake `RetrievalSettings` defaults and `_safe_env_int()` fallbacks from `400/50` to `650/80`.
- In `tests/test_pipeline.py`, update trace-output assertions that expect `{"chunk_size": 400, "chunk_overlap": 50}` to `{"chunk_size": 650, "chunk_overlap": 80}`.

- [ ] **Step 6: Run chunking and retrieval-setting tests**

Run:

```bash
uv run python -m pytest \
  tests/test_chunking.py \
  tests/test_retrieval.py::test_retrieval_settings_defaults_match_accuracy_spec \
  tests/test_pipeline.py::test_run_ingestion_traces_aggregate_lifecycle_without_vector_stage \
  tests/test_pipeline.py::test_run_ingestion_uses_retrieval_chunk_settings \
  -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add src/imperial_rag/ingestion/chunking.py src/imperial_rag/retrieval/service.py tests/test_chunking.py tests/test_retrieval.py tests/test_pipeline.py
git commit -m "feat: add structure-first ingestion chunks"
```

Expected: commit succeeds with only these five files staged.

---

### Task 5: Exact Duplicate Index Policy

**Files:**
- Create: `src/imperial_rag/ingestion/dedupe.py`
- Create: `tests/test_ingestion_dedupe.py`
- Modify: `src/imperial_rag/ingestion/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing dedupe unit tests**

Create `tests/test_ingestion_dedupe.py` with:

```python
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from imperial_rag.ingestion.dedupe import exact_duplicate_decisions, indexable_chunks


def _record(file_id: str, relative_path: str, sha256: str):
    return SimpleNamespace(
        file_id=file_id,
        relative_path=Path(relative_path),
        sha256=sha256,
        duplicate_group_id=f"sha256:{sha256}",
    )


def test_exact_duplicate_decisions_keep_lowest_path_as_canonical():
    decisions = exact_duplicate_decisions(
        [
            _record("b", "z-policy.docx", "same"),
            _record("a", "a-policy.docx", "same"),
            _record("c", "unique.docx", "other"),
        ]
    )

    assert decisions["a"].action == "canonical"
    assert decisions["a"].canonical_file_id == "a"
    assert decisions["b"].action == "skip_exact_duplicate"
    assert decisions["b"].canonical_file_id == "a"
    assert decisions["c"].action == "canonical"


def test_indexable_chunks_excludes_skipped_exact_duplicates():
    decisions = exact_duplicate_decisions(
        [
            _record("a", "a-policy.docx", "same"),
            _record("b", "z-policy.docx", "same"),
        ]
    )
    chunks = [
        Document(page_content="canonical", metadata={"file_id": "a", "chunk_id": "a1"}),
        Document(page_content="duplicate", metadata={"file_id": "b", "chunk_id": "b1"}),
    ]

    filtered = indexable_chunks(chunks, decisions)

    assert [chunk.metadata["chunk_id"] for chunk in filtered] == ["a1"]
    assert filtered[0].metadata["duplicate_action"] == "canonical"
```

- [ ] **Step 2: Run dedupe tests and verify they fail**

Run:

```bash
uv run python -m pytest tests/test_ingestion_dedupe.py -q
```

Expected: FAIL with missing `imperial_rag.ingestion.dedupe`.

- [ ] **Step 3: Add dedupe implementation**

Create `src/imperial_rag/ingestion/dedupe.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document


@dataclass(frozen=True)
class DuplicateDecision:
    file_id: str
    action: str
    canonical_file_id: str
    duplicate_group_id: str | None


def exact_duplicate_decisions(records: Iterable[Any]) -> dict[str, DuplicateDecision]:
    by_hash: dict[str, list[Any]] = {}
    for record in records:
        by_hash.setdefault(str(getattr(record, "sha256", "")), []).append(record)

    decisions: dict[str, DuplicateDecision] = {}
    for group in by_hash.values():
        ordered = sorted(group, key=lambda record: Path(record.relative_path).as_posix())
        canonical = ordered[0]
        canonical_file_id = str(canonical.file_id)
        for record in ordered:
            file_id = str(record.file_id)
            duplicate_group_id = getattr(record, "duplicate_group_id", None)
            decisions[file_id] = DuplicateDecision(
                file_id=file_id,
                action="canonical" if file_id == canonical_file_id else "skip_exact_duplicate",
                canonical_file_id=canonical_file_id,
                duplicate_group_id=duplicate_group_id,
            )
    return decisions


def duplicate_action_map(decisions: dict[str, DuplicateDecision]) -> dict[str, str]:
    return {file_id: decision.action for file_id, decision in decisions.items()}


def indexable_chunks(chunks: Iterable[Document], decisions: dict[str, DuplicateDecision]) -> list[Document]:
    filtered: list[Document] = []
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        file_id = str(metadata.get("file_id", ""))
        decision = decisions.get(file_id)
        if decision is None:
            metadata["duplicate_action"] = "canonical"
            filtered.append(Document(page_content=chunk.page_content, metadata=metadata))
            continue
        if decision.action == "skip_exact_duplicate":
            continue
        metadata["duplicate_action"] = decision.action
        metadata["duplicate_canonical_file_id"] = decision.canonical_file_id
        filtered.append(Document(page_content=chunk.page_content, metadata=metadata))
    return filtered
```

- [ ] **Step 4: Run dedupe tests**

Run:

```bash
uv run python -m pytest tests/test_ingestion_dedupe.py -q
```

Expected: PASS.

- [ ] **Step 5: Wire dedupe into pipeline indexing**

In `src/imperial_rag/ingestion/pipeline.py`, add imports inside `_load_dependencies()`:

```python
    from imperial_rag.ingestion.dedupe import duplicate_action_map, exact_duplicate_decisions, indexable_chunks
```

Add returned dependency keys:

```python
        "duplicate_action_map": duplicate_action_map,
        "exact_duplicate_decisions": exact_duplicate_decisions,
        "indexable_chunks": indexable_chunks,
```

In `_run()`, immediately after chunks are built:

```python
                duplicate_decisions = deps["exact_duplicate_decisions"](records)
                chunks_for_indexing = deps["indexable_chunks"](chunks, duplicate_decisions)
```

Replace the keyword indexing call:

```python
                keyword_indexed = _replace_keyword_index(deps["KeywordSearchIndex"], settings, chunks)
```

with:

```python
                keyword_indexed = _replace_keyword_index(deps["KeywordSearchIndex"], settings, chunks_for_indexing)
```

Replace the keyword trace output call:

```python
                keyword_span.set_output(_keyword_index_trace_output(settings, chunks, keyword_indexed))
```

with:

```python
                keyword_span.set_output(_keyword_index_trace_output(settings, chunks_for_indexing, keyword_indexed))
```

Replace the vector indexing call:

```python
                        vector_indexed, vector_added_ids = _index_with_vector_store(
                            deps["index_vector_documents"], settings, vector_store, chunks
                        )
```

with:

```python
                        vector_indexed, vector_added_ids = _index_with_vector_store(
                            deps["index_vector_documents"], settings, vector_store, chunks_for_indexing
                        )
```

Replace the vector trace output call:

```python
                        _vector_index_trace_output(settings, chunks, vector_indexed, vector_added_ids, embedding_model)
```

with:

```python
                        _vector_index_trace_output(settings, chunks_for_indexing, vector_indexed, vector_added_ids, embedding_model)
```

Keep `_write_chunks(extraction_root, chunks)` writing the full extracted chunk set.

Pass `duplicate_action_by_file=deps["duplicate_action_map"](duplicate_decisions)` into `write_corpus_ledger()`.

- [ ] **Step 6: Add pipeline duplicate coverage**

In `tests/test_pipeline.py`, extend `_install_fake_dependencies()` with a fake dedupe module:

```python
    dedupe = ModuleType("imperial_rag.ingestion.dedupe")
    dedupe.exact_duplicate_decisions = lambda records: {
        str(record.file_id): SimpleNamespace(action="canonical", canonical_file_id=str(record.file_id))
        for record in records
    }
    dedupe.duplicate_action_map = lambda decisions: {
        file_id: decision.action for file_id, decision in decisions.items()
    }
    dedupe.indexable_chunks = lambda chunks, decisions: list(chunks)
```

Add `dedupe` to the module registration loop.

Add this test:

```python
def test_run_ingestion_indexes_only_canonical_duplicate_chunks(tmp_path, monkeypatch):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.txt").write_text("Регламент возврата брака.", encoding="utf-8")
    _install_fake_dependencies(monkeypatch)

    dedupe_module = sys.modules["imperial_rag.ingestion.dedupe"]
    dedupe_module.indexable_chunks = lambda chunks, decisions: []

    summary = run_ingestion(settings=FakeSettings(tmp_path), enable_ocr=False, index_vectors=False)

    assert summary.chunk_count == 1
    assert FakeKeywordSearchIndex.last_docs == []
```

- [ ] **Step 7: Run focused pipeline and dedupe tests**

Run:

```bash
uv run python -m pytest tests/test_ingestion_dedupe.py tests/test_pipeline.py::test_run_ingestion_indexes_only_canonical_duplicate_chunks -q
```

Expected: PASS.

- [ ] **Step 8: Commit Task 5**

Run:

```bash
git add src/imperial_rag/ingestion/dedupe.py src/imperial_rag/ingestion/pipeline.py tests/test_ingestion_dedupe.py tests/test_pipeline.py
git commit -m "feat: skip exact duplicates during indexing"
```

Expected: commit succeeds with only these four files staged.

---

### Task 6: Shadow Artifact Roots, Index Flags, And Old-To-New ID Map

**Files:**
- Modify: `src/imperial_rag/config.py`
- Modify: `scripts/ingest.py`
- Modify: `src/imperial_rag/ingestion/pipeline.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_scripts.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing script and pipeline tests**

Add this test to `tests/test_scripts.py`:

```python
def test_ingest_script_exposes_shadow_index_suffix_and_artifact_root_flags():
    source = Path("scripts/ingest.py").read_text(encoding="utf-8")

    assert "--index-suffix" in source
    assert "--artifact-root" in source
    assert "--baseline-artifact-root" in source
    assert "_settings_with_shadow_targets" in source
```

Add this test to `tests/test_config.py`:

```python
def test_settings_allows_extraction_root_override(tmp_path):
    shadow_root = tmp_path / ".imperial_rag" / "extracted-shadow-v2"

    settings = Settings(workspace_root=tmp_path, extraction_root_override=shadow_root)

    assert settings.processed_root == tmp_path / ".imperial_rag"
    assert settings.extraction_root == shadow_root
```

Add this test to `tests/test_pipeline.py`:

```python
def test_run_ingestion_writes_old_to_new_id_map(tmp_path, monkeypatch):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.txt").write_text("Регламент возврата брака.", encoding="utf-8")
    extracted = tmp_path / ".imperial_rag" / "extracted"
    extracted.mkdir(parents=True)
    (extracted / "chunks.jsonl").write_text(
        json.dumps({"page_content": "old", "metadata": {"file_id": "file1", "chunk_id": "old-chunk", "citation_id": "old-citation"}}) + "\n",
        encoding="utf-8",
    )
    _install_fake_dependencies(monkeypatch)

    run_ingestion(settings=FakeSettings(tmp_path), enable_ocr=False, index_vectors=False)

    id_map_path = extracted / "old-to-new-id-map.json"
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    assert id_map["schema_version"] == "old-to-new-id-map-v1"
    assert id_map["rows"][0]["old_chunk_id"] == "old-chunk"
    assert id_map["rows"][0]["new_chunk_id"] == "file1:body:0"
```

In `tests/test_pipeline.py`, extend `FakeSettings` with `extraction_root_override: Path | None = None` and update its `extraction_root` property to return the override when present. Then add this artifact isolation regression test:

```python
def test_run_ingestion_can_write_shadow_artifacts_without_mutating_canonical_root(tmp_path, monkeypatch):
    canonical = tmp_path / ".imperial_rag" / "extracted"
    canonical.mkdir(parents=True)
    (canonical / "chunks.jsonl").write_text(
        json.dumps({"page_content": "canonical", "metadata": {"file_id": "file1", "chunk_id": "old"}}) + "\n",
        encoding="utf-8",
    )
    shadow = tmp_path / ".imperial_rag" / "extracted-shadow-v2"
    _install_fake_dependencies(monkeypatch)

    settings = FakeSettings(tmp_path, extraction_root_override=shadow)
    run_ingestion(settings=settings, enable_ocr=False, index_vectors=False)

    assert "canonical" in (canonical / "chunks.jsonl").read_text(encoding="utf-8")
    assert (shadow / "chunks.jsonl").exists()
    assert (shadow / "old-to-new-id-map.json").exists()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run python -m pytest \
  tests/test_scripts.py::test_ingest_script_exposes_shadow_index_suffix_and_artifact_root_flags \
  tests/test_config.py::test_settings_allows_extraction_root_override \
  tests/test_pipeline.py::test_run_ingestion_writes_old_to_new_id_map \
  tests/test_pipeline.py::test_run_ingestion_can_write_shadow_artifacts_without_mutating_canonical_root \
  -q
```

Expected: FAIL because the flags, extraction-root override, artifact isolation behavior, and ID map are missing.

- [ ] **Step 3: Add extraction-root override and CLI shadow targets**

In `src/imperial_rag/config.py`, add these dataclass fields:

```python
    extraction_root_override: Path | None = None
    baseline_extraction_root: Path | None = None
```

Update the `extraction_root` property:

```python
    @property
    def extraction_root(self) -> Path:
        if self.extraction_root_override is not None:
            return self.extraction_root_override
        return self.processed_root / "extracted"
```

In `scripts/ingest.py`, add this import:

```python
from dataclasses import replace
```

Add this parser argument after `--index-vectors`:

```python
    parser.add_argument(
        "--index-suffix",
        help="Append a suffix to Elasticsearch index and Qdrant collection names for shadow ingestion.",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Write extracted artifacts to this root instead of the canonical .imperial_rag/extracted root.",
    )
    parser.add_argument(
        "--baseline-artifact-root",
        type=Path,
        help="Read old chunks for old-to-new ID mapping from this immutable baseline artifact root.",
    )
```

After `settings = _build_settings(args.workspace_root)`, add:

```python
    settings = _settings_with_shadow_targets(settings, args.index_suffix, args.artifact_root, args.baseline_artifact_root)
```

Add this helper near `_build_settings()`:

```python
def _settings_with_shadow_targets(
    settings: Any,
    suffix: str | None,
    artifact_root: Path | None,
    baseline_artifact_root: Path | None,
) -> Any:
    updates: dict[str, Any] = {}
    if suffix is not None and suffix.strip():
        clean = suffix.strip().replace(" ", "_")
        updates["elasticsearch_index"] = f"{settings.elasticsearch_index}_{clean}"
        updates["qdrant_collection"] = f"{settings.qdrant_collection}_{clean}"
    if artifact_root is not None:
        updates["extraction_root_override"] = _resolve_artifact_root(settings, artifact_root)
    if baseline_artifact_root is not None:
        updates["baseline_extraction_root"] = _resolve_artifact_root(settings, baseline_artifact_root)
    if not updates:
        return settings
    return replace(settings, **updates)


def _resolve_artifact_root(settings: Any, artifact_root: Path) -> Path:
    if artifact_root.is_absolute():
        return artifact_root
    return Path(settings.workspace_root) / artifact_root
```

Canonical ingestion must still use `.imperial_rag/extracted` when `--artifact-root` is omitted. Shadow ingestion must pass `--index-suffix shadow_v2`, `--artifact-root .imperial_rag/extracted-shadow-v2`, and `--baseline-artifact-root .imperial_rag/extracted-baseline`.

- [ ] **Step 4: Add ID-map writer helpers**

In `src/imperial_rag/ingestion/pipeline.py`, add these helpers near `_write_chunks()`:

```python
def _read_existing_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_old_to_new_id_map(extraction_root: Path, old_rows: list[dict[str, Any]], chunks: list[Any]) -> None:
    new_by_file: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        file_id = metadata.get("file_id")
        if file_id is not None:
            new_by_file.setdefault(str(file_id), []).append(metadata)

    rows: list[dict[str, Any]] = []
    for row in old_rows:
        old = dict(row.get("metadata") or {})
        file_id = str(old.get("file_id") or "")
        new_candidates = new_by_file.get(file_id, [])
        new = new_candidates.pop(0) if new_candidates else {}
        rows.append(
            {
                "file_id": file_id,
                "old_chunk_id": old.get("chunk_id"),
                "old_citation_id": old.get("citation_id"),
                "new_chunk_id": new.get("chunk_id"),
                "new_citation_id": new.get("citation_id"),
                "source_locator": new.get("source_locator"),
                "status": "mapped" if new.get("chunk_id") else "unmapped",
            }
        )

    for file_id, remaining in sorted(new_by_file.items()):
        for new in remaining:
            rows.append(
                {
                    "file_id": file_id,
                    "old_chunk_id": None,
                    "old_citation_id": None,
                    "new_chunk_id": new.get("chunk_id"),
                    "new_citation_id": new.get("citation_id"),
                    "source_locator": new.get("source_locator"),
                    "status": "new_only",
                }
            )
    payload = {"schema_version": "old-to-new-id-map-v1", "rows": rows}
    (extraction_root / "old-to-new-id-map.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
```

In `_run()`, before `_write_chunks(extraction_root, chunks)`, add:

```python
                baseline_root = Path(getattr(settings, "baseline_extraction_root", None) or extraction_root)
                previous_chunk_rows = _read_existing_chunks(baseline_root / "chunks.jsonl")
```

Immediately after `_write_chunks(extraction_root, chunks)`, add:

```python
                _write_old_to_new_id_map(extraction_root, previous_chunk_rows, chunks)
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run python -m pytest \
  tests/test_scripts.py::test_ingest_script_exposes_shadow_index_suffix_and_artifact_root_flags \
  tests/test_config.py::test_settings_allows_extraction_root_override \
  tests/test_pipeline.py::test_run_ingestion_writes_old_to_new_id_map \
  tests/test_pipeline.py::test_run_ingestion_can_write_shadow_artifacts_without_mutating_canonical_root \
  -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

Run:

```bash
git add src/imperial_rag/config.py scripts/ingest.py src/imperial_rag/ingestion/pipeline.py tests/test_config.py tests/test_scripts.py tests/test_pipeline.py
git commit -m "feat: add shadow ingestion artifacts"
```

Expected: commit succeeds with only these six files staged.

---

### Task 7: Promotion Gate Comparator

**Files:**
- Create: `src/imperial_rag/ingestion/promotion.py`
- Create: `scripts/check_ingestion_promotion.py`
- Create: `tests/test_ingestion_promotion.py`
- Modify: `tests/test_scripts.py`

- [ ] **Step 1: Write failing promotion tests**

Create `tests/test_ingestion_promotion.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from imperial_rag.ingestion.promotion import PromotionGateResult, check_promotion_gates


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_chunks(path: Path, rows: list[dict]) -> None:
    _write_jsonl(path, [{"page_content": row.get("page_content", "text"), "metadata": row["metadata"]} for row in rows])


def test_check_promotion_gates_accepts_improved_shadow(tmp_path):
    baseline = tmp_path / "baseline"
    shadow = tmp_path / "shadow"
    baseline.mkdir()
    shadow.mkdir()
    _write_jsonl(
        baseline / "corpus-ledger.jsonl",
        [{"file_id": "file-a", "status": "indexed", "chunk_count": 1, "locator_coverage": 0.0, "index_inclusion_reason": "indexable"}],
    )
    _write_jsonl(
        shadow / "corpus-ledger.jsonl",
        [{"file_id": "file-a", "status": "indexed", "chunk_count": 2, "locator_coverage": 1.0, "index_inclusion_reason": "indexable"}],
    )
    _write_chunks(baseline / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "old", "citation_id": "old-citation"}}])
    _write_chunks(shadow / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "new", "citation_id": "new-citation"}}])
    (shadow / "old-to-new-id-map.json").write_text(
        json.dumps({"rows": [{"file_id": "file-a", "old_chunk_id": "old", "old_citation_id": "old-citation", "new_chunk_id": "new", "new_citation_id": "new-citation", "status": "mapped"}]}),
        encoding="utf-8",
    )
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(questions, [{"id": "q1", "reference_context_ids": ["file-a"]}])

    result = check_promotion_gates(baseline, shadow, questions_path=questions)

    assert isinstance(result, PromotionGateResult)
    assert result.passed is True
    assert result.errors == []


def test_check_promotion_gates_rejects_missing_gold_reference_id(tmp_path):
    baseline = tmp_path / "baseline"
    shadow = tmp_path / "shadow"
    baseline.mkdir()
    shadow.mkdir()
    _write_jsonl(baseline / "corpus-ledger.jsonl", [{"file_id": "file-a", "status": "indexed", "chunk_count": 1}])
    _write_jsonl(shadow / "corpus-ledger.jsonl", [{"file_id": "file-b", "status": "indexed", "chunk_count": 1}])
    _write_chunks(baseline / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "old"}}])
    _write_chunks(shadow / "chunks.jsonl", [{"metadata": {"file_id": "file-b", "chunk_id": "new"}}])
    (shadow / "old-to-new-id-map.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(questions, [{"id": "q1", "reference_context_ids": ["file-a"]}])

    result = check_promotion_gates(baseline, shadow, questions_path=questions)

    assert result.passed is False
    assert "gold reference_context_id missing from shadow ledger: file-a" in result.errors


def test_check_promotion_gates_rejects_self_comparison(tmp_path):
    root = tmp_path / "same"
    root.mkdir()
    _write_jsonl(root / "corpus-ledger.jsonl", [{"file_id": "file-a", "status": "indexed", "chunk_count": 1}])
    _write_chunks(root / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "old"}}])
    (root / "old-to-new-id-map.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(questions, [])

    result = check_promotion_gates(root, root, questions_path=questions)

    assert result.passed is False
    assert "baseline and shadow roots must be different" in result.errors


def test_check_promotion_gates_rejects_partial_old_to_new_id_map(tmp_path):
    baseline = tmp_path / "baseline"
    shadow = tmp_path / "shadow"
    baseline.mkdir()
    shadow.mkdir()
    _write_jsonl(baseline / "corpus-ledger.jsonl", [{"file_id": "file-a", "status": "indexed", "chunk_count": 2}])
    _write_jsonl(shadow / "corpus-ledger.jsonl", [{"file_id": "file-a", "status": "indexed", "chunk_count": 1, "locator_coverage": 1.0}])
    _write_chunks(
        baseline / "chunks.jsonl",
        [
            {"metadata": {"file_id": "file-a", "chunk_id": "old-1"}},
            {"metadata": {"file_id": "file-a", "chunk_id": "old-2"}},
        ],
    )
    _write_chunks(shadow / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "new-1"}}])
    (shadow / "old-to-new-id-map.json").write_text(
        json.dumps({"rows": [{"file_id": "file-a", "old_chunk_id": "old-1", "new_chunk_id": "new-1", "status": "mapped"}]}),
        encoding="utf-8",
    )
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(questions, [])

    result = check_promotion_gates(baseline, shadow, questions_path=questions)

    assert result.passed is False
    assert "old chunk has no replacement or reviewed drop: old-2" in result.errors


def test_check_promotion_gates_accepts_reviewed_drop(tmp_path):
    baseline = tmp_path / "baseline"
    shadow = tmp_path / "shadow"
    baseline.mkdir()
    shadow.mkdir()
    _write_jsonl(baseline / "corpus-ledger.jsonl", [{"file_id": "file-a", "status": "indexed", "chunk_count": 1}])
    _write_jsonl(shadow / "corpus-ledger.jsonl", [{"file_id": "file-a", "status": "indexed", "chunk_count": 1, "locator_coverage": 1.0}])
    _write_chunks(baseline / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "old-1"}}])
    _write_chunks(shadow / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "new-1"}}])
    (shadow / "old-to-new-id-map.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    (shadow / "reviewed-drops.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "old_chunk_id": "old-1",
                        "reason": "duplicate content merged into new-1",
                        "reviewed_by": "migration-review",
                        "reviewed_at": "2026-06-25T00:00:00Z",
                        "rollback_impact": "citation redirects to merged chunk",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(questions, [])

    result = check_promotion_gates(baseline, shadow, questions_path=questions)

    assert result.passed is True
    assert result.errors == []
```

Add this import smoke to `tests/test_scripts.py`:

```python
def test_ingestion_promotion_script_imports_and_defines_main():
    module = _load_script("scripts/check_ingestion_promotion.py", "check_ingestion_promotion_script")

    assert hasattr(module, "main")
```

- [ ] **Step 2: Run promotion tests and verify they fail**

Run:

```bash
uv run python -m pytest tests/test_ingestion_promotion.py tests/test_scripts.py::test_ingestion_promotion_script_imports_and_defines_main -q
```

Expected: FAIL because the promotion module and script do not exist.

- [ ] **Step 3: Add promotion comparator**

Create `src/imperial_rag/ingestion/promotion.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PromotionGateResult:
    passed: bool
    errors: list[str]
    summary: dict[str, int | float]


def check_promotion_gates(
    baseline_root: Path,
    shadow_root: Path,
    *,
    questions_path: Path,
    min_locator_coverage: float = 0.95,
) -> PromotionGateResult:
    errors: list[str] = []
    if baseline_root.resolve() == shadow_root.resolve():
        errors.append("baseline and shadow roots must be different")

    baseline_rows = _read_jsonl(baseline_root / "corpus-ledger.jsonl")
    shadow_rows = _read_jsonl(shadow_root / "corpus-ledger.jsonl")
    baseline_chunks = _read_jsonl(baseline_root / "chunks.jsonl")
    id_map = _read_json(shadow_root / "old-to-new-id-map.json")
    reviewed_drops = _read_optional_json(shadow_root / "reviewed-drops.json", default={"rows": []})
    questions = _read_jsonl(questions_path)

    baseline_ids = {str(row.get("file_id")) for row in baseline_rows}
    shadow_ids = {str(row.get("file_id")) for row in shadow_rows}
    if not shadow_ids.issuperset(baseline_ids):
        for file_id in sorted(baseline_ids - shadow_ids):
            errors.append(f"baseline file missing from shadow ledger: {file_id}")

    baseline_indexed = sum(1 for row in baseline_rows if row.get("status") == "indexed")
    shadow_indexed = sum(1 for row in shadow_rows if row.get("status") == "indexed")
    if shadow_indexed < baseline_indexed:
        errors.append(f"shadow indexed file count regressed: {shadow_indexed} < {baseline_indexed}")

    shadow_chunk_count = sum(int(row.get("chunk_count") or 0) for row in shadow_rows)
    if shadow_chunk_count == 0:
        errors.append("shadow chunk count is zero")

    locator_rows = [row for row in shadow_rows if int(row.get("chunk_count") or 0) > 0]
    locator_coverage = _mean(float(row.get("locator_coverage") or 0.0) for row in locator_rows)
    if locator_rows and locator_coverage < min_locator_coverage:
        errors.append(f"shadow locator coverage below gate: {locator_coverage} < {min_locator_coverage}")

    baseline_chunk_ids = _metadata_values(baseline_chunks, "chunk_id")
    baseline_citation_ids = _metadata_values(baseline_chunks, "citation_id")
    mapped_old_chunk_ids = {
        str(row.get("old_chunk_id"))
        for row in id_map.get("rows", [])
        if row.get("old_chunk_id") and row.get("new_chunk_id")
    }
    mapped_old_citation_ids = {
        str(row.get("old_citation_id"))
        for row in id_map.get("rows", [])
        if row.get("old_citation_id") and row.get("new_citation_id")
    }
    reviewed_drop_chunk_ids, reviewed_drop_citation_ids = _reviewed_drop_ids(reviewed_drops, errors)
    unmapped_old_chunk_ids = baseline_chunk_ids - mapped_old_chunk_ids - reviewed_drop_chunk_ids
    for old_chunk_id in sorted(unmapped_old_chunk_ids):
        errors.append(f"old chunk has no replacement or reviewed drop: {old_chunk_id}")

    for context_id in _reference_context_ids(questions):
        if context_id in shadow_ids:
            continue
        if context_id in baseline_chunk_ids:
            if context_id not in mapped_old_chunk_ids and context_id not in reviewed_drop_chunk_ids:
                errors.append(f"gold old chunk ID has no replacement or reviewed drop: {context_id}")
            continue
        if context_id in baseline_citation_ids:
            if context_id not in mapped_old_citation_ids and context_id not in reviewed_drop_citation_ids:
                errors.append(f"gold old citation ID has no replacement or reviewed drop: {context_id}")
            continue
        errors.append(f"gold reference_context_id missing from shadow ledger: {context_id}")

    summary = {
        "baseline_files": len(baseline_rows),
        "shadow_files": len(shadow_rows),
        "baseline_indexed_files": baseline_indexed,
        "shadow_indexed_files": shadow_indexed,
        "baseline_chunk_ids": len(baseline_chunk_ids),
        "shadow_chunk_count": shadow_chunk_count,
        "shadow_locator_coverage": locator_coverage,
        "mapped_old_chunk_ids": len(mapped_old_chunk_ids),
        "reviewed_drop_chunk_ids": len(reviewed_drop_chunk_ids),
        "unmapped_old_chunk_ids": len(unmapped_old_chunk_ids),
    }
    return PromotionGateResult(passed=not errors, errors=errors, summary=summary)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path, *, default: dict) -> dict:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_values(rows: list[dict], key: str) -> set[str]:
    values: set[str] = set()
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        value = metadata.get(key)
        if value is not None and str(value).strip():
            values.add(str(value))
    return values


def _reviewed_drop_ids(payload: dict, errors: list[str]) -> tuple[set[str], set[str]]:
    required = {"old_chunk_id", "reason", "reviewed_by", "reviewed_at", "rollback_impact"}
    chunk_ids: set[str] = set()
    citation_ids: set[str] = set()
    for index, row in enumerate(payload.get("rows", [])):
        missing = sorted(field for field in required if not str(row.get(field) or "").strip())
        if missing:
            errors.append(f"reviewed drop row {index} missing fields: {', '.join(missing)}")
            continue
        chunk_ids.add(str(row["old_chunk_id"]))
        if row.get("old_citation_id"):
            citation_ids.add(str(row["old_citation_id"]))
    return chunk_ids, citation_ids


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(sum(materialized) / len(materialized), 4)


def _reference_context_ids(rows: list[dict]) -> list[str]:
    ids: list[str] = []
    for row in rows:
        for context_id in row.get("reference_context_ids") or []:
            value = str(context_id).strip()
            if value:
                ids.append(value)
    return ids
```

- [ ] **Step 4: Add the CLI wrapper**

Create `scripts/check_ingestion_promotion.py`:

```python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    _ensure_src_on_path()
    parser = argparse.ArgumentParser(description="Check whether a shadow ingestion run can be promoted.")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--questions-path", type=Path, default=Path("evals/questions.jsonl"))
    parser.add_argument("--min-locator-coverage", type=float, default=0.95)
    args = parser.parse_args(argv)

    from imperial_rag.ingestion.promotion import check_promotion_gates

    result = check_promotion_gates(
        args.baseline_root,
        args.shadow_root,
        questions_path=args.questions_path,
        min_locator_coverage=args.min_locator_coverage,
    )
    print(json.dumps({"passed": result.passed, "errors": result.errors, "summary": result.summary}, ensure_ascii=False, indent=2))
    if not result.passed:
        raise SystemExit(1)


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run promotion tests**

Run:

```bash
uv run python -m pytest tests/test_ingestion_promotion.py tests/test_scripts.py::test_ingestion_promotion_script_imports_and_defines_main -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 7**

Run:

```bash
git add src/imperial_rag/ingestion/promotion.py scripts/check_ingestion_promotion.py tests/test_ingestion_promotion.py tests/test_scripts.py
git commit -m "feat: add ingestion promotion gates"
```

Expected: commit succeeds with only these four files staged.

---

### Task 8: Docs And End-To-End Verification

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `docs/superpowers/plans/2026-06-25-ingestion-chunking-migration.md`

- [ ] **Step 1: Document new artifacts in README**

In `README.md`, update the local artifacts table to include:

```markdown
| Corpus ledger | `.imperial_rag/extracted/corpus-ledger.jsonl` |
| Corpus ledger summary | `.imperial_rag/extracted/corpus-ledger-summary.json` |
| Baseline artifacts | `.imperial_rag/extracted-baseline/` |
| Shadow artifacts | `.imperial_rag/extracted-shadow-v2/` |
| Old-to-new ID map | `.imperial_rag/extracted-shadow-v2/old-to-new-id-map.json` |
```

Add this migration workflow section near the ingestion docs:

````markdown
### Shadow Ingestion Migration

Capture baseline artifacts before running a shadow migration:

```bash
rm -rf .imperial_rag/extracted-baseline
cp -R .imperial_rag/extracted .imperial_rag/extracted-baseline
```

Use a suffix to build separate Elasticsearch and Qdrant targets, and use `--artifact-root` so shadow extraction output does not overwrite `.imperial_rag/extracted`:

```bash
uv run python scripts/ingest.py \
  --workspace-root /Users/danil/Public/imperial \
  --index-suffix shadow_v2 \
  --artifact-root .imperial_rag/extracted-shadow-v2 \
  --baseline-artifact-root .imperial_rag/extracted-baseline
```

With vectors enabled:

```bash
uv run python scripts/ingest.py \
  --workspace-root /Users/danil/Public/imperial \
  --index-vectors \
  --index-suffix shadow_v2 \
  --artifact-root .imperial_rag/extracted-shadow-v2 \
  --baseline-artifact-root .imperial_rag/extracted-baseline
```

Promotion checks compare immutable baseline artifacts with the isolated shadow artifact directory:

```bash
uv run python scripts/check_ingestion_promotion.py \
  --baseline-root .imperial_rag/extracted-baseline \
  --shadow-root .imperial_rag/extracted-shadow-v2 \
  --questions-path evals/questions.jsonl
```

Do not promote shadow index names to canonical `ELASTICSEARCH_INDEX` or `QDRANT_COLLECTION` until promotion checks pass, every old chunk ID is mapped or explicitly reviewed as dropped, and the eval rows with `reference_context_ids` still resolve to indexed files or mapped IDs.
````

- [ ] **Step 2: Document chunk-budget env names in `.env.example`**

In `.env.example`, update or add:

```bash
# Ingestion chunk budget. These are estimated-token units in the structure-first splitter.
IMPERIAL_RAG_CHUNK_SIZE=650
IMPERIAL_RAG_CHUNK_OVERLAP=80
```

- [ ] **Step 3: Run formatting and focused tests**

Run:

```bash
uv run ruff check src tests scripts
uv run python -m pytest tests/test_ingestion_ledger.py tests/test_ingestion_dedupe.py tests/test_ingestion_promotion.py tests/test_extraction.py tests/test_chunking.py tests/test_pipeline.py tests/test_scripts.py -q
```

Expected: PASS.

- [ ] **Step 4: Run the full unit suite**

Run:

```bash
uv run python -m pytest -q
```

Expected: PASS.

- [ ] **Step 5: Run a local shadow artifact smoke without vectors**

Derive the expected scan count from the same scanner the CLI uses:

```bash
EXPECTED_SCANNED_FILES="$(uv run python - <<'PY'
from pathlib import Path
from imperial_rag.ingestion.manifest import scan_files

print(len(scan_files(Path("documents"))))
PY
)"
printf 'expected_scanned_files=%s\n' "$EXPECTED_SCANNED_FILES"
```

Run:

```bash
test -s .imperial_rag/extracted-baseline/chunks.jsonl
rm -rf .imperial_rag/extracted-shadow-v2
uv run python scripts/ingest.py \
  --workspace-root /Users/danil/Public/imperial \
  --index-suffix shadow_v2 \
  --artifact-root .imperial_rag/extracted-shadow-v2 \
  --baseline-artifact-root .imperial_rag/extracted-baseline
```

Expected stdout includes:

```text
scanned_files=<EXPECTED_SCANNED_FILES from scan_files()>
chunks=
keyword_indexed=True
vector_indexed=False
```

Expected files exist:

```bash
test -s .imperial_rag/extracted-shadow-v2/corpus-ledger.jsonl
test -s .imperial_rag/extracted-shadow-v2/corpus-ledger-summary.json
test -s .imperial_rag/extracted-shadow-v2/old-to-new-id-map.json
test -s .imperial_rag/extracted-shadow-v2/chunks.jsonl
```

- [ ] **Step 6: Run promotion check against the preflight baseline**

Run:

```bash
test -s .imperial_rag/extracted-baseline/baseline-fingerprint.json
uv run python scripts/check_ingestion_promotion.py \
  --baseline-root .imperial_rag/extracted-baseline \
  --shadow-root .imperial_rag/extracted-shadow-v2 \
  --questions-path evals/questions.jsonl
```

Expected: PASS only if `.imperial_rag/extracted-baseline` was created by Task 0 before the migration code changed and `.imperial_rag/extracted-shadow-v2` was produced by the shadow smoke. If the baseline is missing, stop and recapture it from a clean pre-migration checkout or backup; do not copy `.imperial_rag/extracted` after a shadow run.

- [ ] **Step 7: Commit Task 8**

Run:

```bash
git add README.md .env.example docs/superpowers/plans/2026-06-25-ingestion-chunking-migration.md
git commit -m "docs: document ingestion migration workflow"
```

Expected: commit succeeds with only these three files staged.

---

## Rollback And Promotion Rules

- Canonical default ingestion still uses `ELASTICSEARCH_INDEX` and `QDRANT_COLLECTION` with no suffix.
- Shadow ingestion uses `--index-suffix shadow_v2`, producing separate target names such as `imperial_keyword_chunks_shadow_v2` and `imperial_chunks_qwen_shadow_v2`, and `--artifact-root .imperial_rag/extracted-shadow-v2` so extracted artifacts stay isolated from `.imperial_rag/extracted`.
- If promotion gates fail, leave canonical environment values unchanged and keep the shadow artifacts for diagnosis.
- Promotion requires:
  - `uv run python -m pytest -q` passes.
  - `corpus-ledger.jsonl` accounts for every scanned file.
  - Indexed shadow file count is not lower than baseline.
  - Shadow chunk count is nonzero.
  - Locator coverage for chunked files is at least `0.95`.
  - `old-to-new-id-map.json` maps every baseline old chunk ID to at least one new chunk ID, unless the old chunk appears in `reviewed-drops.json` with reason, reviewer, timestamp, and rollback impact.
  - Every gold `reference_context_ids` value in `evals/questions.jsonl` exists in the shadow ledger when it is a file ID, or maps through the old-to-new ID map when it is an old chunk/citation ID.
  - A separate eval run confirms grounded answers before changing default index names.

## Follow-Up Plan Boundary

After this migration passes, create a separate retrieval plan for parent-child or small-to-big retrieval. That plan must rerun Context7 against the current LangChain Python docs and choose between a documented vector-store retriever, a custom LangChain runnable retriever, or a current parent-document retriever API based on the installed package surface.

## Self-Review

Spec coverage:

- Corpus reconciliation ledger: Task 1 and Task 2.
- Coverage repair taxonomy for archives, `.doc`, OCR candidates, junk/no-text, unsupported files: Task 1 ledger fields and proposed actions.
- Stable evidence IDs and locators before chunking: Task 3.
- Structure-first chunks, contextual headers, `add_start_index=True`, Russian separators, noise dropping: Task 4.
- Exact dedupe before indexing: Task 5.
- Shadow Elasticsearch/Qdrant targets, versioned artifacts, old-to-new ID map, rollback: Task 6 and rollback rules.
- Promotion gates with gold `reference_context_ids`: Task 7.
- ParentDocumentRetriever-style retrieval after evidence stability: explicitly moved to a follow-up plan boundary.

Placeholder scan:

- Checked the plan for empty task language, missing file paths, and missing commands.
- Each task has explicit files, test commands, implementation snippets, verification commands, and commit commands.

Type consistency:

- `CorpusLedgerRow`, `DuplicateDecision`, and `PromotionGateResult` names are used consistently.
- `source_doc_id`, `source_locator`, `element_id`, `chunk_id`, `citation_id`, `start_index`, and `body_start_index` are introduced before later tasks rely on them.
- `duplicate_action_by_file` is produced by `duplicate_action_map()` and consumed by `write_corpus_ledger()`.
