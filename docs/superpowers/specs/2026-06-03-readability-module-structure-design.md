# Readability Module Structure Design

Date: 2026-06-03
Workspace: `/Users/danil/Public/imperial`
Scope: readability-only Python package restructuring

## Status

Design approved in chat and written for user review. Implementation planning has not started.

This workspace is not currently a Git repository, so the normal Superpowers commit step cannot be completed here.

## Context

The Imperial RAG package is currently a small flat Python package under `src/imperial_rag/`. The current files are understandable individually, but the flat structure makes the main workflows harder for new readers to see at a glance.

The project already has clear conceptual areas:

- corpus discovery, manifest storage, extraction, OCR, and chunking;
- ingestion orchestration and artifact writing;
- keyword and vector indexing;
- query-time retrieval, answer generation, citation validation, and runtime wiring;
- Phoenix tracing and evaluation;
- Streamlit UI entrypoint code.

The goal is not to redesign behavior. The goal is to make those existing areas visible in the filesystem and reduce the mental load for readers.

## Goals

- Reorganize modules around the main reader journeys: corpus, ingestion, indexing, query, evaluation, observability, and app UI.
- Split only files that currently mix several reader concerns.
- Preserve existing behavior.
- Preserve existing public imports through compatibility wrappers.
- Keep scripts and tests working throughout the migration.
- Avoid broad refactoring, API redesign, or feature work.

## Non-Goals

- No behavioral changes to ingestion, retrieval, evaluation, tracing, or UI.
- No replacement of Qdrant, SQLite, Phoenix, LangChain, LangGraph, or Streamlit.
- No new abstractions unless they are needed to move code cleanly.
- No deletion of compatibility modules in this migration.
- No test-suite redesign.

## Chosen Approach

Use a hybrid module structure: workflow-level packages for navigation, with small internal modules only where an existing file is carrying multiple responsibilities.

Target package map:

```text
src/imperial_rag/
  config.py
  corpus/
  ingestion/
  indexing/
  query/
  evals/
  observability/
  app/
```

Old public import paths remain available during this migration. Most moved top-level modules become thin compatibility wrappers. The existing `imperial_rag.indexing` import path becomes the `indexing/__init__.py` re-export surface, because Python should not have both `indexing.py` and an `indexing/` package competing for the same import name.

## Package Responsibilities

`corpus/` owns source-file concepts:

- manifest records and status enums;
- file scanning and duplicate grouping;
- document extraction;
- OCR client and OCR cache;
- chunk construction.

`ingestion/` owns the end-to-end ingestion pipeline:

- running discovery, extraction, chunking, and indexing;
- writing extracted artifacts;
- updating manifest and index statuses;
- producing ingestion summaries.

`indexing/` owns searchable storage:

- keyword normalization and SQLite full-text search;
- stable chunk IDs;
- Qdrant vector store creation and vector indexing;
- Qdrant health checks.

`query/` owns RAG query behavior:

- hybrid retrieval merge and ranking;
- evidence prompt construction;
- strict citation validation;
- LangGraph query workflow;
- runtime dependency wiring.

`evals/` owns deterministic regression evaluation and Phoenix experiment helpers.

`observability/` owns Phoenix tracing setup.

`app/` owns Streamlit UI code.

## File-Level Target

```text
src/imperial_rag/
  corpus/
    __init__.py
    manifest.py
    scanning.py
    extraction.py
    ocr.py
    chunking.py

  ingestion/
    __init__.py
    pipeline.py
    artifacts.py
    status.py
    workflow.py

  indexing/
    __init__.py
    ids.py
    keyword.py
    vector.py
    health.py

  query/
    __init__.py
    answering.py
    retrieval.py
    workflow.py
    runtime.py

  evals/
    __init__.py
    phoenix.py

  observability/
    __init__.py
    phoenix.py

  app/
    __init__.py
    web.py

  answering.py
  chunking.py
  extraction.py
  manifest.py
  ocr.py
  pipeline.py
  runtime.py
  tracing.py
  web_app.py
  workflows.py
```

The remaining top-level files are compatibility wrappers. They should import and re-export the moved public symbols rather than duplicate logic. The former `indexing.py` module is the exception: it becomes `indexing/__init__.py`.

## Existing File Mapping

Move current code as follows:

- `manifest.py`
  - `FileStatus`, `IndexStatus`, `FileRecord`, and `ManifestStore` move to `corpus/manifest.py`.
  - `hash_file`, `stable_file_id`, `scan_files`, and `assign_duplicate_groups` move to `corpus/scanning.py`.
- `extraction.py` moves to `corpus/extraction.py`.
- `ocr.py` moves to `corpus/ocr.py`.
- `chunking.py` moves to `corpus/chunking.py`.
- `pipeline.py`
  - `IngestionSummary`, `run_ingestion`, and `ingest_corpus` move to `ingestion/pipeline.py`.
  - artifact-writing helpers move to `ingestion/artifacts.py`.
  - summary/status helpers move to `ingestion/status.py`.
- `indexing.py`
  - stable chunk ID helpers move to `indexing/ids.py`.
  - `KeywordIndex` and keyword normalization/search helpers move to `indexing/keyword.py`.
  - Qdrant store creation and vector indexing move to `indexing/vector.py`.
  - Qdrant health helpers move to `indexing/health.py`.
- `answering.py` moves to `query/answering.py`.
- `workflows.py`
  - query workflow and state move to `query/workflow.py`.
  - retrieval merge/ranking helpers move to `query/retrieval.py`.
  - ingestion workflow moves to `ingestion/workflow.py`.
- `runtime.py` moves to `query/runtime.py`.
- `tracing.py` moves to `observability/phoenix.py`.
- `web_app.py` moves to `app/web.py`.
- `scripts/run_phoenix_eval.py` can stay as a CLI script, but reusable loading, scoring, target, and Phoenix experiment helpers move to `evals/phoenix.py`.

## Compatibility Strategy

The first migration pass preserves old import paths. Compatibility modules should stay small and boring.

Examples:

```python
# src/imperial_rag/pipeline.py
from imperial_rag.ingestion.pipeline import *
```

```python
# src/imperial_rag/runtime.py
from imperial_rag.query.runtime import *
```

```python
# src/imperial_rag/web_app.py
from imperial_rag.app.web import *
```

Top-level wrappers should be kept for the current public module names:

- `imperial_rag.answering`;
- `imperial_rag.chunking`;
- `imperial_rag.extraction`;
- `imperial_rag.manifest`;
- `imperial_rag.ocr`;
- `imperial_rag.pipeline`;
- `imperial_rag.runtime`;
- `imperial_rag.tracing`;
- `imperial_rag.web_app`;
- `imperial_rag.workflows`.

The `imperial_rag.indexing` compatibility surface is slightly larger because one current module will split into several files. It should be implemented as `src/imperial_rag/indexing/__init__.py`, re-exporting public names from `indexing.ids`, `indexing.keyword`, `indexing.vector`, and `indexing.health`.

## Migration Plan Shape

Implement in two passes.

Pass 1:

- Create package folders with `__init__.py` files.
- Move code into the new modules.
- Update internal imports to use new paths.
- Add top-level compatibility wrappers.
- Replace `src/imperial_rag/indexing.py` with the `src/imperial_rag/indexing/` package.
- Keep scripts and tests mostly unchanged.
- Run the full test suite.

Pass 2:

- Update scripts and tests to use the clearer new imports where it improves readability.
- Keep compatibility wrappers in place.
- Avoid deleting old import paths.
- Run the full test suite again.

## Error Handling And Risk Controls

The main risk is import drift, not behavior drift. The migration should treat import errors as the first class of failures to resolve.

Risk controls:

- Move one concern at a time.
- Keep top-level wrappers until a later cleanup.
- Avoid changing function signatures.
- Avoid editing algorithmic code while moving it.
- Prefer direct imports from the new package paths inside moved modules.
- Preserve test expectations unless they are explicitly about import locations.

## Testing

Because this is readability-only, the existing test suite is the primary verification gate:

```bash
uv run --extra dev python -m pytest -q
```

If failures occur, resolve import errors first, then any accidental behavior changes.

No large new test suite is required. Add only focused import-compatibility tests if the move exposes a gap in existing coverage.

## Approval Notes

The user selected the hybrid structure option and approved:

- the high-level package map;
- the file-level split;
- the two-pass compatibility-first migration strategy.
