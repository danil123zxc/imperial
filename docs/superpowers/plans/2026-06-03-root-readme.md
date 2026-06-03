# Root README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a concise, practical hybrid root `README.md` for the Imperial RAG repository.

**Architecture:** Add one root documentation file that explains the live local RAG workflow, command surfaces, service dependencies, project layout, configuration, privacy boundaries, and first-run troubleshooting. No application code changes are needed.

**Tech Stack:** Markdown, Python 3.12+, uv, LangChain/LangGraph, Qdrant, Phoenix, Streamlit, pytest.

---

## File Structure

- Create: `README.md`
  - Responsibility: first-page project overview and developer onboarding for the current local/private Imperial RAG system.
- Read only: `.env.example`
  - Responsibility: source of truth for environment variable names and defaults.
- Read only: `pyproject.toml`
  - Responsibility: source of truth for runtime dependencies and Python version.
- Read only: `scripts/ingest.py`, `scripts/query.py`, `scripts/run_phoenix_eval.py`, `scripts/start_qdrant.sh`
  - Responsibility: source of truth for command names and flags.
- Read only: `compose.yaml`
  - Responsibility: source of truth for Phoenix local service setup.

## Task 1: Create The Root README

**Files:**
- Create: `README.md`
- Verify: `.env.example`
- Verify: `pyproject.toml`
- Verify: `scripts/ingest.py`
- Verify: `scripts/query.py`
- Verify: `scripts/run_phoenix_eval.py`
- Verify: `scripts/start_qdrant.sh`
- Verify: `compose.yaml`

- [ ] **Step 1: Confirm the README does not already exist**

Run:

```bash
test ! -f README.md
```

Expected: command exits with status `0`. If it exits with status `1`, stop and inspect the existing `README.md` before replacing or editing it.

- [ ] **Step 2: Create `README.md`**

Create `README.md` with this exact content:

````markdown
# Imperial RAG

Imperial RAG is a local/private retrieval-augmented generation system for the Imperial document corpus. It ingests files from `documents/`, extracts searchable text, builds local keyword and optional vector indexes, and answers questions only from retrieved evidence with source citations.

The project is designed for private local operation: source files stay in the workspace, generated state lives under `.imperial_rag/`, Qdrant runs locally when vector search is needed, and Phoenix can be self-hosted locally for tracing and evaluation storage.

## What It Does

- Scans every file under `documents/` into a manifest.
- Extracts text from supported documents, spreadsheets, PDFs, images, and OCR-backed sources.
- Writes extracted artifacts and chunks under `.imperial_rag/`.
- Builds a SQLite full-text keyword index for exact Russian/company terminology.
- Optionally indexes chunks into local Qdrant for semantic vector search.
- Retrieves hybrid evidence and generates strict citation-based answers.
- Provides a CLI query path and a local Streamlit chat UI.
- Runs deterministic citation/refusal/source-hint evaluations, with optional Phoenix experiment storage.

## Architecture

```text
documents/
  -> manifest + extraction + OCR
  -> .imperial_rag/extracted/ artifacts and chunks
  -> .imperial_rag/keyword.sqlite3 SQLite FTS
  -> optional local Qdrant collection
  -> retrieval + strict citation answer generation
  -> CLI / Streamlit UI
  -> optional Phoenix traces and eval experiments
```

Core package code lives in `src/imperial_rag/`:

- `pipeline.py`, `extraction.py`, `ocr.py`, and `chunking.py` handle ingestion and text preparation.
- `manifest.py` tracks discovered files, extraction status, duplicate groups, and index status.
- `indexing.py` owns SQLite FTS keyword indexing and Qdrant vector indexing helpers.
- `retrieval.py`, `answering.py`, `workflows.py`, and `runtime.py` own query-time RAG behavior.
- `tracing.py` configures Phoenix tracing.
- `web_app.py` provides the Streamlit UI.

## Quickstart

Install dependencies:

```bash
uv sync --extra dev
```

Create a local environment file:

```bash
cp .env.example .env
```

Fill in local secrets such as `OPENAI_API_KEY` in `.env` or your shell environment. Do not commit real keys.

Run ingestion without vector indexing:

```bash
uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial
```

Ask a question against the processed local state:

```bash
uv run python scripts/query.py "question text"
```

Run the local UI:

```bash
uv run python -m streamlit run src/imperial_rag/web_app.py --server.address 127.0.0.1 --server.port 8501
```

Then open `http://127.0.0.1:8501`.

## Local Services

### Qdrant

Qdrant is optional unless you want vector indexing and semantic retrieval. Start it locally before running ingestion with `--index-vectors`:

```bash
./scripts/start_qdrant.sh
```

In another terminal, index vectors:

```bash
uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial --index-vectors
```

Defaults:

- URL: `http://localhost:6333`
- collection: `imperial_chunks`
- storage: `.imperial_rag/qdrant_storage`

### Phoenix

Phoenix is optional for local tracing and evaluation storage.

Start the self-hosted Phoenix service:

```bash
docker compose up phoenix
```

Phoenix UI:

```text
http://localhost:6006
```

Run a query or ingestion command with tracing enabled:

```bash
uv run python scripts/query.py "question text" --trace-phoenix
uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial --trace-phoenix
```

## Evaluation

Gold questions live in `evals/questions.jsonl`.

Run deterministic local citation/refusal/source-hint checks:

```bash
uv run python scripts/run_phoenix_eval.py
```

Store the dataset and experiment in local Phoenix:

```bash
uv run python scripts/run_phoenix_eval.py --use-phoenix
```

Phoenix mode requires the Phoenix service to be reachable at `PHOENIX_CLIENT_ENDPOINT`, which defaults to `http://localhost:6006`.

## Testing

Run the full test suite:

```bash
uv run python -m pytest -q
```

Run the live Qdrant health test only when local Qdrant is intentionally running:

```bash
IMPERIAL_RAG_LIVE_QDRANT=1 uv run python -m pytest tests/test_qdrant_health.py -q
```

## Project Layout

```text
src/imperial_rag/          Python package code
scripts/                   Ingestion, query, eval, and service helper scripts
tests/                     pytest suite
evals/questions.jsonl      Deterministic evaluation questions
docs/superpowers/          Design specs and implementation plans
documents/                 Private source corpus
.imperial_rag/             Generated local state, indexes, extracted text, caches
compose.yaml               Local Phoenix service
pyproject.toml             Python package and dependency configuration
```

## Configuration

Important environment variables are documented in `.env.example`.

Common settings:

- `OPENAI_API_KEY`: required for answer generation, OpenAI embeddings, vector indexing, and OCR-backed paths.
- `IMPERIAL_RAG_WORKSPACE_ROOT`: workspace root, defaulting to `/Users/danil/Public/imperial`.
- `QDRANT_URL`: Qdrant endpoint, defaulting to `http://localhost:6333`.
- `QDRANT_COLLECTION`: Qdrant collection, defaulting to `imperial_chunks`.
- `PHOENIX_PROJECT_NAME`: Phoenix project name, defaulting to `imperial-rag`.
- `PHOENIX_COLLECTOR_ENDPOINT`: Phoenix trace collector endpoint.
- `PHOENIX_CLIENT_ENDPOINT`: Phoenix client/UI endpoint.
- `PHOENIX_TRACING_ENABLED` or `IMPERIAL_RAG_TRACING_ENABLED`: enables tracing when set to a truthy value.

Retrieval and chunking tuning variables are also listed in `.env.example`, including chunk size, overlap, vector fetch limits, keyword limits, reranker choices, and final evidence limits.

## Privacy And Local State

Treat these paths as private:

- `documents/`
- `.imperial_rag/`
- local Qdrant storage
- Phoenix traces and experiment data
- `.env` files containing real secrets

Do not commit API keys, generated corpus artifacts, local indexes, OCR cache data, or private traces.

## Troubleshooting

If query answers always refuse or lack useful evidence, run ingestion first and confirm `.imperial_rag/extracted/chunks.jsonl` and `.imperial_rag/keyword.sqlite3` exist.

If vector indexing fails, start Qdrant with `./scripts/start_qdrant.sh` before running ingestion with `--index-vectors`.

If Phoenix experiment mode fails, start Phoenix with `docker compose up phoenix` and confirm `http://localhost:6006` is reachable.

If semantic search, embeddings, answer generation, or OCR-backed paths fail, confirm `OPENAI_API_KEY` or the intended provider key is present in your local environment.

If live Qdrant tests fail during normal unit testing, make sure `IMPERIAL_RAG_LIVE_QDRANT` is unset or set to `0`.
````

- [ ] **Step 3: Check referenced files and command names**

Run:

```bash
test -f README.md
test -f .env.example
test -f pyproject.toml
test -f scripts/ingest.py
test -f scripts/query.py
test -f scripts/run_phoenix_eval.py
test -f scripts/start_qdrant.sh
test -f compose.yaml
rg -n "trace-phoenix|index-vectors" scripts/ingest.py scripts/query.py
rg -n "use-phoenix|questions-path|experiment-name" scripts/run_phoenix_eval.py
```

Expected:

- all `test -f` commands exit with status `0`;
- `rg` finds `--trace-phoenix` and `--index-vectors` in the script sources;
- `rg` finds `--use-phoenix`, `--questions-path`, and `--experiment-name` in `scripts/run_phoenix_eval.py`.

- [ ] **Step 4: Run a focused verification suite**

Run:

```bash
uv run python -m pytest tests/test_config.py tests/test_scripts.py -q
```

Expected: all selected tests pass. If dependency setup is missing, first run `uv sync --extra dev`, then repeat the test command.

- [ ] **Step 5: Inspect the README diff**

Run:

```bash
git diff -- README.md
git diff --check
```

Expected:

- the README contains only project onboarding documentation;
- no API keys, private corpus text, generated artifact payloads, or trace data appear;
- `git diff --check` exits with status `0`.

- [ ] **Step 6: Commit the README**

Run:

```bash
git status --short
git add README.md
git commit -m "docs: add root readme"
```

Expected:

- `README.md` is the only unstaged or staged implementation file before commit;
- commit succeeds with message `docs: add root readme`.
