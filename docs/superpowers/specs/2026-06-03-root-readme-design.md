# Root README Design

Date: 2026-06-03
Workspace: `/Users/danil/Public/imperial`
Scope: root `README.md` for the Imperial RAG repository

## Status

Design approved in chat and written for user review. Implementation planning has not started.

## Context

The repository does not currently have a root `README.md`. The project is a Python 3.12+ local/private RAG system for the Imperial document corpus. The live repo includes:

- package code under `src/imperial_rag/`;
- ingestion, extraction, OCR, chunking, indexing, retrieval, answering, tracing, and Streamlit UI modules;
- operational scripts under `scripts/`;
- deterministic evaluation prompts under `evals/questions.jsonl`;
- tests under `tests/`;
- local services for Qdrant and Phoenix;
- generated private local state under `.imperial_rag/`.

The README should help a reader quickly understand what the repo is and how to run the current local workflows.

## Chosen Approach

Use a practical hybrid README: concise project overview plus concrete local commands.

The README should serve both:

- a developer opening the repository and needing setup/run/test commands;
- a reader trying to understand the architecture and privacy boundaries.

It should not become a full internal runbook or a speculative product document.

## Goals

- Explain that Imperial RAG is a local/private RAG system for the Imperial corpus.
- Show the main data flow from source documents to cited answers.
- Provide accurate setup, ingestion, query, UI, local service, evaluation, and test commands.
- Summarize the project layout and key generated/private paths.
- Document the most important environment variables and privacy constraints.
- Keep the README concise enough to be useful as first-page onboarding.

## Non-Goals

- No detailed documentation of private corpus contents.
- No sensitive examples, API keys, generated traces, or corpus artifacts.
- No exhaustive internal runbook.
- No future roadmap beyond what the repo can run today.
- No unrelated contributing, changelog, license, or deployment boilerplate unless it already exists in the repo.

## README Structure

The root `README.md` should contain these sections:

1. `# Imperial RAG`
2. Short project summary.
3. Capabilities: ingest, extract, index, query, cite, evaluate, trace.
4. Architecture and data flow.
5. Quickstart.
6. Local services: Qdrant and Phoenix.
7. CLI, UI, and evaluation commands.
8. Project layout.
9. Configuration.
10. Testing.
11. Privacy and local-state notes.
12. Troubleshooting.

## Content Requirements

The README should describe the current code-derived behavior:

- Source documents live under `documents/`.
- Generated state lives under `.imperial_rag/`.
- Ingestion creates a manifest, extracted artifacts, chunks, and a keyword index.
- SQLite FTS provides keyword search.
- Qdrant provides optional vector search when running locally and vector indexing is requested.
- The query runtime answers only from retrieved evidence and uses strict citation prompts.
- The Streamlit UI is available through `src/imperial_rag/web_app.py`.
- Phoenix is optional for tracing and evaluation storage.
- Deterministic evals live in `evals/questions.jsonl` and run through `scripts/run_phoenix_eval.py`.

## Commands To Include

Use the repository's actual command surfaces:

```bash
uv sync --extra dev
cp .env.example .env
uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial
./scripts/start_qdrant.sh
uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial --index-vectors
uv run python scripts/query.py "question text"
uv run python -m streamlit run src/imperial_rag/web_app.py --server.address 127.0.0.1 --server.port 8501
docker compose up phoenix
uv run python scripts/run_phoenix_eval.py
uv run python -m pytest -q
```

The README should make clear which commands require secrets, Qdrant, Phoenix, Docker, or generated local state.

## Error Handling And Troubleshooting

Include concise guidance for likely first-run issues:

- missing `OPENAI_API_KEY` limits answer generation, embeddings, and OCR-backed paths;
- Qdrant must be running before `--index-vectors` can succeed;
- Phoenix must be running before `--use-phoenix` eval mode or trace viewing;
- source files and generated local artifacts should not be committed;
- live Qdrant tests are opt-in with `IMPERIAL_RAG_LIVE_QDRANT=1`.

## Verification

After writing the README:

- check that referenced paths and command names exist;
- inspect the diff for accidental private data or overlong internal detail;
- run a lightweight relevant check such as `uv run python -m pytest tests/test_config.py tests/test_scripts.py -q`, unless dependency/runtime state blocks it;
- inspect `git status --short` and `git diff`;
- commit only README-related files from this session.
