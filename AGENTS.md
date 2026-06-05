# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12+ local RAG project for the Imperial document corpus. Core package code lives in `src/imperial_rag/`: ingestion and extraction in `pipeline.py` and `extraction.py`, chunking/indexing in `chunking.py` and `indexing.py`, query runtime in `runtime.py`, workflows in `workflows.py`, and the Streamlit UI in `web_app.py`. Tests live in `tests/` and mirror the subsystem names with `test_*.py` files. Operational scripts live in `scripts/`, eval prompts in `evals/questions.jsonl`, design/planning notes in `docs/superpowers/`, source documents in `documents/`, and generated local state in `.imperial_rag/`.

## Build, Test, and Development Commands

- `uv sync --extra dev` installs runtime dependencies plus pytest.
- `uv run python -m pytest -q` runs the full test suite configured by `pyproject.toml`.
- `./scripts/start_qdrant.sh` starts local-only Qdrant on `127.0.0.1:6333`.
- `docker compose up phoenix` starts Phoenix tracing/eval storage on port `6006`.
- `uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial` ingests the corpus; add `--index-vectors` after Qdrant is running.
- `uv run python scripts/query.py "question text"` queries the processed RAG state.
- `uv run python -m streamlit run src/imperial_rag/web_app.py --server.address 127.0.0.1 --server.port 8501` runs the local UI.

To run the server locally, use the Streamlit command above from the repository root, then open `http://127.0.0.1:8501`. If port `8501` is already in use, choose another local port with `--server.port <port>` and report the final URL. Verify startup with `curl -fsS -I http://127.0.0.1:<port>/` or a browser smoke check before telling the user it is running.

## Coding Style & Naming Conventions

Use Python type hints, `from __future__ import annotations`, `pathlib.Path`, and dataclasses where they match existing code. Keep four-space indentation, `snake_case` for modules/functions, `PascalCase` for classes/dataclasses, and uppercase constants. No formatter or linter is configured, so match the existing concise style and run pytest before handing off.

## Testing Guidelines

The project uses pytest with `pythonpath = ["src"]` and `testpaths = ["tests"]`. Add focused tests beside related subsystem tests, name files `test_<module>.py`, and name test functions `test_<behavior>()`. Prefer `tmp_path`, fakes, and `monkeypatch` for local state. Live Qdrant checks are opt-in: `IMPERIAL_RAG_LIVE_QDRANT=1 uv run python -m pytest tests/test_qdrant_health.py -q`.

## Commit & Pull Request Guidelines

Use short imperative commit subjects, optionally Conventional Commit style such as `feat: add phoenix tracing` or `fix: preserve citation sources`. Before editing code, capture `git status --short` as the session baseline. After every code-changing task or checkpoint, run the relevant tests or checks, inspect `git status --short` and `git diff`, then commit. Stage only files or hunks changed by the current agent session; avoid `git add .`. Do not commit pre-existing user changes, unrelated generated artifacts, secrets, corpus artifacts, or local state. If a file contains mixed user and session edits that cannot be safely separated, stop and ask before committing. PRs should describe the change, list test commands run, note corpus/config impacts, and include screenshots for UI changes.

## Security & Configuration Tips

Treat `documents/`, `.imperial_rag/`, eval outputs, and Phoenix traces as private. Do not commit secrets or generated corpus artifacts. Configure services with environment variables such as `QDRANT_URL`, `QDRANT_COLLECTION`, `PHOENIX_CLIENT_ENDPOINT`, and OCR/OpenAI keys only in local environment files or shell state. Keep Qdrant bound to localhost.
