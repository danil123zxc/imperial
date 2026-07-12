# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12+ local RAG project for the Imperial document corpus. Core package code lives in directories under `src/imperial_rag/`: ingestion and extraction in `ingestion/`, vector indexing in `indexing/`, retrieval in `retrieval/`, query runtime and workflows in `answering/`, provider integrations in `integrations/`, observability in `observability/`, and the Streamlit UI in `app/`. Tests live in `tests/` and mirror the subsystem names with `test_*.py` files. Operational scripts live in `scripts/`, eval prompts in `evals/questions.jsonl`, design/planning notes in `docs/superpowers/`, source documents in `documents/`, and generated local state in `.imperial_rag/`.

## Build, Test, and Development Commands

- `uv sync --extra dev` installs runtime dependencies plus pytest.
- `uv run python -m pytest -q` runs the full test suite configured by `pyproject.toml`.
- `./scripts/start_qdrant.sh` starts local-only Qdrant on `127.0.0.1:6333`.
- `docker compose up phoenix` starts Phoenix tracing/eval storage on port `6006`.
- `uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial` ingests the corpus; add `--index-vectors` after Qdrant is running.
- `uv run python scripts/query.py "question text"` queries the processed RAG state.
- `uv run python -m streamlit run src/imperial_rag/app/web.py --server.address 127.0.0.1 --server.port 8501` runs the local UI.

To run the server locally, use the Streamlit command above from the repository root, then open `http://127.0.0.1:8501`. If port `8501` is already in use, choose another local port with `--server.port <port>` and report the final URL. Verify startup with `curl -fsS -I http://127.0.0.1:<port>/` or a browser smoke check before telling the user it is running.

The main query and UI entrypoints (`scripts/query.py` and `src/imperial_rag/app/web.py`) call `load_project_env()` and automatically load the repository `.env` without overriding already-exported shell variables. Vector search still requires Qdrant to be running and `DASHSCOPE_API_KEY` to be present in `.env` or the process environment. For ad hoc Python probes that import runtime/provider modules directly, call `from imperial_rag.env import load_project_env; load_project_env()` before creating `Settings()` or checking provider state.

## Coding Style & Naming Conventions

Use Python type hints, `from __future__ import annotations`, `pathlib.Path`, and dataclasses where they match existing code. Keep four-space indentation, `snake_case` for modules/functions, `PascalCase` for classes/dataclasses, and uppercase constants. Always try to reuse the project's implemented LangChain flows, adapters, and other existing integrations when they fit the task before adding bespoke code. Prefer well-maintained existing libraries, SDKs, and framework integrations over custom implementations, especially for service APIs, storage, parsing, retrieval, orchestration, and evaluation. No formatter or linter is configured, so match the existing concise style and run pytest before handing off.

## Testing Guidelines

The project uses pytest with `pythonpath = ["src"]` and `testpaths = ["tests"]`. Add focused tests beside related subsystem tests, name files `test_<module>.py`, and name test functions `test_<behavior>()`. Prefer `tmp_path`, fakes, and `monkeypatch` for local state. Live Qdrant checks are opt-in: `IMPERIAL_RAG_LIVE_QDRANT=1 uv run python -m pytest tests/test_qdrant_health.py -q`.

Evaluation workflows should be async-first: write eval runners, provider calls, and eval tests with async-compatible code paths instead of blocking synchronous loops when touching retrieval, model, tracing, or Phoenix-backed evaluation behavior.

## Obsidian Documentation Gate

After every code, configuration, schema, or operational-script change, use the repo-local `sync-imperial-obsidian-docs` skill at `.agents/skills/sync-imperial-obsidian-docs/SKILL.md` before final checks and commit. Treat the `Second brain/1. Projects/Imperial RAG/` note set as detailed human project documentation and `README.md` as the concise repo/operator guide.

Inspect the session diff against the starting `git status --short`, assess durable newcomer-facing impact, and update only affected registered notes through the skill's `obsidian_docs.py` adapter. Architecture, ownership, database/RAG schemas, ingestion and retrieval strategies, pipelines, evaluation, privacy, commands, failure modes, deliberate tradeoffs, current state, and roadmap changes are documentation-relevant. Test-only or behavior-preserving changes may be a no-op unless they change a documented invariant, module map, command, or troubleshooting path.

Read before writing, pass the adapter's SHA-256 optimistic-lock value, and read every changed note back through Obsidian CLI. Keep implemented code capability, generated-state snapshots, currently running services, and planned direction explicitly separate. Never copy secrets, corpus text, extracted chunks, prompts, answers, Phoenix payloads, eval outputs, auth rows, or chat transcripts into notes. If final checks cause another code change, repeat the documentation-impact assessment.

In the final handoff, list updated note titles or state `Obsidian docs: no durable documentation impact` with a short reason. If Obsidian is closed or the vault check fails, the code commit may proceed, but prominently report `Obsidian docs: pending`, the CLI error, and the candidate notes that still need synchronization.

## Log Inspection Guidelines

When diagnosing a runtime, UI, ingestion, eval, or service issue, check the live logs before guessing from code. In the Compose stack, the app emits structured JSON logs to stderr and Docker stores them with the `json-file` driver and rotation from `compose.yaml`. Start with `docker compose ps`, then inspect the relevant service with `docker compose logs --tail=200 app` or `docker compose logs --tail=200 <service>`; use `-f` only when actively watching a repro. If you need the underlying log file path, run `container_id=$(docker compose ps -q app) && docker inspect --format='{{.LogPath}}' "$container_id"` and read that returned Docker `LogPath`; the usual Docker path shape is `/var/lib/docker/containers/<container-id>/<container-id>-json.log`, which may live inside Docker Desktop's VM on macOS.

Searchable event logs are optional and separate from the stderr/Docker log stream. Check whether they are enabled with `rg -n "IMPERIAL_RAG_EVENTLOG" .env .env.example compose.yaml` and, when enabled, query Elasticsearch data streams such as `imperial-rag-events-v1` and `imperial-rag-eval-summaries-v1` through `http://127.0.0.1:9200`. Phoenix traces are also private diagnostics, not ordinary app logs; when a bug involves retrieval, model calls, or missing UI answers, correlate Docker logs with fresh Phoenix traces at `http://127.0.0.1:6006` or with `uv run python scripts/validate_phoenix_trace.py`.

## Commit & Pull Request Guidelines

Use short imperative commit subjects, optionally Conventional Commit style such as `feat: add phoenix tracing` or `fix: preserve citation sources`. Before editing code, capture `git status --short` as the session baseline. After every code-changing task or checkpoint, run the relevant tests or checks, inspect `git status --short` and `git diff`, then create a commit containing only the changes made by the agent in the current session. Stage only files or hunks changed by the current agent session; avoid `git add .`. Do not commit pre-existing user changes, unrelated generated artifacts, secrets, corpus artifacts, or local state. If a file contains mixed user and session edits that cannot be safely separated, stop and ask before committing. PRs should describe the change, list test commands run, note corpus/config impacts, and include screenshots for UI changes.

## Security & Configuration Tips

Treat `documents/`, `.imperial_rag/`, eval outputs, and Phoenix traces as private. Do not commit secrets or generated corpus artifacts. Configure services with environment variables such as `QDRANT_URL`, `QDRANT_COLLECTION`, `PHOENIX_CLIENT_ENDPOINT`, and OCR/OpenAI keys only in local environment files or shell state. Keep Qdrant bound to localhost.
