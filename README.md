# Imperial RAG

Imperial RAG is a local/private retrieval-augmented generation system for the Imperial document corpus. It scans private files in `documents/`, extracts searchable text, builds local keyword and optional vector indexes, and answers questions from retrieved evidence with citations.

The project is designed to run on one trusted machine. Source files stay in the checkout, generated state lives under `.imperial_rag/`, Elasticsearch and Qdrant stay loopback-bound, and Phoenix can be used locally for tracing and evaluation storage.

## What It Does

- Ingests files from `documents/` into a SQLite manifest and extracted artifacts.
- Extracts text from common document, spreadsheet, PDF, image, and OCR-backed formats.
- Chunks extracted text into `.imperial_rag/extracted/chunks.jsonl`.
- Builds an Elasticsearch keyword index for exact terminology and Russian/company-name matching.
- Optionally indexes chunks into Qdrant for semantic vector retrieval.
- Uses DashScope/Qwen by default for chat, embeddings, OCR, and reranking when `DASHSCOPE_API_KEY` is configured.
- Produces strict citation-based answers through a CLI and a private Telegram bot.
- Returns a structured `no_relevant_documents` error without source links when retrieval is empty or the strict answer model rejects the retrieved evidence as insufficient.
- Supports deterministic evals, optional Ragas metrics, local structured logs, and Phoenix traces.

## Architecture

```text
documents/
  -> manifest scan
  -> extraction and optional OCR
  -> chunks and lineage under .imperial_rag/extracted/
  -> Elasticsearch keyword index
  -> optional Qdrant vector collection
  -> hybrid retrieval, reranking, and strict answer generation
  -> scripts/query.py or Telegram bot
  -> optional Phoenix traces and eval experiments
```

Core code lives in `src/imperial_rag/`:

- `ingestion/`: file scanning, extraction, OCR, manifests, chunking, and corpus ingestion.
- `indexing/`: Qdrant vector indexing helpers and stable chunk identifiers.
- `retrieval/`: Elasticsearch keyword search, vector/keyword fusion, and reranking.
- `answering/`: query runtime, LangGraph workflows, and strict answer formatting.
- `integrations/`: DashScope/Qwen provider adapters and legacy provider escape hatches.
- `observability/`: structured logs, event logs, Phoenix tracing, and privacy controls.
- `app/`: Telegram delivery, retained Streamlit UI/auth, and local chat history.

## Requirements

- Python 3.12+
- `uv`
- Docker or Docker Desktop for Elasticsearch, Qdrant, Phoenix, Kibana, and the Compose app
- A local `.env` copied from `.env.example`
- `DASHSCOPE_API_KEY` for hosted Qwen answer generation, embeddings, OCR, reranking, and vector indexing

Generated corpus state, service data, traces, and secrets are private. Do not commit `.env`, `documents/`, `.imperial_rag/`, local indexes, OCR caches, Phoenix data, or exported traces.

## Quickstart

Install the Python environment:

```bash
uv sync --extra dev
```

Create local configuration:

```bash
cp .env.example .env
```

Fill in `DASHSCOPE_API_KEY`, `TELEGRAM_BOT_TOKEN`, and
`IMPERIAL_RAG_TELEGRAM_ALLOWED_USER_IDS`. The allowlist is a comma-separated set of trusted numeric
Telegram user IDs. The bot ignores group chats and never runs retrieval for users outside the allowlist.

Start Elasticsearch for keyword retrieval:

```bash
./scripts/start_elasticsearch.sh
```

Ingest the corpus:

```bash
uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial
```

Ask a question:

```bash
uv run python scripts/query.py "question text"
```

Questions that are unrelated to the indexed corpus, or whose retrieved chunks do not support an answer, return the strict refusal text. The result carries `error.type=no_relevant_documents`, reports `retrieval.final_evidence=0`, and omits citations and retrieved-file links so weak matches are not presented as sources.

Run the Telegram bot:

```bash
uv run python -m imperial_rag.app.telegram
```

Verify the bot process is ready:

```bash
curl -fsS http://127.0.0.1:8501/healthz
```

Send `/start` to the bot from an allowlisted Telegram account, then send a document question. The bot
persists the exchange in `.imperial_rag/chat_history.sqlite3`, returns citation labels, and uploads each
unique retrieved source file that passes the existing documents-root and 50 MB checks. Those full source
files leave the trusted host and are stored by Telegram. The former Streamlit entrypoint remains at
`src/imperial_rag/app/web.py` for rollback but is not served by the deployed `app` service.

## Vector Search

Qdrant is optional for basic keyword-backed querying, but required for semantic vector retrieval. Start it before vector indexing:

```bash
./scripts/start_qdrant.sh
```

Index vectors:

```bash
uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial --index-vectors
```

If the embedding model or dimensions change, recreate the target Qdrant collection before reindexing:

```bash
uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial --index-vectors --recreate-qdrant-collection
```

## Private Compose Deployment

The private Compose stack runs the Telegram bot, Elasticsearch, Kibana, Qdrant, and Phoenix on one host. Published ports are bound to `127.0.0.1`; the bot itself reaches the Telegram Bot API over outbound HTTPS.

Elasticsearch, Kibana, and Phoenix in this stack are unauthenticated by default and are safe only while bound to `127.0.0.1` on a trusted host. Do not bind them to `0.0.0.0`, expose them through a public proxy, or share broad tunnels unless authentication and TLS are added.

Prepare the checkout:

```bash
cp .env.example .env
mkdir -p documents .imperial_rag/qdrant_storage
```

Fill `.env` with `DASHSCOPE_API_KEY`, `TELEGRAM_BOT_TOKEN`, `IMPERIAL_RAG_TELEGRAM_ALLOWED_USER_IDS`, and any model or tracing settings needed on that machine. Host-local commands can keep the `localhost` defaults from `.env.example`; `compose.yaml` overrides service endpoints inside containers.

Start the runtime stack:

```bash
docker compose up -d elasticsearch qdrant phoenix app kibana
```

Verify local endpoints:

```bash
curl -fsS http://127.0.0.1:8501/healthz
curl -fsS http://127.0.0.1:9200
curl -fsS http://127.0.0.1:5601/api/status
curl -fsS http://127.0.0.1:6333/healthz
curl -I --max-time 3 http://127.0.0.1:6006/
```

Run ingestion inside Compose when documents change:

```bash
docker compose --profile ingest up ingest
```

### Automatic application deployment

A successful GitHub Actions `Quality` job for a push to protected `main` deploys that exact commit to the production host over Tailscale and command-restricted SSH. The deployment rebuilds and replaces only the `app` service:

```bash
docker compose build app
docker compose up -d --no-deps app
```

The deploy command waits for the container health check and `http://127.0.0.1:8501/healthz`. The endpoint becomes ready only after Telegram accepts the bot token during application initialization. A failed build leaves the existing container running. A failed startup or health check restores the previously healthy commit and reports a failed GitHub deployment.

Automatic deployment does not run ingestion, restart Qdrant, Elasticsearch, Kibana, or Phoenix, or modify `.env`, `documents/`, `.imperial_rag/`, or persistent volumes. Apply corpus ingestion and dependency-service configuration changes explicitly as separate operator actions.

The production GitHub environment owns `TS_OAUTH_CLIENT_ID`, `TS_AUDIENCE`, `DEPLOY_SSH_KEY`, and `DEPLOY_KNOWN_HOSTS`. The Tailscale identity uses `tag:github-ci` and may reach only SSH on the production node. Deployment audit records and failure logs stay private on the server under `/home/server1/.local/state/imperial-deploy/`.

Telegram deployment notifications use the production secrets `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`. The interactive bot reads `TELEGRAM_BOT_TOKEN` from the server `.env`; it may use the same bot because deployment notifications only send messages and do not consume updates. `TELEGRAM_CHAT_ID` does not grant RAG access. After every production attempt, the workflow reports whether the commit was deployed, already healthy, superseded by a newer `main` commit, or failed, together with the repository, server, short commit SHA, triggering actor, and Actions run link. Telegram delivery is best-effort and cannot change the deployment result.

From a normal operator SSH session, roll back to the recorded previous healthy commit with:

```bash
/home/server1/.local/bin/imperial-deploy rollback
```

The CI-only SSH key cannot invoke rollback or arbitrary shell commands.

Inspect logs:

```bash
docker compose logs -f app
docker compose logs -f ingest
```

Stop the stack:

```bash
docker compose down
```

## Common Commands

```bash
# Install runtime and dev dependencies
uv sync --extra dev

# Run the default offline test suite
uv run python -m pytest -q

# Run the local quality gate: Ruff, mypy, pytest with coverage, and whitespace diff checks
./scripts/check.sh

# Rebuild keyword artifacts from documents/
uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial

# Rebuild keyword artifacts and vector index
uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial --index-vectors

# Build a fully isolated candidate (artifacts, manifest, OCR cache, Elasticsearch, and Qdrant)
uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial --enable-ocr --index-vectors --shadow-run migration-v1

# Validate the candidate and switch active Elasticsearch/Qdrant aliases plus the local pointer
uv run python scripts/promote_ingestion.py migration-v1 --workspace-root /Users/danil/Public/imperial

# Query processed state
uv run python scripts/query.py "question text"

# Run all configured evals
uv run python scripts/run_all_evals.py
```

## Services And State

| Surface | Default | Purpose |
| --- | --- | --- |
| Telegram bot | outbound Bot API; local `http://127.0.0.1:8501/healthz` | Private chat delivery and readiness |
| Elasticsearch | `http://localhost:9200` | Keyword search index `imperial_keyword_chunks` |
| Kibana | `http://127.0.0.1:5601` | Local inspection of Elasticsearch data |
| Qdrant | `http://localhost:6333` | Optional vector collection `imperial_chunks_qwen` |
| Phoenix | `http://localhost:6006` | Optional traces and eval experiments |
| `.imperial_rag/manifest.sqlite3` | local file | Corpus manifest and per-file status |
| `.imperial_rag/extracted/` | local directory | Extracted text, chunks, ledger, and lineage |
| `.imperial_rag/shadow-runs/<id>/` | local directory | Isolated candidate artifacts, manifest, OCR cache, and run descriptor |
| `.imperial_rag/active-ingestion.json` | local file | Atomically replaced pointer to the promoted artifacts and search aliases |
| `.imperial_rag/auth.sqlite3` | local file | Retained legacy Streamlit users and browser sessions |
| `.imperial_rag/chat_history.sqlite3` | local file | Local chat history |

Use the live files, database tables, and service health checks as source of truth for generated state. Snapshot counts in documentation drift quickly after corpus rebuilds.

Document authority overrides live in `docs/document-authority.json`. Each optional row is keyed by `relative_path` and may define `department`, `document_type`, `status` (`active`, `draft`, or `archived`), effective dates, `owner`, `authoritative_rank`, `supersedes`, and `version_group`. Exact-file duplicates are indexed once; every original path remains in the canonical chunk's `provenance_paths` metadata.

## Configuration

Important settings are documented in `.env.example`.

| Variable | Notes |
| --- | --- |
| `DASHSCOPE_API_KEY` | Required for Qwen chat, embeddings, OCR, reranking, and Ragas model-backed metrics |
| `IMPERIAL_RAG_WORKSPACE_ROOT` | Workspace root; defaults to this checkout in host runs and `/app` in Compose |
| `TELEGRAM_BOT_TOKEN` | Required Telegram Bot API token; keep only in local/server environment configuration |
| `IMPERIAL_RAG_TELEGRAM_ALLOWED_USER_IDS` | Required comma-separated allowlist of trusted numeric Telegram user IDs |
| `IMPERIAL_RAG_ADMIN_EMAIL` / `IMPERIAL_RAG_ADMIN_PASSWORD` | Retained legacy Streamlit admin access |
| `ELASTICSEARCH_URL` / `ELASTICSEARCH_INDEX` | Keyword search endpoint and index |
| `QDRANT_URL` / `QDRANT_COLLECTION` | Optional vector search endpoint and collection |
| `PHOENIX_CLIENT_ENDPOINT` / `PHOENIX_COLLECTOR_ENDPOINT` | Phoenix UI/client endpoint and OTLP trace collector |
| `PHOENIX_TRACING_ENABLED` / `IMPERIAL_RAG_TRACING_ENABLED` | Enable tracing without passing `--trace-phoenix` |
| `IMPERIAL_RAG_TRACE_*` | Trace run IDs, privacy/detail controls, and retrieval-debug options |
| `OPENINFERENCE_HIDE_*` | OpenInference redaction controls for prompts, outputs, images, and text |
| `IMPERIAL_RAG_LOG_*` | Local structured log level, format, service name, and environment |
| `IMPERIAL_RAG_EVENTLOG_*` | Optional local Elasticsearch event-log settings |
| `IMPERIAL_RAG_CHUNK_*`, `IMPERIAL_RAG_VECTOR_*`, `IMPERIAL_RAG_KEYWORD_LIMIT`, `IMPERIAL_RAG_RERANK_*` | Retrieval, chunking, and reranking tuning |

Legacy OpenAI, Azure OpenAI, and Cohere keys are compatibility escape hatches only. They must be enabled explicitly with `IMPERIAL_RAG_ALLOW_LEGACY_OPENAI` or `IMPERIAL_RAG_ALLOW_LEGACY_COHERE`.

## Tracing And Logs

Phoenix is optional for local tracing and eval storage:

```bash
docker compose up -d phoenix
uv run python scripts/query.py "question text" --trace-phoenix
```

Set `IMPERIAL_RAG_TRACE_RUN_ID` when you want a stable marker for filtering or validation:

```bash
IMPERIAL_RAG_TRACE_RUN_ID=readability-smoke uv run python scripts/query.py "question text" --trace-phoenix
uv run python scripts/validate_phoenix_trace.py --run-id readability-smoke
```

Phoenix traces are private diagnostic records. Depending on `OPENINFERENCE_HIDE_*` and `IMPERIAL_RAG_TRACE_*` settings, spans can include raw questions, model prompts, model answers, selected evidence text, candidate chunks, citations, source paths, and document metadata. Treat Phoenix access as access to private corpus-derived data.

The app emits newline-delimited structured logs to stderr. In Compose, Docker's `json-file` driver is the short-term log store, capped by `max-size: "10m"` and `max-file: "10"`. Use:

```bash
docker compose logs -f app
```

### Searchable Event Logs

Searchable event logs are optional and local-only. When enabled, the app writes closed-schema operational events to Elasticsearch after normal stderr logging. It does not scrape Docker logs and it does not index free-form log payloads.

Enable event-log indexes:

```bash
uv run python scripts/setup_event_logs.py
IMPERIAL_RAG_EVENTLOG_ELASTICSEARCH_ENABLED=true docker compose up -d app
```

Default data streams:

- `imperial-rag-events-v1`: query, web query, ingest, dependency, and app operational events.
- `imperial-rag-eval-summaries-v1`: eval summary events without private text.

Allowed event fields are operational metadata such as timings, counts, statuses, provider names, request/session IDs, pseudonymous user hashes, Phoenix trace IDs, and build provenance. They must not include raw questions, answers, prompts, messages, document text, snippets, citations, source lists, filenames, paths, raw document metadata, raw exception messages, tracebacks, credentials, or provider API responses. Redaction is a cleanup layer; closed schema validation is the privacy boundary.

## Evaluation

Gold questions live in `evals/questions.jsonl`.

Run the full configured eval suite:

```bash
uv run python scripts/run_all_evals.py
```

Run deterministic citation/refusal/source-hint checks:

```bash
uv run python scripts/run_phoenix_eval.py
```

Store a deterministic-only Phoenix experiment:

```bash
uv run python scripts/run_phoenix_eval.py --use-phoenix --ragas-metrics none
```

Run standalone Ragas checks:

```bash
uv run python scripts/run_ragas_eval.py
```

Ragas metrics need the dev dependencies and model credentials configured in `.env`.

## Testing

Run the normal offline suite:

```bash
uv run python -m pytest -q
```

Run the repo quality gate:

```bash
./scripts/check.sh
```

Live service tests are opt-in:

```bash
IMPERIAL_RAG_LIVE_QDRANT=1 uv run python -m pytest tests/test_qdrant_health.py -q
IMPERIAL_RAG_LIVE_ELASTICSEARCH=1 uv run python -m pytest tests/test_elasticsearch_live.py -q
```

Keep those flags unset during ordinary offline testing.

## Project Layout

```text
src/imperial_rag/          Application package
scripts/                   Ingestion, query, eval, tracing, and service helpers
tests/                     pytest suite
evals/questions.jsonl      Evaluation questions
docs/superpowers/          Planning and implementation notes
documents/                 Private source corpus
.imperial_rag/             Generated private local state
compose.yaml               Telegram bot, Elasticsearch, Kibana, Qdrant, and Phoenix stack
Dockerfile                 Compose app image
pyproject.toml             Python package, dependency, and tool configuration
```

## Troubleshooting

If answers refuse or return no useful evidence, confirm ingestion has run, `.imperial_rag/extracted/chunks.jsonl` exists, and Elasticsearch is reachable at `ELASTICSEARCH_URL`.

If vector search is unavailable, start Qdrant and rerun ingestion with `--index-vectors`. If the vector provider metadata no longer matches the configured embedding model or dimensions, recreate the collection.

If model-backed chat, OCR, embeddings, reranking, or Ragas metrics fail, confirm `DASHSCOPE_API_KEY` is present in `.env` or the process environment.

If the Telegram bot is unhealthy, confirm `TELEGRAM_BOT_TOKEN` and
`IMPERIAL_RAG_TELEGRAM_ALLOWED_USER_IDS`, then inspect `docker compose logs --tail=200 app`. A `200`
response from `/healthz` proves that the process initialized with Telegram; it does not prove that Qwen,
Elasticsearch, or Qdrant will answer a new question.

If Phoenix validation fails, start Phoenix, run a fresh traced query with a stable `IMPERIAL_RAG_TRACE_RUN_ID`, then validate that run ID.

If Compose services are unreachable, check that their ports remain bound to `127.0.0.1`, inspect `docker compose ps`, and read `docker compose logs -f <service>`.
