# Imperial RAG

Imperial RAG is a local/private retrieval-augmented generation system for the Imperial document corpus. It ingests files from `documents/`, extracts searchable text, builds local keyword and optional vector indexes, and answers questions only from retrieved evidence with source citations.

The project is designed for private local operation: source files stay in the workspace, generated filesystem state lives under `.imperial_rag/`, Elasticsearch keyword service state lives in the local Docker volume `imperial_elasticsearch_data`, Qdrant runs locally when vector search is needed, and Phoenix can be self-hosted locally for tracing and evaluation storage.

## What It Does

- Scans every file under `documents/` into a manifest.
- Extracts text from supported documents, spreadsheets, PDFs, images, and OCR-backed sources.
- Writes extracted artifacts and chunks under `.imperial_rag/`.
- Builds a local Elasticsearch keyword index for exact Russian/company terminology.
- Optionally indexes chunks into local Qdrant for semantic vector search.
- Retrieves hybrid evidence and generates strict citation-based answers.
- Provides a CLI query path and a local Streamlit chat UI.
- Runs deterministic citation/refusal/source-hint evaluations, with optional Phoenix experiment storage.

## Architecture

```text
documents/
  -> manifest + extraction + OCR
  -> .imperial_rag/extracted/ artifacts and chunks
  -> local Elasticsearch keyword index
  -> optional local Qdrant collection
  -> retrieval + strict citation answer generation
  -> CLI / Streamlit UI
  -> optional Phoenix traces and eval experiments
```

Core package code lives in `src/imperial_rag/`:

- `pipeline.py`, `extraction.py`, `ocr.py`, and `chunking.py` handle ingestion and text preparation.
- `manifest.py` tracks discovered files, extraction status, duplicate groups, and index status.
- `elasticsearch_keyword.py` owns Elasticsearch keyword indexing, and `indexing.py` owns Qdrant vector indexing helpers plus stable chunk ids.
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

Fill in local secrets such as `DASHSCOPE_API_KEY` in `.env` or your shell environment. Set `IMPERIAL_RAG_ADMIN_EMAIL` and `IMPERIAL_RAG_ADMIN_PASSWORD` to bootstrap the first approved Streamlit admin account. Do not commit real keys or passwords.

Start the required local Elasticsearch keyword service:

```bash
./scripts/start_elasticsearch.sh
```

In another terminal, run ingestion without vector indexing:

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

Then open `http://127.0.0.1:8501`. Sign in with the bootstrap admin account, then approve pending signup requests from the sidebar access panel. Auth state is stored locally in `.imperial_rag/auth.sqlite3`.

## Private Compose Deployment

The private Compose stack runs the Streamlit app, Elasticsearch, Kibana, Qdrant, and Phoenix on one machine with all published ports bound to `127.0.0.1`.

Prepare the server checkout:

```bash
cp .env.example .env
mkdir -p documents .imperial_rag/qdrant_storage
```

Fill `.env` with `DASHSCOPE_API_KEY`, `IMPERIAL_RAG_ADMIN_EMAIL`, `IMPERIAL_RAG_ADMIN_PASSWORD`, and any provider settings needed for the deployed machine. Host-local commands can keep the `localhost` defaults from `.env.example`; `compose.yaml` overrides service endpoints inside the app containers.

Start the runtime stack:

```bash
docker compose up -d elasticsearch qdrant phoenix app kibana
```

Verify the private endpoints from the host:

```bash
curl -fsS http://127.0.0.1:8501/_stcore/health
curl -fsS http://127.0.0.1:9200
curl -fsS http://127.0.0.1:5601/api/status
curl -fsS http://127.0.0.1:6333/healthz
curl -I --max-time 3 http://127.0.0.1:6006/
```

Open the app and Kibana through the local machine or an SSH tunnel:

```text
http://127.0.0.1:8501
http://127.0.0.1:5601
```

Run ingestion explicitly when documents change:

```bash
docker compose --profile ingest up ingest
```

Inspect logs:

```bash
docker compose logs -f app
docker compose logs -f ingest
```

Stop the stack:

```bash
docker compose down
```

## Local Services

### Elasticsearch

Elasticsearch is required for keyword search. Start it locally before running ingestion or querying the processed corpus:

```bash
./scripts/start_elasticsearch.sh
```

Defaults:

- URL: `http://localhost:9200`
- index: `imperial_keyword_chunks`
- Docker volume: `imperial_elasticsearch_data`

Ingestion rebuilds the Elasticsearch keyword index from `.imperial_rag/extracted/chunks.jsonl`. The old `.imperial_rag/keyword.sqlite3` file is obsolete generated state and is not read by the application after the Elasticsearch migration.

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
- collection: `imperial_chunks_qwen`
- provider metadata: `.imperial_rag/vector_provider.json`
- storage: `.imperial_rag/qdrant_storage`

Changing embedding providers or embedding dimensions requires a clean vector collection. For Qwen vectors, use `QDRANT_COLLECTION=imperial_chunks_qwen`; if reusing an older collection name, recreate that collection before indexing. At runtime, semantic search is disabled when `.imperial_rag/vector_provider.json` does not match the configured Qwen embedding model and dimensions.

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

Query traces use a domain-first hierarchy: `imperial_rag.query` contains retrieval and answer phases,
retrieval has child spans for vector search, keyword search, fusion, reranking, and final evidence selection,
and answer generation has child spans for context preparation, model call, and citation validation. For rich local
debugging, set `IMPERIAL_RAG_TRACE_FULL_FINAL_EVIDENCE=true` to attach full final evidence documents to the
Phoenix-native `retrieval.select_evidence` document panel. Candidate spans remain compact. `OPENINFERENCE_HIDE_*`
redaction settings still override document text and outputs.

### Local Logs

The app emits local newline-delimited JSON logs for CLI runs and Streamlit query handling. Logs go to process stderr, so they are captured wherever the process is launched. The current detached Streamlit run redirects stderr into `/tmp/imperial-streamlit-8501.log`.

Phoenix remains the trace and evaluation system. This v1 logging layer does not send logs or alerts to Sentry or any other external service.

## Evaluation

Gold questions live in `evals/questions.jsonl`.

Run all currently runnable evals and create one Phoenix experiment:

```bash
uv run python scripts/run_all_evals.py
```

Phoenix must already be reachable at `PHOENIX_CLIENT_ENDPOINT`, which defaults to `http://localhost:6006`. If it is not running, start it separately:

```bash
docker compose up -d phoenix
```

By default, the all-evals command stores deterministic citation/refusal/source-hint checks plus Ragas faithfulness and answer relevancy in the same Phoenix experiment. Answer relevancy uses both evaluator chat and embedding calls through the DashScope/OpenAI-compatible settings. To create a deterministic-only Phoenix experiment for troubleshooting:

```bash
uv run python scripts/run_all_evals.py --ragas-metrics none
```

Run deterministic local citation/refusal/source-hint checks:

```bash
uv run python scripts/run_phoenix_eval.py
```

Store only the legacy Phoenix eval runner output in local Phoenix:

```bash
uv run python scripts/run_phoenix_eval.py --use-phoenix
```

By default, Phoenix experiments from `run_phoenix_eval.py` include deterministic citation/refusal/source-hint checks plus Ragas faithfulness and answer relevancy. To store only deterministic scores:

```bash
uv run python scripts/run_phoenix_eval.py --use-phoenix --ragas-metrics none
```

Run standalone Ragas quality checks over the same gold questions without creating a Phoenix experiment:

```bash
uv run python scripts/run_ragas_eval.py
```

The Ragas runner is part of the dev/eval toolchain, so run `uv sync --extra dev` first. It defaults to `faithfulness,answer_relevancy`; reference-based metrics such as `context_recall` and `factual_correctness` use the `reference_answer` values in `evals/questions.jsonl`.

Write Ragas scores to JSONL:

```bash
uv run python scripts/run_ragas_eval.py --output-path .imperial_rag/evals/ragas-faithfulness.jsonl
```

## Testing

Run the full test suite:

```bash
uv run python -m pytest -q
```

Live tests are opt-in so the default suite stays offline and free of paid network calls. The live API and live corpus consent flags must be set in the process environment; `.env` is used only for secrets after those flags are present.

Run live DashScope/Qwen provider smoke and fixture integration tests only when real credentials are available:

```bash
IMPERIAL_RAG_LIVE_API=1 uv run python -m pytest tests/test_live_provider_smoke.py tests/test_live_rag_integration.py -q
```

When running from an isolated worktree, point live tests at the trusted env file instead of copying secrets:

```bash
IMPERIAL_RAG_LIVE_API=1 IMPERIAL_RAG_LIVE_ENV_PATH=/Users/danil/Public/imperial/.env uv run python -m pytest tests/test_live_provider_smoke.py tests/test_live_rag_integration.py -q
```

Run the real generated Imperial corpus health check only when `.imperial_rag` is present and you intentionally want to test it:

```bash
IMPERIAL_RAG_LIVE_API=1 IMPERIAL_RAG_LIVE_CORPUS=1 uv run python -m pytest tests/test_live_real_corpus.py -q
```

From an isolated worktree, point the test back at the main checkout env file so it can use the main checkout's generated state:

```bash
IMPERIAL_RAG_LIVE_API=1 IMPERIAL_RAG_LIVE_CORPUS=1 IMPERIAL_RAG_LIVE_ENV_PATH=/Users/danil/Public/imperial/.env uv run python -m pytest tests/test_live_real_corpus.py -q
```

Run the live Qdrant health test only when local Qdrant is intentionally running:

```bash
IMPERIAL_RAG_LIVE_QDRANT=1 uv run python -m pytest tests/test_qdrant_health.py -q
```

Run the live Elasticsearch test only when local Elasticsearch is intentionally running:

```bash
IMPERIAL_RAG_LIVE_ELASTICSEARCH=1 uv run python -m pytest tests/test_elasticsearch_live.py -q
```

## Project Layout

```text
src/imperial_rag/          Python package code
scripts/                   Ingestion, query, eval, and service helper scripts
tests/                     pytest suite
evals/questions.jsonl      Deterministic evaluation questions
docs/superpowers/          Design specs and implementation plans
documents/                 Private source corpus
.imperial_rag/             Generated local filesystem state, extracted text, caches
imperial_elasticsearch_data  Local Docker volume for keyword index state
compose.yaml               Local Phoenix, Qdrant, and Elasticsearch services
pyproject.toml             Python package and dependency configuration
```

## Configuration

Important environment variables are documented in `.env.example`.

Common settings:

- `DASHSCOPE_API_KEY`: required for hosted Qwen answer generation, embeddings/vector indexing, OCR, and reranking.
- `IMPERIAL_RAG_WORKSPACE_ROOT`: workspace root, defaulting to `/Users/danil/Public/imperial`.
- `QDRANT_URL`: Qdrant endpoint, defaulting to `http://localhost:6333`.
- `QDRANT_COLLECTION`: Qdrant collection, defaulting to `imperial_chunks_qwen`.
- `ELASTICSEARCH_URL`: Elasticsearch endpoint, defaulting to `http://localhost:9200`.
- `ELASTICSEARCH_INDEX`: Elasticsearch keyword index, defaulting to `imperial_keyword_chunks`.
- `PHOENIX_PROJECT_NAME`: Phoenix project name, defaulting to `imperial-rag`.
- `PHOENIX_COLLECTOR_ENDPOINT`: Phoenix trace collector endpoint.
- `PHOENIX_CLIENT_ENDPOINT`: Phoenix client/UI endpoint.
- `PHOENIX_TRACING_ENABLED` or `IMPERIAL_RAG_TRACING_ENABLED`: enables tracing when set to a truthy value.
- `IMPERIAL_RAG_TRACE_SESSION_ID`: optional CLI Phoenix `session.id`; omitted values generate a per-run `cli_<uuid>`.
- `IMPERIAL_RAG_TRACE_FULL_METADATA`: include full document metadata in retrieval/reranker traces when explicitly enabled.
- `IMPERIAL_RAG_TRACE_FULL_FINAL_EVIDENCE`: attach uncapped final evidence document text to `retrieval.select_evidence` when explicitly enabled.
- `IMPERIAL_RAG_TRACE_DOCUMENT_LIMIT` and `IMPERIAL_RAG_TRACE_DOCUMENT_CONTENT_CHARS`: bound traced document count and content length.
- `IMPERIAL_RAG_TRACE_USER_HASH_SECRET`: optional local secret for HMAC-based Phoenix user IDs; leave unset to preserve current deterministic local trace correlation.
- `OPENINFERENCE_HIDE_*` and `OTEL_BSP_*`: optional OpenInference privacy and batch-export controls; see `.env.example`.
- `IMPERIAL_RAG_LOG_LEVEL`: local structured log level, defaulting to `INFO`.
- `IMPERIAL_RAG_LOG_FORMAT`: local structured log format; v1 supports `json`.
- `IMPERIAL_RAG_ADMIN_EMAIL` and `IMPERIAL_RAG_ADMIN_PASSWORD`: bootstrap the first approved Streamlit admin account for granting chat access.
- `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY`, and `COHERE_API_KEY`: legacy debugging compatibility only when `IMPERIAL_RAG_ALLOW_LEGACY_OPENAI` or `IMPERIAL_RAG_ALLOW_LEGACY_COHERE` is enabled.

Retrieval and chunking tuning variables are also listed in `.env.example`, including chunk size, overlap, vector fetch limits, keyword limits, reranker choices, and final evidence limits.

## Privacy And Local State

Treat these paths as private:

- `documents/`
- `.imperial_rag/`
- `.imperial_rag/auth.sqlite3`
- Elasticsearch Docker volume `imperial_elasticsearch_data`
- local Qdrant storage
- Phoenix traces and experiment data
- `.env` files containing real secrets

Do not commit API keys, generated corpus artifacts, local indexes, OCR cache data, or private traces.

## Troubleshooting

If query answers always refuse or lack useful evidence, run ingestion first, confirm `.imperial_rag/extracted/chunks.jsonl` exists, and make sure local Elasticsearch is running.

If vector indexing fails, start Qdrant with `./scripts/start_qdrant.sh` before running ingestion with `--index-vectors`.

If Phoenix experiment mode fails, start Phoenix with `docker compose up phoenix` and confirm `http://localhost:6006` is reachable.

If semantic search, embeddings, answer generation, OCR, or reranking fail under the defaults, confirm `DASHSCOPE_API_KEY` is present in your local environment.

If live Qdrant tests fail during normal unit testing, make sure `IMPERIAL_RAG_LIVE_QDRANT` is unset or set to `0`.
