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
- Produces strict citation-based answers through a CLI, Streamlit, and a private Telegram webhook.
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
  -> scripts/query.py, Streamlit, or Russian Telegram job worker
  -> answer text and source labels through Render to Telegram
  -> optional Phoenix traces and eval experiments
```

Core code lives in `src/imperial_rag/`:

- `ingestion/`: file scanning, extraction, OCR, manifests, chunking, and corpus ingestion.
- `indexing/`: Qdrant vector indexing helpers and stable chunk identifiers.
- `retrieval/`: Elasticsearch keyword search, vector/keyword fusion, and reranking.
- `answering/`: query runtime, LangGraph workflows, and strict answer formatting.
- `integrations/`: DashScope/Qwen provider adapters.
- `observability/`: structured logs, event logs, Phoenix tracing, and privacy controls.
- `app/`: Streamlit UI/auth, the Render Telegram adapter, the Russian job API/worker, and local chat history.

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

Fill in `DASHSCOPE_API_KEY` and the Streamlit credentials. Telegram additionally needs the six variables
listed under [Render Telegram deployment](#render-telegram-deployment). The allowlist is a comma-separated
set of trusted numeric Telegram user IDs. Both HTTP boundaries reject groups and users outside that list.

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

Run the website:

```bash
uv run python -m streamlit run src/imperial_rag/app/web.py --server.address 127.0.0.1 --server.port 8501
```

Run the Russian Telegram API and sequential RAG worker:

```bash
uv run uvicorn imperial_rag.app.telegram_backend:app --host 127.0.0.1 --port 8502
```

The backend persists Telegram jobs and chat history in `.imperial_rag/chat_history.sqlite3`. It sends only
answer text and textual source labels to Render; source files, retrieved chunks, documents, indexes, and
Phoenix traces remain on the Russian host.

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

The private Compose stack runs Streamlit, the Telegram job API/worker, Elasticsearch, Kibana, Qdrant, and
Phoenix on the Russian host. Every published port remains bound to `127.0.0.1`. Render handles Telegram
webhooks and response delivery; it never mounts the private corpus or generated state.

Elasticsearch, Kibana, and Phoenix in this stack are unauthenticated by default and are safe only while bound to `127.0.0.1` on a trusted host. Do not bind them to `0.0.0.0`, expose them through a public proxy, or share broad tunnels unless authentication and TLS are added.

Prepare the checkout:

```bash
cp .env.example .env
mkdir -p documents .imperial_rag/qdrant_storage
```

Fill `.env` with `DASHSCOPE_API_KEY`, Streamlit credentials, `IMPERIAL_RAG_TELEGRAM_PHONE_HASH_SECRET`,
and `IMPERIAL_RAG_TELEGRAM_SERVICE_TOKEN`, plus any model or tracing settings needed on that machine.
Both secrets must be 32–256 URL-safe letters, digits, underscores, or hyphens; the service token must match
Render, while the phone-hash secret stays only on the Russian host. Numeric IDs in
`IMPERIAL_RAG_TELEGRAM_ALLOWED_USER_IDS` remain an optional bootstrap fallback. Host-local commands can keep
the `localhost` defaults from `.env.example`; `compose.yaml` overrides service endpoints inside containers.

Start the runtime stack:

```bash
docker compose up -d elasticsearch qdrant phoenix app telegram-api kibana
```

Verify local endpoints:

```bash
curl -fsS http://127.0.0.1:8501/_stcore/health
curl -fsS -H "Authorization: Bearer $IMPERIAL_RAG_TELEGRAM_SERVICE_TOKEN" http://127.0.0.1:8502/healthz
curl -fsS http://127.0.0.1:9200
curl -fsS http://127.0.0.1:5601/api/status
curl -fsS http://127.0.0.1:6333/healthz
curl -I --max-time 3 http://127.0.0.1:6006/
```

Run ingestion inside Compose when documents change:

```bash
docker compose --profile ingest up ingest
```

### Russian HTTPS boundary

Expose only `/internal/telegram/` from the existing authenticated HTTPS reverse proxy to
`http://127.0.0.1:8502`. Keep `/healthz` local and do not proxy Qdrant, Elasticsearch, Kibana, Phoenix,
SQLite, `documents/`, or `.imperial_rag/`. For Nginx, the application location is:

```nginx
location /internal/telegram/ {
    proxy_pass http://127.0.0.1:8502;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto https;
}
```

The API independently requires `Authorization: Bearer <IMPERIAL_RAG_TELEGRAM_SERVICE_TOKEN>` on every
endpoint and enforces JSON content type, a 16 KiB body limit, positive IDs, and the Russian access database.
The protected Streamlit admin panel can pre-authorize `@username` or an international `+phone`, list pending
or bound grants, and revoke them. Phone numbers are stored only as a keyed digest and masked label.

### Render Telegram deployment

`render.yaml` defines one paid `standard` Docker Web Service in Frankfurt, binds Uvicorn to
`0.0.0.0:$PORT`, checks `/healthz`, and disables automatic deploys. Enter these values in Render; the
Blueprint deliberately contains no secret values:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_WEBHOOK_URL` (the exact Render HTTPS URL ending in `/telegram/webhook`)
- `TELEGRAM_WEBHOOK_SECRET` (32–256 URL-safe letters, digits, underscores, or hyphens)
- `IMPERIAL_RAG_TELEGRAM_BACKEND_URL` (the Russian HTTPS origin, without the internal path)
- `IMPERIAL_RAG_TELEGRAM_SERVICE_TOKEN` (the same strong value as on the Russian host)

At startup, Render registers the webhook with `allowed_updates=["message"]`, `drop_pending_updates=false`,
and the secret-token header. For each private message, Render first sends only the Telegram user ID,
username, and any explicitly shared contact to the authenticated Russian access endpoint. Question text is
submitted only after authorization succeeds. A matching username binds automatically; a phone grant binds
only when Telegram marks the shared contact as belonging to the sender. Accepted questions are idempotently
stored in Russia before Render returns `200`. The Russian worker processes jobs sequentially; Render leases
completed responses every two seconds and acknowledges delivery after Telegram accepts all message chunks.
Expired processing and delivery leases recover after restarts. Delivery is at least once, so a reply can
rarely be duplicated if Telegram accepts it but the completion acknowledgement fails.

Render also registers the private-chat command menu in Russian. `/start` welcomes the user, `/help` lists
the available commands, and `/new` creates a fresh conversation on the Russian backend without querying the
RAG runtime. Unknown slash commands show the help text; ordinary text continues through the durable job flow.

Questions, answers, and source labels transit Render and Telegram. Source documents and retrieved evidence
do not. Deploy the Russian API and HTTPS route first, then deploy Render. Production deployment and webhook
activation require explicit operator authorization.

### Automatic application deployment

A successful GitHub Actions `Quality` job for a push to protected `main` deploys that exact commit to the
production host over Tailscale and command-restricted SSH. The deployment builds and replaces Streamlit and
the Telegram API as one rollback unit:

```bash
docker compose build app telegram-api
docker compose up -d --no-deps app telegram-api
```

The deploy command waits for both container health checks and
`http://127.0.0.1:8501/_stcore/health`. A failed build leaves both existing containers running. A failed
startup or health check restores both services to the previously healthy commit and reports a failed GitHub
deployment.

Automatic deployment does not deploy Render, register the webhook, run ingestion, restart Qdrant,
Elasticsearch, Kibana, or Phoenix, or modify `.env`, `documents/`, `.imperial_rag/`, or persistent volumes.
Apply those operator actions explicitly.

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
docker compose logs -f telegram-api
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
uv run python scripts/run_all_evals.py --snapshot .imperial_rag/evidence-eval/snapshot.json --annotations .imperial_rag/evidence-eval/annotations.jsonl
```

## Services And State

| Surface | Default | Purpose |
| --- | --- | --- |
| Streamlit website | `http://127.0.0.1:8501` | Existing private website |
| Telegram webhook | Render HTTPS `/telegram/webhook` | Stateless validation, submission, and Telegram delivery |
| Telegram job API | local `http://127.0.0.1:8502`; proxied only at `/internal/telegram/` | Durable jobs and sequential RAG work |
| Elasticsearch | `http://localhost:9200` | Keyword search index `imperial_keyword_chunks` |
| Kibana | `http://127.0.0.1:5601` | Local inspection of Elasticsearch data |
| Qdrant | `http://localhost:6333` | Optional vector collection `imperial_chunks_qwen` |
| Phoenix | `http://localhost:6006` | Optional traces and eval experiments |
| `.imperial_rag/manifest.sqlite3` | local file | Corpus manifest and per-file status |
| `.imperial_rag/extracted/` | local directory | Extracted text, chunks, ledger, and lineage |
| `.imperial_rag/shadow-runs/<id>/` | local directory | Isolated candidate artifacts, manifest, OCR cache, and run descriptor |
| `.imperial_rag/active-ingestion.json` | local file | Atomically replaced pointer to the promoted artifacts and search aliases |
| `.imperial_rag/auth.sqlite3` | local file | Streamlit users, browser sessions, and Telegram access grants |
| `.imperial_rag/chat_history.sqlite3` | local file | Local chat history |
| `telegram_jobs` | table in chat-history SQLite | Durable Telegram job, result, attempt, and lease state |
| `telegram_access_grants` | table in auth SQLite | Username/phone grants, masked labels, and bound Telegram IDs |

Use the live files, database tables, and service health checks as source of truth for generated state. Snapshot counts in documentation drift quickly after corpus rebuilds.

Document authority overrides live in `docs/document-authority.json`. Each optional row is keyed by `relative_path` and may define `department`, `document_type`, `status` (`active`, `draft`, or `archived`), effective dates, `owner`, `authoritative_rank`, `supersedes`, and `version_group`. Exact-file duplicates are indexed once; every original path remains in the canonical chunk's `provenance_paths` metadata.

## Configuration

Important settings are documented in `.env.example`.

| Variable | Notes |
| --- | --- |
| `DASHSCOPE_API_KEY` | Required for Qwen chat, embeddings, OCR, reranking, and Phoenix/Ragas model-backed metrics |
| `IMPERIAL_RAG_WORKSPACE_ROOT` | Workspace root; defaults to this checkout in host runs and `/app` in Compose |
| `TELEGRAM_BOT_TOKEN` | Required Telegram Bot API token; keep only in local/server environment configuration |
| `IMPERIAL_RAG_TELEGRAM_ALLOWED_USER_IDS` | Optional comma-separated bootstrap allowlist of trusted numeric Telegram user IDs |
| `IMPERIAL_RAG_TELEGRAM_PHONE_HASH_SECRET` | Required Russia-only HMAC secret for phone-number grants |
| `TELEGRAM_WEBHOOK_URL` / `TELEGRAM_WEBHOOK_SECRET` | Exact Render webhook URL and strong Telegram secret-token value |
| `IMPERIAL_RAG_TELEGRAM_BACKEND_URL` / `IMPERIAL_RAG_TELEGRAM_SERVICE_TOKEN` | Russian HTTPS origin and shared bearer secret |
| `IMPERIAL_RAG_ADMIN_EMAIL` / `IMPERIAL_RAG_ADMIN_PASSWORD` | Streamlit admin access |
| `ELASTICSEARCH_URL` / `ELASTICSEARCH_INDEX` | Keyword search endpoint and index |
| `QDRANT_URL` / `QDRANT_COLLECTION` | Optional vector search endpoint and collection |
| `PHOENIX_CLIENT_ENDPOINT` / `PHOENIX_COLLECTOR_ENDPOINT` | Phoenix UI/client endpoint and OTLP trace collector |
| `PHOENIX_TRACING_ENABLED` / `IMPERIAL_RAG_TRACING_ENABLED` | Enable tracing without passing `--trace-phoenix` |
| `IMPERIAL_RAG_TRACE_*` | Trace run IDs, privacy/detail controls, and retrieval-debug options |
| `OPENINFERENCE_HIDE_*` | OpenInference redaction controls for prompts, outputs, images, and text |
| `IMPERIAL_RAG_LOG_*` | Local structured log level, format, service name, and environment |
| `IMPERIAL_RAG_EVENTLOG_*` | Optional local Elasticsearch event-log settings |
| `IMPERIAL_RAG_CHUNK_*`, `IMPERIAL_RAG_VECTOR_*`, `IMPERIAL_RAG_KEYWORD_LIMIT`, `IMPERIAL_RAG_RERANK_*` | Retrieval, chunking, and reranking tuning |

The default runtime uses DashScope/Qwen. Direct `build_query_workflow` callers must supply `chat_model` or `generate` to generate an answer; empty evidence still returns a refusal without a model.

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

Run the full configured eval suite with a frozen snapshot and reviewed evidence annotations:

```bash
uv run python scripts/run_all_evals.py --snapshot .imperial_rag/evidence-eval/snapshot.json --annotations .imperial_rag/evidence-eval/annotations.jsonl
```

Both evidence inputs are required; corrected questions alone do not enable a run.
See [the evidence evaluation guide](docs/evidence-evaluation.md) to prepare them.
This runner scores source evidence at k=1/3/5/10 and budgets 1000/2000/4000, using
ranked retrieval before answer packing. ID-based checks appear as `legacy_*` diagnostics.
Phoenix mode judges each of the first `--retrieval-k` ranked chunks with Phoenix's
`DocumentRelevanceEvaluator` and the configured Qwen model (default cutoff: 5).
The experiment output's `retrieval_evaluation` contains status and trace/span IDs.
Open the `evaluation.retrieval_relevance` span to inspect document labels,
explanations and Phoenix-native nDCG, MRR, precision and hit rate. No ranking
arithmetic is implemented locally, and MAP is no longer emitted. These scores
are not comparable to historical complete-evidence ranking scores.
Full evidence evaluation still validates the active corpus against the frozen
snapshot and reports evidence recall/completeness separately; local chunk
comparisons retain only those diagnostics. See the
[metric and trace contract](docs/evidence-evaluation.md#phoenix-llm-retrieval-ranking).
The separate `chunk_recall` evaluator and its chunk hit/precision metrics are removed;
`id_recall` and optional Ragas `id_context_recall` remain available.
Invalid source mappings or degraded retrieval fail the run; existing indexes may need
rebuilding to carry valid `source_spans`. `--ragas-metrics none` disables Ragas judges,
but Phoenix retrieval judging and the query answer model still run when applicable.

Run deterministic citation/refusal/source-hint checks:

```bash
uv run python scripts/run_phoenix_eval.py
```

Store a Phoenix experiment with retrieval judging and no Ragas answer judges:

```bash
uv run python scripts/run_phoenix_eval.py --use-phoenix --ragas-metrics none
```

Run standalone Ragas checks:

```bash
uv run python scripts/run_ragas_eval.py
```

Ragas metrics need the dev dependencies and model credentials configured in `.env`.

### Chunk-independent evidence comparison

`scripts/compare_chunking.py` compares isolated indexes against one frozen extraction
snapshot and reviewed source-span annotations. The existing gold questions and ID
metrics are unchanged. See [the evidence evaluation guide](docs/evidence-evaluation.md)
for the annotation contract, metrics, and provider boundaries.

```bash
uv run python scripts/compare_chunking.py freeze --output .imperial_rag/evidence-eval/snapshot.json
uv run python scripts/generate_eval_evidence_packets.py --snapshot .imperial_rag/evidence-eval/snapshot.json --output-path .imperial_rag/evidence-eval/annotations.jsonl
# Review annotations against the snapshot before validation or running providers.
uv run python scripts/compare_chunking.py validate --snapshot .imperial_rag/evidence-eval/snapshot.json --annotations .imperial_rag/evidence-eval/annotations.jsonl
uv run python scripts/compare_chunking.py run --snapshot .imperial_rag/evidence-eval/snapshot.json --annotations .imperial_rag/evidence-eval/annotations.jsonl --output .imperial_rag/evidence-eval/runs/first
uv run python scripts/compare_chunking.py answers --run .imperial_rag/evidence-eval/runs/first
uv run python scripts/compare_chunking.py phoenix --run .imperial_rag/evidence-eval/runs/first
```

`freeze` and annotation preparation are local. `run` calls embedding/query/reranking
providers and writes fresh shadow indexes; `answers` separately calls the answer
model. `phoenix` publishes saved results without repeating retrieval. No command
promotes indexes or changes active aliases. Keep every generated artifact private
under `.imperial_rag/`; use a new output directory for each run.

Application context packing is opt-in with `IMPERIAL_RAG_CONTEXT_TOKEN_BUDGET`.
It counts the rendered evidence using the fixed `cl100k_base` proxy tokenizer,
including source labels and separators; these are not exact Qwen billing tokens.
Unset the variable to retain existing application behavior.

### Phoenix datasets as experiment input

Maintain questions, reference answers and reviewed evidence in an **existing**
Phoenix dataset instead of local questions/annotation JSONL. `snapshot.json` stays
local and frozen: it is still the source for rechunking and exact evidence-span
validation. No dataset is created, uploaded, overwritten or marked reviewed when
Phoenix is the input source.

All four eval entrypoints accept `--phoenix-dataset-name NAME` or
`--phoenix-dataset-id ID`, plus optional `--phoenix-dataset-version-id VERSION_ID`.
Omitting the version resolves latest once per command; the returned immutable
version is retained for every configuration. Use an explicit version to bind
separate validation and execution commands to the same reviewed dataset.
`--dataset-name` retains its existing **upload destination** meaning and conflicts
with Phoenix input. Explicit local questions, local annotations, and a second
Phoenix selector also conflict; a version flag requires a Phoenix selector.
Without a Phoenix selector, existing local defaults and commands above still work.

Each Phoenix example uses these objects (the Phoenix example ID is separate from
the stable Imperial question ID in metadata):

| Object | Fields |
| --- | --- |
| `input` | `question`: nonempty string |
| `output` | `reference_answer`, `expected_behavior`, `lane`; optional `expected_source_hints`, `reference_context_ids`, `quarantine_reason`, `evidence` |
| `metadata` | `id`, `suite`; optional string-list `tags`; evidence runs also require `split`, `review_status`, `question_hash`, `snapshot_hash` |

The existing question contract applies: `expected_behavior` is `cite_answer`,
`surface_conflict` or `refuse_if_not_found`, and `lane` must match that behavior.
For compatibility, `lane` and `quarantine_reason` may also live in metadata;
duplicate values must agree. Optional fields remain absent when omitted.
`expected_source_hints` and legacy `reference_context_ids` are lists of strings.
Question IDs must be unique, nonempty strings; suite and reference answer are required.

Evidence has the existing shape:

```text
evidence: [
  {evidence_id: "claim-1", support_sets: [
    [{source_id, text_sha256, start, end, quote}],
    [{source_id, text_sha256, start, end, quote}]
  ]}
]
```

Each support set requires all its spans; any complete support set supports the
unit. Offsets are Python Unicode character indices with exclusive ends. Exact
source identity, text hash, bounds and quote must match the frozen snapshot.
Answerable questions require evidence, conflict questions need at least two units,
and refusals require an explicit empty evidence list. `split` is `dev` or `test`;
`review_status` must explicitly be `reviewed`. The importer never fills in review
approval or repairs hashes. All rows are validated before selecting a comparison split.

`question_hash` is `imperial_rag.ingestion.provenance.digest(mapped_question)`:
combine `input.question`, the listed question fields from output (excluding
`evidence`), and `metadata.id/suite/tags/lane/quarantine_reason` when present.
Review fields and Phoenix example IDs are excluded. Do not insert absent optional
fields before hashing. `snapshot_hash` is the verified snapshot's `snapshot_hash`.
After editing a question or its reference fields, re-review its evidence and
update the hash in Phoenix. This read-only helper prints current question hashes;
it does not approve annotations:

```bash
uv run python - <<'PY'
import asyncio
from phoenix.client import AsyncClient
from imperial_rag.config import Settings
from imperial_rag.env import load_project_env
from imperial_rag.evals.dataset_input import map_phoenix_examples
from imperial_rag.ingestion.provenance import digest
async def main():
    load_project_env()
    dataset = await AsyncClient(base_url=Settings().phoenix_client_endpoint).datasets.get_dataset(dataset="imperial-reviewed-questions")
    print("dataset_id=", dataset.id, "version_id=", dataset.version_id)
    for question in map_phoenix_examples(dataset)[0]:
        print(question["id"], digest(question))
asyncio.run(main())
PY
```

Basic evaluation accepts datasets without evidence annotations. It uses the
question/reference contract; source-evidence validation and metrics belong to
`run_all_evals.py` and `compare_chunking.py validate/run`:

```bash
# Basic evaluation; --use-phoenix stores an experiment on the original dataset.
uv run python scripts/run_phoenix_eval.py --phoenix-dataset-name imperial-reviewed-questions --use-phoenix --ragas-metrics none
uv run python scripts/run_ragas_eval.py --phoenix-dataset-name imperial-reviewed-questions --output-path .imperial_rag/ragas-phoenix.jsonl

# Use the dataset/version IDs reported by the helper or shown in Phoenix.
PHOENIX_INPUT_ID='replace-with-dataset-id'
PHOENIX_INPUT_VERSION='replace-with-version-id'
uv run python scripts/compare_chunking.py validate --snapshot .imperial_rag/evidence-eval/snapshot.json --phoenix-dataset-id "$PHOENIX_INPUT_ID" --phoenix-dataset-version-id "$PHOENIX_INPUT_VERSION"
uv run python scripts/run_all_evals.py --snapshot .imperial_rag/evidence-eval/snapshot.json --phoenix-dataset-id "$PHOENIX_INPUT_ID" --phoenix-dataset-version-id "$PHOENIX_INPUT_VERSION" --ragas-metrics none
uv run python scripts/compare_chunking.py run --snapshot .imperial_rag/evidence-eval/snapshot.json --phoenix-dataset-id "$PHOENIX_INPUT_ID" --phoenix-dataset-version-id "$PHOENIX_INPUT_VERSION" --configs 400:50,256:0 --output .imperial_rag/evidence-eval/runs/phoenix-first
uv run python scripts/compare_chunking.py answers --run .imperial_rag/evidence-eval/runs/phoenix-first
uv run python scripts/compare_chunking.py phoenix --run .imperial_rag/evidence-eval/runs/phoenix-first
```

Phoenix-backed `validate` only reads Phoenix and the local snapshot. Other eval/run
commands retain their provider requirements: disabling Ragas judges does not
disable answer generation or retrieval providers. Comparison indexes remain
isolated; active indexes and aliases are unchanged.

The `phoenix_dataset` binding records endpoint, dataset ID/name, version ID and an
example-content hash in comparison benchmarks/manifests/results, answer artifacts,
Ragas records and Phoenix experiment metadata; CLI output also reports the binding.
Comparison `answers` uses saved retrieval and retains the binding without reading
latest. `phoenix` fetches the recorded version, verifies its content and endpoint,
and replays the selected split on the original dataset with original example IDs.
Deletion/unavailability or a mismatch fails publication rather than re-uploading.
Local-file comparison publication retains its existing upload-once/pinned replay.
All snapshots, bindings and outputs remain private under `.imperial_rag/`.

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
compose.yaml               Streamlit, Telegram backend, Elasticsearch, Kibana, Qdrant, and Phoenix stack
render.yaml                Stateless Render Telegram Web Service Blueprint
Dockerfile                 Compose app image
pyproject.toml             Python package, dependency, and tool configuration
```

## Troubleshooting

If answers refuse or return no useful evidence, confirm ingestion has run, `.imperial_rag/extracted/chunks.jsonl` exists, and Elasticsearch is reachable at `ELASTICSEARCH_URL`.

If vector search is unavailable, start Qdrant and rerun ingestion with `--index-vectors`. If the vector provider metadata no longer matches the configured embedding model or dimensions, recreate the collection.

If model-backed chat, OCR, embeddings, reranking, or Ragas metrics fail, confirm `DASHSCOPE_API_KEY` is present in `.env` or the process environment.

If the Telegram backend is unhealthy, confirm `IMPERIAL_RAG_TELEGRAM_PHONE_HASH_SECRET` and
`IMPERIAL_RAG_TELEGRAM_SERVICE_TOKEN`, and ensure the optional numeric allowlist contains only integers.
Then inspect `docker compose logs --tail=200 telegram-api`. An authenticated `200` from local port `8502`
proves that the API and worker initialized; it does not prove that Qwen, Elasticsearch, or Qdrant will
answer a new question. On Render, verify `/healthz` and Telegram `getWebhookInfo`, including the exact URL
and an empty `last_error_message`.

If Phoenix validation fails, start Phoenix, run a fresh traced query with a stable `IMPERIAL_RAG_TRACE_RUN_ID`, then validate that run ID.

If Compose services are unreachable, check that their ports remain bound to `127.0.0.1`, inspect `docker compose ps`, and read `docker compose logs -f <service>`.
