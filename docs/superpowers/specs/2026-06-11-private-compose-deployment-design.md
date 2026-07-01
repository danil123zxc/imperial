# Private Compose Deployment Design

## Purpose

Prepare Imperial RAG for a private single-machine deployment where Docker Compose owns the Streamlit app, Qdrant, Phoenix, and an explicit one-shot ingestion job. The design keeps the current local RAG architecture intact: Streamlit calls the Python runtime directly, SQLite remains file-backed under `.imperial_rag/`, Qdrant stores vectors, and Phoenix stores traces and evals.

## Decisions

- Deployment target: one private machine.
- Access model: all published ports bind to `127.0.0.1`.
- Runtime services: `app`, `qdrant`, and `phoenix`.
- Ingestion model: explicit `ingest` profile, not automatic startup ingestion.
- Data ownership: host-mounted `documents/` and `.imperial_rag/`.
- SQL direction: keep SQLite files; do not add Postgres in v1.
- API direction: do not add a FastAPI/Flask backend in v1.

## Service Architecture

### app

The `app` service will build from the repository using a new Python/uv Dockerfile. It runs:

```bash
uv run python -m streamlit run src/imperial_rag/web_app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
```

The host port should publish as `127.0.0.1:8501:8501`. Inside Compose, the app uses service DNS for dependencies:

- `QDRANT_URL=http://qdrant:6333`
- `PHOENIX_CLIENT_ENDPOINT=http://phoenix:6006`
- `PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006/v1/traces`

The app image contains code and dependencies only. It must not bake private documents, generated indexes, local traces, or real secrets into the image.

### qdrant

The `qdrant` service remains the vector database. It publishes `127.0.0.1:6333:6333` for host access and persists storage at:

```text
./.imperial_rag/qdrant_storage:/qdrant/storage
```

The deployment keeps the current collection default `imperial_chunks_qwen`.

### phoenix

The `phoenix` service remains the self-hosted tracing and evaluation store. It publishes:

- `127.0.0.1:6006:6006` for the Phoenix UI/API
- `127.0.0.1:4317:4317` for OTLP gRPC when needed

Phoenix keeps `PHOENIX_WORKING_DIR=/mnt/data` and persists data in the existing named volume.

### ingest

The `ingest` service uses the same image and environment as `app`, but runs as a one-shot job through a Compose profile:

```bash
docker compose --profile ingest up ingest
```

The command should run:

```bash
uv run python scripts/ingest.py --workspace-root /app --index-vectors
```

This job reads mounted documents, rebuilds generated SQLite/extracted state under `.imperial_rag/`, and writes vectors to Qdrant. It should depend on Qdrant. Phoenix tracing can stay optional and env-controlled.

## Data Flow

The server filesystem remains the source of truth for private data:

- `./documents` mounts read-only at `/app/documents`.
- `./.imperial_rag` mounts read-write at `/app/.imperial_rag`.
- `.env` supplies secrets and runtime settings, especially `DASHSCOPE_API_KEY`.

Normal startup is:

```bash
docker compose up -d app qdrant elasticsearch phoenix
```

The app loads environment values, reads manifest/OCR SQLite and generated artifacts from `/app/.imperial_rag`, uses Elasticsearch through `http://elasticsearch:9200` for keyword search, uses Qdrant through `http://qdrant:6333`, and sends Phoenix traces only when tracing flags are enabled.

Reindexing is explicit:

```bash
docker compose --profile ingest up ingest
```

This avoids slow or costly corpus processing during ordinary app restarts.

## Reliability And Health Checks

Compose should include cheap service checks where practical:

- `app`: `http://127.0.0.1:8501/_stcore/health` from inside the container.
- `qdrant`: `http://127.0.0.1:6333/healthz` from inside the container.
- `elasticsearch`: `http://127.0.0.1:9200/` from inside the container.
- `phoenix`: simple HTTP check against port `6006`.

The app should wait for Qdrant and Elasticsearch health when Compose can express that cleanly, because Qdrant is required for vector search and Elasticsearch is required for keyword search. Phoenix should start with the normal stack, but app startup must not be gated on Phoenix health; tracing remains optional, and Phoenix downtime should not prevent ordinary querying when tracing is disabled. If runtime provider checks disable semantic search, keyword-only fallback still depends on Elasticsearch.

## Configuration

`.env.example` should stay useful for both host-local commands and Compose. The implementation should make the Compose-specific defaults explicit without hiding the current host defaults:

- Host-local default: `QDRANT_URL=http://localhost:6333`
- Compose default: `QDRANT_URL=http://qdrant:6333`
- Host-local default: `ELASTICSEARCH_URL=http://localhost:9200`
- Compose default: `ELASTICSEARCH_URL=http://elasticsearch:9200`
- Host-local default: `PHOENIX_CLIENT_ENDPOINT=http://localhost:6006`
- Compose default: `PHOENIX_CLIENT_ENDPOINT=http://phoenix:6006`
- Host-local default: `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces`
- Compose default: `PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006/v1/traces`

Secrets stay in `.env` or the process environment and must not be copied into Docker images.

## Build Context

Add a `.dockerignore` so image builds exclude private and generated data:

- `.env` and local env variants
- `documents/`
- `.imperial_rag/`
- Python caches and test caches
- git metadata and local editor files
- eval outputs or traces if present

The Dockerfile should install project dependencies reproducibly from `pyproject.toml` and `uv.lock`, then run the app from the repository package code.

## Verification

The implementation should be verified with:

```bash
docker compose config
docker compose --profile ingest config
docker compose build app
docker compose up -d qdrant phoenix app
curl -fsS http://127.0.0.1:8501/_stcore/health
curl -fsS http://127.0.0.1:6333/healthz
curl -I --max-time 3 http://127.0.0.1:6006/
```

If build time and local Docker state allow it, also run focused Python tests for config/env and Streamlit behavior.

## Non-Goals

- No public internet exposure in v1.
- No reverse proxy, TLS, or auth layer in v1.
- No Postgres migration in v1.
- No backend API split in v1.
- No automatic ingestion on app startup.
- No Kubernetes or multi-node orchestration.
