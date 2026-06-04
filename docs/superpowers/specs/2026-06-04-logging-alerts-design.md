# Logging And Alerts Design

Date: 2026-06-04
Workspace: `/Users/danil/Public/imperial`
Related system: local Imperial RAG application

## Status

Design approved in chat and written for user review. Implementation planning has not started.

## Context

The Imperial RAG repo already has a local observability backend for traces and evals:

- Phoenix runs from `compose.yaml`;
- `src/imperial_rag/tracing.py` configures Phoenix OpenTelemetry tracing;
- `scripts/run_phoenix_eval.py` stores deterministic eval datasets and experiments in Phoenix when requested;
- README and `.env.example` document Phoenix trace/eval configuration.

The repo does not yet have one application logging layer or an alerting sink. CLI scripts currently print concise summaries, and runtime failures are mostly surfaced through normal Python exceptions. The user wants logging plus alerts, chose Sentry for alerts, selected failure-only alerting, and confirmed that Sentry must not receive raw questions, answers, document text, or file paths by default.

Sentry Python SDK documentation consulted through Context7 confirms:

- `sentry_sdk.init(...)` enables the SDK with a DSN;
- `capture_exception` captures exceptions;
- `capture_message` captures manual message events;
- `before_send` can modify or drop events before transmission;
- logging integration exists, but v1 should not promote broad logging records into Sentry events.

## Goals

Add a small logging and alerting layer that:

- keeps Phoenix as the trace and evaluation system;
- uses Sentry only for failure alerts;
- emits local structured JSON logs for run completion and failure diagnostics;
- captures failures in ingestion, query, eval, and Streamlit query handling;
- alerts when ingestion completes with one or more failed files;
- preserves existing CLI behavior and exit semantics;
- avoids sending private corpus or prompt data to Sentry.

## Non-Goals

- No Sentry performance tracing in v1; Phoenix remains the tracing system.
- No broad `ERROR` log forwarding to Sentry.
- No Sentry cron monitors or scheduled check-ins in v1.
- No external log aggregation backend.
- No alert rules managed from code; Sentry issue alert rules are configured in Sentry itself.
- No logging of raw corpus text, questions, answers, retrieved documents, citations, or file paths.

## Recommended Approach

Use failure-only Sentry alerts plus local JSON logs.

The application configures local logging for every supported entrypoint. Sentry initializes only when alerting is enabled and a DSN is present. CLI and UI boundaries explicitly capture failed operations, attach sanitized metadata, and then preserve the current failure behavior.

This avoids noisy alert streams and keeps the local/private nature of the project intact. Sentry becomes a sparse incident signal; Phoenix remains the detailed trace/eval tool; local JSON logs remain the first debugging trail.

## Configuration

Extend `Settings` with:

- `sentry_dsn`: from `SENTRY_DSN`, default empty;
- `sentry_environment`: from `SENTRY_ENVIRONMENT`, default `local`;
- `sentry_release`: from `SENTRY_RELEASE`, default empty;
- `sentry_enabled`: from `IMPERIAL_RAG_SENTRY_ENABLED`, default true only when `SENTRY_DSN` is present;
- `log_level`: from `IMPERIAL_RAG_LOG_LEVEL`, default `INFO`;
- `log_format`: from `IMPERIAL_RAG_LOG_FORMAT`, default `json`.

Update `.env.example` and README with these variables. The DSN remains secret and must stay local.

## Architecture

Add one observability module:

- `src/imperial_rag/observability.py`
  - configures local logging once per process;
  - initializes Sentry once per process;
  - exposes `configure_observability(settings)`;
  - exposes `log_event(event_name, **fields)` for local structured events;
  - exposes `capture_failure(operation, exc=None, **fields)` for local failure logs plus Sentry failure alerts;
  - sanitizes all metadata before it reaches Sentry.

Existing Phoenix tracing stays in `src/imperial_rag/tracing.py`. The new observability module may live beside it for now; a future module-structure cleanup can move both under `imperial_rag/observability/` if that older readability plan is implemented.

## Entrypoints

Update these entrypoints to call `configure_observability(settings)` after settings are built:

- `scripts/ingest.py`;
- `scripts/query.py`;
- `scripts/run_phoenix_eval.py`;
- `src/imperial_rag/web_app.py`.

Each CLI script wraps its core operation:

1. record start time and operation name;
2. run existing behavior;
3. on success, emit one local completion event;
4. on failure, capture a failure alert and re-raise or exit with the same semantics as today.

`scripts/ingest.py` also checks the returned summary. If `failed_files > 0`, it emits a local failure event and sends a Sentry alert with safe counts, even if the command otherwise completes.

The Streamlit app captures query/runtime exceptions and shows a user-safe error. It must not send chat text, retrieved evidence, answers, citations, source names, or file paths to Sentry.

## Sentry Event Policy

Sentry captures:

- unhandled exceptions in ingestion, query, eval, and Streamlit query handling;
- explicit failed-run alerts such as ingestion with `failed_files > 0`;
- operation name, component, status, exception type, duration, and safe counts;
- environment and release when configured.

Sentry does not capture:

- raw user questions;
- generated answers;
- retrieved document text;
- source document text;
- citations or sources;
- absolute or relative corpus paths;
- filenames;
- API keys, DSNs, provider request payloads, authorization headers, tokens, or Phoenix trace data.

The sanitizer recursively drops sensitive keys including:

- `question`;
- `answer`;
- `page_content`;
- `documents`;
- `sources`;
- `citations`;
- `path`;
- `file_path`;
- `absolute_path`;
- `relative_path`;
- `file_name`;
- `filename`;
- `api_key`;
- `dsn`;
- `authorization`;
- `token`;
- `secret`.

It preserves safe scalar fields such as:

- `operation`;
- `component`;
- `status`;
- `duration_ms`;
- `total_files`;
- `failed_files`;
- `chunk_count`;
- `keyword_indexed`;
- `vector_indexed`;
- `final_evidence`;
- `vector_candidates`;
- `keyword_candidates`;
- `reranker`;
- `fallback_count`.

## Local Log Shape

Local logs are newline-delimited JSON by default. Each event includes:

- `timestamp`;
- `level`;
- `event`;
- `operation`;
- `status`;
- `duration_ms` when available;
- `component`;
- `workspace_root` only if it is not a corpus/document path;
- safe counts and booleans relevant to the operation.

Local logs may include slightly more operational context than Sentry, but should still avoid raw document text, questions, answers, and secrets. If text content is ever needed for debugging, it should be inspected from local files or Phoenix traces intentionally, not emitted automatically.

## Data Flow

CLI flow:

1. entrypoint builds `Settings`;
2. entrypoint configures local logging and optional Sentry;
3. Phoenix tracing remains configured only through existing tracing flags/env;
4. operation runs normally;
5. completion emits one local JSON event;
6. failures emit one local JSON event and one sanitized Sentry event when enabled;
7. current terminal output and exit behavior are preserved.

Streamlit flow:

1. app startup builds `Settings`;
2. app configures local logging and optional Sentry;
3. Phoenix tracing remains opt-in through existing tracing env;
4. each query gets a generated local operation id;
5. success logs status and safe counts;
6. failure captures sanitized Sentry context and displays a user-safe error.

## Error Handling

Sentry setup should never break a local run. If `sentry-sdk` is missing, the DSN is invalid, or SDK initialization fails, the app writes a local warning and continues without Sentry alerts.

Explicit operation failures should continue to behave as they do today. The alerting layer observes and reports failures; it should not convert successful commands into nonzero exits except where the existing code already does so.

For ingestion summaries with `failed_files > 0`, v1 sends a failure alert but keeps the existing command completion behavior.

## Testing

Add focused tests for:

- settings defaults and environment overrides for Sentry/logging variables;
- Sentry disabled when no DSN is configured;
- Sentry initializes once when enabled with a DSN;
- Sentry initialization failure does not break local execution;
- local JSON log shape for success and failure events;
- sanitizer recursively drops sensitive fields;
- `capture_failure` never sends private prompt, answer, document, citation, path, or secret fields;
- CLI failure wrappers capture exceptions and preserve failure behavior;
- ingestion summary with `failed_files > 0` sends a failure alert with safe counts;
- Streamlit query failure capture excludes raw question and answer text.

Focused verification:

```bash
uv run python -m pytest tests/test_config.py tests/test_scripts.py tests/test_web_app.py -q
```

Full verification:

```bash
uv run python -m pytest -q
```

## Documentation

Update README and `.env.example` to state:

- Phoenix is for local traces and eval experiments;
- Sentry is for failure-only alerts;
- Sentry requires `sentry-sdk`, `SENTRY_DSN`, and local opt-in configuration;
- alert rules are configured in Sentry itself;
- Sentry events are sanitized by default and should not contain private corpus or prompt data.

## Implementation Notes

The implementation plan should start with tests for settings, sanitizer, and Sentry-disabled behavior. Then add the observability module, wire CLI entrypoints, wire Streamlit, and update docs/config. Keep commits scoped so the alerting layer is reviewable independently from any later monitor/check-in work.
