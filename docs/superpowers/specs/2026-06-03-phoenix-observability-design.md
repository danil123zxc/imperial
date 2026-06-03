# Phoenix Observability Migration Design

Date: 2026-06-03
Workspace: `/Users/danil/Public/imperial`
Related system: local Imperial RAG application

## Status

Design approved in chat and written for user review. Implementation planning has not started.

## Context

The Imperial RAG project currently uses LangSmith as its planned observability and evaluation backend. The live code has a small LangSmith footprint:

- `pyproject.toml` depends on `langsmith`.
- `src/imperial_rag/config.py` exposes `Settings.langsmith_project`.
- `src/imperial_rag/indexing.py` preserves `langsmith_project` when constructing a temporary `Settings` object.
- `scripts/run_langsmith_eval.py` runs local deterministic evaluations and optionally uploads results through LangSmith.
- `tests/test_config.py`, `tests/test_evals.py`, and `tests/test_scripts.py` assert LangSmith naming and API shape.
- The previous Superpowers spec and plan mention LangSmith as the tracing and evaluation platform.

The user wants Phoenix instead of LangSmith, and wants the observability backend to be fully self-hosted.

Phoenix documentation consulted through Context7 describes:

- self-hosted Phoenix with `arizephoenix/phoenix:latest`;
- Phoenix UI and HTTP OTLP collector on port `6006`;
- OTLP gRPC collector on port `4317`;
- Python tracing through `phoenix.otel.register(project_name=..., auto_instrument=True)`;
- Phoenix experiments through `run_experiment(dataset=..., task=..., evaluators=...)`;
- Phoenix automatically logging experiment results.

## Goals

Replace LangSmith with Phoenix for:

- tracing ingestion, retrieval, ranking, answer generation, and evaluation runs;
- local self-hosted observability UI;
- local datasets and experiment result storage;
- deterministic citation/refusal/source-hint regression checks.

Keep the migration small and repo-shaped. The project should still run local evals without requiring Phoenix, but Phoenix experiment mode should store datasets and results in the self-hosted Phoenix instance.

## Non-Goals

- No Phoenix Cloud configuration.
- No LangSmith compatibility layer.
- No broad rewrite of the RAG pipeline.
- No custom OpenTelemetry framework beyond Phoenix setup glue.
- No replacement of Qdrant, SQLite manifest storage, or keyword search.
- No new LLM-as-judge evaluator in this migration.

## Decisions

- Phoenix is bundled into the local self-hosted stack for this project.
- Phoenix defaults are local:
  - `PHOENIX_PROJECT_NAME=imperial-rag`
  - `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006/v1/traces`
  - `PHOENIX_CLIENT_ENDPOINT=http://localhost:6006`
- The code remains configurable so it can point at another self-hosted Phoenix later.
- LangSmith package usage, environment variables, script names, and tests are removed or renamed.
- The existing deterministic evaluators remain the evaluation source of truth:
  - citation behavior;
  - refusal behavior;
  - source hint matching.
- Phoenix stores the eval dataset and experiment results when Phoenix mode is requested.
- Local eval mode remains available and prints pass/fail counts without requiring Phoenix.

## Architecture

Add a small observability layer:

- `src/imperial_rag/tracing.py`
  - owns Phoenix tracing configuration;
  - exposes an idempotent `configure_phoenix_tracing(settings, enabled=True)` function;
  - calls `phoenix.otel.register(project_name=settings.phoenix_project_name, endpoint=settings.phoenix_collector_endpoint, auto_instrument=True)`;
  - handles missing optional Phoenix packages with clear errors when tracing is explicitly requested.

Update settings:

- replace `Settings.langsmith_project` with `Settings.phoenix_project_name`;
- add `Settings.phoenix_collector_endpoint`;
- add `Settings.phoenix_client_endpoint`;
- preserve current defaults for workspace root and Qdrant.

Update scripts:

- rename `scripts/run_langsmith_eval.py` to `scripts/run_phoenix_eval.py`;
- keep `load_questions`, `run_target`, `run_local_eval`, `citation_behavior`, and `source_hint_behavior`;
- replace LangSmith upload mode with Phoenix experiment mode;
- create or load the Phoenix dataset from `evals/questions.jsonl`;
- run a Phoenix experiment with the existing deterministic evaluators;
- print the dataset name, experiment name or id, and number of examples.

Add local stack support:

- add `compose.yaml` with a `phoenix` service using `arizephoenix/phoenix:latest`;
- expose ports `6006` and `4317`;
- persist Phoenix data in a local named volume;
- keep Qdrant startup as-is unless a later implementation plan chooses to merge services.

## Data Flow

Normal query path:

1. A script or app creates `Settings`.
2. If tracing is enabled, the app calls `configure_phoenix_tracing(settings)`.
3. Phoenix auto-instrumentation captures supported LangChain, LangGraph, and OpenAI spans.
4. Spans are exported to the self-hosted Phoenix collector.
5. The user reviews traces in the local Phoenix UI.

Evaluation path:

1. `scripts/run_phoenix_eval.py` loads `evals/questions.jsonl`.
2. Local mode invokes the RAG runtime and scores rows in-process.
3. Phoenix mode creates or loads a Phoenix dataset.
4. Phoenix runs an experiment using the RAG task and deterministic evaluators.
5. Phoenix stores experiment outputs and evaluator scores locally.
6. The script prints a concise summary for the terminal.

## Error Handling

Local eval mode must not require Phoenix.

Phoenix experiment mode should fail fast when:

- the Phoenix Python client package is missing;
- the Phoenix server is not reachable;
- dataset creation fails;
- experiment execution fails.

Tracing setup should be idempotent so scripts can call it safely more than once. If automatic instrumentation is unavailable, the error should name the missing Phoenix/OpenInference dependency instead of silently disabling explicitly requested tracing.

## Testing

Update tests to verify:

- settings expose Phoenix defaults and environment overrides;
- `pyproject.toml` no longer depends on `langsmith`;
- Phoenix packages are declared;
- the Phoenix eval script imports and defines `main`, `citation_behavior`, and `source_hint_behavior`;
- the eval script source uses Phoenix dataset/experiment APIs rather than LangSmith APIs;
- local eval behavior remains deterministic;
- no live Phoenix server is required for unit tests.

Run focused tests first:

```bash
python -m pytest tests/test_config.py tests/test_evals.py tests/test_scripts.py tests/test_indexing.py -q
```

Then run the full suite:

```bash
python -m pytest -q
```

Optional live verification after Phoenix is started:

```bash
docker compose up -d phoenix
python scripts/run_phoenix_eval.py --use-phoenix
```

Expected result: Phoenix UI is available at `http://localhost:6006`, traces are visible under project `imperial-rag`, and the eval experiment is recorded locally.

## Migration Checklist

1. Replace LangSmith settings with Phoenix settings.
2. Replace LangSmith dependency with Phoenix tracing/evaluation dependencies.
3. Add Phoenix tracing setup module.
4. Rename and rewrite the eval runner for Phoenix datasets/experiments.
5. Update tests.
6. Add self-hosted Phoenix compose service.
7. Update old spec/plan references or add a short note that this Phoenix spec supersedes their LangSmith observability decision.

## Open Constraints

This workspace is not currently a Git repository, so the Superpowers requirement to commit the design document cannot be completed unless the workspace is initialized or moved into a repo.
