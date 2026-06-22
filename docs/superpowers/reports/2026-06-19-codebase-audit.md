# Imperial RAG Codebase Audit

Date: 2026-06-19  
Scope: tracked application code, scripts, tests, configuration, Docker/Compose, README, and current repository runtime conventions under `/Users/danil/Public/imperial`. I did not change application code.

## Verification performed

- `git status --short` baseline captured before audit. The working tree already contained unrelated `.agents/skills/*` deletions/modifications, `skills-lock.json` changes, and an untracked observability plan before this report was written.
- `uv run python -m pytest -q` -> **324 passed, 2 skipped, 7 warnings** in 5.45s.
- `uv run python -m compileall -q src scripts tests` -> **passed**.
- `uv run python -m pytest -q tests/test_document_ids.py tests/test_tracing.py tests/test_workflows.py` -> **60 passed, 2 warnings** after final workspace verification.
- Current local toolchain observed: Python **3.14.3**, `uv` **0.10.10**.
- No live Qdrant, Elasticsearch, Phoenix, Docker Compose, or paid DashScope calls were run for this audit.

## Executive summary

The codebase is in good functional shape: the full unit suite passes, generated/local state is separated from source, the RAG pipeline is reasonably modular, and the project already uses LangChain, LangGraph, Qdrant, Elasticsearch, Phoenix/OpenInference, Streamlit, and Ragas rather than building everything from scratch.

The main issues are not catastrophic bugs; they are mostly maintainability, observability, privacy, and performance risks that will become more painful as the app grows. The highest-value improvements are:

1. Stop silently treating vector backend failures as empty vector results.
2. Cache the Streamlit runtime/dependencies instead of rebuilding clients/models per question.
3. Tighten Phoenix redaction so retrieval preview text cannot bypass `OPENINFERENCE_HIDE_INPUT_TEXT` through span outputs.
4. Add clear SQLite resource lifecycles/context managers.
5. Replace duplicated env parsing and custom retrieval/search plumbing with framework-native pieces where practical.

## Findings and recommendations

### P1 - Vector backend failures are hidden as normal empty results

**Evidence**

- `src/imperial_rag/runtime.py:51-59` catches any vector-store construction error and replaces it with `_NoopVectorSearch()`.
- `_NoopVectorSearch` in `src/imperial_rag/runtime.py:20-22` returns `[]` and has no status marker.
- `HybridRetriever.retrieve()` then sees an empty but otherwise normal vector search and records `vector_search_status = "empty"` rather than `"unavailable"`.

**Why it matters**

If Qdrant is down, the Qdrant collection is missing, credentials are wrong, or the client creation fails, the user can get keyword-only answers without a clear warning in diagnostics/logs/UI. This makes retrieval quality regressions harder to diagnose.

**Recommendation**

Use separate sentinel retrievers for intentionally disabled semantic search and unavailable vector search, for example `DisabledVectorSearch` and `UnavailableVectorSearch(provider_error_type=...)`, and propagate a diagnostic status such as `vector_search_status="unavailable"` plus a degraded fallback tag. Also log a sanitized warning once when vector dependency construction fails.

---

### P1 - Streamlit rebuilds runtime dependencies per question

**Evidence**

- `src/imperial_rag/web_app.py:60-67` calls `create_runtime(settings).query(question)` each time.
- `create_runtime()` builds a new workflow and dependency cache for that one runtime instance in `src/imperial_rag/runtime.py:96-166`.
- The UI does not use `st.cache_resource` around runtime, Elasticsearch client, Qdrant vector store/retriever, or chat model creation.

**Why it matters**

Every chat turn can recreate clients and lazy model wrappers. For a local RAG app this is unnecessary latency and makes failures/noisy setup work happen on the user interaction path.

**Recommendation**

Cache a runtime resource keyed by stable settings and provider metadata, e.g. a small `get_cached_runtime(settings_key)` wrapper using Streamlit `st.cache_resource`. Invalidate it after ingestion or when relevant env/settings change. Keep per-user trace/session context outside the cached runtime.

---

### P2 - OpenInference input-text hiding can still leak retrieval preview text in outputs

**Evidence**

- `OpenInferenceTraceSpan.set_output()` hides outputs only when `OPENINFERENCE_HIDE_OUTPUTS` is truthy (`src/imperial_rag/tracing.py:60-64`).
- `_set_documents_span_output()` adds `top_documents` previews to `output.value` via `retrieval_documents_preview()` (`src/imperial_rag/retrieval.py:642-651`).
- `retrieval_documents_preview()` includes a text `preview` field from `page_content` (`src/imperial_rag/tracing.py:257-280`).
- `OPENINFERENCE_HIDE_INPUT_TEXT` hides document-content attributes, but it does not hide these output previews unless `OPENINFERENCE_HIDE_OUTPUTS` is also enabled.

**Why it matters**

A user can reasonably expect `OPENINFERENCE_HIDE_INPUT_TEXT=true` to remove corpus text from traces. Today document content attributes are hidden, but compact preview text can still be serialized into `output.value` on retrieval spans.

**Recommendation**

When input text hiding is enabled, make retrieval span outputs metadata-only: keep counts, status, IDs, file names, ranks, and scores, but omit `preview`. Add a focused regression test for `OPENINFERENCE_HIDE_INPUT_TEXT=true` without `OPENINFERENCE_HIDE_OUTPUTS=true`.

---

### P2 - SQLite-backed stores lack context-manager lifecycles

**Evidence**

- `ManifestStore` opens a persistent SQLite connection in `src/imperial_rag/manifest.py:100-106`; callers often do not close it, e.g. `load_status_summary()` at `src/imperial_rag/web_app.py:41-57`.
- `OcrCache` opens a persistent SQLite connection in `src/imperial_rag/ocr.py:101-106`; ingestion builds it via `src/imperial_rag/pipeline.py:319-324` without a guaranteed close.
- `AuthStore` is safer because it opens short-lived connections per operation, but it repeatedly calls `initialize()` from most public methods.

**Why it matters**

The current pattern is fine in short tests, but Streamlit reruns and long ingestion jobs can leak file descriptors or hold SQLite locks longer than necessary. This also makes the intended ownership of database connections less readable.

**Recommendation**

Add context-manager support (`__enter__`, `__exit__`) or convert methods to short-lived connection helpers. Use `with ManifestStore(...) as store:` and `with OcrCache(...) as cache:` in ingestion/UI paths. If the stores grow, consider SQLModel/SQLAlchemy for explicit sessions and migrations.

---

### P2 - Invalid citation answers are shown unchanged

**Evidence**

- `validate_citations()` runs in `src/imperial_rag/workflows.py:302-327` and sets `citations_valid` / `invalid_citations`.
- The workflow still returns the generated answer unchanged even when validation fails.
- Tests intentionally preserve this diagnostic behavior in `tests/test_workflows.py:453-482`.
- The Streamlit UI renders `message["content"]` without a visible invalid-citation warning in `src/imperial_rag/web_app.py:383-391`.

**Why it matters**

For a strict-citation RAG assistant, a model answer with missing or invalid citations can still appear to the user as a normal answer. The diagnostics exist but are not used as a guardrail in the UI.

**Recommendation**

Keep diagnostics for eval/debug, but add a product-mode guardrail: either replace invalid-citation answers with the refusal text, retry once with a stricter prompt, or display a prominent Streamlit warning and hide the answer behind an expander. This can be optional behind a setting if tests need the current preservation behavior.

---

### P2 - Retrieved-file previews can load and render full extracted documents

**Evidence**

- Ingestion writes full extracted document content to `.imperial_rag/extracted/documents/<file_id>.json` in `src/imperial_rag/pipeline.py:361-374`.
- The UI reads the whole JSON artifact and joins all `page_content` strings in `src/imperial_rag/web_app.py:539-559`.
- `_download_button_payload()` also reads full source files into memory for each rendered download button in `src/imperial_rag/web_app.py:418-424`.

**Why it matters**

Large PDFs/spreadsheets can produce very large Streamlit messages, slow rerenders, and high memory use. This also duplicates private corpus text into UI session state more aggressively than necessary.

**Recommendation**

Store or compute bounded previews separately, e.g. first N characters/pages per file, and load full content only on explicit user action. Cache preview reads with `st.cache_data` keyed by `file_id` plus artifact mtime. Consider using Streamlit file-like objects for downloads where possible, or cap large downloads with a warning.

---

### P2 - Configuration parsing is duplicated and weakly validated

**Evidence**

- Env parsing exists in several places: `src/imperial_rag/config.py`, `src/imperial_rag/providers.py`, `src/imperial_rag/retrieval.py`, `src/imperial_rag/tracing.py`, and duplicated script helpers.
- `RetrievalSettings.from_env()` in `src/imperial_rag/retrieval.py:40-66` directly converts ints/floats and can crash on invalid env values.
- There is no central validation for relationships such as `chunk_overlap < chunk_size`, positive `top_n`, positive `k`, or valid URL fields.

**Why it matters**

The project is already configuration-heavy. Scattered parsing makes behavior harder to read and makes bad env values fail at arbitrary runtime points.

**Recommendation**

Move settings to a framework/library-backed config layer such as `pydantic-settings` with typed fields, validators, aliases for env var names, and one `Settings.load()` entrypoint. Keep lightweight dataclasses only for internal immutable result objects.

---

### P2 - Retrieval has too much custom orchestration around framework primitives

**Evidence**

- `src/imperial_rag/retrieval.py` is 727 lines and defines `HybridRetriever`, `CandidateMerger`, `RrfCandidateFusion`, `FallbackRanker`, `Reranker`, and `RetrievalService`.
- It already uses LangChain's `EnsembleRetriever` for RRF (`src/imperial_rag/retrieval.py:313-321`) and DashScope reranker integration (`src/imperial_rag/providers.py:226-232`).
- `src/imperial_rag/workflows.py` also has a legacy ranking path in `rank_hybrid_candidates()`.

**Why it matters**

The custom classes are tested, but they make the retrieval path harder for a new maintainer to understand. The code reads like a framework reimplementation in places even though LangChain retrievers, compressors, `RunnableParallel`, and `ContextualCompressionRetriever` can express much of this pipeline.

**Recommendation**

Refactor incrementally toward framework-native composition:

1. Keep existing public diagnostics/tests as the contract.
2. Express vector + keyword retrieval as retrievers/runnables.
3. Use `EnsembleRetriever` directly for fusion where possible.
4. Wrap reranking with LangChain compression abstractions.
5. Keep only the genuinely project-specific pieces: metadata normalization, privacy-safe document IDs, and diagnostics mapping.

---

### P2 - Keyword matching relies on a custom stemmer/stopword/query builder

**Evidence**

- `src/imperial_rag/keyword.py:10-77` defines a regex ending stripper, stopword list, and fuzzy options.
- `build_elasticsearch_token_query()` in `src/imperial_rag/keyword.py:147-179` requires every remaining token to match exactly-or-fuzzily.
- Elasticsearch is already the keyword backend, but analysis/stemming is handled mostly in Python rather than with index/search analyzers.

**Why it matters**

Custom stemming is hard to read, hard to tune, and likely less robust than Elasticsearch analyzers for Russian/company terminology. It also duplicates logic between indexing and query construction.

**Recommendation**

Move more language handling into Elasticsearch mappings/settings: use Russian analyzers or custom analyzer chains for normalized fields, keep structured metadata boosts, and reserve Python query logic for high-level query shape. This would align with the preference for framework/library behavior over custom text processing.

---

### P3 - Hidden/system files are included in corpus scans

**Evidence**

- `scan_files()` indexes every file under `documents/` in `src/imperial_rag/manifest.py:63-84`.
- The current workspace contains `documents/.DS_Store`.

**Why it matters**

Common OS/temp files will show up as unsupported/no-text manifest rows and can confuse ingestion status. Office lock files like `~$foo.docx` are another likely source of noise.

**Recommendation**

Add an ignore policy for hidden files, `.DS_Store`, temp files, lock files, and optionally a small allowlist of supported extensions. Keep unsupported real documents visible, but filter generated/system noise before hashing.

---

### P3 - CLI scripts duplicate setup boilerplate

**Evidence**

- `_ensure_src_on_path()`, `_build_settings()`, `_load_project_env()`, `_configure_observability()`, `_configure_tracing()`, `_duration_ms()`, and logging helpers are repeated across scripts such as `scripts/query.py`, `scripts/ingest.py`, `scripts/run_phoenix_eval.py`, and `scripts/run_ragas_eval.py`.

**Why it matters**

Repeated script plumbing makes small behavior changes easy to apply inconsistently. It also makes the scripts longer than the actual task logic.

**Recommendation**

Create a shared `imperial_rag.cli` module or move to a CLI framework such as Typer/Click. A common command context can load `.env`, build settings, configure observability/tracing, and handle sanitized failure logging.

---

### P3 - Eval code bridges async work through a custom thread runner

**Evidence**

- `src/imperial_rag/ragas_eval.py:684-711` implements `_run_coroutine()` and runs awaitables in a new thread if an event loop is already running.
- The repository guideline says evaluation workflows should be async-first when touching retrieval, model, tracing, or Phoenix-backed evaluation behavior.

**Why it matters**

The helper works in tests, but nested loops and thread hops can make cancellation, context propagation, and tracing behavior surprising.

**Recommendation**

Make eval runners async-first and use a maintained async utility (`anyio` is a good fit) at sync CLI boundaries. Keep sync wrappers only at the outermost script entrypoints.

---

### P3 - Tooling does not enforce readability automatically

**Evidence**

- `pyproject.toml` has pytest config but no formatter, linter, import sorter, type checker, coverage thresholds, or CI command definitions.
- The tests pass, but issues like duplicated env parsing, resource lifecycles, broad exception handling, and large modules require manual review to catch.

**Why it matters**

For a codebase where human readability is a priority, automated style and type feedback will prevent regressions and make future refactors safer.

**Recommendation**

Add a minimal dev tooling layer:

- `ruff` for linting/import sorting/formatting.
- `ty`, `pyright`, or `mypy` for type checks on `src/` and selected scripts.
- `pytest --cov` for coverage visibility.
- Optional `deptry` or similar to catch unused/missing dependencies.

---

### P3 - Dependency version policy is broad for fast-moving RAG libraries

**Evidence**

- `pyproject.toml` leaves most fast-moving libraries unconstrained: `langchain`, `langgraph`, `qdrant-client`, `streamlit`, `ragas` in dev, and Phoenix packages except `arize-phoenix-otel>=0.16.0`.
- The current `uv.lock` pins the installed set, but the declared compatibility range is much broader than what tests actually exercise.

**Why it matters**

LangChain, Ragas, Phoenix, and Streamlit APIs change quickly. Broad declarations make fresh locks or dependency updates more likely to break imports or behavior unexpectedly.

**Recommendation**

Keep `uv.lock` as the source of reproducible installs, but add explicit upper/lower compatibility ranges for fast-moving packages and schedule dependency-update PRs with the full test/eval suite.

## Framework/library-first refactor opportunities

These are the changes most aligned with the stated preference: "frameworks and libraries should be prioritised over custom classes and functions."

1. **Settings/config:** use `pydantic-settings` for all env parsing and validation.
2. **CLI:** use Typer or Click plus a shared command context instead of script-local boilerplate.
3. **Retrieval composition:** use LangChain retrievers/runnables/compressors for hybrid retrieval, RRF, and reranking; keep project-specific diagnostics as thin adapters.
4. **Keyword analysis:** move stemming/fuzzy/token behavior into Elasticsearch analyzers and mappings where possible.
5. **SQLite stores:** either add context-manager lifecycles to current stores or use SQLModel/SQLAlchemy once relationships/migrations grow.
6. **Async evals:** use async-first runners plus `anyio` at sync boundaries.
7. **Observability:** keep current Phoenix/OpenInference manual spans for unsupported span kinds, but replace applicable chain/LLM wrappers with public Phoenix decorators where they preserve the current trace hierarchy and privacy behavior.

## Suggested implementation order

### Phase 1: correctness and observability hardening

1. Add vector unavailable/degraded diagnostics and logging.
2. Fix trace preview redaction under `OPENINFERENCE_HIDE_INPUT_TEXT`.
3. Add Streamlit runtime caching.
4. Add SQLite context-manager lifecycles and close existing UI/ingestion stores.
5. Add bounded retrieved-file previews.

### Phase 2: readability and framework adoption

1. Introduce `pydantic-settings` and migrate env parsing.
2. Extract shared CLI context or migrate scripts to Typer/Click.
3. Refactor retrieval one step at a time around LangChain retrievers/runnables/compressors.
4. Move more keyword language logic into Elasticsearch analyzers.

### Phase 3: tooling and regression protection

1. Add Ruff formatting/linting.
2. Add a type checker for `src/`.
3. Add a CI-like local command that runs lint/type/tests.
4. Add focused regression tests for vector outage diagnostics, trace redaction, Streamlit runtime caching, and preview limits.

## Things already working well

- The main suite is strong for a local RAG app: 324 passing tests across ingestion, retrieval, tracing, evals, providers, scripts, auth, web app, and deployment config.
- Generated/private state is consistently kept under `.imperial_rag/` and ignored by git.
- Compose binds sensitive local services to `127.0.0.1` and documents the lack of auth for local Phoenix/Elasticsearch/Kibana.
- The project already uses several mature integrations: LangChain document abstractions, LangGraph workflows, LangChain Elasticsearch/Qdrant integrations, Phoenix/OpenInference tracing, and Ragas evals.
- The recent document-ID and Phoenix privacy hardening is visible in code/tests and should be preserved during refactors.

## Out of scope / not verified live

- I did not run live Qdrant, Elasticsearch, Kibana, Phoenix, Streamlit, Docker Compose, or DashScope network calls.
- I did not run a dependency vulnerability audit.
- I did not inspect private corpus document content.
- I did not modify application source code.
