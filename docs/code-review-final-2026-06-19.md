# Imperial RAG Final Validated Code Review

Date: 2026-06-19

Source reports merged:

- `docs/superpowers/reports/2026-06-19-codebase-audit.md`
- `docs/code-review-2026-06-19.md`

Scope: current repository code under `/Users/danil/Public/imperial`, with emphasis on `src/imperial_rag/`, scripts, tests, configuration, and docs-visible runtime behavior.

Method: every carried finding below was checked against the current checkout. The two source reports were treated as inputs, not as trusted facts. Items that existed but were only style preferences were either downgraded or grouped as low-priority cleanup. No application code was changed.

## Executive Summary

The codebase is functional and well covered by tests, but the merged review found several real risks:

1. Vector backend construction failures can be hidden as normal empty vector results.
2. LLM/provider errors are rendered to users as the same refusal text used for legitimate no-evidence answers.
3. `OPENINFERENCE_HIDE_INPUT_TEXT` does not currently suppress retrieval preview text placed in span outputs.
4. Streamlit rebuilds the query runtime per question instead of caching long-lived clients/resources.
5. SQLite-backed stores have explicit `close()` methods but are used without reliable ownership/lifecycle patterns.

Most other findings are maintainability risks: duplicated environment parsing, duplicated retrieval helper logic, a hardcoded local workspace default, custom keyword stemming/search behavior, repeated CLI setup code, weak toolchain enforcement, and broad dependency ranges for fast-moving RAG libraries.

## Validated Findings

### P1 - Vector store construction failures are hidden as empty vector search results

Status: validated, narrowed.

Evidence:

- `src/imperial_rag/runtime.py:48-59` catches all exceptions while constructing the Qdrant-backed vector retriever and replaces the vector search with `_NoopVectorSearch`.
- `_NoopVectorSearch` at `src/imperial_rag/runtime.py:23-25` returns `[]` and carries no diagnostic marker.
- `HybridRetriever.retrieve()` only reports `vector_search_status="unavailable"` when the vector call itself raises (`src/imperial_rag/retrieval.py:123-128`). If it receives `_NoopVectorSearch`, it sees an empty result and changes status to `"empty"` (`src/imperial_rag/retrieval.py:129-130`).

Reasoning:

Search-time vector failures are already handled correctly, and tests cover that path (`tests/test_retrieval.py:281-298`). The defect is construction-time failure: Qdrant connection errors, collection errors, or vector-store setup exceptions are converted to a no-op before retrieval diagnostics can distinguish outage from empty semantic matches.

Recommendation:

Use separate sentinel retrievers for disabled semantic search and unavailable vector construction, e.g. `DisabledVectorSearch` and `UnavailableVectorSearch(error_type=...)`. Propagate `vector_search_status="unavailable"` plus a fallback tag such as `vector_store_unavailable`, and log one sanitized warning from `build_query_dependencies()`.

### P1 - LLM/provider exceptions are rendered like legitimate no-evidence refusals

Status: validated.

Evidence:

- `generate()` catches all `Exception` from `chat_model.invoke()` and returns `REFUSAL_TEXT` with trace attributes (`src/imperial_rag/runtime.py:140-156`).
- The behavior is explicitly asserted in `tests/test_runtime.py:80-165`.
- The workflow coerces the returned mapping into an answer string and preserves diagnostics separately (`src/imperial_rag/workflows.py:285-327`).
- The web UI displays only `result["answer"]` and stores it as message content (`src/imperial_rag/web_app.py:217-225`).

Reasoning:

The trace contains `answer.model_status="error"`, but the user-facing answer is identical to a normal no-evidence refusal. Network timeouts, invalid API keys, rate limits, or token failures are therefore indistinguishable from a successful refusal unless the operator inspects traces/logs.

Recommendation:

Return a structured error field or raise a typed exception from the model-generation layer. In Streamlit, display an operational error message and avoid treating it as a successful RAG refusal. Keep the trace attributes, but do not make Phoenix the only visible error surface.

### P1 - Retrieval preview text can bypass `OPENINFERENCE_HIDE_INPUT_TEXT`

Status: validated.

Evidence:

- `OpenInferenceTraceSpan.set_output()` only checks `OPENINFERENCE_HIDE_OUTPUTS` (`src/imperial_rag/tracing.py:83-87`).
- `_attribute_hidden()` hides document-content attributes when `OPENINFERENCE_HIDE_INPUT_TEXT` is set (`src/imperial_rag/tracing.py:509-522`), but that logic does not inspect JSON already passed to `set_output()`.
- `retrieval_documents_preview()` copies compact `page_content` into a `preview` field (`src/imperial_rag/tracing.py:257-277`).
- `_set_documents_span_output()` writes those previews into `output.value` as `top_documents` (`src/imperial_rag/retrieval.py:644-652`).
- Existing redaction coverage hides outputs only when `OPENINFERENCE_HIDE_OUTPUTS=true` (`tests/test_tracing.py:504-522`), while preview tests assert the preview contains text (`tests/test_tracing.py:629-655`).

Reasoning:

A user setting `OPENINFERENCE_HIDE_INPUT_TEXT=true` can reasonably expect corpus text to be absent from traces. Document-content attributes are hidden, but retrieval span outputs can still contain preview snippets unless `OPENINFERENCE_HIDE_OUTPUTS=true` is also set.

Recommendation:

When `_hide_input_text()` is true, make retrieval/reranker outputs metadata-only: counts, IDs, filenames, status, scores, and fallback tags are fine; omit `preview`. Add a regression test for `OPENINFERENCE_HIDE_INPUT_TEXT=true` without `OPENINFERENCE_HIDE_OUTPUTS=true`.

### P2 - Streamlit rebuilds runtime dependencies on every question

Status: validated.

Evidence:

- `query_runtime()` calls `create_runtime(settings).query(question)` for each submitted question (`src/imperial_rag/web_app.py:60-67`).
- `create_runtime()` creates dependency and retrieval-service caches inside one runtime instance (`src/imperial_rag/runtime.py:111-163`), so those caches are lost when Streamlit creates a fresh runtime for the next question.
- No `st.cache_resource` wrapper exists around runtime, Qdrant, Elasticsearch, retrieval service, or model setup in `web_app.py`.

Reasoning:

This puts client construction and lazy model setup on the interaction path for every chat turn. It is not a correctness bug, but it is a real latency and stability risk for a local Streamlit app.

Recommendation:

Add a cached runtime resource keyed by stable settings/provider metadata. Keep session/user trace context outside the cached runtime, and invalidate the cache when ingestion or relevant env settings change.

### P2 - SQLite stores lack reliable context-manager lifecycles

Status: validated.

Evidence:

- `ManifestStore` opens a persistent connection in `src/imperial_rag/manifest.py:100-106` and exposes `close()` at `src/imperial_rag/manifest.py:201-202`, but has no `__enter__`/`__exit__`.
- `load_status_summary()` creates `ManifestStore(manifest_path).list_records()` without closing it (`src/imperial_rag/web_app.py:41-57`).
- Ingestion creates `manifest_store` and `ocr_cache` without guaranteed close paths (`src/imperial_rag/pipeline.py:93-95`, `src/imperial_rag/pipeline.py:316-321`).
- `OcrCache` similarly owns a persistent connection and only exposes `close()` (`src/imperial_rag/ocr.py:101-145`).

Reasoning:

Short CLI/test runs may not expose this, but Streamlit reruns and ingestion jobs can leave SQLite connections open longer than intended. This can hold locks and make ownership unclear.

Recommendation:

Add context-manager support to `ManifestStore` and `OcrCache`, then update UI/ingestion/tests to use `with ... as store:`. Alternatively, convert these classes to short-lived connection helpers.

### P2 - Invalid citation diagnostics are not surfaced as a UI guardrail

Status: validated, reframed.

Evidence:

- `validate_citations()` runs and produces `citations_valid` plus `invalid_citations` (`src/imperial_rag/workflows.py:302-327`).
- Tests intentionally preserve unsupported or uncited generated answers while marking diagnostics invalid (`tests/test_workflows.py:453-482`).
- Streamlit stores and renders only the answer text, sources, and retrieved files (`src/imperial_rag/web_app.py:217-225`, `src/imperial_rag/web_app.py:383-391`).

Reasoning:

This is not an accidental workflow bug: tests show the current contract preserves the generated answer for diagnostics. The user-facing gap is that product mode does not warn, retry, or hide the answer when `citations_valid` is false.

Recommendation:

Keep diagnostics for eval/debug, but add a product guardrail in the UI: show a clear warning, retry once with a stricter prompt, or replace invalid-citation answers with refusal text behind a setting.

### P2 - Retrieved-file previews and downloads eagerly load full private files

Status: validated.

Evidence:

- Ingestion writes full extracted page content into per-file JSON artifacts (`src/imperial_rag/pipeline.py:356-371`).
- Streamlit preview loading reads the artifact and joins all extracted document `page_content` values (`src/imperial_rag/web_app.py:539-559`).
- The download button payload reads the entire source file into bytes during render (`src/imperial_rag/web_app.py:418-424`).

Reasoning:

Large PDFs/spreadsheets can create heavy Streamlit rerenders and copy more corpus text into UI/session memory than necessary.

Recommendation:

Store or compute bounded previews, cache preview reads by artifact mtime, and load full source bytes only on explicit download action where Streamlit allows it. Add limits or warnings for large files.

### P2 - Configuration parsing is duplicated and weakly validated

Status: validated.

Evidence:

- `Settings` parses core env values in `src/imperial_rag/config.py:21-40`.
- Retrieval parsing has separate `_env_int`, `_env_float`, and `_env_str` helpers (`src/imperial_rag/retrieval.py:21-72`).
- Provider parsing has another set of `_env_*` helpers (`src/imperial_rag/providers.py:41-73`).
- Tracing has separate `_env_flag` and `_env_int` behavior (`src/imperial_rag/tracing.py:446-460`).
- Behavior diverges: retrieval integer parsing raises on invalid values, tracing integer parsing falls back to defaults, and provider string parsing strips values while retrieval string parsing returns raw strings.

Reasoning:

The app is now configuration-heavy. Scattered parsing means invalid env values fail at arbitrary runtime points, and cross-field relationships such as positive limits or `chunk_overlap < chunk_size` are not centralized.

Recommendation:

Move env parsing to a single typed settings layer, such as `pydantic-settings`, with validators and one load entrypoint. Keep small internal dataclasses only for derived runtime objects.

### P2 - Hardcoded local workspace root is the default

Status: validated.

Evidence:

- `DEFAULT_WORKSPACE_ROOT` is `/Users/danil/Public/imperial` (`src/imperial_rag/config.py:8`).
- `Settings.workspace_root` uses that value whenever `IMPERIAL_RAG_WORKSPACE_ROOT` is unset (`src/imperial_rag/config.py:21-25`).
- Tests assert the local developer path as the default (`tests/test_config.py:7-24`).

Reasoning:

This is acceptable for a private local app on this machine, but it is a portability and CI risk. On another checkout, ingestion/query commands can point at a nonexistent or wrong corpus root unless the env var is set.

Recommendation:

Derive the default from the repository root relative to `config.py`, or use `Path.cwd()` only for CLI entrypoints. Keep `IMPERIAL_RAG_WORKSPACE_ROOT` as the explicit override.

### P2 - Retrieval code has too much custom orchestration around framework primitives

Status: validated as a maintainability issue.

Evidence:

- `src/imperial_rag/retrieval.py` defines custom `HybridRetriever`, `CandidateMerger`, `RrfCandidateFusion`, `FallbackRanker`, `Reranker`, and `RetrievalService`.
- It already uses LangChain `BaseRetriever` and `EnsembleRetriever` (`src/imperial_rag/retrieval.py:7-9`, `src/imperial_rag/retrieval.py:312-357`).
- `workflows.py` still keeps a legacy ranking path, `rank_hybrid_candidates()` (`src/imperial_rag/workflows.py:99-126`), alongside `RetrievalService`.

Reasoning:

The custom code is tested and not broken by itself. The issue is long-term readability: retrieval behavior is split between project-specific orchestration and framework retriever/reranker abstractions that could carry more of the pipeline.

Recommendation:

Refactor incrementally, preserving diagnostics/tests. Use LangChain retrievers/runnables/compressors where they fit; keep only project-specific metadata normalization, privacy-safe IDs, and diagnostic mapping as local adapters.

### P2 - Keyword matching relies on custom stemming and token query logic

Status: validated.

Evidence:

- `stem_token()` is a regex-based iterative suffix stripper (`src/imperial_rag/keyword.py:10-100`).
- The stopword list includes `найт`, which is an implementation artifact of stemming `найти` (`src/imperial_rag/keyword.py:11-49`).
- Elasticsearch token queries require each remaining token to match exactly or fuzzily (`src/imperial_rag/keyword.py:147-178`).
- Elasticsearch stores a Python-computed `normalized_text` field using this custom normalization (`src/imperial_rag/elasticsearch_keyword.py:215-220`).

Reasoning:

This is not necessarily wrong for the current corpus, but it couples search quality to custom morphology and stopword behavior. It is a real maintainability and relevance risk, especially for Russian terms.

Recommendation:

Move more language behavior into Elasticsearch analyzers/mappings where practical. If Python-side stemming remains, replace the custom stemmer with a maintained Russian stemming/morphology library and test the high-value query terms.

### P2 - `IMPERIAL_RAG_LOG_FORMAT` is a dead switch

Status: validated, narrowed.

Evidence:

- `_log_format_from_env()` returns `"json"` regardless of env value (`src/imperial_rag/config.py:16-18`).
- `Settings.log_format` stores that value (`src/imperial_rag/config.py:39-40`).
- `configure_observability()` uses `log_level`, but always installs `JsonEventFormatter()` and never reads `log_format` (`src/imperial_rag/observability.py:43-57`).
- README documents that v1 supports JSON only (`README.md:395-396`), and tests assert `IMPERIAL_RAG_LOG_FORMAT=plain` still resolves to `"json"` (`tests/test_config.py:27-52`).

Reasoning:

This is not a runtime bug because JSON-only logging is deliberate in docs/tests. The issue is that the env var and settings field look configurable but are intentionally inert.

Recommendation:

Either remove the env var/settings field from the active config surface, or implement a real alternate formatter.

### P3 - Hidden/system files are included in corpus scans

Status: validated.

Evidence:

- `scan_files()` walks every file under `documents_root` and does not filter hidden/system/temp files (`src/imperial_rag/manifest.py:63-84`).
- The current workspace contains `documents/.DS_Store`.

Reasoning:

Unsupported real documents should remain visible, but OS artifacts such as `.DS_Store` and Office lock files add noise to manifests and ingestion status.

Recommendation:

Filter hidden files, `.DS_Store`, temp files, and lock files before hashing. Consider an allowlist of supported extensions while still reporting unsupported real corpus files intentionally.

### P3 - `assign_duplicate_groups()` is called twice during ingestion

Status: validated, downgraded.

Evidence:

- `scan_files()` returns `assign_duplicate_groups(records)` (`src/imperial_rag/manifest.py:63-84`).
- `pipeline._run()` wraps `scan_files()` in another `assign_duplicate_groups()` call (`src/imperial_rag/pipeline.py:85-91`).

Reasoning:

This is currently idempotent because grouping is based on stable SHA-256 hashes. It is not a correctness bug today, but the API contract is ambiguous and the second pass is redundant.

Recommendation:

Make `scan_files()` pure file listing and keep duplicate grouping as one explicit pipeline step, or remove the outer call and document that `scan_files()` returns grouped records.

### P3 - Duplicate retrieval helper logic can diverge

Status: validated.

Evidence:

- `_document_key()` and `_content_key()` are defined in both `workflows.py` (`src/imperial_rag/workflows.py:85-91`) and `retrieval.py` (`src/imperial_rag/retrieval.py:221-248`).
- `rank_hybrid_candidates()` inlines searchable-text construction (`src/imperial_rag/workflows.py:112-124`).
- `retrieval._searchable_text()` repeats the same five-field join with `.casefold()` (`src/imperial_rag/retrieval.py:255-265`).
- `keyword.searchable_document_text()` is the canonical shared version without casefolding (`src/imperial_rag/keyword.py:181-191`).

Reasoning:

The current implementations match closely, but if document ID or searchable metadata rules change in one module, the legacy workflow path and retrieval service can silently diverge.

Recommendation:

Move document/content/searchable-text helper functions to one shared module and import them everywhere. Deprecate or remove the legacy workflow ranking path once `RetrievalService` is the only supported route.

### P3 - CLI scripts duplicate setup and logging boilerplate

Status: validated.

Evidence:

- `scripts/query.py`, `scripts/ingest.py`, `scripts/run_phoenix_eval.py`, `scripts/run_ragas_eval.py`, and `scripts/run_all_evals.py` repeat variants of env loading, observability setup, tracing setup, duration calculation, and failure logging.
- `rg` confirms repeated helpers such as `_configure_observability`, `_configure_tracing`, `_load_project_env`, `_duration_ms`, and `_log_failure` across those scripts.

Reasoning:

This is not a functional defect, but behavior changes are easy to apply inconsistently across CLI entrypoints.

Recommendation:

Extract a shared `imperial_rag.cli` command context, or migrate to Typer/Click with common setup hooks.

### P3 - Ragas evals bridge async work through a custom thread runner

Status: validated.

Evidence:

- `_resolve_awaitable()` delegates awaitables to `_run_coroutine()` (`src/imperial_rag/ragas_eval.py:684-687`).
- `_run_coroutine()` starts a new thread when an event loop is already running (`src/imperial_rag/ragas_eval.py:690-711`).

Reasoning:

The helper may work for current tests, but nested event loops and thread hops make cancellation, tracing context, and error propagation more surprising.

Recommendation:

Make eval runners async-first and use a maintained boundary utility such as `anyio` at sync CLI edges.

### P3 - Tooling does not enforce style, imports, typing, or coverage

Status: validated.

Evidence:

- `pyproject.toml` has project metadata and pytest configuration only (`pyproject.toml:1-42`).
- There is no configured formatter, linter/import sorter, type checker, or coverage command.

Reasoning:

The tests are useful, but several findings here are exactly the kind of drift lint/type tooling helps catch earlier: duplicated helpers, broad exception surfaces, and unused/dead configuration.

Recommendation:

Add Ruff for formatting/linting/import sorting, a type checker for `src/`, and a documented local check command that runs lint/type/tests. Add coverage reporting once the suite stabilizes.

### P3 - Dependency version policy is broad for fast-moving RAG libraries

Status: validated.

Evidence:

- `pyproject.toml` leaves many fast-moving dependencies unconstrained: `langchain`, `langgraph`, `qdrant-client`, `streamlit`, `ragas`, and several Phoenix/OpenInference packages (`pyproject.toml:6-38`).
- The lockfile pins the current environment, but declared compatibility is much broader than what tests necessarily exercise.

Reasoning:

Fresh locks or dependency update batches are more likely to break behavior when public APIs move quickly.

Recommendation:

Keep `uv.lock` for reproducible local installs, but add explicit compatibility ranges for the most volatile dependencies and update them intentionally with the full test/eval suite.

### P3 - `trace_user_id_from_email(None)` hashes `"none"`

Status: validated, low risk.

Evidence:

- `trace_user_id_from_email()` coerces input with `str(email).strip().casefold()` (`src/imperial_rag/tracing.py:243-254`).
- Passing `None` therefore hashes the string `"none"` instead of returning `""`.
- Current Streamlit usage passes `current_user.email`, which comes from the auth flow (`src/imperial_rag/web_app.py:191-196`), so the visible path is likely safe.

Reasoning:

This is a latent helper bug rather than an observed runtime failure.

Recommendation:

Accept `email: str | None` and return `""` before coercion when the input is not a non-empty string.

## Validated Low-Priority Cleanup

These source-report observations are true but should not distract from the findings above:

- `RrfCandidateFusion.fuse()` instantiates `CandidateMerger()` twice even though it is stateless (`src/imperial_rag/retrieval.py:338-347`).
- `_annotate_retrieval_documents()` creates an intermediate `Document` only so `_retrieval_id()` can read metadata (`src/imperial_rag/retrieval.py:236-243`).
- `make_qdrant_store()` creates one `Settings()` object, then another partial `Settings(...)` object (`src/imperial_rag/indexing.py:75-87`).
- `create_runtime()` uses single-slot dictionaries for closure caches (`src/imperial_rag/runtime.py:111-129`). This is harmless but could be simplified with `nonlocal` or a small cached object.
- Raw SQLite row mapping is hand-written. The standalone issue is lifecycle/ownership, not an immediate need to migrate to SQLAlchemy/SQLModel.

## Claims Reframed Or Not Promoted

- Search-time vector failures are not hidden; `HybridRetriever` marks them `"unavailable"` and tests cover that. The real issue is construction-time vector-store failure being converted to `_NoopVectorSearch`.
- Invalid citations being preserved in the workflow is intentional and tested. The real issue is missing UI/product handling when `citations_valid` is false.
- `Settings.log_level` is not dead; `configure_observability()` reads it. Only `log_format` is a dead/inert switch.
- Custom null-object vector classes are not a standalone problem. They matter because the no-op and unavailable cases are not distinguishable in diagnostics.

## Suggested Implementation Order

1. Fix vector construction diagnostics and add a regression test for construction-time Qdrant/vector-store failure.
2. Surface model/provider errors distinctly from no-evidence refusals in runtime results and Streamlit.
3. Remove retrieval preview text from span outputs when `OPENINFERENCE_HIDE_INPUT_TEXT=true`.
4. Cache the Streamlit runtime resource with safe invalidation.
5. Add context managers for `ManifestStore` and `OcrCache`, then update callers.
6. Add UI handling for invalid citation diagnostics.
7. Bound retrieved-file previews/download reads.
8. Centralize settings/env parsing and validation.
9. Consolidate duplicated retrieval/search helper functions.
10. Address lower-priority CLI/tooling/dependency cleanup.

## Verification Notes

- Verified by source inspection against the current checkout.
- Post-write check: `uv run python -m pytest -q` -> 324 passed, 2 skipped, 7 warnings.
- No live Qdrant, Elasticsearch, Phoenix, Kibana, Streamlit, Docker Compose, or DashScope calls were required for this merge.
- The final report intentionally preserves only current-code findings; source-report claims that were too broad were narrowed in the text above.
