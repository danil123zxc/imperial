# Live RAG Smoke And Integration Tests Design

Date: 2026-06-05
Workspace: `/Users/danil/Public/imperial`
Status: Approved in chat. Written spec awaiting user review before implementation planning.

## Context

Imperial RAG already has broad offline pytest coverage for configuration, extraction, indexing, retrieval, runtime wiring, provider factories, Phoenix eval wrappers, and Ragas plumbing. Those tests intentionally avoid paid network calls and use fakes for provider behavior.

The current live checkout also has:

- `evals/questions.jsonl` with 30 Russian evaluation cases.
- `.imperial_rag/extracted/chunks.jsonl` present with 2,470 chunks.
- `.imperial_rag/keyword.sqlite3` present.
- `.imperial_rag/manifest.sqlite3` present.
- no `.imperial_rag/vector_provider.json` in the current generated state.
- Qdrant not listening on `127.0.0.1:6333` during design review.
- Phoenix reachable on `127.0.0.1:6006`.
- `.env` containing a `DASHSCOPE_API_KEY`, while the current shell did not have that variable exported.

The new test layer should prove that the real hosted provider calls and the local RAG pipeline work together, without turning normal pytest runs into slow, paid, network-dependent checks.

## Goals

- Add opt-in smoke tests that make real DashScope/Qwen API calls.
- Add opt-in integration tests that exercise RAG behavior with real APIs.
- Cover both a stable temporary fixture corpus and the existing generated Imperial corpus state.
- Keep the default test suite offline, deterministic, and cheap.
- Load local `.env` values for live tests without printing secrets.
- Produce clear skip reasons before opt-in and actionable failures after opt-in.

## Non-Goals

- No broad rewrite of provider, retrieval, runtime, Phoenix, or Ragas code.
- No mandatory live API calls in the default pytest suite.
- No CI requirement to have DashScope, Qdrant, Phoenix, or private corpus state available.
- No committing generated corpus artifacts, Qdrant storage, Phoenix traces, eval outputs, or secrets.
- No reference-answer Ragas expansion in this change.

## Chosen Approach

Use a live pytest layer plus a small helper module.

Add a reusable helper, likely `tests/live_support.py`, that:

- loads `.env` from the workspace root when present;
- never prints secret values;
- checks opt-in flags;
- verifies `DASHSCOPE_API_KEY`;
- checks generated corpus artifacts for real-corpus tests;
- checks Qdrant availability when vector-specific assertions are requested;
- exposes focused helpers such as `require_live_api()` and `require_live_corpus()`.

Then add three opt-in test modules:

- `tests/test_live_provider_smoke.py`
- `tests/test_live_rag_integration.py`
- `tests/test_live_real_corpus.py`

This keeps live behavior discoverable in pytest while preserving the existing offline suite.

## Gating

Live tests are enabled only by environment flags:

| Flag | Meaning |
| --- | --- |
| `IMPERIAL_RAG_LIVE_API=1` | Enables real DashScope/Qwen API smoke and fixture integration tests. |
| `IMPERIAL_RAG_LIVE_CORPUS=1` | Enables tests against the existing `.imperial_rag` Imperial corpus state. Requires `IMPERIAL_RAG_LIVE_API=1`. |
| `IMPERIAL_RAG_LIVE_QDRANT=1` | Existing Qdrant live-health opt-in. Also useful for vector-specific integration checks. |

Default behavior:

- `uv run python -m pytest -q` skips all live API and live corpus tests.
- Missing `.env` or missing `DASHSCOPE_API_KEY` before opt-in should skip with a clear reason.
- Missing `DASHSCOPE_API_KEY` after `IMPERIAL_RAG_LIVE_API=1` should still skip with a clear reason, because no live request can be attempted. Provider auth, quota, model, network, or response-shape errors after a request is attempted should fail.

## Provider Smoke Tests

`tests/test_live_provider_smoke.py` should call the real provider at the thinnest useful layer:

- Chat: call `create_chat_model().invoke(...)` with a tiny deterministic prompt and assert a non-empty response.
- Embeddings: call `create_embeddings().embed_query(...)` and assert a non-empty numeric vector with the configured dimension when available.
- Rerank: call `create_reranker(top_n=1).compress_documents(...)` over two tiny `Document` objects and assert the expected document ranks first.
- OCR: create a tiny readable image in `tmp_path`, call the Qwen OCR client, and assert non-empty extracted text. Keep the assertion flexible enough for OCR variance, but strict enough to catch empty or malformed provider responses.

These smoke tests validate provider credentials, endpoints, model names, SDK wiring, response parsing, and secret sanitization boundaries.

## Fixture Corpus Integration

`tests/test_live_rag_integration.py` should build a tiny temporary corpus and run the RAG flow with real APIs.

The fixture corpus should include a small supported text-bearing document such as a DOCX or RTF with a unique sentinel fact. The test should:

1. Create the fixture corpus under `tmp_path/documents`.
2. Build `Settings(workspace_root=tmp_path)`.
3. Run ingestion using real provider behavior where needed.
4. Build the runtime.
5. Ask a question whose answer is supported by the fixture text.
6. Assert:
   - answer is non-empty;
   - answer is not the strict refusal text;
   - citations are present;
   - evidence/documents are present;
   - retrieval diagnostics are present;
   - final evidence count is greater than zero.

When local Qdrant is unavailable, the fixture test may still prove keyword retrieval plus real chat/rerank behavior. Vector-specific assertions should run only when Qdrant is available and explicitly enabled.

## Real Corpus Integration

`tests/test_live_real_corpus.py` should run against the existing generated state in `.imperial_rag`.

Before running, it should verify:

- `.imperial_rag/extracted/chunks.jsonl` exists and has at least one row;
- `.imperial_rag/keyword.sqlite3` exists;
- `.imperial_rag/manifest.sqlite3` exists;
- `DASHSCOPE_API_KEY` is available;
- `IMPERIAL_RAG_LIVE_API=1` and `IMPERIAL_RAG_LIVE_CORPUS=1` are set.

The test should use one or a few curated questions from `evals/questions.jsonl`, preferably direct policy questions with `expected_behavior="cite_answer"`. It should query the real runtime and assert:

- non-refusal answer;
- citations are present;
- sources or evidence are present;
- retrieval diagnostics include key fields such as `keyword_candidates`, `merged_candidates`, `final_evidence`, and `reranker`;
- source hints or document text include at least one expected hint when practical.

The real-corpus test is a health check for the current local generated state, not a replacement for the deterministic eval runner.

## Error Handling

Skip before opt-in:

- live API flag not set;
- live corpus flag not set for real-corpus tests;
- `.env` missing and no credential in the shell;
- local generated state missing for tests that were not explicitly opted into.

Fail after opt-in:

- provider authentication, quota, model, or network response errors;
- malformed provider responses;
- empty chat, embedding, rerank, or OCR outputs;
- broken generated corpus state when `IMPERIAL_RAG_LIVE_CORPUS=1`;
- real-corpus answer contract failures such as no citations or no retrieval diagnostics.

Qdrant handling:

- If Qdrant is not running, skip vector-specific assertions.
- Do not require Qdrant for keyword-backed fixture or real-corpus smoke unless the specific test name promises vector behavior.

Secret handling:

- Do not print API keys.
- Do not include key values in assertion messages.
- Keep using existing provider error sanitization for provider exceptions.

## Commands

Default offline suite:

```bash
uv run python -m pytest -q
```

Live provider and fixture integration:

```bash
IMPERIAL_RAG_LIVE_API=1 uv run python -m pytest tests/test_live_provider_smoke.py tests/test_live_rag_integration.py -q
```

Live real-corpus integration:

```bash
IMPERIAL_RAG_LIVE_API=1 IMPERIAL_RAG_LIVE_CORPUS=1 uv run python -m pytest tests/test_live_real_corpus.py -q
```

Optional Qdrant live health:

```bash
IMPERIAL_RAG_LIVE_QDRANT=1 uv run python -m pytest tests/test_qdrant_health.py -q
```

## Acceptance Criteria

- The normal full pytest suite remains offline and skips all paid/network tests.
- Provider smoke tests prove chat, embeddings, rerank, and OCR real API calls.
- Fixture integration proves ingestion, retrieval, generation, citations, and diagnostics work with real APIs.
- Real-corpus integration proves the current generated Imperial corpus can answer curated questions with citations and retrieval diagnostics.
- Tests load `.env` safely when needed.
- Skip reasons are explicit.
- Failures after opt-in are actionable.
- No secrets, generated corpus artifacts, local state, traces, or eval outputs are committed.

## Implementation Boundaries

Implement this in stages:

1. Add `tests/live_support.py` with environment loading and opt-in gates.
2. Add provider smoke tests.
3. Add fixture-corpus integration tests.
4. Add real-corpus integration tests.
5. Update `.env.example` and README testing docs only if the new flags or commands are not already clear.
6. Verify default pytest and the focused live test commands as far as local services and credentials allow.
