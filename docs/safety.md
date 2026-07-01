# Loop Safety - Imperial RAG

Imperial RAG is a private local RAG system. Loops must protect source documents, generated corpus state, auth databases, chat history, traces, eval outputs, and provider credentials.

## Default Posture

- L1 loops are report-only.
- No auto-merge, auto-push, auto-fix, dependency bump, runtime restart, ingestion promotion, or provider-backed eval run is allowed without explicit human approval.
- Loop tools are workflow scaffolding only; do not add `loop-engineering` packages to the Imperial runtime.

## Protected Data

Never copy these into external tools, state files, PR comments, or run logs:

- `.env` values, API keys, passwords, cookies, bearer tokens, and service credentials.
- Raw `documents/` content or extracted chunk text from `.imperial_rag/`.
- Chat history, auth databases, Phoenix spans, prompt payloads, retrieval previews, provider error payloads, or eval outputs.
- Generated local state that could reveal private corpus details.

Allowed summaries: paths, counts, IDs, checksums, pass/fail status, redacted snippets, and command names.

## Human Gates

Human approval is required for:

- Security, auth, privacy, tracing, provider billing, or deployment changes.
- Changes touching `.env*`, `.imperial_rag/`, `documents/`, auth/chat databases, eval outputs, or Phoenix traces.
- Dependency upgrades or lockfile changes.
- Ingestion promotion, vector/keyword index replacement, corpus artifact regeneration, or Docker Compose changes.
- Any change touching more than 10 files.

## Verification

- Default source verifier: `./scripts/check.sh`.
- Eval dataset verifier: `uv run python scripts/audit_eval_rows.py --strict --output-path <tmp>`.
- Ingestion promotion verifier: `uv run python scripts/check_ingestion_promotion.py --baseline-root <path> --shadow-root <path>`.
- Trace-shape verifier: `uv run python scripts/validate_phoenix_trace.py --run-id <id> --json`.
- If a verifier cannot run, the loop must report `ESCALATE_HUMAN` instead of approving itself.

## Incident Response

If a loop writes unsafe state, leaks private data, or proposes an unsafe change:

1. Add `loop-pause-all` to `loop-budget.md`.
2. Stop all scheduled loop runs.
3. Revert or redact the unsafe output.
4. Record a short post-mortem in `STATE.md`.
5. Tighten `loop-constraints.md` before re-enabling the loop.
