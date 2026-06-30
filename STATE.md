# Loop State - Imperial RAG

Last run: Not started.
Mode: L1 report-only.

This file is the durable memory spine for Imperial automation loops. Week-one loops may update this file with findings, status, and suggested next actions only. They must not edit source, generated corpus state, secrets, or runtime configuration unless a human explicitly asks in the active thread.

## High Priority

- None yet.

## Watch List

- Loop scaffold created for report-only daily triage, CI sweeps, eval drift checks, and ingestion promotion reviews.
- Baseline verifier gate: `./scripts/check.sh`.
- RAG-specific gates to reference when relevant:
  - `uv run python scripts/audit_eval_rows.py --strict --output-path <tmp>`
  - `uv run python scripts/check_ingestion_promotion.py --baseline-root <path> --shadow-root <path>`
  - `uv run python scripts/validate_phoenix_trace.py --run-id <id> --json`

## Noise / Ignore

- Generated local runtime state is private and should be summarized by path and status only, not copied into state.

## Human Decisions

- 2026-06-30: Start Imperial loop engineering in L1 report-only mode.
- No auto-fix, auto-push, or auto-merge is allowed.

## Recent Runs

| Run | Pattern | Outcome | Notes |
| --- | --- | --- | --- |
| Not started | Daily triage | Pending | First scheduled run should populate this table. |
