# Loop State - Imperial RAG

Last run: 2026-07-02T00:52:49+0900 `daily-triage` manual L1.
Mode: L1 report-only.
Config: `LOOP.md`.
Registry: `patterns/registry.yaml`.
Allowed L1 write paths: `STATE.md`, `loop-run-log.md`.

This file is the durable memory spine for Imperial automation loops. L1 loops may update this file with findings, status, and suggested next actions only. They must not edit source, generated corpus state, secrets, or runtime configuration unless a human explicitly asks in the active thread.

## Active Loops

Loop IDs must match `LOOP.md` and `patterns/registry.yaml`.

| Loop ID | Status | Trigger | Level | Notes |
| --- | --- | --- | --- | --- |
| `daily-triage` | Active | Manual or at most once per day | L1 report-only | Reads local repo signals and updates state/logs only. |
| `ci-sweeper-manual` | Planned | Manual or after failed CI | L1 report-only | Summarizes CI/local check failures; no fixes. |
| `eval-regression-check` | Planned | Manual before eval changes | L1 report-only | Summarizes dataset audit or drift signals; provider-backed evals require approval. |
| `ingestion-promotion-review` | Planned | Manual before promotion | L1 report-only | Summarizes baseline/shadow checks; no promotion. |
| `post-merge-cleanup` | Candidate | Manual after merge review | L1 report-only | Report-only in L1; any cleanup fix needs later L2 approval. |

## Current Findings

- None. Manual L1 triage found no high-priority loop issue after scaffold sync.

## Watch List

- Loop scaffold created for report-only daily triage, CI sweeps, eval drift checks, ingestion promotion reviews, and post-merge cleanup review.
- Baseline verifier gate: `./scripts/check.sh`.
- Operator CLI cadence for `daily-triage` is pinned to manual or exactly `1d`; do not use the upstream default `1d-2h` window for Imperial.
- Pre-existing unrelated local dirt remains outside loop ownership: `.DS_Store` is modified, and local council/planning docs are untracked.
- RAG-specific gates to reference when relevant:
  - `uv run python scripts/audit_eval_rows.py --strict --output-path <tmp>`
  - `uv run python scripts/check_ingestion_promotion.py --baseline-root <path> --shadow-root <path>`
  - `uv run python scripts/validate_phoenix_trace.py --run-id <id> --json`

## Noise / Ignore

- Generated local runtime state is private and should be summarized by path and status only, not copied into state.
- No connectors, provider-backed evals, live ingestion, Docker restarts, or Phoenix queries were used for the first manual L1 run.

## Human Decisions

- 2026-06-30: Start Imperial loop engineering in L1 report-only mode.
- No auto-fix, auto-push, or auto-merge is allowed.
- 2026-07-02: Pin `daily-triage` to manual or once daily; keep high-frequency CI sweeps disabled.

## Pause State

Pause flag: none.

If `loop-pause-all` appears here or in `loop-budget.md`, every loop must exit after appending a skipped-run entry to `loop-run-log.md`.

## Recent Runs

| Run ID | Loop ID | Level | Outcome | Notes |
| --- | --- | --- | --- | --- |
| 2026-07-02T00:52:49+0900 | `daily-triage` | L1 | Success | `CONSTRAINTS_REPORT_ONLY`; `BUDGET_OK` at 23k/100k tokens; `loop-sync` clean; state/log privacy diff passed. |
