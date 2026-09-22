# Loop State - Imperial RAG

Last run: 2026-07-02T16:40:20+0900 `ingestion-promotion-review` manual L1.
Mode: L1 report-only loops plus default L2 assisted publish for code-changing tasks.
Config: `LOOP.md`.
Registry: `patterns/registry.yaml`.
Allowed L1 write paths: `STATE.md`, `loop-run-log.md`. L2 writes are limited to the explicitly approved task scope.

This file is the durable memory spine for Imperial automation loops. L1 loops may update this file with findings, status, and suggested next actions only. They must not edit source, generated corpus state, secrets, or runtime configuration unless a human explicitly asks in the active thread.

## Current Decision

On 2026-07-02, active manual L1 report-only loops are `daily-triage`, `ci-sweeper-manual`, `eval-regression-check`, and `ingestion-promotion-review`. In this file, "active" means allowing a manual L1 report-only run when the loop trigger and gates pass. It does not mean scheduling, connector access, source edits, provider-backed runs, ingestion promotion, or PR/GitHub writes.

`daily-triage` is active L1 report-only, but it already ran twice on 2026-07-02. Run it again only on explicit human request or on a later day, after checking the pause flag, cadence, daily token estimate, 0-subagent rule, allowed writes, and privacy diff. If there is no new signal, append a short skipped/no-signal run entry and stop.

Event-gated active loops remain gated:

- `ci-sweeper-manual`: run only with failed CI/local-check evidence or explicit request.
- `eval-regression-check`: run only before eval changes or on explicit request; no provider evals/dataset edits without approval.
- `ingestion-promotion-review`: run only with baseline/shadow context or explicit request; never promote artifacts.
- `post-merge-cleanup`: candidate only; GitHub PR metadata read requires approval before use.

Future ideas such as `dependency-sweeper`, `issue-triage`, `changelog-drafter`, and `pr-babysitter` are not configured loops. They need registry entries, budgets, connector policy, allowed writes, and human gates before use.

On 2026-07-13, the human made `agent-assisted-publish` the default for code-changing tasks. The standing repository instruction authorizes commit, task-branch push, and draft-PR create/update after all safety and verifier gates pass. `Local-only`, `Do not publish`, read-only/review-only tasks, and no-diff outcomes do not publish. New draft-PR creation removes the clean linked worktree but retains the branch and PR. This does not enable scheduling, auto-merge, or mutation by recurring L1 loops.

## Active Loops

Loop IDs must match `LOOP.md` and `patterns/registry.yaml`.

| Loop ID | Status | Trigger | Level | Notes |
| --- | --- | --- | --- | --- |
| `daily-triage` | Active | Manual or at most once per day | L1 report-only | Reads local repo signals and updates state/logs only. |
| `ci-sweeper-manual` | Active | Manual or after failed CI | L1 report-only | Summarizes CI/local check failures; no fixes. |
| `eval-regression-check` | Active | Manual before eval changes | L1 report-only | Summarizes dataset audit or drift signals; provider-backed evals require approval. |
| `ingestion-promotion-review` | Active | Manual before promotion | L1 report-only | Summarizes baseline/shadow checks; no promotion. |
| `post-merge-cleanup` | Candidate | Manual after merge review | L1 report-only | Report-only in L1; any cleanup fix needs later L2 approval. |
| `agent-assisted-publish` | Active | Code-changing task without opt-out | L2 assisted | Fresh `codex/*` worktree, scoped commit, independent verification, task-branch push, draft PR, then clean linked-worktree removal. |

## Current Findings

- `daily-triage`: Pre-run dirty baseline included human-gated source/dependency/eval paths, but post-run status narrowed to `.DS_Store`, `STATE.md`, `evals/questions.jsonl`, `loop-run-log.md`, `tests/fixtures/eval_corpus_chunks.jsonl`, and untracked docs/plans. `./scripts/check.sh` passed; remaining eval-file changes are outside L1 loop ownership and need an explicit review/implementation workflow before commit/push.
- `ci-sweeper-manual`: Local check gate passed (`./scripts/check.sh`: 477 passed, 2 skipped), so there is no failing CI/local-check item to map to a fix loop.
- `eval-regression-check`: Strict eval audit passed with 21 rows kept and 0 warnings/errors; no provider-backed eval, judge calibration, or dataset edit was performed.
- `ingestion-promotion-review`: Full promotion review skipped because no baseline/shadow roots were provided; no `.env`, `.imperial_rag/`, or `documents/` changes appeared in the scoped status check.

## Watch List

- Loop scaffold active for report-only daily triage, CI sweeps, eval drift checks, and ingestion promotion reviews; post-merge cleanup remains candidate.
- Baseline verifier gate: `./scripts/check.sh`.
- `./scripts/check.sh` is green but emitted resource/deprecation warnings; monitor if they become blocking or point to a concrete leak.
- Operator CLI cadence for `daily-triage` is pinned to manual or exactly `1d`; do not use the upstream default `1d-2h` window for Imperial.
- Before any manual run, capture `git status --short`; on 2026-07-02 the baseline already included modified `STATE.md` and `loop-run-log.md`, plus unrelated local dirt, so preserve in-flight changes and compare the post-run diff against allowed writes.
- Pre-existing unrelated local dirt remains outside loop ownership: `.DS_Store` is modified, and local council/planning docs are untracked.
- RAG-specific gates to reference when relevant:
  - `uv run python scripts/audit_eval_rows.py --strict --output-path <tmp>`
  - `uv run python scripts/check_ingestion_promotion.py --baseline-root <path> --shadow-root <path>`
  - `uv run python scripts/validate_phoenix_trace.py --run-id <id> --json`

## Noise / Ignore

- Generated local runtime state is private and should be summarized by path and status only, not copied into state.
- No connectors, provider-backed evals, live ingestion, Docker restarts, or Phoenix queries were used for this L1 run.
- `./scripts/check.sh` was run for this loop batch and passed; use it again as the verifier gate for approved source edits.
- `loop-audit` stayed green at 100/100 readiness; treat that as scaffold readiness, not permission for unattended mutation.
- Ingestion promotion checker was not run because baseline/shadow roots were absent.

## Human Decisions

- 2026-06-30: Start Imperial loop engineering in L1 report-only mode.
- Recurring loops have no auto-fix or auto-push; auto-merge is always disabled.
- 2026-07-13: Make L2 commit, task-branch push, and draft-PR create/update the default for code-changing tasks after all publish gates; allow `Local-only` and `Do not publish` opt-outs and remove clean linked worktrees after new PR creation.
- 2026-07-02: Pin `daily-triage` to manual or once daily; keep high-frequency CI sweeps disabled.
- 2026-07-02: Promote `ci-sweeper-manual`, `eval-regression-check`, and `ingestion-promotion-review` to active manual L1 report-only loops on explicit human request.

## Pause State

Pause flag: none.

If `loop-pause-all` appears here or in `loop-budget.md`, every loop must exit after appending a skipped-run entry to `loop-run-log.md`.

## Recent Runs

| Run ID | Loop ID | Level | Outcome | Notes |
| --- | --- | --- | --- | --- |
| 2026-07-02T16:40:20+0900 | `ingestion-promotion-review` | L1 | Skipped | `CONSTRAINTS_REPORT_ONLY`; `BUDGET_OK` early-exit estimate ~93k/100k daily, 0 subagents; no baseline/shadow roots provided; no generated corpus or document state touched. |
| 2026-07-02T16:40:12+0900 | `eval-regression-check` | L1 | Success | `CONSTRAINTS_REPORT_ONLY`; `BUDGET_OK` ~90.5k/100k daily estimate, 0 subagents; strict audit checked 21 rows, keep 21, 0 warnings/errors. |
| 2026-07-02T16:40:01+0900 | `ci-sweeper-manual` | L1 | Success | `CONSTRAINTS_REPORT_ONLY`; `BUDGET_OK` ~78.5k/100k daily estimate, 0 subagents; `./scripts/check.sh` passed, so no failing local check item to map. |
| 2026-07-02T16:39:49+0900 | `daily-triage` | L1 | Success | `CONSTRAINTS_REPORT_ONLY`; `BUDGET_OK` ~69k/100k daily estimate, 0 subagents; pre-run dirty baseline noted; post-run status narrowed; local check and loop audit passed. |
| 2026-07-02T01:19:03+0900 | `daily-triage` | L1 | Success | `CONSTRAINTS_REPORT_ONLY`; `BUDGET_OK` at ~46k/100k daily estimate, 0 subagents; `loop-sync` 100/100 healthy; `loop-audit` 100/100 readiness; privacy/scope diff passed. |
| 2026-07-02T00:52:49+0900 | `daily-triage` | L1 | Success | `CONSTRAINTS_REPORT_ONLY`; `BUDGET_OK` at 23k/100k tokens; `loop-sync` clean; state/log privacy diff passed. |
