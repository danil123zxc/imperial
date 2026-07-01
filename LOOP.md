# Loop Configuration - Imperial RAG

State: STATE.md.
State file: `STATE.md`.
Pattern registry: `patterns/registry.yaml`.
Mode: L1 report-only.

Loop IDs are stable coordination keys. They must match `STATE.md`, `patterns/registry.yaml`, and loop skill output.

## Active Loops

| Loop ID | Cadence / trigger | Status | Command / prompt |
| --- | --- | --- | --- |
| `daily-triage` | Manual or exactly once daily (`1d`) | Active L1 report-only | Run `$loop-constraints`, `$loop-budget`, then `$loop-triage`. Update `STATE.md` and `loop-run-log.md`; do not edit source. |
| `ci-sweeper-manual` | Manual or after failed CI only | Planned L1 report-only | Run `$loop-constraints`, `$loop-budget`, then `$ci-sweeper-manual`. Read CI/local-check failures, map them to local verifiers, and write findings only. |
| `eval-regression-check` | Manual before eval changes | Planned L1 report-only | Run `$loop-constraints`, `$loop-budget`, then `$eval-regression-check`. Audit eval dataset drift without provider-backed runs or dataset edits. |
| `ingestion-promotion-review` | Manual before promotion | Planned L1 report-only | Run `$loop-constraints`, `$loop-budget`, then `$ingestion-promotion-review`. Compare approved baseline/shadow context; never promote artifacts. |
| `post-merge-cleanup` | Manual after merge review | Candidate L1 report-only | Summarize follow-up cleanup only; any source edit requires later L2 approval. |

## Enablement Terms

- `daily-triage` is the only active loop. All other configured loops remain planned or candidate until a human explicitly promotes them.
- "Enable" means allowing a manual L1 report-only run after trigger, pause, cadence, budget, write-scope, and privacy gates pass.
- "Enable" does not mean scheduling, promotion to active, connector access, source edits, dependency changes, provider-backed evals, ingestion promotion, or PR/GitHub writes.
- Future loop ideas such as `dependency-sweeper`, `issue-triage`, `changelog-drafter`, and `pr-babysitter` are not configured loops until they have registry entries, budgets, connector policy, allowed writes, and human gates.

## Current Findings

- None yet. `STATE.md` is the durable source for active findings.

## Safety Gates

- Auto-merge is disabled.
- Auto-push is disabled.
- Report-only loops may write only `STATE.md`, `loop-run-log.md`, and clearly scoped loop reports unless the user explicitly asks for implementation.
- Any source edit, dependency change, generated corpus rewrite, provider-backed eval run, or runtime restart requires human approval in the active thread.
- High-risk paths and data-egress rules are binding in `loop-constraints.md` and `docs/safety.md`.
- Report-only output must pass a privacy diff review before a loop is considered successful.
- Every manual run must record pre/post `git status --short`, branch, and SHA. If `STATE.md` or `loop-run-log.md` are already modified, preserve the pre-existing entries and explicitly report the dirty baseline.
- L1 success requires a post-run allowed-path diff check. Expected L1 writes are `STATE.md` and `loop-run-log.md` unless the active thread authorizes a narrower report artifact.
- `daily-triage` already ran twice on 2026-07-02; rerun it the same day only on explicit request, and early-exit with a short skipped/no-signal entry when no new signal exists.

## Operator Tooling

- Do not add `loop-engineering` packages to the Python runtime or project dependencies.
- Before scheduled use, record and pin exact `@cobusgreyling/*` CLI versions. Current checked versions: `@cobusgreyling/loop-audit@1.5.2`, `@cobusgreyling/loop-sync@1.0.0`, `@cobusgreyling/loop-cost@1.0.3`.
- Run operator CLIs without exporting `.env` secrets into the command environment.
- Re-check package metadata before scheduled usage or upgrades; unpinned `npx` is allowed only for manual operator verification.

## Watch List

- Keep high-frequency CI sweeps disabled for Imperial. The upstream `ci-sweeper` cadence is too expensive unless it is event-driven and early-exits.
- Treat `loop-audit` `L3` as scaffold readiness only, not permission for unattended mutation.

## Noise / Ignore

- Do not copy private generated state, corpus text, traces, auth data, chat history, provider prompts, or eval outputs into loop state/logs.

## Worktrees

- Use one isolated branch or worktree per assisted fix after L2 is approved.
- The implementer cannot verify its own work.
- The verifier must inspect the diff, confirm no denylist paths changed, and run the relevant checks before a PR or commit is proposed.
- Stop after three failed attempts on the same item and escalate with evidence.

## Connectors (MCP)

- MCP is optional for L1 report-only loops.
- No connectors are required for the first manual L1 run.
- GitHub access, when enabled, should be read-only by default and limited to CI/PR metadata; write scope is limited to comments or draft PRs after explicit approval.
- Do not attach Slack, Drive, Vercel, database, or analytics connectors to L1 loops without a separate approval and scope review.
- No connector should receive raw `.env`, corpus documents, Phoenix traces, auth databases, or private eval outputs.

## Budget

- Daily L1 cap: 100k tokens.
- Daily L2 cap, once approved: 500k tokens.
- Max sub-agent spawns per L1 run: 0.
- Max sub-agent spawns per L2 run: 2.
- Kill switch: if `loop-pause-all` appears in `STATE.md` or `loop-budget.md`, every loop exits after writing a short skipped-run log entry.
- At 80% of the daily cap, switch to report-only for the rest of the day.
- `daily-triage` must be scheduled as manual or `1d`, not the upstream default `1d-2h`.
- Empty or no-signal runs should early-exit after writing a short run-log entry.

## Verification Gates

- Default local quality gate: `./scripts/check.sh`.
- Eval dataset gate: `uv run python scripts/audit_eval_rows.py --strict --output-path <tmp>`.
- Ingestion promotion gate: `uv run python scripts/check_ingestion_promotion.py --baseline-root <path> --shadow-root <path>`.
- Phoenix trace-shape gate: `uv run python scripts/validate_phoenix_trace.py --run-id <id> --json`.
- Provider-backed or live-service gates require explicit human approval because they can touch private traces, generated artifacts, or paid APIs.

## Human Decisions

- 2026-06-30: Start Imperial loop engineering in L1 report-only mode.
- 2026-07-02: Keep `daily-triage` manual or once daily; keep `ci-sweeper-manual` event-driven/manual only.

## Pause State

- Current pause flag: none.
- If unsafe state is written, add `loop-pause-all`, stop scheduled runs, redact or revert unsafe output, record a short post-mortem in `STATE.md`, and tighten constraints before resuming.

## Recent Runs

- See `STATE.md` and `loop-run-log.md`.
