# Loop Configuration - Imperial RAG

## Active Loops

| Pattern | Cadence | Status | Command / prompt |
| --- | --- | --- | --- |
| Daily triage | Daily during active development | L1 report-only | Run `$loop-constraints`, `$loop-budget`, then `$loop-triage`. Update `STATE.md` and `loop-run-log.md`; do not edit source. |
| CI sweeper | Manual or after failed CI | Planned L1 report-only | Read GitHub Actions / local check results, map failures to `./scripts/check.sh`, and write findings only. |
| Eval regression check | Manual before eval changes | Planned L1 report-only | Audit eval dataset quality and summarize drift; provider-backed runs require human approval. |
| Ingestion promotion review | Manual before promotion | Planned L1 report-only | Compare baseline and shadow artifacts with `scripts/check_ingestion_promotion.py`; no direct promotion. |

## Safety Gates

- Auto-merge is disabled.
- Auto-push is disabled.
- Report-only loops may write only `STATE.md`, `loop-run-log.md`, and clearly scoped loop reports unless the user explicitly asks for implementation.
- Any source edit, dependency change, generated corpus rewrite, provider-backed eval run, or runtime restart requires human approval in the active thread.
- High-risk paths and data-egress rules are binding in `loop-constraints.md` and `docs/safety.md`.

## Worktrees

- Use one isolated branch or worktree per assisted fix after L2 is approved.
- The implementer cannot verify its own work.
- The verifier must inspect the diff, confirm no denylist paths changed, and run the relevant checks before a PR or commit is proposed.
- Stop after three failed attempts on the same item and escalate with evidence.

## Connectors (MCP)

- MCP is optional for L1 report-only loops.
- GitHub access, when enabled, should be read-only by default; write scope is limited to comments or draft PRs after explicit approval.
- No connector should receive raw `.env`, corpus documents, Phoenix traces, auth databases, or private eval outputs.

## Budget

- Daily L1 cap: 100k tokens.
- Daily L2 cap, once approved: 500k tokens.
- Max sub-agent spawns per L1 run: 0.
- Max sub-agent spawns per L2 run: 2.
- Kill switch: if `loop-pause-all` appears in `STATE.md` or `loop-budget.md`, every loop exits after writing a short skipped-run log entry.
- At 80% of the daily cap, switch to report-only for the rest of the day.

## Verification Gates

- Default local quality gate: `./scripts/check.sh`.
- Eval dataset gate: `uv run python scripts/audit_eval_rows.py --strict --output-path <tmp>`.
- Ingestion promotion gate: `uv run python scripts/check_ingestion_promotion.py --baseline-root <path> --shadow-root <path>`.
- Phoenix trace-shape gate: `uv run python scripts/validate_phoenix_trace.py --run-id <id> --json`.
- Provider-backed or live-service gates require explicit human approval because they can touch private traces, generated artifacts, or paid APIs.
