# Loop Budget - Imperial RAG

Status: active for L1 report-only loops.
Kill switch: inactive.

## Daily Caps

| Level | Max tokens/day | Max sub-agents/run | Allowed action |
| --- | ---: | ---: | --- |
| L1 report-only | 100k | 0 | Triage, summarize, update state/logs |
| L2 assisted fixes | 500k | 2 | Draft minimal fixes in isolated worktrees after approval |
| L3 unattended | Not approved | 0 | Not allowed |

## Runtime Rules

- If estimated usage reaches 80% of the daily cap, switch to report-only.
- If estimated usage reaches 100% of the daily cap, stop and write a skipped-run entry.
- Empty or no-signal runs should exit quickly after updating `loop-run-log.md`.
- CI sweeper loops must early-exit when CI is green.
- Provider-backed evals, live Phoenix checks, ingestion promotion, and service restarts do not run on a schedule without explicit approval.

## Pause Controls

Write `loop-pause-all` below to pause all loops.

```text
pause flag: none
```
