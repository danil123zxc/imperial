---
name: loop-budget
description: Enforce Imperial RAG loop token caps, sub-agent limits, and pause controls.
user_invocable: true
---

# Loop Budget Skill - Imperial RAG

Read `loop-budget.md` and `LOOP.md` before every loop run.

## Required Checks

- If `loop-pause-all` is present in `STATE.md` or `loop-budget.md`, exit after writing a skipped-run entry to `loop-run-log.md`.
- For L1 report-only loops, do not spawn sub-agents.
- For L2 assisted-fix loops, stop at two sub-agents per run unless the human explicitly approves more.
- If estimated daily usage is above 80% of the active cap, switch to report-only.
- If estimated daily usage is above 100% of the active cap, stop and log the skipped run.

## Output

State one of:

- `BUDGET_OK`
- `BUDGET_REPORT_ONLY`
- `BUDGET_STOP`

Include the active cap, estimated usage if known, and the reason.
