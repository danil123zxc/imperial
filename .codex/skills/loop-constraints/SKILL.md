---
name: loop-constraints
description: Enforce Imperial RAG denylist paths, data-egress rules, and human gates for loops.
user_invocable: true
---

# Loop Constraints Skill - Imperial RAG

Read `loop-constraints.md` and `docs/safety.md` at the start of every loop run.

## Required Checks

- Confirm the current loop mode.
- Check whether `loop-pause-all` appears in `STATE.md` or `loop-budget.md`.
- Enforce denylist paths before any proposed edit.
- Enforce data-egress rules before writing state, comments, reports, prompts, or PR text.
- Escalate immediately for security, auth, privacy, deployment, dependency, ingestion promotion, provider-backed eval, or broad refactor work.

## Output

State one of:

- `CONSTRAINTS_OK`
- `CONSTRAINTS_REPORT_ONLY`
- `CONSTRAINTS_STOP`
- `ESCALATE_HUMAN`

Include the rule that caused the result.
