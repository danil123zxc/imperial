---
name: loop-triage
description: Triage Imperial RAG project health in report-only loop runs.
user_invocable: true
---

# Loop Triage Skill - Imperial RAG

You produce concise, actionable findings for Imperial RAG automation loops.

## Required Inputs

- `STATE.md`
- `LOOP.md`
- `loop-constraints.md`
- `loop-budget.md`
- `loop-run-log.md`
- `docs/safety.md`
- Current `git status --short`
- Current branch name
- Recent CI/check output when available

## Report-Only Rules

- Week one is L1 report-only.
- You may update `STATE.md` and `loop-run-log.md` only.
- Do not edit source, generated artifacts, tests, workflows, dependencies, or runtime configuration.
- Do not run provider-backed evals, live ingestion, Docker restarts, or Phoenix queries unless the human explicitly asked in the active thread.
- Do not include secrets, raw document text, chat history, Phoenix payloads, provider prompts, eval outputs, or extracted chunks in reports.

## Triage Surfaces

Prioritize:

- Failed GitHub Actions or local `./scripts/check.sh` results.
- Dirty worktree changes that touch high-risk files.
- Eval dataset drift or strict audit failures.
- Ingestion promotion risk between baseline and shadow state.
- Phoenix trace-shape regressions only when a trace run ID is provided.
- Stale items in `STATE.md` that need human attention.

## Output Shape

Write or update these sections in `STATE.md`:

```markdown
## High Priority
- Finding, impact, suggested next action, verifier command, human gate if needed.

## Watch List
- Lower urgency item, why it is being watched, when to revisit.

## Noise / Ignore
- Signals checked and intentionally ignored.

## Recent Runs
- Append one short run summary.
```

Append the same run to `loop-run-log.md`.

## Priority Rules

- High Priority means a human should know today.
- Watch List means monitor, but do not act yet.
- Noise means checked and not worth action.
- Do not invent architecture work during triage.
- If uncertain, place the item in Watch or escalate to human.
