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
- `patterns/registry.yaml`
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
- Use stable loop IDs from `LOOP.md` and `patterns/registry.yaml`.
- `daily-triage` may run manually or at most once per day. Do not use high-frequency cadence.

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
## Active Loops
- Keep loop IDs, status, trigger, and level aligned with `LOOP.md`.

## Current Findings
- Finding, impact, suggested next action, verifier command, human gate if needed.

## Watch List
- Lower urgency item, why it is being watched, when to revisit.

## Noise / Ignore
- Signals checked and intentionally ignored.

## Recent Runs
- Append one short run summary.

## Pause State
- Keep pause flag explicit. If `loop-pause-all` appears, stop after a skipped-run log entry.
```

Append the same run to `loop-run-log.md` with run ID, loop ID, branch/SHA, inputs read, changed files, budget verdict, privacy review, verifier result, token estimate, and outcome.

## First-Run Acceptance Checklist

- Record `CONSTRAINTS_*` and `BUDGET_*` verdicts.
- Record the command or prompt used, without secrets or private content.
- Record which inputs were read by filename only.
- Confirm changed files are limited to `STATE.md` and `loop-run-log.md` for report-only runs.
- Inspect the state/log diff for private corpus text, extracted chunks, chat history, auth rows, Phoenix payloads, provider prompts, eval outputs, secrets, cookies, and bearer tokens.
- If a check cannot run, record `ESCALATE_HUMAN` instead of approving the run.

## Priority Rules

- High Priority means a human should know today.
- Watch List means monitor, but do not act yet.
- Noise means checked and not worth action.
- Do not invent architecture work during triage.
- If uncertain, place the item in Watch or escalate to human.
