---
name: ci-sweeper-manual
description: Summarize Imperial RAG CI or local check failures in manual L1 report-only loop runs.
user_invocable: true
---

# CI Sweeper Manual Skill - Imperial RAG

Map failed CI or local check evidence to concise report-only findings. This loop does not fix failures.

## Required Inputs

- `STATE.md`
- `LOOP.md`
- `patterns/registry.yaml`
- `loop-constraints.md`
- `loop-budget.md`
- `loop-run-log.md`
- `docs/safety.md`
- Current `git status --short`
- Current branch name and SHA
- Failed GitHub Actions metadata or failed local check output

## Trigger Gate

- Run only after `$loop-constraints` and `$loop-budget`.
- Continue only when the human explicitly requests this loop or provides failed CI/local-check evidence.
- GitHub access is read-only CI/PR metadata only and requires explicit approval before connector use.
- If CI is green, no failed run is provided, or the only signal is unrelated local dirt, early-exit with a skipped/no-signal entry.

## Report-Only Rules

- You may update `STATE.md` and `loop-run-log.md` only.
- Do not edit source, tests, workflows, dependencies, generated artifacts, or runtime configuration.
- Do not rerun expensive or live-service checks unless the human explicitly asks.
- Do not paste secrets, raw documents, extracted chunks, Phoenix payloads, provider prompts, auth/chat rows, or eval outputs.
- Use paths, job names, command names, exit codes, test names, counts, and redacted snippets only.

## Triage Steps

1. Record the dirty baseline with pre-run `git status --short`, branch, and SHA.
2. Identify the failing command, job, test file, or check stage.
3. Map each failure to the closest local verifier, usually `./scripts/check.sh` or a focused pytest command already shown in the failure.
4. Separate actionable failures from infrastructure/noise. Escalate security, workflow, dependency, or connector-write issues.
5. Write concise findings with impact, suggested next verifier, and required human gate.
6. Append a run entry to `loop-run-log.md` with inputs read, changed files, budget verdict, privacy review, verifier status, token estimate, and outcome.
7. Re-check `git status --short` and confirm L1 writes stayed inside `STATE.md` and `loop-run-log.md`.

## Output

Update `STATE.md` sections as needed:

- `## Current Findings`: failing check, impact, next verifier, human gate.
- `## Watch List`: lower-risk or flaky signals to revisit.
- `## Noise / Ignore`: green checks, unrelated dirt, or non-actionable infrastructure noise.
- `## Recent Runs`: one short summary row.

If the failure needs source edits, dependency changes, workflow changes, credentials, or connector writes, record `ESCALATE_HUMAN` instead of implementing.
