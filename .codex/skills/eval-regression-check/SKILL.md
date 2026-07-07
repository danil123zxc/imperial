---
name: eval-regression-check
description: Summarize Imperial RAG eval dataset drift or audit risk in manual L1 report-only loop runs.
user_invocable: true
---

# Eval Regression Check Skill - Imperial RAG

Review eval-change readiness without running provider-backed evals or editing datasets.

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
- Eval-related diff, planned eval change, or explicit human request

## Trigger Gate

- Run only after `$loop-constraints` and `$loop-budget`.
- Continue only before eval changes, after eval-related diffs, or on explicit human request.
- Do not run provider-backed evals, judge calibration, Phoenix-backed evaluation, or paid APIs without approval.
- Do not edit `evals/questions.jsonl`, calibration data, prompts, or provider settings in L1.
- If no eval signal is present, early-exit with a skipped/no-signal entry.

## Report-Only Rules

- You may update `STATE.md` and `loop-run-log.md` only.
- Treat eval rows and outputs as private. Report row IDs, counts, statuses, and file paths; do not paste raw expected answers, private corpus text, or judge outputs.
- Prefer existing local evidence. Run `uv run python scripts/audit_eval_rows.py --strict --output-path <tmp>` only when the human requested the eval check or eval files are already in scope.
- Write audit outputs to a temporary path unless the human explicitly asks for a committed artifact.

## Review Steps

1. Record the dirty baseline with pre-run `git status --short`, branch, and SHA.
2. Identify changed eval files or the requested eval surface.
3. Check for missing `reference_context_ids`, weak refusal boundaries, provider-backed requirements, or dataset edits that need approval.
4. If running the strict audit gate, summarize only counts, row IDs, and failure categories.
5. Write findings with impact, suggested verifier, and human gate.
6. Append a run entry to `loop-run-log.md` with inputs read, changed files, budget verdict, privacy review, verifier status, token estimate, and outcome.
7. Re-check `git status --short` and confirm L1 writes stayed inside `STATE.md` and `loop-run-log.md`.

## Output

Update `STATE.md` sections as needed:

- `## Current Findings`: eval drift, strict-audit failure category, missing evidence, or approval-needed item.
- `## Watch List`: lower-risk dataset hygiene or future calibration work.
- `## Noise / Ignore`: non-eval diffs, optional checks not run, or already-gated provider work.
- `## Recent Runs`: one short summary row.

If the next step is a dataset edit, provider-backed eval, judge calibration change, or Phoenix-backed evaluation, record `ESCALATE_HUMAN` instead of taking the action.
