---
name: ingestion-promotion-review
description: Summarize Imperial RAG baseline versus shadow ingestion promotion risk in manual L1 report-only loop runs.
user_invocable: true
---

# Ingestion Promotion Review Skill - Imperial RAG

Review promotion readiness between baseline and shadow ingestion artifacts without promoting or rewriting corpus state.

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
- Human-provided baseline and shadow artifact roots, or explicit promotion-review context

## Trigger Gate

- Run only after `$loop-constraints` and `$loop-budget`.
- Continue only when baseline/shadow context is provided or the human explicitly requests an ingestion promotion review.
- Do not run ingestion, OCR, indexing, artifact promotion, Docker restarts, or provider calls in L1.
- Do not write `.imperial_rag/`, `documents/`, indexes, manifests, or generated corpus artifacts.
- If baseline/shadow context is missing, early-exit with a skipped/no-signal entry.

## Report-Only Rules

- You may update `STATE.md` and `loop-run-log.md` only.
- Treat corpus state as private. Report paths, counts, IDs, checksums, and pass/fail statuses; do not paste raw document text, extracted chunks, OCR text, prompts, or traces.
- Run `uv run python scripts/check_ingestion_promotion.py --baseline-root <path> --shadow-root <path>` only when both roots are provided and the human approved the review.
- Summarize verifier output by status and category, not by private content.

## Review Steps

1. Record the dirty baseline with pre-run `git status --short`, branch, and SHA.
2. Confirm baseline and shadow roots, or early-exit if they are absent.
3. Check that no generated corpus, document, index, or manifest path is being modified by the loop.
4. If running the promotion checker, quote the command name and summarize counts/statuses only.
5. Write findings with impact, suggested verifier, and human gate.
6. Append a run entry to `loop-run-log.md` with inputs read, changed files, budget verdict, privacy review, verifier status, token estimate, and outcome.
7. Re-check `git status --short` and confirm L1 writes stayed inside `STATE.md` and `loop-run-log.md`.

## Output

Update `STATE.md` sections as needed:

- `## Current Findings`: promotion blocker, missing baseline/shadow evidence, or approval-needed item.
- `## Watch List`: lower-risk artifact drift or future promotion prerequisite.
- `## Noise / Ignore`: absent roots, unrelated dirt, or checks intentionally skipped.
- `## Recent Runs`: one short summary row.

If the next step is ingestion, OCR, vector/keyword index replacement, artifact promotion, or generated-state edits, record `ESCALATE_HUMAN` instead of taking the action.
