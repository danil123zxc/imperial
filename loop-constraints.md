# Loop Constraints - Imperial RAG

The `loop-constraints` skill must read this file at the start of every loop run. These constraints are binding.

## Mode

- Week one is L1 report-only.
- Report-only loops may update `STATE.md` and `loop-run-log.md`.
- Do not implement fixes, rewrite files, run formatters, stage changes, commit, push, open PRs, or restart services unless the human explicitly asks in the active thread.

## Run Hygiene

- Before every manual loop run, capture `git status --short`, the current branch, and the current SHA.
- If `STATE.md` or `loop-run-log.md` are already modified, treat them as in-flight state, preserve existing entries, and report that the loop state files were dirty before the run.
- After every loop run, capture `git status --short` again and inspect the diff. L1 success requires changes to stay within the allowed write paths and pass the privacy review.
- If same-day `daily-triage` entries already exist, rerun only on explicit human request; otherwise wait for a later-day cadence window. Early-exit when there is no new signal.

## Push & Merge

- Never auto-merge.
- Never push without telling the human first and receiving approval.
- Draft PRs are allowed only after the project has graduated to L2 assisted fixes.

## Denylist Paths

Never auto-edit these paths:

```text
.env
.env.*
.DS_Store
.imperial_rag/**
documents/**
**/secrets/**
**/credentials/**
**/*_key*
**/*_secret*
**/auth.sqlite3
**/chat_history.sqlite3
**/eval_outputs/**
**/phoenix/**
```

Human approval is also required before editing:

```text
compose.yaml
Dockerfile
uv.lock
.github/workflows/**
pyproject.toml
evals/questions.jsonl
evals/russian_judge_calibration.jsonl
scripts/ingest.py
scripts/run_*eval*.py
src/imperial_rag/observability/**
src/imperial_rag/app/**
src/imperial_rag/ingestion/**
src/imperial_rag/retrieval/**
src/imperial_rag/answering/**
```

## Data Egress

- Do not paste document text, extracted chunks, chat history, auth rows, Phoenix spans, provider errors containing prompts, or eval outputs into external tools.
- When reporting on private artifacts, cite counts, paths, checksums, IDs, or redacted snippets only.
- Do not include secrets, credentials, API keys, passwords, cookies, or bearer tokens in state, run logs, prompts, PRs, or comments.

## Verification

- Never disable or weaken tests to make a loop green.
- Never increase timeouts without a root-cause note.
- Use `./scripts/check.sh` as the default verifier gate for source changes.
- Use focused RAG gates only when relevant and approved:
  - `uv run python scripts/audit_eval_rows.py --strict --output-path <tmp>`
  - `uv run python scripts/check_ingestion_promotion.py --baseline-root <path> --shadow-root <path>`
  - `uv run python scripts/validate_phoenix_trace.py --run-id <id> --json`
## Escalation

- Escalate after three failed attempts on the same item.
- Escalate immediately for security, auth, privacy, provider billing, deployment, ingestion promotion, dependency upgrades, or broad refactors.
- If `loop-pause-all` appears in `STATE.md` or `loop-budget.md`, exit immediately after logging the skipped run.
