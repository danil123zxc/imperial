# Loop Constraints - Imperial RAG

The `loop-constraints` skill must read this file at the start of every loop run. These constraints are binding.

## Mode

- Recurring loops remain L1 report-only.
- L2 assisted publishing is the default for code-changing tasks unless the task is read-only/review-only, produces no changes, or contains `Local-only` or `Do not publish`.
- Report-only loops may update `STATE.md` and `loop-run-log.md`.
- Do not implement fixes, rewrite files, run formatters, stage changes, commit, push, open PRs, or restart services unless the human explicitly asks in the active thread.

## Run Hygiene

- Before every manual loop run, capture `git status --short`, the current branch, and the current SHA.
- If `STATE.md` or `loop-run-log.md` are already modified, treat them as in-flight state, preserve existing entries, and report that the loop state files were dirty before the run.
- After every loop run, capture `git status --short` again and inspect the diff. L1 success requires changes to stay within the allowed write paths and pass the privacy review.
- If same-day `daily-triage` entries already exist, rerun only on explicit human request; otherwise wait for a later-day cadence window. Early-exit when there is no new signal.

## Push & Merge

- Never auto-merge.
- The standing repository instruction authorizes pushing the verified task's scoped `codex/*` branch and creating/updating its draft PR. It does not authorize high-risk paths, ready-for-review transitions, or merges.
- Before publishing, require a fresh branch based on current `origin/main`, a clean worktree, scoped commits, `./scripts/check.sh`, a base-range whitespace check, and an independent verifier `APPROVE` verdict.
- Refuse `main`, `dev`, detached HEADs, non-`codex/*` branches, and branches associated with merged or closed PRs.
- Draft PR create/update is the only allowed automated GitHub write. Publishing to a ready-for-review PR or merging requires a new human action.
- Use `scripts/publish_agent_pr.sh --verifier-approved` as the final publish boundary. The publisher always targets `main`; its base is not operator-configurable.
- After creating a new draft PR, the publisher may remove only its clean linked worktree, without `--force`. It must never remove the primary worktree, and it must retain the branch and PR. Existing draft-PR updates retain their worktree.

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
- Before an L2 push, also run `git diff --check origin/main...HEAD` and scan the complete base diff for denylisted paths.
- More than 10 changed files or any human-gated path requires separate explicit approval; default publishing does not override these gates.
- Use focused RAG gates only when relevant and approved:
  - `uv run python scripts/audit_eval_rows.py --strict --output-path <tmp>`
  - `uv run python scripts/check_ingestion_promotion.py --baseline-root <path> --shadow-root <path>`
  - `uv run python scripts/validate_phoenix_trace.py --run-id <id> --json`
## Escalation

- Escalate after three failed attempts on the same item.
- Escalate immediately for security, auth, privacy, provider billing, deployment, ingestion promotion, dependency upgrades, or broad refactors.
- If `loop-pause-all` appears in `STATE.md` or `loop-budget.md`, exit immediately after logging the skipped run.
