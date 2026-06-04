# Ragas Eval Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a first Ragas evaluation runner that reuses the existing Imperial RAG runtime and JSONL gold questions.

**Architecture:** Keep Phoenix as the existing deterministic eval and observability path. Add a sidecar `scripts/run_ragas_eval.py` that prepares Ragas rows from runtime outputs, runs Ragas metrics locally, and writes optional JSONL results without changing the Phoenix runner.

**Tech Stack:** Python, pytest, uv, Ragas, LangChain model wrapper, Imperial runtime helpers.

---

### Task 1: Pin The Runner Contract With Tests

**Files:**
- Modify: `tests/test_dependencies.py`
- Create: `tests/test_ragas_eval.py`

- [ ] **Step 1: Add a dependency test**

Add a test asserting `ragas` is declared under `project.optional-dependencies.dev`.

- [ ] **Step 2: Add runner behavior tests**

Create tests that load `scripts/run_ragas_eval.py`, verify it imports without Ragas at module import time, prepares supported rows from fake runtime outputs, rejects reference-required metrics when `reference_answer` is missing, and calls an injected `evaluate` function with dataset, metric objects, and evaluator LLM.

- [ ] **Step 3: Run the new tests and confirm RED**

Run: `uv run python -m pytest tests/test_dependencies.py tests/test_ragas_eval.py -q`

Expected: failure because `ragas` is not declared and `scripts/run_ragas_eval.py` does not exist.

### Task 2: Implement The Ragas Sidecar

**Files:**
- Create: `scripts/run_ragas_eval.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `ragas` as a dev dependency**

Declare `ragas` in `[project.optional-dependencies].dev`.

- [ ] **Step 2: Create the runner**

Implement a CLI with `--questions-path`, `--workspace-root`, `--metrics`, and `--output-path`. Reuse `run_phoenix_eval.load_questions`, `build_runtime`, and `run_target` to avoid a second RAG invocation path.

- [ ] **Step 3: Prepare Ragas rows**

Build rows with `user_input`, `response`, `retrieved_contexts`, `expected_behavior`, and optional `reference`. Skip `refuse_if_not_found` rows and rows without retrieved contexts.

- [ ] **Step 4: Run Ragas**

Support `faithfulness` immediately. Gate `factual_correctness` and `context_recall` behind `reference_answer` so the runner does not fabricate ground truth.

### Task 3: Verification And Docs

**Files:**
- Modify: `README.md`
- Modify: `uv.lock`

- [ ] **Step 1: Update docs**

Document the deterministic Phoenix runner and the new Ragas quality runner separately.

- [ ] **Step 2: Refresh lockfile**

Run: `uv lock`

- [ ] **Step 3: Verify**

Run:

```bash
uv run python -m pytest tests/test_dependencies.py tests/test_ragas_eval.py -q
uv run python -m pytest tests/test_evals.py tests/test_scripts.py -q
uv run python -m pytest -q
git diff --check
```

- [ ] **Step 4: Commit scoped changes**

Stage only files changed for this Ragas eval runner and commit with:

```bash
git commit -m "feat: add ragas eval runner"
```
