# Live RAG Smoke And Integration Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in smoke and integration tests that prove Imperial RAG works with real DashScope/Qwen API calls while keeping normal pytest offline.

**Architecture:** Add a small `tests/live_support.py` helper for `.env` loading, opt-in gates, live-service checks, and generated-corpus validation. Build three focused live test modules on top of it: provider smoke, fixture-corpus integration, and real-corpus integration. Keep live tests skipped unless explicit environment flags are set.

**Tech Stack:** Python, pytest, pathlib, socket, DashScope/Qwen provider wrappers, LangChain `Document`, PIL, Imperial ingestion/runtime APIs.

---

## File Structure

- Create `tests/live_support.py`: shared helper for live test gates, `.env` loading, Qdrant reachability, and generated-corpus checks.
- Create `tests/test_live_support.py`: offline unit tests for `tests/live_support.py`; these run in the normal suite.
- Create `tests/test_live_provider_smoke.py`: opt-in real DashScope/Qwen chat, embeddings, rerank, and OCR smoke tests.
- Create `tests/test_live_rag_integration.py`: opt-in temporary fixture corpus integration using real chat/rerank API calls.
- Create `tests/test_live_real_corpus.py`: opt-in health check against the current `.imperial_rag` Imperial corpus state.
- Modify `.env.example`: document `IMPERIAL_RAG_LIVE_API` and `IMPERIAL_RAG_LIVE_CORPUS`.
- Modify `README.md`: document live test commands and skip behavior.

Context7 docs for `/dashscope/dashscope-sdk-python` were checked before this plan. The current docs reinforce the existing repo-wrapper shape: check response status codes for API calls, use `TextEmbedding.call` for embeddings, and use `MultiModalConversation.call` for multimodal/OCR calls. The implementation should test through the repo wrappers, not duplicate SDK calls directly.

### Task 1: Add Live Test Support Helper

**Files:**
- Create: `tests/live_support.py`
- Create: `tests/test_live_support.py`

- [ ] **Step 1: Write offline tests for `.env` loading and gate behavior**

Create `tests/test_live_support.py` with this content:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from live_support import (
    generated_corpus_state,
    load_dotenv,
    qdrant_is_available,
    require_live_api,
    require_live_corpus,
)


def test_load_dotenv_sets_missing_values_without_overwriting(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# local secrets",
                "DASHSCOPE_API_KEY=from-file",
                "IMPERIAL_RAG_LIVE_API=1",
                "QUOTED_VALUE='quoted'",
                'DOUBLE_QUOTED_VALUE="double quoted"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "from-shell")
    monkeypatch.delenv("IMPERIAL_RAG_LIVE_API", raising=False)
    monkeypatch.delenv("QUOTED_VALUE", raising=False)
    monkeypatch.delenv("DOUBLE_QUOTED_VALUE", raising=False)

    load_dotenv(env_path)

    assert __import__("os").environ["DASHSCOPE_API_KEY"] == "from-shell"
    assert __import__("os").environ["IMPERIAL_RAG_LIVE_API"] == "1"
    assert __import__("os").environ["QUOTED_VALUE"] == "quoted"
    assert __import__("os").environ["DOUBLE_QUOTED_VALUE"] == "double quoted"


def test_require_live_api_skips_without_opt_in(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("DASHSCOPE_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.delenv("IMPERIAL_RAG_LIVE_API", raising=False)

    with pytest.raises(pytest.skip.Exception, match="IMPERIAL_RAG_LIVE_API=1"):
        require_live_api(env_path=env_path)


def test_require_live_api_skips_without_dashscope_key_after_opt_in(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("IMPERIAL_RAG_LIVE_API=1\nDASHSCOPE_API_KEY=\n", encoding="utf-8")
    monkeypatch.delenv("IMPERIAL_RAG_LIVE_API", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(pytest.skip.Exception, match="DASHSCOPE_API_KEY"):
        require_live_api(env_path=env_path)


def test_require_live_corpus_fails_for_broken_generated_state_when_opted_in(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "IMPERIAL_RAG_LIVE_API=1\nIMPERIAL_RAG_LIVE_CORPUS=1\nDASHSCOPE_API_KEY=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("IMPERIAL_RAG_LIVE_API", raising=False)
    monkeypatch.delenv("IMPERIAL_RAG_LIVE_CORPUS", raising=False)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(AssertionError, match="chunks.jsonl"):
        require_live_corpus(root=tmp_path, env_path=env_path)


def test_generated_corpus_state_accepts_minimal_present_state(tmp_path: Path) -> None:
    processed = tmp_path / ".imperial_rag"
    extracted = processed / "extracted"
    extracted.mkdir(parents=True)
    (extracted / "chunks.jsonl").write_text('{"page_content": "x", "metadata": {}}\n', encoding="utf-8")
    (processed / "keyword.sqlite3").write_bytes(b"sqlite marker")
    (processed / "manifest.sqlite3").write_bytes(b"sqlite marker")

    state = generated_corpus_state(tmp_path)

    assert state["chunks_path"] == extracted / "chunks.jsonl"
    assert state["keyword_db_path"] == processed / "keyword.sqlite3"
    assert state["manifest_db_path"] == processed / "manifest.sqlite3"
    assert state["chunk_rows"] == 1


def test_qdrant_is_available_returns_false_for_closed_local_port() -> None:
    assert qdrant_is_available(host="127.0.0.1", port=1, timeout=0.05) is False
```

- [ ] **Step 2: Run the new helper tests and confirm they fail because the helper does not exist**

Run:

```bash
uv run python -m pytest tests/test_live_support.py -q
```

Expected: FAIL with an import error for `live_support`.

- [ ] **Step 3: Implement the helper**

Create `tests/live_support.py` with this content:

```python
from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import pytest


TRUTHY = {"1", "true", "yes", "on"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().casefold() in TRUTHY


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", maxsplit=1)
        name = name.strip()
        if name.startswith("export "):
            name = name.removeprefix("export ").strip()
        if not name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(name, value)


def require_live_api(*, env_path: Path | None = None) -> None:
    load_dotenv(env_path)
    if not truthy_env("IMPERIAL_RAG_LIVE_API"):
        pytest.skip("set IMPERIAL_RAG_LIVE_API=1 to run live DashScope/Qwen API tests")
    if not os.environ.get("DASHSCOPE_API_KEY", "").strip():
        pytest.skip("DASHSCOPE_API_KEY is required for live DashScope/Qwen API tests")


def require_live_corpus(*, root: Path | None = None, env_path: Path | None = None) -> dict[str, Any]:
    require_live_api(env_path=env_path)
    if not truthy_env("IMPERIAL_RAG_LIVE_CORPUS"):
        pytest.skip("set IMPERIAL_RAG_LIVE_CORPUS=1 to run live tests against .imperial_rag")
    return generated_corpus_state(root or REPO_ROOT)


def generated_corpus_state(root: Path) -> dict[str, Any]:
    processed = root / ".imperial_rag"
    chunks_path = processed / "extracted" / "chunks.jsonl"
    keyword_db_path = processed / "keyword.sqlite3"
    manifest_db_path = processed / "manifest.sqlite3"
    assert chunks_path.exists(), f"missing generated chunks.jsonl at {chunks_path}"
    assert keyword_db_path.exists(), f"missing keyword index at {keyword_db_path}"
    assert manifest_db_path.exists(), f"missing manifest database at {manifest_db_path}"
    chunk_rows = sum(1 for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip())
    assert chunk_rows > 0, f"generated chunks.jsonl has no rows at {chunks_path}"
    return {
        "chunks_path": chunks_path,
        "keyword_db_path": keyword_db_path,
        "manifest_db_path": manifest_db_path,
        "chunk_rows": chunk_rows,
    }


def qdrant_is_available(host: str = "127.0.0.1", port: int = 6333, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
```

- [ ] **Step 4: Run the helper tests and confirm they pass**

Run:

```bash
uv run python -m pytest tests/test_live_support.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the helper**

Run:

```bash
git add tests/live_support.py tests/test_live_support.py
git commit -m "test: add live test gates"
```

Expected: commit includes only `tests/live_support.py` and `tests/test_live_support.py`.

### Task 2: Add Provider Smoke Tests

**Files:**
- Create: `tests/test_live_provider_smoke.py`

- [ ] **Step 1: Create provider smoke tests**

Create `tests/test_live_provider_smoke.py` with this content:

```python
from __future__ import annotations

from numbers import Real
from pathlib import Path

from langchain_core.documents import Document
from PIL import Image, ImageDraw

from live_support import require_live_api


def test_live_qwen_chat_smoke() -> None:
    require_live_api()

    from imperial_rag.providers import create_chat_model

    model = create_chat_model()
    response = model.invoke(
        [
            {"role": "system", "content": "Reply briefly. Do not include secrets."},
            {"role": "user", "content": "Say: imperial live api ok"},
        ]
    )
    content = str(getattr(response, "content", response)).strip()

    assert content
    assert "DASHSCOPE_API_KEY" not in content


def test_live_qwen_embedding_smoke() -> None:
    require_live_api()

    from imperial_rag.providers import QwenProviderSettings, create_embeddings

    settings = QwenProviderSettings.from_env()
    vector = create_embeddings(settings=settings).embed_query("imperial rag live embedding smoke")

    assert isinstance(vector, list)
    assert len(vector) >= 64
    if settings.embedding_dimensions is not None:
        assert len(vector) == settings.embedding_dimensions
    assert all(isinstance(value, Real) for value in vector[:16])


def test_live_qwen_rerank_smoke() -> None:
    require_live_api()

    from imperial_rag.providers import create_reranker

    documents = [
        Document(page_content="Возврат брака оформляется актом и согласованием склада.", metadata={"id": "match"}),
        Document(page_content="График отпусков утверждается отдельным приказом.", metadata={"id": "other"}),
    ]
    reranked = list(
        create_reranker(top_n=1).compress_documents(
            documents=documents,
            query="Как оформить возврат брака?",
        )
    )

    assert len(reranked) == 1
    assert reranked[0].metadata.get("id") == "match"


def test_live_qwen_ocr_smoke(tmp_path: Path) -> None:
    require_live_api()

    from imperial_rag.ocr import QwenOcrClient

    image_path = tmp_path / "live-ocr.png"
    image = Image.new("RGB", (420, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.text((30, 55), "LIVE OCR 2468", fill="black")
    image.save(image_path)

    result = QwenOcrClient().extract_image_text(image_path)

    assert result.text.strip()
    assert result.method.startswith("dashscope:")
```

- [ ] **Step 2: Run provider smoke tests without opt-in and confirm clean skips**

Run:

```bash
uv run python -m pytest tests/test_live_provider_smoke.py -q
```

Expected: 4 skipped with reasons mentioning `IMPERIAL_RAG_LIVE_API=1`.

- [ ] **Step 3: Run provider smoke tests with live API opt-in when credentials are available**

Run:

```bash
IMPERIAL_RAG_LIVE_API=1 uv run python -m pytest tests/test_live_provider_smoke.py -q
```

Expected when `.env` contains a valid `DASHSCOPE_API_KEY`: PASS. Expected when the key is absent: 4 skipped with a `DASHSCOPE_API_KEY` reason. Expected when the provider rejects the request: FAIL with the sanitized provider error.

- [ ] **Step 4: Commit provider smoke tests**

Run:

```bash
git add tests/test_live_provider_smoke.py
git commit -m "test: add live provider smoke tests"
```

Expected: commit includes only `tests/test_live_provider_smoke.py`.

### Task 3: Add Fixture Corpus Integration Test

**Files:**
- Create: `tests/test_live_rag_integration.py`

- [ ] **Step 1: Create temporary fixture integration test**

Create `tests/test_live_rag_integration.py` with this content:

```python
from __future__ import annotations

from pathlib import Path

from live_support import qdrant_is_available, require_live_api


def test_live_fixture_corpus_answers_with_citations_and_diagnostics(tmp_path: Path) -> None:
    require_live_api()

    from imperial_rag.answering import REFUSAL_TEXT
    from imperial_rag.config import Settings
    from imperial_rag.pipeline import ingest_corpus
    from imperial_rag.runtime import create_runtime

    documents_root = tmp_path / "documents"
    documents_root.mkdir()
    (documents_root / "live_policy.rtf").write_text(
        r"{\rtf1\ansi Контрольный код LIVE-RAG-2468 означает, что тестовый склад проверен.}",
        encoding="utf-8",
    )
    settings = Settings(workspace_root=tmp_path)

    summary = ingest_corpus(settings=settings, vector_store=None)
    assert summary.indexed_files == 1
    assert summary.chunk_count >= 1
    assert summary.keyword_indexed is True

    result = create_runtime(settings).query("Что означает контрольный код LIVE-RAG-2468?")
    answer = str(result.get("answer", "")).strip()
    citations = list(result.get("citations") or [])
    evidence = list(result.get("evidence") or result.get("retrieved_documents") or [])
    retrieval = dict(result.get("retrieval") or {})

    assert answer
    assert answer != REFUSAL_TEXT
    assert citations
    assert evidence
    assert retrieval["keyword_candidates"] >= 1
    assert retrieval["final_evidence"] >= 1
    assert retrieval["reranker"] in {"dashscope:qwen3-rerank", "fallback:deterministic"} or retrieval["reranker"].startswith("dashscope:")
    if qdrant_is_available():
        assert retrieval["vector_search_status"] in {"provider_mismatch", "ok", "empty", "unavailable"}
```

- [ ] **Step 2: Run fixture integration without opt-in and confirm clean skip**

Run:

```bash
uv run python -m pytest tests/test_live_rag_integration.py -q
```

Expected: 1 skipped with a reason mentioning `IMPERIAL_RAG_LIVE_API=1`.

- [ ] **Step 3: Run fixture integration with live API opt-in**

Run:

```bash
IMPERIAL_RAG_LIVE_API=1 uv run python -m pytest tests/test_live_rag_integration.py -q
```

Expected when `.env` contains a valid `DASHSCOPE_API_KEY`: PASS. The current generated Imperial corpus is not used by this test.

- [ ] **Step 4: Commit fixture integration test**

Run:

```bash
git add tests/test_live_rag_integration.py
git commit -m "test: add live rag fixture integration"
```

Expected: commit includes only `tests/test_live_rag_integration.py`.

### Task 4: Add Real Corpus Integration Test

**Files:**
- Create: `tests/test_live_real_corpus.py`

- [ ] **Step 1: Create real corpus integration test**

Create `tests/test_live_real_corpus.py` with this content:

```python
from __future__ import annotations

import json

from live_support import REPO_ROOT, require_live_corpus


def test_live_real_imperial_corpus_answers_curated_question() -> None:
    corpus_state = require_live_corpus()
    assert corpus_state["chunk_rows"] > 0

    from imperial_rag.answering import REFUSAL_TEXT
    from imperial_rag.config import Settings
    from imperial_rag.runtime import create_runtime

    example = _first_cite_answer_example()
    result = create_runtime(Settings(workspace_root=REPO_ROOT)).query(example["question"])
    answer = str(result.get("answer", "")).strip()
    citations = list(result.get("citations") or [])
    evidence = list(result.get("evidence") or result.get("retrieved_documents") or [])
    retrieval = dict(result.get("retrieval") or {})

    assert answer
    assert answer != REFUSAL_TEXT
    assert citations
    assert evidence
    for key in ("keyword_candidates", "merged_candidates", "final_evidence", "reranker"):
        assert key in retrieval
    assert retrieval["keyword_candidates"] >= 1
    assert retrieval["final_evidence"] >= 1

    haystack = "\n".join(
        [
            answer,
            *(str(citation) for citation in citations),
            *(str(getattr(document, "page_content", document)) for document in evidence),
        ]
    ).casefold()
    hints = [str(hint).casefold() for hint in example.get("expected_source_hints", [])]
    assert not hints or any(hint in haystack for hint in hints)


def _first_cite_answer_example() -> dict:
    for line in (REPO_ROOT / "evals" / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("expected_behavior") == "cite_answer":
            return payload
    raise AssertionError("evals/questions.jsonl has no cite_answer example")
```

- [ ] **Step 2: Run real corpus test without opt-in and confirm clean skip**

Run:

```bash
uv run python -m pytest tests/test_live_real_corpus.py -q
```

Expected: 1 skipped with a reason mentioning `IMPERIAL_RAG_LIVE_API=1`.

- [ ] **Step 3: Run real corpus test with API flag only and confirm corpus gate**

Run:

```bash
IMPERIAL_RAG_LIVE_API=1 uv run python -m pytest tests/test_live_real_corpus.py -q
```

Expected when `.env` contains a valid `DASHSCOPE_API_KEY`: 1 skipped with a reason mentioning `IMPERIAL_RAG_LIVE_CORPUS=1`.

- [ ] **Step 4: Run real corpus test with full opt-in**

Run:

```bash
IMPERIAL_RAG_LIVE_API=1 IMPERIAL_RAG_LIVE_CORPUS=1 uv run python -m pytest tests/test_live_real_corpus.py -q
```

Expected when `.env` contains a valid `DASHSCOPE_API_KEY` and generated state is present: PASS. If `.imperial_rag/extracted/chunks.jsonl`, `.imperial_rag/keyword.sqlite3`, or `.imperial_rag/manifest.sqlite3` is missing, the test fails with a path-specific assertion.

- [ ] **Step 5: Commit real corpus integration test**

Run:

```bash
git add tests/test_live_real_corpus.py
git commit -m "test: add live corpus rag integration"
```

Expected: commit includes only `tests/test_live_real_corpus.py`.

### Task 5: Document Live Test Flags And Commands

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: Update `.env.example` live-test flags**

In `.env.example`, add these lines immediately before the existing `IMPERIAL_RAG_LIVE_QDRANT=0` line:

```dotenv
# Set to 1 only when intentionally running paid/network DashScope smoke and integration tests.
IMPERIAL_RAG_LIVE_API=0

# Set to 1 only when intentionally testing the existing generated `.imperial_rag` corpus state.
IMPERIAL_RAG_LIVE_CORPUS=0
```

- [ ] **Step 2: Update README testing section**

In `README.md`, replace the current live-Qdrant-only testing section:

````markdown
Run the live Qdrant health test only when local Qdrant is intentionally running:

```bash
IMPERIAL_RAG_LIVE_QDRANT=1 uv run python -m pytest tests/test_qdrant_health.py -q
```
````

with:

````markdown
Live tests are opt-in so the default suite stays offline and free of paid network calls.

Run live DashScope/Qwen provider smoke and fixture integration tests only when real credentials are available:

```bash
IMPERIAL_RAG_LIVE_API=1 uv run python -m pytest tests/test_live_provider_smoke.py tests/test_live_rag_integration.py -q
```

Run the real generated Imperial corpus health check only when `.imperial_rag` is present and you intentionally want to test it:

```bash
IMPERIAL_RAG_LIVE_API=1 IMPERIAL_RAG_LIVE_CORPUS=1 uv run python -m pytest tests/test_live_real_corpus.py -q
```

Run the live Qdrant health test only when local Qdrant is intentionally running:

```bash
IMPERIAL_RAG_LIVE_QDRANT=1 uv run python -m pytest tests/test_qdrant_health.py -q
```
````

- [ ] **Step 3: Run docs diff check**

Run:

```bash
git diff --check -- .env.example README.md
```

Expected: no output.

- [ ] **Step 4: Commit docs**

Run:

```bash
git add .env.example README.md
git commit -m "docs: document live rag tests"
```

Expected: commit includes only `.env.example` and `README.md`.

### Task 6: Final Verification

**Files:**
- Review all files changed by Tasks 1-5.

- [ ] **Step 1: Run offline helper and default-suite checks**

Run:

```bash
uv run python -m pytest tests/test_live_support.py -q
uv run python -m pytest tests/test_live_provider_smoke.py tests/test_live_rag_integration.py tests/test_live_real_corpus.py -q
uv run python -m pytest -q
```

Expected:

- `tests/test_live_support.py` passes.
- The three live test modules skip without opt-in.
- The full suite passes, with the existing Qdrant live test skipped unless `IMPERIAL_RAG_LIVE_QDRANT=1` is set.

- [ ] **Step 2: Run live API checks when credentials are available**

Run:

```bash
IMPERIAL_RAG_LIVE_API=1 uv run python -m pytest tests/test_live_provider_smoke.py tests/test_live_rag_integration.py -q
```

Expected when `.env` contains a valid `DASHSCOPE_API_KEY`: provider smoke and fixture integration pass.

- [ ] **Step 3: Run live real-corpus check when generated state should be validated**

Run:

```bash
IMPERIAL_RAG_LIVE_API=1 IMPERIAL_RAG_LIVE_CORPUS=1 uv run python -m pytest tests/test_live_real_corpus.py -q
```

Expected when `.env` contains a valid `DASHSCOPE_API_KEY` and `.imperial_rag` has generated state: real-corpus health check passes.

- [ ] **Step 4: Inspect scoped diff and status**

Run:

```bash
git diff --stat
git status --short
```

Expected: no unstaged changes from the live-test implementation remain after the task commits. Pre-existing unrelated changes may still appear and must not be staged.

- [ ] **Step 5: If verification requires a final checkpoint commit, create it with only current-session files**

Run this only if there are current-session edits not already committed by Tasks 1-5:

```bash
git add tests/live_support.py tests/test_live_support.py tests/test_live_provider_smoke.py tests/test_live_rag_integration.py tests/test_live_real_corpus.py .env.example README.md
git commit -m "test: add live rag smoke integration"
```

Expected: commit includes only files changed by this implementation plan.
