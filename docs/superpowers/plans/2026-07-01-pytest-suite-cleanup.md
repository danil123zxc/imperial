# Pytest Suite Conservative Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove redundant/stale pytest coverage, replace weak script import smokes with behavior checks, and add narrow tests for known coverage gaps without changing runtime public APIs.

**Architecture:** Keep the cleanup conservative: delete only tests that are superseded by stronger behavior tests, fold useful Phoenix Compose assertions into the existing private deployment test file, and add focused coverage around shared CLI helpers, SQLite connection lifecycle, and ingestion workflow edge cases. Live DashScope OCR tests are isolated in a final opt-in phase so normal local pytest remains offline and deterministic.

**Tech Stack:** Python 3.12+, pytest, uv, sqlite3, Streamlit app stores, LangGraph ingestion workflow, DashScope/Qwen OCR for optional live tests.

---

## File Structure

- Modify `tests/test_all_evals.py`
  - Remove the import-only `test_all_evals_script_imports_and_defines_main`.
- Modify `tests/test_scripts.py`
  - Remove redundant import-only script tests for ingest/query/Phoenix eval/Phoenix trace validator/event-log setup.
  - Replace the ingestion-promotion import smoke with behavior coverage for `scripts/check_ingestion_promotion.py::main`.
- Modify `tests/test_private_compose_deployment.py`
  - Add the still-useful Phoenix persistence assertions currently duplicated in `tests/test_phoenix_stack.py`.
- Delete `tests/test_phoenix_stack.py`
  - Its useful Compose persistence checks move to `tests/test_private_compose_deployment.py`; the old docs-supersession assertion is intentionally dropped.
- Create `tests/test_cli.py`
  - Cover shared helpers in `src/imperial_rag/cli.py`.
- Modify `tests/test_auth.py`
  - Add lifecycle coverage proving `AuthStore` closes SQLite connections.
- Modify `tests/test_chat_history.py`
  - Add lifecycle coverage proving `ChatHistoryStore` closes SQLite connections.
- Modify `src/imperial_rag/app/auth.py`
  - Add an internal connection context manager and use it instead of raw `with self._connect()`.
- Modify `src/imperial_rag/app/chat_history.py`
  - Add the same internal connection context manager and use it for all DB operations.
- Modify `tests/test_workflows.py`
  - Add ingestion workflow edge coverage for explicit `counts`, object summaries, and state-accepting pipeline callables.
- Optional live phase:
  - Modify `pyproject.toml` to register the `live_api` pytest marker.
  - Create `tests/live_support.py`.
  - Create `tests/test_live_provider_smoke.py`.
  - Create `tests/test_live_rag_integration.py`.

---

### Task 1: Remove Redundant Import-Only Script Tests

**Files:**
- Modify: `tests/test_all_evals.py`
- Modify: `tests/test_scripts.py`

- [ ] **Step 1: Remove the all-evals import-only smoke**

Delete this function from `tests/test_all_evals.py`:

```python
def test_all_evals_script_imports_and_defines_main():
    module = _load_all_evals_runner()

    assert hasattr(module, "main")
```

- [ ] **Step 2: Remove redundant import-only smokes from `tests/test_scripts.py`**

Delete these functions from `tests/test_scripts.py`:

```python
def test_ingest_script_imports_and_defines_main():
    module = _load_script("scripts/ingest.py", "ingest_script")

    assert hasattr(module, "main")
    assert hasattr(module, "print_summary")


def test_query_script_imports_and_defines_main():
    module = _load_script("scripts/query.py", "query_script")

    assert hasattr(module, "main")


def test_phoenix_eval_script_imports_and_defines_main():
    module = _load_script("scripts/run_phoenix_eval.py", "run_phoenix_eval_script")

    assert hasattr(module, "main")
    assert hasattr(module, "citation_behavior")


def test_phoenix_trace_validator_script_imports_and_defines_main():
    module = _load_script("scripts/validate_phoenix_trace.py", "validate_phoenix_trace_script")

    assert hasattr(module, "main")
    assert hasattr(module, "validate_span_records")


def test_event_log_setup_script_imports_and_defines_main():
    module = _load_script("scripts/setup_event_logs.py", "setup_event_logs_script")

    assert hasattr(module, "main")
    assert hasattr(module, "setup_event_log_streams")
```

- [ ] **Step 3: Run the reduced script/eval slice**

Run:

```bash
uv run --extra dev python -m pytest tests/test_scripts.py tests/test_all_evals.py -q
```

Expected: the remaining behavior tests pass without any import-only script smoke tests.

- [ ] **Step 4: Inspect the diff**

Run:

```bash
git diff -- tests/test_all_evals.py tests/test_scripts.py
```

Expected: only the redundant import-only functions are removed; behavior tests remain.

---

### Task 2: Replace Ingestion Promotion Import Smoke With CLI Behavior Coverage

**Files:**
- Modify: `tests/test_scripts.py`

- [ ] **Step 1: Add `json` import**

At the top of `tests/test_scripts.py`, add:

```python
import json
```

- [ ] **Step 2: Replace the ingestion-promotion import smoke**

Delete:

```python
def test_ingestion_promotion_script_imports_and_defines_main():
    module = _load_script("scripts/check_ingestion_promotion.py", "check_ingestion_promotion_script")

    assert hasattr(module, "main")
```

Add this behavior test near the other script tests:

```python
def test_ingestion_promotion_cli_outputs_json_and_exits_one_when_gates_fail(monkeypatch, capsys, tmp_path):
    module = _load_script("scripts/check_ingestion_promotion.py", "check_ingestion_promotion_cli")
    calls: dict[str, object] = {}

    @dataclass(frozen=True)
    class FakePromotionGateResult:
        passed: bool
        errors: list[str]
        summary: dict[str, object]

    def check_promotion_gates(
        baseline_root,
        shadow_root,
        *,
        questions_path,
        min_locator_coverage,
        expected_keyword_index,
        expected_qdrant_collection,
    ):
        calls.update(
            {
                "baseline_root": baseline_root,
                "shadow_root": shadow_root,
                "questions_path": questions_path,
                "min_locator_coverage": min_locator_coverage,
                "expected_keyword_index": expected_keyword_index,
                "expected_qdrant_collection": expected_qdrant_collection,
            }
        )
        return FakePromotionGateResult(
            passed=False,
            errors=["shadow locator coverage below gate: 0.5 < 0.95"],
            summary={"shadow_locator_coverage": 0.5},
        )

    promotion_module = _fake_module("imperial_rag.ingestion.promotion")
    promotion_module.check_promotion_gates = check_promotion_gates
    monkeypatch.setitem(sys.modules, "imperial_rag.ingestion.promotion", promotion_module)

    baseline_root = tmp_path / "baseline"
    shadow_root = tmp_path / "shadow"
    questions_path = tmp_path / "questions.jsonl"

    with pytest.raises(SystemExit) as exc_info:
        module.main(
            [
                "--baseline-root",
                str(baseline_root),
                "--shadow-root",
                str(shadow_root),
                "--questions-path",
                str(questions_path),
                "--min-locator-coverage",
                "0.95",
                "--expected-keyword-index",
                "keyword_shadow",
                "--expected-qdrant-collection",
                "vectors_shadow",
            ]
        )

    assert exc_info.value.code == 1
    assert calls == {
        "baseline_root": baseline_root,
        "shadow_root": shadow_root,
        "questions_path": questions_path,
        "min_locator_coverage": 0.95,
        "expected_keyword_index": "keyword_shadow",
        "expected_qdrant_collection": "vectors_shadow",
    }
    assert json.loads(capsys.readouterr().out) == {
        "passed": False,
        "errors": ["shadow locator coverage below gate: 0.5 < 0.95"],
        "summary": {"shadow_locator_coverage": 0.5},
    }
```

- [ ] **Step 3: Run the new behavior test**

Run:

```bash
uv run --extra dev python -m pytest tests/test_scripts.py::test_ingestion_promotion_cli_outputs_json_and_exits_one_when_gates_fail -q
```

Expected: pass.

- [ ] **Step 4: Run all script tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_scripts.py -q
```

Expected: pass.

---

### Task 3: Consolidate Phoenix Compose Coverage

**Files:**
- Modify: `tests/test_private_compose_deployment.py`
- Delete: `tests/test_phoenix_stack.py`

- [ ] **Step 1: Move Phoenix persistence assertions**

In `tests/test_private_compose_deployment.py`, inside `test_compose_defines_private_app_and_ingest_services()`, add these snippets to `required_snippets`:

```python
"PHOENIX_WORKING_DIR: /mnt/data",
"phoenix_data:/mnt/data",
"  phoenix_data:\n    driver: local",
```

- [ ] **Step 2: Delete stale Phoenix stack file**

Delete `tests/test_phoenix_stack.py` entirely.

Do not move this historical-doc assertion:

```python
def test_old_superpowers_docs_point_to_phoenix_supersession_spec():
    spec = Path("docs/superpowers/specs/2026-06-02-local-rag-system-design.md").read_text(encoding="utf-8")
    plan = Path("docs/superpowers/plans/2026-06-02-local-rag-system.md").read_text(encoding="utf-8")

    assert "`2026-06-03-phoenix-observability-design.md` supersedes" in spec
    assert "`2026-06-03-phoenix-observability-design.md` supersedes" in plan
```

- [ ] **Step 3: Run private Compose coverage**

Run:

```bash
uv run --extra dev python -m pytest tests/test_private_compose_deployment.py -q
```

Expected: pass.

- [ ] **Step 4: Confirm deleted test file is gone**

Run:

```bash
test ! -e tests/test_phoenix_stack.py
```

Expected: exit code `0`.

---

### Task 4: Add Shared CLI Helper Coverage

**Files:**
- Create: `tests/test_cli.py`

- [ ] **Step 1: Create `tests/test_cli.py`**

Create the file with this content:

```python
from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace

from imperial_rag import cli


def test_build_settings_falls_back_to_workspace_root_env(monkeypatch, tmp_path):
    class FakeSettings:
        def __init__(self, **kwargs):
            if "workspace_root" in kwargs:
                raise TypeError("workspace_root is not accepted")

    config_module = types.ModuleType("imperial_rag.config")
    config_module.Settings = FakeSettings
    monkeypatch.setitem(sys.modules, "imperial_rag.config", config_module)
    monkeypatch.delenv("IMPERIAL_RAG_WORKSPACE_ROOT", raising=False)

    result = cli.build_settings(tmp_path)

    assert isinstance(result, FakeSettings)
    assert os.environ["IMPERIAL_RAG_WORKSPACE_ROOT"] == str(tmp_path)


def test_configure_tracing_maps_flags_to_enabled(monkeypatch):
    settings = SimpleNamespace()
    calls = []
    tracing_module = types.ModuleType("imperial_rag.observability.phoenix")
    tracing_module.configure_phoenix_tracing = lambda settings, *, enabled=None: calls.append((settings, enabled))
    monkeypatch.setitem(sys.modules, "imperial_rag.observability.phoenix", tracing_module)

    cli.configure_tracing(settings, trace_phoenix=True)
    cli.configure_tracing(settings, trace_phoenix=False)
    cli.configure_tracing(settings, enabled=False)

    assert calls == [(settings, True), (settings, None), (settings, False)]


def test_trace_context_adds_entrypoint_metadata_and_tags(monkeypatch):
    calls = []
    tracing_module = types.ModuleType("imperial_rag.observability.phoenix")

    @contextmanager
    def phoenix_trace_context(session_id, **kwargs):
        calls.append({"session_id": session_id, **kwargs})
        yield

    tracing_module.phoenix_trace_context = phoenix_trace_context
    monkeypatch.setitem(sys.modules, "imperial_rag.observability.phoenix", tracing_module)

    with cli.trace_context("session-1", entrypoint="eval", tags=["custom"]):
        calls.append({"entered": True})

    assert calls == [
        {
            "session_id": "session-1",
            "metadata": {"entrypoint": "eval"},
            "tags": ["custom"],
        },
        {"entered": True},
    ]


def test_log_failure_forwards_cli_component_duration_and_fields(monkeypatch):
    calls = []
    observability_module = types.ModuleType("imperial_rag.observability")
    observability_module.log_failure = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, "imperial_rag.observability", observability_module)
    monkeypatch.setattr(cli, "duration_ms", lambda started_at: 123)
    exc = RuntimeError("boom")

    cli.log_failure("query", exc, 1.0, phoenix_session_id="session-1")

    assert calls == [
        (
            ("query", exc),
            {
                "component": "cli",
                "duration_ms": 123,
                "phoenix_session_id": "session-1",
            },
        )
    ]
```

- [ ] **Step 2: Run CLI helper tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_cli.py -q
```

Expected: `4 passed`.

---

### Task 5: Add SQLite Connection Lifecycle Tests And Internal Closure Fix

**Files:**
- Modify: `tests/test_auth.py`
- Modify: `tests/test_chat_history.py`
- Modify: `src/imperial_rag/app/auth.py`
- Modify: `src/imperial_rag/app/chat_history.py`

- [ ] **Step 1: Add lifecycle test to `tests/test_auth.py`**

Add imports:

```python
import sqlite3

import imperial_rag.app.auth as auth_module
```

Add this test:

```python
def test_auth_store_closes_sqlite_connections(monkeypatch, tmp_path):
    original_connect = sqlite3.connect
    connections = []

    class TrackingConnection(sqlite3.Connection):
        closed = False

        def close(self):
            self.closed = True
            super().close()

    def connect(*args, **kwargs):
        kwargs["factory"] = TrackingConnection
        conn = original_connect(*args, **kwargs)
        connections.append(conn)
        return conn

    monkeypatch.setattr(auth_module.sqlite3, "connect", connect)
    store = AuthStore(tmp_path / "auth.sqlite3")

    store.bootstrap_admin("admin@example.com", "admin-password")
    store.register_user("user@example.com", "user-password", "User", "Testing")
    store.authenticate("admin@example.com", "admin-password")
    store.list_pending_users()

    assert connections
    assert all(conn.closed for conn in connections)
```

- [ ] **Step 2: Add lifecycle test to `tests/test_chat_history.py`**

Add imports:

```python
import sqlite3

import imperial_rag.app.chat_history as chat_history_module
```

Add this test:

```python
def test_chat_history_store_closes_sqlite_connections(monkeypatch, tmp_path):
    original_connect = sqlite3.connect
    connections = []

    class TrackingConnection(sqlite3.Connection):
        closed = False

        def close(self):
            self.closed = True
            super().close()

    def connect(*args, **kwargs):
        kwargs["factory"] = TrackingConnection
        conn = original_connect(*args, **kwargs)
        connections.append(conn)
        return conn

    monkeypatch.setattr(chat_history_module.sqlite3, "connect", connect)
    store = ChatHistoryStore(tmp_path / "chat_history.sqlite3")

    conversation = store.create_conversation("user@example.com", "Question", phoenix_session_id="trace-1")
    store.add_message("user@example.com", conversation.id, "user", "Question")
    store.list_conversations("user@example.com")
    store.get_conversation("user@example.com", conversation.id)
    store.list_messages("user@example.com", conversation.id)

    assert connections
    assert all(conn.closed for conn in connections)
```

- [ ] **Step 3: Run lifecycle tests and confirm failure**

Run:

```bash
uv run --extra dev python -m pytest \
  tests/test_auth.py::test_auth_store_closes_sqlite_connections \
  tests/test_chat_history.py::test_chat_history_store_closes_sqlite_connections \
  -q
```

Expected before implementation: failures showing at least one tracked SQLite connection remains open.

- [ ] **Step 4: Add internal connection context manager to `AuthStore`**

In `src/imperial_rag/app/auth.py`, add imports:

```python
from collections.abc import Iterator
from contextlib import contextmanager
```

Add this method inside `AuthStore`, just above `_connect()`:

```python
    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()
```

Replace every:

```python
with self._connect() as conn:
```

with:

```python
with self._connection() as conn:
```

- [ ] **Step 5: Add internal connection context manager to `ChatHistoryStore`**

In `src/imperial_rag/app/chat_history.py`, add imports:

```python
from collections.abc import Iterator
from contextlib import contextmanager
```

Add this method inside `ChatHistoryStore`, just above `_connect()`:

```python
    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()
```

Replace every:

```python
with self._connect() as conn:
```

with:

```python
with self._connection() as conn:
```

- [ ] **Step 6: Run lifecycle tests again**

Run:

```bash
uv run --extra dev python -m pytest \
  tests/test_auth.py::test_auth_store_closes_sqlite_connections \
  tests/test_chat_history.py::test_chat_history_store_closes_sqlite_connections \
  -q
```

Expected after implementation: pass.

- [ ] **Step 7: Run full auth/chat history tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_auth.py tests/test_chat_history.py -q
```

Expected: pass.

---

### Task 6: Add Ingestion Workflow Edge Coverage

**Files:**
- Modify: `tests/test_workflows.py`

- [ ] **Step 1: Add explicit counts mapping test**

Add near `test_ingestion_workflow_invokes_pipeline_and_returns_status_counts()`:

```python
def test_ingestion_workflow_prefers_explicit_counts_from_mapping_summary():
    def run_pipeline():
        return {
            "status": "partial",
            "documents": 99,
            "counts": {"documents": "2", "chunks": "5"},
        }

    workflow = build_ingestion_workflow(run_pipeline=run_pipeline)

    result = workflow.invoke({})

    assert result["status"] == "partial"
    assert result["counts"] == {"documents": 2, "chunks": 5}
```

- [ ] **Step 2: Add object summary test**

Add:

```python
def test_ingestion_workflow_counts_object_summary_fields():
    class Summary:
        status = "completed"
        total_files = 3
        chunk_count = 8
        indexed_count = 2

    workflow = build_ingestion_workflow(run_pipeline=lambda: Summary())

    result = workflow.invoke({})

    assert result["status"] == "completed"
    assert result["counts"] == {"files": 3, "chunks": 8, "indexed": 2}
```

- [ ] **Step 3: Add state-accepting callable test**

Add:

```python
def test_ingestion_workflow_passes_state_to_pipeline_callable_that_accepts_it():
    received = {}

    def run_pipeline(state):
        received.update(state)
        return {"counts": {"documents": 1}}

    workflow = build_ingestion_workflow(run_pipeline=run_pipeline)

    result = workflow.invoke({"settings": "settings", "ocr_client": "ocr-client"})

    assert received["settings"] == "settings"
    assert received["ocr_client"] == "ocr-client"
    assert result["counts"] == {"documents": 1}
```

- [ ] **Step 4: Run workflow tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_workflows.py -q
```

Expected: pass.

---

### Task 7: Optional Opt-In Live OCR Coverage

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/live_support.py`
- Create: `tests/test_live_provider_smoke.py`
- Create: `tests/test_live_rag_integration.py`

This task is optional and should be a separate commit from the cleanup if the user wants a very small review surface first.

- [ ] **Step 1: Register the pytest marker**

In `pyproject.toml`, change:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

to:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = [
  "live_api: paid/network DashScope API tests; require IMPERIAL_RAG_LIVE_API=1",
]
```

- [ ] **Step 2: Create live test support helper**

Create `tests/live_support.py`:

```python
from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont


LIVE_SENTINEL = "86420"


def require_live_api() -> None:
    if os.environ.get("IMPERIAL_RAG_LIVE_API") != "1":
        pytest.skip("live DashScope API tests are opt-in with IMPERIAL_RAG_LIVE_API=1")
    env_path = os.environ.get("IMPERIAL_RAG_LIVE_ENV_PATH", "").strip()
    if env_path:
        load_dotenv(Path(env_path), override=False)
    if not os.environ.get("DASHSCOPE_API_KEY"):
        pytest.skip("DASHSCOPE_API_KEY is required for live DashScope OCR tests")


def normalized_digits(text: str) -> str:
    return "".join(character for character in str(text) if character.isdigit())


def write_sentinel_image(path: Path, text: str = LIVE_SENTINEL) -> None:
    image = Image.new("RGB", (520, 180), "white")
    draw = ImageDraw.Draw(image)
    font = _large_font()
    draw.text((48, 44), text, fill="black", font=font)
    image.save(path)


def _large_font():
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, 72)
    return ImageFont.load_default()
```

- [ ] **Step 3: Create direct provider live smoke test**

Create `tests/test_live_provider_smoke.py`:

```python
from __future__ import annotations

import pytest

from imperial_rag.ingestion.ocr import QwenOcrClient
from tests.live_support import LIVE_SENTINEL, normalized_digits, require_live_api, write_sentinel_image


pytestmark = pytest.mark.live_api


def test_live_qwen_ocr_reads_numeric_sentinel(tmp_path):
    require_live_api()
    image_path = tmp_path / "sentinel.png"
    write_sentinel_image(image_path)

    result = QwenOcrClient().extract_image_text(image_path)

    assert LIVE_SENTINEL in normalized_digits(result.text)
    assert result.method.startswith("dashscope:")
```

- [ ] **Step 4: Create live OCR-through-ingestion test**

Create `tests/test_live_rag_integration.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from imperial_rag.config import Settings
from imperial_rag.ingestion.ocr import QwenOcrClient
from imperial_rag.ingestion.pipeline import ingest_corpus
from tests.live_support import LIVE_SENTINEL, normalized_digits, require_live_api, write_sentinel_image


pytestmark = pytest.mark.live_api


class FakeKeywordSearchIndex:
    documents: list[Any] = []

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def replace_all(self, documents: list[Any]) -> None:
        FakeKeywordSearchIndex.documents = list(documents)


def test_live_qwen_ocr_flows_through_ingestion_chunks(tmp_path, monkeypatch):
    require_live_api()
    docs = tmp_path / "documents"
    docs.mkdir()
    write_sentinel_image(docs / "scan.png")
    settings = Settings(workspace_root=tmp_path)
    FakeKeywordSearchIndex.documents = []
    monkeypatch.setattr("imperial_rag.retrieval.elasticsearch.ElasticsearchKeywordIndex", FakeKeywordSearchIndex)

    summary = ingest_corpus(settings=settings, ocr_client=QwenOcrClient(), vector_store=None)

    assert summary.total_files == 1
    assert summary.indexed_files == 1
    chunk_text = "\n".join(
        json.loads(line)["page_content"]
        for line in (settings.extraction_root / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    )
    assert LIVE_SENTINEL in normalized_digits(chunk_text)
    assert FakeKeywordSearchIndex.documents
```

- [ ] **Step 5: Verify default mode skips live tests**

Run:

```bash
uv run --extra dev python -m pytest tests/test_live_provider_smoke.py tests/test_live_rag_integration.py -q
```

Expected without `IMPERIAL_RAG_LIVE_API=1`: both tests skip.

- [ ] **Step 6: Run live tests only when intentionally enabled**

Run only when a trusted DashScope key is available:

```bash
IMPERIAL_RAG_LIVE_API=1 uv run --extra dev python -m pytest tests/test_live_provider_smoke.py tests/test_live_rag_integration.py -q
```

Expected with valid credentials: pass. If credentials are missing, tests skip with a clear reason. If the provider returns an auth or quota error, report the provider failure plainly and do not treat the live phase as verified.

---

## Verification Plan

- [ ] **Run cleanup-focused tests**

```bash
uv run --extra dev python -m pytest tests/test_scripts.py tests/test_all_evals.py tests/test_private_compose_deployment.py -q
```

Expected: pass.

- [ ] **Run added focused tests**

```bash
uv run --extra dev python -m pytest tests/test_cli.py tests/test_auth.py tests/test_chat_history.py tests/test_workflows.py -q
```

Expected: pass.

- [ ] **Run full suite with durations**

```bash
uv run --extra dev python -m pytest -q --durations=25
```

Expected: pass, with live Qdrant/Elasticsearch/API tests skipped unless explicitly enabled.

- [ ] **Run canonical local gate**

```bash
./scripts/check.sh
```

Expected: pass.

- [ ] **Optional live OCR verification**

```bash
IMPERIAL_RAG_LIVE_API=1 uv run --extra dev python -m pytest tests/test_live_provider_smoke.py tests/test_live_rag_integration.py -q
```

Expected: pass only when credentials and provider access are valid.

---

## Commit Plan

- [ ] **Commit 1: Conservative cleanup and behavior replacement**

Stage only:

```bash
git add tests/test_all_evals.py tests/test_scripts.py tests/test_private_compose_deployment.py
git rm tests/test_phoenix_stack.py
git commit -m "test: remove redundant pytest smokes"
```

- [ ] **Commit 2: Focused helper and lifecycle coverage**

Stage only:

```bash
git add tests/test_cli.py tests/test_auth.py tests/test_chat_history.py src/imperial_rag/app/auth.py src/imperial_rag/app/chat_history.py tests/test_workflows.py
git commit -m "test: cover cli helpers and sqlite store lifecycle"
```

- [ ] **Commit 3: Optional live OCR tests**

Stage only if Task 7 is implemented:

```bash
git add pyproject.toml tests/live_support.py tests/test_live_provider_smoke.py tests/test_live_rag_integration.py
git commit -m "test: add opt-in live qwen ocr coverage"
```

Do not stage `.DS_Store`, untracked council reports/transcripts, generated corpus state, secrets, or unrelated planning docs.
