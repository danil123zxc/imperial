# Phoenix Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LangSmith with a fully self-hosted Phoenix observability path for traces, evaluation datasets, and experiment results.

**Architecture:** Add Phoenix settings and a small idempotent tracing module, then wire scripts and Streamlit startup through that module. Replace the LangSmith eval runner with a Phoenix runner that preserves local deterministic scoring while adding Phoenix Python SDK dataset creation and experiment execution. Add a local Phoenix Docker Compose service with persistent SQLite-backed storage.

**Tech Stack:** Python 3.12, LangChain, LangGraph, Qdrant, Phoenix client SDK, Phoenix OTel SDK, OpenInference instrumentation, Docker Compose, pytest.

---

## File Structure

- `pyproject.toml` declares Phoenix client/tracing dependencies and removes `langsmith`.
- `src/imperial_rag/config.py` owns Phoenix configuration defaults and environment overrides.
- `src/imperial_rag/tracing.py` owns Phoenix tracing setup and is the only module that imports `phoenix.otel`.
- `src/imperial_rag/indexing.py` keeps Qdrant helper settings construction compatible with the new `Settings` fields.
- `scripts/query.py`, `scripts/ingest.py`, and `src/imperial_rag/web_app.py` call the tracing setup when tracing is enabled.
- `scripts/run_phoenix_eval.py` replaces `scripts/run_langsmith_eval.py`.
- `tests/test_config.py`, `tests/test_dependencies.py`, `tests/test_tracing.py`, `tests/test_evals.py`, and `tests/test_scripts.py` cover the migration without requiring a live Phoenix server.
- `compose.yaml` runs local Phoenix with persistent data.
- Existing Superpowers spec/plan docs get a short supersession note pointing to the Phoenix design.

## Git Note

`/Users/danil/Public/imperial` is not currently a Git repository. Each task includes a commit step for future repo execution; in the current workspace, verify `git status` reports `fatal: not a git repository` and skip the commit.

### Task 1: Replace LangSmith Configuration And Dependencies

**Files:**
- Modify: `/Users/danil/Public/imperial/tests/test_config.py`
- Create: `/Users/danil/Public/imperial/tests/test_dependencies.py`
- Modify: `/Users/danil/Public/imperial/pyproject.toml`
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/config.py`
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/indexing.py`

- [ ] **Step 1: Write failing settings and dependency tests**

Replace `tests/test_config.py` with:

```python
from pathlib import Path

from imperial_rag.config import Settings


def test_settings_defaults_to_workspace_documents():
    settings = Settings()

    assert settings.workspace_root == Path("/Users/danil/Public/imperial")
    assert settings.documents_root == Path("/Users/danil/Public/imperial/documents")
    assert settings.processed_root == Path("/Users/danil/Public/imperial/.imperial_rag")
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "imperial_chunks"
    assert settings.phoenix_project_name == "imperial-rag"
    assert settings.phoenix_collector_endpoint == "http://localhost:6006/v1/traces"
    assert settings.phoenix_client_endpoint == "http://localhost:6006"
    assert settings.manifest_db_path == Path("/Users/danil/Public/imperial/.imperial_rag/manifest.sqlite3")
    assert settings.keyword_db_path == Path("/Users/danil/Public/imperial/.imperial_rag/keyword.sqlite3")
    assert settings.extraction_root == Path("/Users/danil/Public/imperial/.imperial_rag/extracted")


def test_settings_reads_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("IMPERIAL_RAG_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_chunks")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "test-project")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix.internal:6006/v1/traces")
    monkeypatch.setenv("PHOENIX_CLIENT_ENDPOINT", "http://phoenix.internal:6006")

    settings = Settings()

    assert settings.workspace_root == tmp_path
    assert settings.documents_root == tmp_path / "documents"
    assert settings.processed_root == tmp_path / ".imperial_rag"
    assert settings.qdrant_url == "http://127.0.0.1:6333"
    assert settings.qdrant_collection == "test_chunks"
    assert settings.phoenix_project_name == "test-project"
    assert settings.phoenix_collector_endpoint == "http://phoenix.internal:6006/v1/traces"
    assert settings.phoenix_client_endpoint == "http://phoenix.internal:6006"
```

Create `tests/test_dependencies.py`:

```python
from __future__ import annotations

import tomllib
from pathlib import Path


def test_project_uses_phoenix_dependencies_instead_of_langsmith():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(pyproject["project"]["dependencies"])

    assert "langsmith" not in dependencies
    assert "arize-phoenix-client" in dependencies
    assert "arize-phoenix-otel" in dependencies
    assert "openinference-instrumentation-langchain" in dependencies
    assert "openinference-instrumentation-openai" in dependencies
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_config.py tests/test_dependencies.py -q
```

Expected: FAIL because `Settings.phoenix_project_name` does not exist, `langsmith` is still present, and Phoenix dependencies are missing.

- [ ] **Step 3: Update dependencies and settings**

In `pyproject.toml`, replace the dependency list with this list:

```toml
dependencies = [
  "langchain",
  "langchain-community",
  "langchain-core",
  "langchain-openai",
  "langchain-qdrant",
  "langchain-text-splitters",
  "langgraph",
  "arize-phoenix-client",
  "arize-phoenix-otel",
  "openinference-instrumentation-langchain",
  "openinference-instrumentation-openai",
  "qdrant-client",
  "python-docx",
  "openpyxl",
  "pypdf",
  "pymupdf",
  "pillow",
  "striprtf",
  "streamlit",
]
```

Replace `src/imperial_rag/config.py` with:

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_WORKSPACE_ROOT = Path("/Users/danil/Public/imperial")


@dataclass(frozen=True)
class Settings:
    workspace_root: Path = field(
        default_factory=lambda: Path(os.environ.get("IMPERIAL_RAG_WORKSPACE_ROOT", DEFAULT_WORKSPACE_ROOT))
    )
    qdrant_url: str = field(default_factory=lambda: os.environ.get("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(default_factory=lambda: os.environ.get("QDRANT_COLLECTION", "imperial_chunks"))
    phoenix_project_name: str = field(default_factory=lambda: os.environ.get("PHOENIX_PROJECT_NAME", "imperial-rag"))
    phoenix_collector_endpoint: str = field(
        default_factory=lambda: os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
    )
    phoenix_client_endpoint: str = field(
        default_factory=lambda: os.environ.get("PHOENIX_CLIENT_ENDPOINT", "http://localhost:6006")
    )

    @property
    def documents_root(self) -> Path:
        return self.workspace_root / "documents"

    @property
    def processed_root(self) -> Path:
        return self.workspace_root / ".imperial_rag"

    @property
    def manifest_db_path(self) -> Path:
        return self.processed_root / "manifest.sqlite3"

    @property
    def keyword_db_path(self) -> Path:
        return self.processed_root / "keyword.sqlite3"

    @property
    def extraction_root(self) -> Path:
        return self.processed_root / "extracted"
```

In `src/imperial_rag/indexing.py`, replace `make_qdrant_store` with:

```python
def make_qdrant_store(qdrant_url: str, collection_name: str, embeddings: object | None = None) -> QdrantVectorStore:
    settings = Settings()
    return create_qdrant_vector_store(
        Settings(
            workspace_root=settings.workspace_root,
            qdrant_url=qdrant_url,
            qdrant_collection=collection_name,
            phoenix_project_name=settings.phoenix_project_name,
            phoenix_collector_endpoint=settings.phoenix_collector_endpoint,
            phoenix_client_endpoint=settings.phoenix_client_endpoint,
        ),
        embeddings=embeddings,
    )
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m pytest tests/test_config.py tests/test_dependencies.py tests/test_indexing.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit or record no-git status**

Run:

```bash
git status --short
```

Expected in the current workspace: `fatal: not a git repository (or any of the parent directories): .git`.

If this plan is executed inside a Git repository, commit:

```bash
git add pyproject.toml src/imperial_rag/config.py src/imperial_rag/indexing.py tests/test_config.py tests/test_dependencies.py
git commit -m "chore: replace langsmith settings with phoenix"
```

### Task 2: Add Phoenix Tracing Setup And Wire Entrypoints

**Files:**
- Create: `/Users/danil/Public/imperial/tests/test_tracing.py`
- Modify: `/Users/danil/Public/imperial/tests/test_scripts.py`
- Create: `/Users/danil/Public/imperial/src/imperial_rag/tracing.py`
- Modify: `/Users/danil/Public/imperial/scripts/query.py`
- Modify: `/Users/danil/Public/imperial/scripts/ingest.py`
- Modify: `/Users/danil/Public/imperial/src/imperial_rag/web_app.py`

- [ ] **Step 1: Write failing tracing tests**

Create `tests/test_tracing.py`:

```python
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from imperial_rag.config import Settings
from imperial_rag.tracing import _reset_phoenix_tracing_for_tests, configure_phoenix_tracing


def test_configure_phoenix_tracing_returns_none_when_disabled(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    monkeypatch.delenv("PHOENIX_TRACING_ENABLED", raising=False)
    settings = Settings(workspace_root=tmp_path)

    assert configure_phoenix_tracing(settings, enabled=False) is None
    assert configure_phoenix_tracing(settings, enabled=None) is None


def test_configure_phoenix_tracing_registers_once(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    calls: list[dict[str, object]] = []
    provider = object()
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel = types.ModuleType("phoenix.otel")

    def register(**kwargs):
        calls.append(kwargs)
        return provider

    fake_otel.register = register
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.otel", fake_otel)
    settings = Settings(
        workspace_root=tmp_path,
        phoenix_project_name="trace-project",
        phoenix_collector_endpoint="http://localhost:6006/v1/traces",
    )

    assert configure_phoenix_tracing(settings, enabled=True) is provider
    assert configure_phoenix_tracing(settings, enabled=True) is provider
    assert calls == [
        {
            "project_name": "trace-project",
            "endpoint": "http://localhost:6006/v1/traces",
            "auto_instrument": True,
        }
    ]


def test_configure_phoenix_tracing_can_be_enabled_by_env(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    provider = object()
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel = types.ModuleType("phoenix.otel")
    fake_otel.register = lambda **kwargs: provider
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.otel", fake_otel)
    monkeypatch.setenv("PHOENIX_TRACING_ENABLED", "true")

    assert configure_phoenix_tracing(Settings(workspace_root=tmp_path), enabled=None) is provider


def test_configure_phoenix_tracing_errors_clearly_when_dependency_missing(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    monkeypatch.delitem(sys.modules, "phoenix.otel", raising=False)
    settings = Settings(workspace_root=tmp_path)

    with pytest.raises(RuntimeError, match="Phoenix tracing dependencies are missing"):
        configure_phoenix_tracing(settings, enabled=True)
```

Replace `tests/test_scripts.py` with:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path


def test_ingest_script_imports_and_defines_main():
    module = _load_script("scripts/ingest.py", "ingest_script")

    assert hasattr(module, "main")
    assert hasattr(module, "print_summary")


def test_query_script_imports_and_defines_main():
    module = _load_script("scripts/query.py", "query_script")

    assert hasattr(module, "main")


def test_langsmith_eval_script_imports_and_defines_main_until_phoenix_replacement():
    module = _load_script("scripts/run_langsmith_eval.py", "run_langsmith_eval_script")

    assert hasattr(module, "main")
    assert hasattr(module, "citation_behavior")


def test_entrypoint_scripts_expose_phoenix_tracing_flag():
    assert "--trace-phoenix" in Path("scripts/ingest.py").read_text(encoding="utf-8")
    assert "--trace-phoenix" in Path("scripts/query.py").read_text(encoding="utf-8")


def _load_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_tracing.py tests/test_scripts.py -q
```

Expected: FAIL because `imperial_rag.tracing` does not exist and entrypoint scripts do not expose `--trace-phoenix`.

- [ ] **Step 3: Create Phoenix tracing module**

Create `src/imperial_rag/tracing.py`:

```python
from __future__ import annotations

import os
from typing import Any

from imperial_rag.config import Settings


_CONFIGURED_PROVIDER: object | None = None
_CONFIGURED_KEY: tuple[str, str] | None = None


def configure_phoenix_tracing(settings: Settings | None = None, enabled: bool | None = None) -> object | None:
    """Configure Phoenix OpenTelemetry tracing once for the current process."""

    if enabled is None:
        enabled = _env_flag("PHOENIX_TRACING_ENABLED") or _env_flag("IMPERIAL_RAG_TRACING_ENABLED")
    if not enabled:
        return None

    resolved_settings = settings or Settings()
    key = (resolved_settings.phoenix_project_name, resolved_settings.phoenix_collector_endpoint)
    global _CONFIGURED_PROVIDER, _CONFIGURED_KEY
    if _CONFIGURED_PROVIDER is not None and _CONFIGURED_KEY == key:
        return _CONFIGURED_PROVIDER

    try:
        from phoenix.otel import register
    except ImportError as exc:
        raise RuntimeError(
            "Phoenix tracing dependencies are missing. Install arize-phoenix-otel and OpenInference instrumentors."
        ) from exc

    _CONFIGURED_PROVIDER = register(
        project_name=resolved_settings.phoenix_project_name,
        endpoint=resolved_settings.phoenix_collector_endpoint,
        auto_instrument=True,
    )
    _CONFIGURED_KEY = key
    return _CONFIGURED_PROVIDER


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def _reset_phoenix_tracing_for_tests() -> None:
    global _CONFIGURED_PROVIDER, _CONFIGURED_KEY
    _CONFIGURED_PROVIDER = None
    _CONFIGURED_KEY = None
```

- [ ] **Step 4: Wire query script tracing**

In `scripts/query.py`, add this parser argument after the `--workspace-root` argument:

```python
    parser.add_argument("--trace-phoenix", action="store_true", help="Send this run's traces to configured Phoenix.")
```

After `settings = _build_settings(args.workspace_root)`, add:

```python
    _configure_tracing(settings, args.trace_phoenix)
```

Add this helper before `_build_settings`:

```python
def _configure_tracing(settings: Any, trace_phoenix: bool) -> None:
    from imperial_rag.tracing import configure_phoenix_tracing

    configure_phoenix_tracing(settings, enabled=True if trace_phoenix else None)
```

- [ ] **Step 5: Wire ingest script tracing**

In `scripts/ingest.py`, add this parser argument after the `--index-vectors` argument:

```python
    parser.add_argument("--trace-phoenix", action="store_true", help="Send this run's traces to configured Phoenix.")
```

After `settings = _build_settings(args.workspace_root)`, add:

```python
    _configure_tracing(settings, args.trace_phoenix)
```

Add this helper before `_run`:

```python
def _configure_tracing(settings: Any, trace_phoenix: bool) -> None:
    from imperial_rag.tracing import configure_phoenix_tracing

    configure_phoenix_tracing(settings, enabled=True if trace_phoenix else None)
```

- [ ] **Step 6: Wire Streamlit app tracing**

In `src/imperial_rag/web_app.py`, after `settings = Settings()`, add:

```python
    from imperial_rag.tracing import configure_phoenix_tracing

    configure_phoenix_tracing(settings)
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
python -m pytest tests/test_tracing.py tests/test_scripts.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit or record no-git status**

Run:

```bash
git status --short
```

Expected in the current workspace: `fatal: not a git repository (or any of the parent directories): .git`.

If this plan is executed inside a Git repository, commit:

```bash
git add src/imperial_rag/tracing.py src/imperial_rag/web_app.py scripts/query.py scripts/ingest.py tests/test_tracing.py tests/test_scripts.py
git commit -m "feat: add phoenix tracing setup"
```

### Task 3: Replace LangSmith Eval Runner With Phoenix Experiments

**Files:**
- Modify: `/Users/danil/Public/imperial/tests/test_evals.py`
- Modify: `/Users/danil/Public/imperial/tests/test_scripts.py`
- Delete: `/Users/danil/Public/imperial/scripts/run_langsmith_eval.py`
- Create: `/Users/danil/Public/imperial/scripts/run_phoenix_eval.py`

- [ ] **Step 1: Write failing Phoenix eval tests**

Replace `tests/test_evals.py` with:

```python
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace


def test_eval_questions_are_russian_jsonl_with_expected_behavior():
    lines = Path("evals/questions.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(lines) >= 3
    for line in lines:
        payload = json.loads(line)
        assert payload["question"]
        assert payload["expected_behavior"] in {"cite_answer", "refuse_if_not_found", "surface_conflict"}
        assert isinstance(payload.get("expected_source_hints", []), list)


def test_eval_runner_deterministic_citation_behavior():
    module = _load_eval_runner()

    assert module.citation_behavior(
        {"question": "x"},
        {"answer": "Ответ. [/docs/a.docx#chunk]", "citations": ["[/docs/a.docx#chunk] body"]},
        {"expected_behavior": "cite_answer"},
    )["score"] is True
    assert module.citation_behavior(
        {"question": "x"},
        {"answer": "I could not find this clearly in the indexed documents.", "citations": []},
        {"expected_behavior": "refuse_if_not_found"},
    )["score"] is True
    assert module.citation_behavior(
        {"question": "x"},
        {"answer": "Документы противоречат друг другу. [a] [b]", "citations": ["[a] body", "[b] body"]},
        {"expected_behavior": "surface_conflict"},
    )["score"] is True


def test_phoenix_evaluator_wrappers_accept_phoenix_bound_keywords():
    module = _load_eval_runner()

    assert module.phoenix_citation_behavior(
        output={"answer": "Ответ. [/docs/a.docx#chunk]", "citations": ["[/docs/a.docx#chunk] body"]},
        expected={"expected_behavior": "cite_answer"},
    ) is True
    assert module.phoenix_source_hint_behavior(
        output={"sources": ["source contains брак"]},
        expected={"expected_source_hints": ["брак"]},
    ) is True


def test_eval_runner_uses_phoenix_dataset_and_experiment_api_shape():
    source = Path("scripts/run_phoenix_eval.py").read_text(encoding="utf-8")

    assert "from phoenix.client import Client" in source
    assert "client.datasets.create_dataset" in source
    assert "inputs=inputs" in source
    assert "outputs=outputs" in source
    assert "metadata=metadata" in source
    assert "client.experiments.run_experiment" in source
    assert "langsmith" not in source.casefold()


def test_phoenix_dataset_rows_have_stable_metadata_ids():
    module = _load_eval_runner()
    example = {
        "question": "Что делать с браком?",
        "expected_behavior": "cite_answer",
        "expected_source_hints": ["брак"],
    }

    inputs, outputs, metadata = module._to_phoenix_dataset_rows([example])

    assert metadata[0]["id"]
    assert inputs == [{"question": "Что делать с браком?"}]
    assert outputs == [{
        "expected_behavior": "cite_answer",
        "expected_source_hints": ["брак"],
    }]
    assert metadata[0]["row_index"] == 0
    assert metadata[0]["source"] == "evals/questions.jsonl"


def test_phoenix_experiment_uses_documented_python_dataset_arguments(monkeypatch):
    module = _load_eval_runner()
    captured: dict[str, object] = {}

    class FakeDatasets:
        def create_dataset(self, **kwargs):
            captured["dataset"] = kwargs
            return {"dataset_id": "dataset-1"}

    class FakeExperiments:
        def run_experiment(self, **kwargs):
            captured["experiment"] = kwargs
            return SimpleNamespace(id="experiment-1")

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.datasets = FakeDatasets()
            self.experiments = FakeExperiments()

    class FakeRuntime:
        def query(self, question: str) -> dict[str, object]:
            return {"answer": f"Ответ на {question}", "citations": ["[/docs/a.docx#chunk] body"]}

    fake_phoenix = types.ModuleType("phoenix")
    fake_client_module = types.ModuleType("phoenix.client")
    fake_client_module.Client = FakeClient
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.client", fake_client_module)
    monkeypatch.setattr(module, "build_runtime", lambda settings=None: FakeRuntime())

    module._run_phoenix_experiment(
        examples=[
            {
                "question": "Что делать с браком?",
                "expected_behavior": "cite_answer",
                "expected_source_hints": ["брак"],
            }
        ],
        settings=SimpleNamespace(phoenix_client_endpoint="http://localhost:6006"),
        dataset_name="imperial-rag-gold-questions",
        experiment_name="imperial-rag-citation-grounding",
    )

    assert captured["client"] == {"base_url": "http://localhost:6006"}
    dataset_args = captured["dataset"]
    assert dataset_args["name"] == "imperial-rag-gold-questions"
    assert dataset_args["inputs"] == [{"question": "Что делать с браком?"}]
    assert dataset_args["outputs"] == [
        {"expected_behavior": "cite_answer", "expected_source_hints": ["брак"]}
    ]
    assert dataset_args["metadata"][0]["id"]
    assert dataset_args["metadata"][0]["row_index"] == 0
    assert "examples" not in dataset_args
    experiment_args = captured["experiment"]
    assert experiment_args["dataset"] == {"dataset_id": "dataset-1"}
    assert callable(experiment_args["task"])
    assert experiment_args["evaluators"] == [
        module.phoenix_citation_behavior,
        module.phoenix_source_hint_behavior,
    ]


def _load_eval_runner():
    spec = importlib.util.spec_from_file_location("run_phoenix_eval_for_test", Path("scripts/run_phoenix_eval.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

Replace `tests/test_scripts.py` with:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path


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


def test_entrypoint_scripts_expose_phoenix_tracing_flag():
    assert "--trace-phoenix" in Path("scripts/ingest.py").read_text(encoding="utf-8")
    assert "--trace-phoenix" in Path("scripts/query.py").read_text(encoding="utf-8")


def _load_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_evals.py tests/test_scripts.py -q
```

Expected: FAIL because `scripts/run_phoenix_eval.py` does not exist and `scripts/run_langsmith_eval.py` still exists.

- [ ] **Step 3: Rename and adapt the eval runner to Phoenix**

Create `scripts/run_phoenix_eval.py` by adapting the existing `scripts/run_langsmith_eval.py` deterministic core. Keep `load_questions`, `run_target`, `build_runtime`, `citation_behavior`, `source_hint_behavior`, and local eval mode; replace only the LangSmith upload adapter with Phoenix client dataset and experiment calls. Use the documented Phoenix Python dataset shape: `inputs`, `outputs`, and `metadata`. Do not use the TypeScript SDK `examples=[{"input": ..., "output": ...}]` shape in Python code.

```python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_QUESTIONS_PATH = Path("evals/questions.jsonl")
DEFAULT_EXPERIMENT_NAME = "imperial-rag-citation-grounding"
REFUSAL_FALLBACKS = (
    "I could not find this clearly in the indexed documents.",
    "не удалось найти",
    "не найдено",
    "нет в проиндексированных документах",
)


def load_questions(path: Path = DEFAULT_QUESTIONS_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not payload.get("question"):
            raise ValueError(f"missing question on line {line_number}")
        rows.append(payload)
    return rows


def target(inputs: dict[str, Any]) -> dict[str, Any]:
    return run_target(inputs)


def run_target(inputs: dict[str, Any], runtime: Any | None = None) -> dict[str, Any]:
    question = str(inputs["question"])
    resolved_runtime = runtime or build_runtime()
    result = _coerce_result(resolved_runtime.query(question))
    evidence = result.get("evidence", []) or result.get("documents", [])
    return {
        "answer": str(result.get("answer", "")),
        "citations": list(result.get("citations") or result.get("sources") or []),
        "sources": list(result.get("sources") or result.get("citations") or []),
        "documents": [_document_payload(document) for document in evidence],
    }


def build_runtime(settings: Any | None = None) -> Any:
    try:
        from imperial_rag.runtime import create_runtime
    except (ImportError, AttributeError):
        create_runtime = None
    if create_runtime is not None:
        return create_runtime(settings) if settings is not None else create_runtime()

    try:
        from imperial_rag.runtime import Runtime
    except (ImportError, AttributeError):
        Runtime = None

    if Runtime is not None:
        return Runtime(settings=settings) if settings is not None else Runtime()

    from imperial_rag.runtime import build_live_query_workflow

    workflow = build_live_query_workflow(settings) if settings is not None else build_live_query_workflow()

    class WorkflowRuntime:
        def query(self, question: str) -> dict[str, Any]:
            return _coerce_result(workflow.invoke({"question": question}))

    return WorkflowRuntime()


def citation_behavior(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = (reference_outputs or inputs).get("expected_behavior")
    answer = str(outputs.get("answer", ""))
    citations = outputs.get("citations") or outputs.get("sources") or []

    if expected == "refuse_if_not_found":
        score = _looks_like_refusal(answer) and not citations
    elif expected == "cite_answer":
        score = bool(citations) and not _looks_like_refusal(answer)
    elif expected == "surface_conflict":
        score = bool(citations) and _mentions_conflict(answer)
    else:
        score = False
    return {"key": "citation_behavior", "score": bool(score)}


def source_hint_behavior(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference = reference_outputs or inputs
    hints = [str(hint).casefold() for hint in reference.get("expected_source_hints", [])]
    if not hints:
        return {"key": "source_hint_behavior", "score": True}
    haystack = "\n".join(
        [
            *(str(source) for source in outputs.get("sources", []) or outputs.get("citations", []) or []),
            *(_document_search_text(document) for document in outputs.get("documents", []) or []),
        ]
    ).casefold()
    return {"key": "source_hint_behavior", "score": any(hint in haystack for hint in hints)}


def phoenix_citation_behavior(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> bool:
    return bool(citation_behavior(input or {}, output, expected)["score"])


def phoenix_source_hint_behavior(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> bool:
    return bool(source_hint_behavior(input or {}, output, expected)["score"])


def run_local_eval(examples: list[dict[str, Any]], runtime: Any | None = None) -> list[dict[str, Any]]:
    resolved_runtime = runtime or build_runtime()
    rows: list[dict[str, Any]] = []
    for example in examples:
        inputs = {"question": example["question"]}
        reference_outputs = {
            "expected_behavior": example["expected_behavior"],
            "expected_source_hints": example.get("expected_source_hints", []),
        }
        outputs = run_target(inputs, runtime=resolved_runtime)
        rows.append(
            {
                "question": example["question"],
                "citation_behavior": citation_behavior(inputs, outputs, reference_outputs)["score"],
                "source_hint_behavior": source_hint_behavior(inputs, outputs, reference_outputs)["score"],
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    _ensure_src_on_path()
    parser = argparse.ArgumentParser(description="Run Imperial RAG citation/refusal evaluations.")
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--use-phoenix", action="store_true", help="Store dataset and experiment results in Phoenix.")
    parser.add_argument("--trace-phoenix", action="store_true", help="Send this run's traces to configured Phoenix.")
    args = parser.parse_args(argv)

    settings = _build_settings(args.workspace_root)
    if args.trace_phoenix or args.use_phoenix:
        _configure_tracing(settings, enabled=True)
    examples = load_questions(args.questions_path)

    if args.use_phoenix:
        _run_phoenix_experiment(
            examples=examples,
            settings=settings,
            dataset_name=args.dataset_name or f"{settings.phoenix_project_name}-gold-questions",
            experiment_name=args.experiment_name,
        )
        return

    rows = run_local_eval(examples, runtime=build_runtime(settings=settings))
    passed = sum(1 for row in rows if row["citation_behavior"] and row["source_hint_behavior"])
    print(f"local_eval_examples={len(rows)}")
    print(f"local_eval_passed={passed}")


def _run_phoenix_experiment(
    examples: list[dict[str, Any]],
    settings: Any,
    dataset_name: str,
    experiment_name: str,
) -> None:
    try:
        from phoenix.client import Client
    except ImportError as exc:
        raise SystemExit("Phoenix client is not installed; install arize-phoenix-client.") from exc

    client = Client(base_url=settings.phoenix_client_endpoint)
    inputs, outputs, metadata = _to_phoenix_dataset_rows(examples)
    dataset = client.datasets.create_dataset(
        name=dataset_name,
        dataset_description="Imperial RAG gold questions loaded from evals/questions.jsonl.",
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
    )
    runtime = build_runtime(settings=settings)

    def bound_target(inputs: dict[str, Any]) -> dict[str, Any]:
        return run_target(inputs, runtime=runtime)

    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=bound_target,
        evaluators=[phoenix_citation_behavior, phoenix_source_hint_behavior],
        experiment_name=experiment_name,
        experiment_description="Imperial RAG deterministic citation, refusal, and source-hint regression checks.",
    )
    print(f"phoenix_dataset={dataset_name}")
    print(f"phoenix_examples={len(examples)}")
    print(f"phoenix_experiment={_experiment_identifier(experiment)}")


def _to_phoenix_dataset_rows(
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for row_index, example in enumerate(examples):
        expected = {
            "expected_behavior": example["expected_behavior"],
            "expected_source_hints": example.get("expected_source_hints", []),
        }
        stable_payload = json.dumps(
            {"question": example["question"], "expected": expected},
            ensure_ascii=False,
            sort_keys=True,
        )
        example_id = str(example.get("id") or hashlib.sha1(stable_payload.encode("utf-8")).hexdigest())
        inputs.append({"question": example["question"]})
        outputs.append(expected)
        metadata.append({"id": example_id, "row_index": row_index, "source": str(DEFAULT_QUESTIONS_PATH)})
    return inputs, outputs, metadata


def _configure_tracing(settings: Any, enabled: bool) -> None:
    from imperial_rag.tracing import configure_phoenix_tracing

    configure_phoenix_tracing(settings, enabled=enabled)


def _build_settings(workspace_root: Path | None) -> Any:
    from imperial_rag.config import Settings

    if workspace_root is None:
        return Settings()
    try:
        return Settings(workspace_root=workspace_root)
    except TypeError:
        os.environ["IMPERIAL_RAG_WORKSPACE_ROOT"] = str(workspace_root)
        return Settings()


def _looks_like_refusal(answer: str) -> bool:
    normalized = answer.casefold()
    return any(text.casefold() in normalized for text in REFUSAL_FALLBACKS)


def _mentions_conflict(answer: str) -> bool:
    normalized = answer.casefold()
    return any(marker in normalized for marker in ("противореч", "конфликт", "disagree", "conflict"))


def _coerce_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {
        "answer": getattr(result, "answer", ""),
        "citations": getattr(result, "citations", []),
        "sources": getattr(result, "sources", []),
        "evidence": getattr(result, "evidence", []),
    }


def _document_payload(document: Any) -> dict[str, Any]:
    if isinstance(document, dict):
        return document
    return {
        "page_content": str(getattr(document, "page_content", "")),
        "metadata": dict(getattr(document, "metadata", {}) or {}),
    }


def _document_search_text(document: dict[str, Any]) -> str:
    metadata = document.get("metadata", {}) or {}
    return " ".join(
        [
            str(document.get("page_content", "")),
            *(str(metadata.get(field, "")) for field in ("relative_path", "file_name", "parent_folder", "section_heading")),
        ]
    )


def _experiment_identifier(experiment: Any) -> str:
    for field in ("id", "experiment_id", "name"):
        value = getattr(experiment, field, None)
        if value:
            return str(value)
    return str(experiment)


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Delete the LangSmith eval runner**

Run:

```bash
rm scripts/run_langsmith_eval.py
```

Expected: `scripts/run_langsmith_eval.py` is gone and `scripts/run_phoenix_eval.py` exists.

- [ ] **Step 5: Run focused eval tests**

Run:

```bash
python -m pytest tests/test_evals.py tests/test_scripts.py -q
```

Expected: PASS.

- [ ] **Step 6: Run local eval smoke test**

Run:

```bash
python scripts/run_phoenix_eval.py
```

Expected: output includes `local_eval_examples=` and `local_eval_passed=`.

- [ ] **Step 7: Commit or record no-git status**

Run:

```bash
git status --short
```

Expected in the current workspace: `fatal: not a git repository (or any of the parent directories): .git`.

If this plan is executed inside a Git repository, commit:

```bash
git add scripts/run_phoenix_eval.py tests/test_evals.py tests/test_scripts.py
git rm scripts/run_langsmith_eval.py
git commit -m "feat: run evaluations with phoenix"
```

### Task 4: Add Self-Hosted Phoenix Stack And Supersession Notes

**Files:**
- Create: `/Users/danil/Public/imperial/compose.yaml`
- Create: `/Users/danil/Public/imperial/tests/test_phoenix_stack.py`
- Modify: `/Users/danil/Public/imperial/docs/superpowers/specs/2026-06-02-local-rag-system-design.md`
- Modify: `/Users/danil/Public/imperial/docs/superpowers/plans/2026-06-02-local-rag-system.md`

- [ ] **Step 1: Write failing stack/documentation tests**

Create `tests/test_phoenix_stack.py`:

```python
from __future__ import annotations

from pathlib import Path


def test_compose_defines_persistent_self_hosted_phoenix():
    compose = Path("compose.yaml").read_text(encoding="utf-8")

    assert "arizephoenix/phoenix:latest" in compose
    assert '"6006:6006"' in compose
    assert '"4317:4317"' in compose
    assert "PHOENIX_WORKING_DIR=/mnt/data" in compose
    assert "phoenix_data:/mnt/data" in compose


def test_old_superpowers_docs_point_to_phoenix_supersession_spec():
    spec = Path("docs/superpowers/specs/2026-06-02-local-rag-system-design.md").read_text(encoding="utf-8")
    plan = Path("docs/superpowers/plans/2026-06-02-local-rag-system.md").read_text(encoding="utf-8")

    assert "2026-06-03-phoenix-observability-design.md supersedes" in spec
    assert "2026-06-03-phoenix-observability-design.md supersedes" in plan
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_phoenix_stack.py -q
```

Expected: FAIL because `compose.yaml` does not exist and old docs do not yet point to the Phoenix supersession spec.

- [ ] **Step 3: Add Phoenix Compose service**

Create `compose.yaml`:

```yaml
services:
  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"
      - "4317:4317"
    environment:
      - PHOENIX_WORKING_DIR=/mnt/data
    volumes:
      - phoenix_data:/mnt/data

volumes:
  phoenix_data:
    driver: local
```

- [ ] **Step 4: Add supersession notes to old Superpowers docs**

In `docs/superpowers/specs/2026-06-02-local-rag-system-design.md`, add this paragraph after the `## Status` heading:

```markdown
Supersession note: `2026-06-03-phoenix-observability-design.md` supersedes the LangSmith tracing and evaluation decisions in this document. The RAG architecture remains otherwise unchanged.
```

In `docs/superpowers/plans/2026-06-02-local-rag-system.md`, add this paragraph after the first header block or opening summary:

```markdown
Supersession note: `2026-06-03-phoenix-observability-design.md` supersedes the LangSmith tracing and evaluation tasks in this plan. Use the Phoenix observability implementation plan for that slice.
```

- [ ] **Step 5: Run stack/documentation tests**

Run:

```bash
python -m pytest tests/test_phoenix_stack.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit or record no-git status**

Run:

```bash
git status --short
```

Expected in the current workspace: `fatal: not a git repository (or any of the parent directories): .git`.

If this plan is executed inside a Git repository, commit:

```bash
git add compose.yaml tests/test_phoenix_stack.py docs/superpowers/specs/2026-06-02-local-rag-system-design.md docs/superpowers/plans/2026-06-02-local-rag-system.md
git commit -m "chore: add self-hosted phoenix stack"
```

### Task 5: Full Migration Verification

**Files:**
- Inspect: `/Users/danil/Public/imperial`

- [ ] **Step 1: Search for code-level LangSmith references**

Run:

```bash
rg -n "langsmith|LangSmith|LANGSMITH|LANGCHAIN_TRACING" pyproject.toml src scripts tests
```

Expected: no output.

- [ ] **Step 2: Run focused migration tests**

Run:

```bash
python -m pytest tests/test_config.py tests/test_dependencies.py tests/test_tracing.py tests/test_evals.py tests/test_scripts.py tests/test_indexing.py tests/test_phoenix_stack.py -q
```

Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS.

- [ ] **Step 4: Run local eval mode**

Run:

```bash
python scripts/run_phoenix_eval.py
```

Expected: output includes:

```text
local_eval_examples=
local_eval_passed=
```

- [ ] **Step 5: Start Phoenix locally**

Run:

```bash
docker compose up -d phoenix
```

Expected: Docker starts a `phoenix` service exposing `http://localhost:6006`.

- [ ] **Step 6: Run Phoenix eval experiment**

Run:

```bash
python scripts/run_phoenix_eval.py --use-phoenix
```

Expected: output includes:

```text
phoenix_dataset=imperial-rag-gold-questions
phoenix_examples=
phoenix_experiment=
```

- [ ] **Step 7: Verify Phoenix health manually from CLI**

Run:

```bash
curl -I http://localhost:6006
```

Expected: HTTP response headers are returned from Phoenix. A `200`, `302`, or other Phoenix-generated HTTP status is acceptable.

- [ ] **Step 8: Stop Phoenix if this was a temporary smoke test**

Run:

```bash
docker compose down
```

Expected: Phoenix container stops. The `phoenix_data` Docker volume remains so local datasets, traces, and experiments persist.

- [ ] **Step 9: Commit or record no-git status**

Run:

```bash
git status --short
```

Expected in the current workspace: `fatal: not a git repository (or any of the parent directories): .git`.

If this plan is executed inside a Git repository and previous task commits were not made, commit all remaining migration files:

```bash
git add pyproject.toml compose.yaml src scripts tests docs/superpowers
git commit -m "feat: migrate observability to self-hosted phoenix"
```

## Self-Review

Spec coverage:

- Phoenix replaces LangSmith in settings, dependency declarations, tests, and eval runner: Tasks 1 and 3.
- Self-hosted Phoenix stack with persistent local storage: Task 4.
- Phoenix tracing via `phoenix.otel.register(..., auto_instrument=True)`: Task 2.
- Phoenix datasets and experiments for eval result storage: Task 3.
- Local evals continue without a live Phoenix server: Task 3 and Task 5.
- Old LangSmith docs are explicitly superseded: Task 4.

Placeholder scan:

- The plan contains no placeholder implementation steps.
- Every source-changing step includes concrete code or exact text to insert.

Type consistency:

- `Settings.phoenix_project_name`, `Settings.phoenix_collector_endpoint`, and `Settings.phoenix_client_endpoint` are introduced in Task 1 and reused consistently afterward.
- `configure_phoenix_tracing(settings, enabled=...)` is introduced in Task 2 and reused by scripts in Tasks 2 and 3.
- `scripts/run_phoenix_eval.py` keeps the existing local evaluator names while adding thin Phoenix evaluator wrappers over the same deterministic checks.
- Phoenix evaluator wrappers accept Phoenix-bound keyword arguments `output`, `expected`, and optional `input`; tests call them with those keyword names.
- The Phoenix dataset call uses the documented Python client arguments `inputs`, `outputs`, and `metadata`; the fake-client test rejects the TypeScript-style `examples` argument.
