# Qwen Model Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move all default model-backed behavior in Imperial RAG from OpenAI/Cohere to hosted Qwen through Alibaba DashScope.

**Architecture:** Add a focused provider module that owns DashScope/Qwen configuration, credentials, endpoints, model factories, OCR payload parsing, and vector provider metadata. Existing ingestion, indexing, retrieval, runtime, and scripts call the provider module instead of importing model SDKs directly. Keyword-only retrieval remains available when DashScope is not configured, while vector indexing and vector retrieval refuse mismatched embedding collections.

**Tech Stack:** Python 3.12+, uv, pytest, LangChain, LangGraph, Qdrant, DashScope Python SDK, `langchain-qwq`, `langchain-community` DashScope embeddings/reranker where still required.

---

## Files And Responsibilities

- Modify: `pyproject.toml` - add `dashscope` and `langchain-qwq`.
- Modify: `.env.example` - replace OpenAI/Cohere defaults with DashScope/Qwen settings and keep legacy toggles clearly opt-in.
- Modify: `README.md` - update provider setup and Qdrant collection guidance.
- Create: `src/imperial_rag/providers.py` - provider config, capability checks, Qwen chat/embedding/rerank factories, OCR response parsing, vector metadata helpers, and provider mismatch errors.
- Modify: `src/imperial_rag/ocr.py` - replace default OpenAI OCR client with Qwen OCR client using the provider payload/parser.
- Modify: `src/imperial_rag/indexing.py` - use provider embeddings by default, guard vector collection metadata, and record Qwen provider metadata after indexing.
- Modify: `src/imperial_rag/runtime.py` - use Qwen chat by default and gate semantic search on DashScope provider readiness plus vector metadata compatibility.
- Modify: `src/imperial_rag/retrieval.py` - replace Cohere rerank defaults with DashScope `qwen3-rerank`, preserving deterministic fallback.
- Modify: `src/imperial_rag/pipeline.py` - build Qwen OCR and vector clients through provider-aware helpers and record embedding metadata in manifest/index status.
- Modify: `src/imperial_rag/workflows.py` - remove default top-level OpenAI import and keep any OpenAI compatibility only behind explicit legacy opt-in.
- Modify: `scripts/ingest.py` - fail fast for `--index-vectors` without DashScope key, and build Qwen OCR/vector clients through the provider path.
- Modify: `scripts/query.py` - no direct provider work expected, but include in verification because it exercises runtime creation.
- Create: `tests/test_providers.py` - offline tests for provider config, factories, OCR parsing, and vector metadata helpers.
- Modify: `tests/test_dependencies.py` - dependency and import checks.
- Modify: `tests/test_indexing.py` - default Qwen embeddings and vector metadata guard tests.
- Modify: `tests/test_runtime.py` - DashScope semantic readiness, provider mismatch diagnostics, and Qwen chat default tests.
- Modify: `tests/test_retrieval.py` - DashScope rerank and deterministic fallback tests.
- Modify: `tests/test_scripts.py` - ingest script key-gate behavior.
- Modify: `tests/test_pipeline.py` and `tests/test_pipeline_integration.py` - manifest embedding model/status behavior after Qwen indexing.

## Task 1: Dependencies And Provider Config Tests

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `tests/test_dependencies.py`
- Create: `tests/test_providers.py`

- [ ] **Step 1: Write failing dependency tests**

Append these tests to `tests/test_dependencies.py`:

```python
def test_project_includes_dashscope_qwen_dependencies():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {_normalize_dependency_name(dependency) for dependency in pyproject["project"]["dependencies"]}

    assert "dashscope" in dependencies
    assert "langchain-qwq" in dependencies


def test_dashscope_qwen_imports_are_available_after_sync():
    import dashscope
    from langchain_qwq import ChatQwen

    assert hasattr(dashscope, "TextEmbedding")
    assert hasattr(dashscope, "TextReRank")
    assert hasattr(dashscope, "MultiModalConversation")
    assert ChatQwen is not None
```

- [ ] **Step 2: Write failing provider config tests**

Create `tests/test_providers.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from imperial_rag.config import Settings


def clear_provider_env(monkeypatch):
    for name in (
        "DASHSCOPE_API_KEY",
        "IMPERIAL_RAG_DASHSCOPE_REGION",
        "IMPERIAL_RAG_DASHSCOPE_BASE_URL",
        "IMPERIAL_RAG_DASHSCOPE_COMPAT_BASE_URL",
        "IMPERIAL_RAG_QWEN_CHAT_MODEL",
        "IMPERIAL_RAG_QWEN_VISION_MODEL",
        "IMPERIAL_RAG_QWEN_OCR_TASK",
        "IMPERIAL_RAG_QWEN_OCR_MIN_PIXELS",
        "IMPERIAL_RAG_QWEN_OCR_MAX_PIXELS",
        "IMPERIAL_RAG_QWEN_OCR_ENABLE_ROTATE",
        "IMPERIAL_RAG_QWEN_EMBEDDING_MODEL",
        "IMPERIAL_RAG_QWEN_EMBEDDING_DIMENSIONS",
        "IMPERIAL_RAG_QWEN_RERANK_MODEL",
        "IMPERIAL_RAG_ALLOW_LEGACY_OPENAI",
        "IMPERIAL_RAG_ALLOW_LEGACY_COHERE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_qwen_provider_settings_defaults(monkeypatch):
    clear_provider_env(monkeypatch)

    from imperial_rag.providers import QwenProviderSettings

    settings = QwenProviderSettings.from_env()

    assert settings.api_key is None
    assert settings.region == "beijing"
    assert settings.base_url == "https://dashscope.aliyuncs.com/api/v1"
    assert settings.compat_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert settings.chat_model == "qwen3.7-max"
    assert settings.vision_model == "qwen-vl-ocr-2025-11-20"
    assert settings.ocr_task == "multi_lan"
    assert settings.ocr_min_pixels is None
    assert settings.ocr_max_pixels is None
    assert settings.ocr_enable_rotate is None
    assert settings.embedding_model == "text-embedding-v4"
    assert settings.embedding_dimensions == 2048
    assert settings.rerank_model == "qwen3-rerank"
    assert settings.allow_legacy_openai is False
    assert settings.allow_legacy_cohere is False


def test_qwen_provider_settings_read_environment(monkeypatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setenv("IMPERIAL_RAG_DASHSCOPE_REGION", "singapore")
    monkeypatch.setenv("IMPERIAL_RAG_DASHSCOPE_BASE_URL", "https://workspace.ap-southeast-1.maas.aliyuncs.com/api/v1")
    monkeypatch.setenv("IMPERIAL_RAG_DASHSCOPE_COMPAT_BASE_URL", "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_CHAT_MODEL", "qwen-test-chat")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_VISION_MODEL", "qwen-test-ocr")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_OCR_TASK", "text_recognition")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_OCR_MIN_PIXELS", "3072")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_OCR_MAX_PIXELS", "8388608")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_OCR_ENABLE_ROTATE", "true")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_EMBEDDING_MODEL", "text-embedding-v4")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_RERANK_MODEL", "qwen-test-rerank")
    monkeypatch.setenv("IMPERIAL_RAG_ALLOW_LEGACY_OPENAI", "1")
    monkeypatch.setenv("IMPERIAL_RAG_ALLOW_LEGACY_COHERE", "yes")

    from imperial_rag.providers import QwenProviderSettings

    settings = QwenProviderSettings.from_env()

    assert settings.api_key == "dashscope-test-key"
    assert settings.region == "singapore"
    assert settings.base_url == "https://workspace.ap-southeast-1.maas.aliyuncs.com/api/v1"
    assert settings.compat_base_url == "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    assert settings.chat_model == "qwen-test-chat"
    assert settings.vision_model == "qwen-test-ocr"
    assert settings.ocr_task == "text_recognition"
    assert settings.ocr_min_pixels == 3072
    assert settings.ocr_max_pixels == 8388608
    assert settings.ocr_enable_rotate is True
    assert settings.embedding_dimensions == 1024
    assert settings.rerank_model == "qwen-test-rerank"
    assert settings.allow_legacy_openai is True
    assert settings.allow_legacy_cohere is True


def test_dashscope_configured_requires_key(monkeypatch):
    clear_provider_env(monkeypatch)

    from imperial_rag.providers import dashscope_configured

    assert dashscope_configured() is False
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    assert dashscope_configured() is True
```

- [ ] **Step 3: Run the new tests and verify failure**

Run:

```bash
uv run python -m pytest tests/test_dependencies.py tests/test_providers.py -q
```

Expected: FAIL because `dashscope` and `langchain-qwq` are not in `pyproject.toml`, and `imperial_rag.providers` does not exist.

- [ ] **Step 4: Add dependencies**

Modify `pyproject.toml` dependencies:

```toml
dependencies = [
  "dashscope",
  "langchain",
  "langchain-community",
  "langchain-core",
  "langchain-openai",
  "langchain-qdrant",
  "langchain-cohere",
  "langchain-qwq",
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

Run:

```bash
uv sync --extra dev
```

Expected: dependencies resolve and `uv.lock` changes.

- [ ] **Step 5: Add provider config module**

Create `src/imperial_rag/providers.py` with this initial content:

```python
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_DASHSCOPE_COMPAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_CHAT_MODEL = "qwen3.7-max"
DEFAULT_QWEN_VISION_MODEL = "qwen-vl-ocr-2025-11-20"
DEFAULT_QWEN_OCR_TASK = "multi_lan"
DEFAULT_QWEN_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_QWEN_EMBEDDING_DIMENSIONS = 2048
DEFAULT_QWEN_RERANK_MODEL = "qwen3-rerank"
VECTOR_PROVIDER = "dashscope"


class MissingDashScopeKeyError(RuntimeError):
    pass


class VectorProviderMismatchError(RuntimeError):
    pass


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _env_optional_str(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()


def _env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _env_optional_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class QwenProviderSettings:
    api_key: str | None
    region: str = "beijing"
    base_url: str = DEFAULT_DASHSCOPE_BASE_URL
    compat_base_url: str = DEFAULT_DASHSCOPE_COMPAT_BASE_URL
    chat_model: str = DEFAULT_QWEN_CHAT_MODEL
    vision_model: str = DEFAULT_QWEN_VISION_MODEL
    ocr_task: str = DEFAULT_QWEN_OCR_TASK
    ocr_min_pixels: int | None = None
    ocr_max_pixels: int | None = None
    ocr_enable_rotate: bool | None = None
    embedding_model: str = DEFAULT_QWEN_EMBEDDING_MODEL
    embedding_dimensions: int | None = DEFAULT_QWEN_EMBEDDING_DIMENSIONS
    rerank_model: str = DEFAULT_QWEN_RERANK_MODEL
    allow_legacy_openai: bool = False
    allow_legacy_cohere: bool = False

    @classmethod
    def from_env(cls) -> "QwenProviderSettings":
        return cls(
            api_key=_env_optional_str("DASHSCOPE_API_KEY"),
            region=_env_str("IMPERIAL_RAG_DASHSCOPE_REGION", "beijing"),
            base_url=_env_str("IMPERIAL_RAG_DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL),
            compat_base_url=_env_str("IMPERIAL_RAG_DASHSCOPE_COMPAT_BASE_URL", DEFAULT_DASHSCOPE_COMPAT_BASE_URL),
            chat_model=_env_str("IMPERIAL_RAG_QWEN_CHAT_MODEL", DEFAULT_QWEN_CHAT_MODEL),
            vision_model=_env_str("IMPERIAL_RAG_QWEN_VISION_MODEL", DEFAULT_QWEN_VISION_MODEL),
            ocr_task=_env_str("IMPERIAL_RAG_QWEN_OCR_TASK", DEFAULT_QWEN_OCR_TASK),
            ocr_min_pixels=_env_optional_int("IMPERIAL_RAG_QWEN_OCR_MIN_PIXELS"),
            ocr_max_pixels=_env_optional_int("IMPERIAL_RAG_QWEN_OCR_MAX_PIXELS"),
            ocr_enable_rotate=_env_optional_bool("IMPERIAL_RAG_QWEN_OCR_ENABLE_ROTATE"),
            embedding_model=_env_str("IMPERIAL_RAG_QWEN_EMBEDDING_MODEL", DEFAULT_QWEN_EMBEDDING_MODEL),
            embedding_dimensions=_env_optional_int("IMPERIAL_RAG_QWEN_EMBEDDING_DIMENSIONS")
            if os.environ.get("IMPERIAL_RAG_QWEN_EMBEDDING_DIMENSIONS", "").strip()
            else DEFAULT_QWEN_EMBEDDING_DIMENSIONS,
            rerank_model=_env_str("IMPERIAL_RAG_QWEN_RERANK_MODEL", DEFAULT_QWEN_RERANK_MODEL),
            allow_legacy_openai=_env_bool("IMPERIAL_RAG_ALLOW_LEGACY_OPENAI"),
            allow_legacy_cohere=_env_bool("IMPERIAL_RAG_ALLOW_LEGACY_COHERE"),
        )

    def require_api_key(self) -> str:
        if not self.api_key:
            raise MissingDashScopeKeyError("DASHSCOPE_API_KEY is required for hosted Qwen/DashScope behavior.")
        return self.api_key

    def vector_metadata(self) -> "VectorProviderMetadata":
        return VectorProviderMetadata(
            provider=VECTOR_PROVIDER,
            embedding_model=self.embedding_model,
            embedding_dimensions=self.embedding_dimensions,
            distance="cosine",
        )


@dataclass(frozen=True)
class VectorProviderMetadata:
    provider: str
    embedding_model: str
    embedding_dimensions: int | None
    distance: str = "cosine"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VectorProviderMetadata":
        return cls(
            provider=str(payload["provider"]),
            embedding_model=str(payload["embedding_model"]),
            embedding_dimensions=payload.get("embedding_dimensions"),
            distance=str(payload.get("distance", "cosine")),
        )


def dashscope_configured(settings: QwenProviderSettings | None = None) -> bool:
    resolved = settings or QwenProviderSettings.from_env()
    return bool(resolved.api_key)


def configure_dashscope_sdk(settings: QwenProviderSettings | None = None) -> None:
    resolved = settings or QwenProviderSettings.from_env()
    api_key = resolved.require_api_key()
    import dashscope

    dashscope.api_key = api_key
    dashscope.base_http_api_url = resolved.base_url


def vector_metadata_path(settings: Any) -> Path:
    return Path(settings.processed_root) / "vector_provider.json"


def read_vector_metadata(settings: Any) -> VectorProviderMetadata | None:
    path = vector_metadata_path(settings)
    if not path.exists():
        return None
    return VectorProviderMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_vector_metadata(settings: Any, metadata: VectorProviderMetadata) -> None:
    path = vector_metadata_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def vector_metadata_matches_config(settings: Any, provider_settings: QwenProviderSettings | None = None) -> bool:
    existing = read_vector_metadata(settings)
    if existing is None:
        return False
    expected = (provider_settings or QwenProviderSettings.from_env()).vector_metadata()
    return existing == expected


def ensure_vector_metadata_compatible(settings: Any, provider_settings: QwenProviderSettings | None = None) -> None:
    existing = read_vector_metadata(settings)
    if existing is None:
        return
    expected = (provider_settings or QwenProviderSettings.from_env()).vector_metadata()
    if existing != expected:
        raise VectorProviderMismatchError(
            "Qdrant vector provider metadata mismatch: "
            f"existing={existing.to_dict()} expected={expected.to_dict()}"
        )
```

- [ ] **Step 6: Update `.env.example`**

Replace the AI provider credential section and Qdrant collection default with:

```dotenv
# AI provider credentials
# Required for hosted Qwen answer generation, embeddings, vector indexing, OCR, and reranking.
DASHSCOPE_API_KEY=

# DashScope endpoints.
# Beijing defaults:
IMPERIAL_RAG_DASHSCOPE_REGION=beijing
IMPERIAL_RAG_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1
IMPERIAL_RAG_DASHSCOPE_COMPAT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# Hosted Qwen models.
IMPERIAL_RAG_QWEN_CHAT_MODEL=qwen3.7-max
IMPERIAL_RAG_QWEN_VISION_MODEL=qwen-vl-ocr-2025-11-20
IMPERIAL_RAG_QWEN_OCR_TASK=multi_lan
IMPERIAL_RAG_QWEN_OCR_MIN_PIXELS=
IMPERIAL_RAG_QWEN_OCR_MAX_PIXELS=
IMPERIAL_RAG_QWEN_OCR_ENABLE_ROTATE=
IMPERIAL_RAG_QWEN_EMBEDDING_MODEL=text-embedding-v4
IMPERIAL_RAG_QWEN_EMBEDDING_DIMENSIONS=2048
IMPERIAL_RAG_QWEN_RERANK_MODEL=qwen3-rerank

# Legacy provider escape hatches for debugging only.
IMPERIAL_RAG_ALLOW_LEGACY_OPENAI=false
IMPERIAL_RAG_ALLOW_LEGACY_COHERE=false
OPENAI_API_KEY=
AZURE_OPENAI_API_KEY=
COHERE_API_KEY=

# Qdrant vector store
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=imperial_chunks_qwen
```

Keep the remaining local workspace, Phoenix, live Qdrant, and retrieval tuning lines unchanged.

- [ ] **Step 7: Run tests and commit**

Run:

```bash
uv run python -m pytest tests/test_dependencies.py tests/test_providers.py -q
git status --short
git diff -- pyproject.toml .env.example tests/test_dependencies.py tests/test_providers.py src/imperial_rag/providers.py
git add pyproject.toml uv.lock .env.example tests/test_dependencies.py tests/test_providers.py src/imperial_rag/providers.py
git commit -m "feat: add qwen provider configuration"
```

Expected: provider config tests pass, import test passes after `uv sync`, and only the listed files are staged.

## Task 2: Provider Factories And OCR Parser

**Files:**
- Modify: `src/imperial_rag/providers.py`
- Modify: `tests/test_providers.py`

- [ ] **Step 1: Add failing provider factory tests**

Append to `tests/test_providers.py`:

```python
def test_qwen_chat_factory_uses_chatqwen(monkeypatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")

    created = {}

    class FakeChatQwen:
        def __init__(self, model, temperature):
            created["model"] = model
            created["temperature"] = temperature

    import imperial_rag.providers as providers

    monkeypatch.setattr(providers, "_import_chat_qwen", lambda: FakeChatQwen)

    model = providers.create_chat_model()

    assert isinstance(model, FakeChatQwen)
    assert created == {"model": "qwen3.7-max", "temperature": 0}


def test_qwen_embedding_factory_uses_dimension_aware_wrapper(monkeypatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_EMBEDDING_DIMENSIONS", "2048")

    from imperial_rag.providers import DashScopeTextEmbeddings, create_embeddings

    embeddings = create_embeddings()

    assert isinstance(embeddings, DashScopeTextEmbeddings)
    assert embeddings.model == "text-embedding-v4"
    assert embeddings.dimensions == 2048


def test_qwen_reranker_factory_uses_dashscope_rerank(monkeypatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    created = {}

    class FakeDashScopeRerank:
        def __init__(self, model, top_n, api_key):
            created["model"] = model
            created["top_n"] = top_n
            created["api_key"] = api_key

    import imperial_rag.providers as providers

    monkeypatch.setattr(providers, "_import_dashscope_rerank", lambda: FakeDashScopeRerank)

    reranker = providers.create_reranker(top_n=7)

    assert isinstance(reranker, FakeDashScopeRerank)
    assert created == {"model": "qwen3-rerank", "top_n": 7, "api_key": "dashscope-test-key"}


def test_dashscope_text_embeddings_call_sdk_with_dimensions(monkeypatch):
    calls = []

    class FakeTextEmbedding:
        @staticmethod
        def call(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={"embeddings": [{"embedding": [1.0, 2.0]}, {"embedding": [3.0, 4.0]}]},
            )

    from imperial_rag.providers import DashScopeTextEmbeddings, QwenProviderSettings

    settings = QwenProviderSettings(api_key="key", embedding_dimensions=2048)
    embeddings = DashScopeTextEmbeddings(settings=settings, client=FakeTextEmbedding)

    assert embeddings.embed_documents(["a", "b"]) == [[1.0, 2.0], [3.0, 4.0]]
    assert calls == [
        {
            "model": "text-embedding-v4",
            "input": ["a", "b"],
            "text_type": "document",
            "dimension": 2048,
            "api_key": "key",
        }
    ]


def test_dashscope_text_embeddings_raise_clean_error_without_secret():
    class FakeTextEmbedding:
        @staticmethod
        def call(**kwargs):
            return SimpleNamespace(status_code=401, code="InvalidApiKey", message="bad key sk-secret")

    from imperial_rag.providers import DashScopeProviderError, DashScopeTextEmbeddings, QwenProviderSettings

    embeddings = DashScopeTextEmbeddings(
        settings=QwenProviderSettings(api_key="sk-secret"),
        client=FakeTextEmbedding,
    )

    with pytest.raises(DashScopeProviderError) as exc:
        embeddings.embed_query("question")

    assert "sk-secret" not in str(exc.value)
    assert "InvalidApiKey" in str(exc.value)


def test_build_qwen_ocr_message_includes_base64_and_options(tmp_path, monkeypatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_OCR_MIN_PIXELS", "3072")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_OCR_MAX_PIXELS", "8388608")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_OCR_ENABLE_ROTATE", "false")
    image_path = tmp_path / "scan.jpg"
    image_path.write_bytes(b"fake-image")

    from imperial_rag.providers import build_qwen_ocr_message, QwenProviderSettings

    message = build_qwen_ocr_message(image_path, QwenProviderSettings.from_env())

    content = message["content"][0]
    assert message["role"] == "user"
    assert content["image"].startswith("data:image/jpeg;base64,")
    assert content["min_pixels"] == 3072
    assert content["max_pixels"] == 8388608
    assert content["enable_rotate"] is False


def test_parse_qwen_ocr_response_extracts_text():
    from imperial_rag.providers import parse_qwen_ocr_response

    response = {
        "output": {
            "choices": [
                {
                    "message": SimpleNamespace(
                        content=[
                            {"text": " Распознанный текст "},
                        ]
                    )
                }
            ]
        }
    }

    assert parse_qwen_ocr_response(response) == "Распознанный текст"
```

- [ ] **Step 2: Run provider factory tests and verify failure**

Run:

```bash
uv run python -m pytest tests/test_providers.py -q
```

Expected: FAIL because factories, embedding wrapper, OCR message builder, parser, and error class are missing.

- [ ] **Step 3: Add provider factories and OCR parser**

Append this code to `src/imperial_rag/providers.py`:

```python
import base64
import mimetypes

from langchain_core.embeddings import Embeddings


class DashScopeProviderError(RuntimeError):
    pass


def _import_chat_qwen():
    from langchain_qwq import ChatQwen

    return ChatQwen


def _import_dashscope_rerank():
    from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank

    return DashScopeRerank


def create_chat_model(settings: QwenProviderSettings | None = None) -> Any:
    resolved = settings or QwenProviderSettings.from_env()
    resolved.require_api_key()
    chat_cls = _import_chat_qwen()
    return chat_cls(model=resolved.chat_model, temperature=0)


def create_reranker(top_n: int, settings: QwenProviderSettings | None = None) -> Any:
    resolved = settings or QwenProviderSettings.from_env()
    api_key = resolved.require_api_key()
    reranker_cls = _import_dashscope_rerank()
    return reranker_cls(model=resolved.rerank_model, top_n=top_n, api_key=api_key)


class DashScopeTextEmbeddings(Embeddings):
    def __init__(self, settings: QwenProviderSettings | None = None, client: Any | None = None) -> None:
        self.settings = settings or QwenProviderSettings.from_env()
        self.api_key = self.settings.require_api_key()
        self.model = self.settings.embedding_model
        self.dimensions = self.settings.embedding_dimensions
        if client is None:
            import dashscope

            client = dashscope.TextEmbedding
        self.client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, text_type="document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], text_type="query")[0]

    def _embed(self, texts: list[str], text_type: str) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "text_type": text_type,
            "api_key": self.api_key,
        }
        if self.dimensions is not None:
            kwargs["dimension"] = self.dimensions
        response = self.client.call(**kwargs)
        status_code = getattr(response, "status_code", None)
        if status_code != 200:
            code = getattr(response, "code", "dashscope_error")
            message = _sanitize_provider_message(getattr(response, "message", "DashScope request failed"), self.api_key)
            raise DashScopeProviderError(f"DashScope embedding failed: status_code={status_code} code={code} message={message}")
        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")
        embeddings = output["embeddings"]
        return [list(item["embedding"]) for item in embeddings]


def create_embeddings(settings: QwenProviderSettings | None = None) -> Embeddings:
    resolved = settings or QwenProviderSettings.from_env()
    if resolved.embedding_dimensions is not None:
        return DashScopeTextEmbeddings(settings=resolved)
    from langchain_community.embeddings.dashscope import DashScopeEmbeddings

    return DashScopeEmbeddings(model=resolved.embedding_model, dashscope_api_key=resolved.require_api_key())


def _sanitize_provider_message(message: str, api_key: str | None) -> str:
    sanitized = str(message)
    if api_key:
        sanitized = sanitized.replace(api_key, "[redacted]")
    return sanitized


def build_qwen_ocr_message(image_path: Path, settings: QwenProviderSettings | None = None) -> dict[str, Any]:
    resolved = settings or QwenProviderSettings.from_env()
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    image_payload: dict[str, Any] = {"image": f"data:{mime_type};base64,{encoded}"}
    if resolved.ocr_min_pixels is not None:
        image_payload["min_pixels"] = resolved.ocr_min_pixels
    if resolved.ocr_max_pixels is not None:
        image_payload["max_pixels"] = resolved.ocr_max_pixels
    if resolved.ocr_enable_rotate is not None:
        image_payload["enable_rotate"] = resolved.ocr_enable_rotate
    return {"role": "user", "content": [image_payload]}


def parse_qwen_ocr_response(response: Any) -> str:
    output = _response_get(response, "output")
    choices = _response_get(output, "choices") or []
    if not choices:
        return ""
    message = _response_get(choices[0], "message")
    content = _response_get(message, "content") or []
    if isinstance(content, str):
        return content.strip()
    text_parts: list[str] = []
    for item in content:
        text = _response_get(item, "text")
        if text:
            text_parts.append(str(text))
    return "\n".join(part.strip() for part in text_parts if part.strip()).strip()


def _response_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
uv run python -m pytest tests/test_providers.py tests/test_dependencies.py -q
git add src/imperial_rag/providers.py tests/test_providers.py
git commit -m "feat: add qwen provider factories"
```

Expected: tests pass, and the provider module supports config, factories, OCR payload building, OCR parsing, and sanitized provider errors.

## Task 3: Qwen OCR Client And Ingestion OCR Gate

**Files:**
- Modify: `src/imperial_rag/ocr.py`
- Modify: `src/imperial_rag/pipeline.py`
- Modify: `scripts/ingest.py`
- Modify: `tests/test_providers.py`
- Modify: `tests/test_extraction.py`
- Modify: `tests/test_scripts.py`

- [ ] **Step 1: Add failing OCR client tests**

Append to `tests/test_providers.py`:

```python
def test_qwen_ocr_client_calls_multimodal_conversation(tmp_path, monkeypatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(b"fake-image")
    calls = []

    class FakeConversation:
        @staticmethod
        def call(**kwargs):
            calls.append(kwargs)
            return {
                "output": {
                    "choices": [
                        {"message": SimpleNamespace(content=[{"text": "OCR text"}])}
                    ]
                }
            }

    from imperial_rag.ocr import OcrResult, QwenOcrClient
    from imperial_rag.providers import QwenProviderSettings

    client = QwenOcrClient(settings=QwenProviderSettings.from_env(), conversation_client=FakeConversation)
    result = client.extract_image_text(image_path)

    assert result == OcrResult(text="OCR text", method="dashscope:qwen-vl-ocr-2025-11-20")
    assert calls[0]["model"] == "qwen-vl-ocr-2025-11-20"
    assert calls[0]["api_key"] == "dashscope-test-key"
    assert calls[0]["ocr_options"] == {"task": "multi_lan"}
    assert calls[0]["messages"][0]["role"] == "user"
```

Append to `tests/test_scripts.py`:

```python
def test_ingest_ocr_gate_uses_dashscope_key(monkeypatch):
    module = _load_script("scripts/ingest.py", "ingest_script_ocr_gate")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-legacy-key")

    assert module._ocr_appears_configured() is False

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    assert module._ocr_appears_configured() is True
```

- [ ] **Step 2: Run OCR tests and verify failure**

Run:

```bash
uv run python -m pytest tests/test_providers.py::test_qwen_ocr_client_calls_multimodal_conversation tests/test_scripts.py::test_ingest_ocr_gate_uses_dashscope_key -q
```

Expected: FAIL because `QwenOcrClient` is missing and the script still gates OCR on OpenAI/Azure keys.

- [ ] **Step 3: Replace default OCR client**

Modify `src/imperial_rag/ocr.py` so the default client is Qwen-backed and legacy OpenAI is opt-in:

```python
class QwenOcrClient:
    def __init__(self, settings=None, conversation_client=None) -> None:
        from imperial_rag.providers import QwenProviderSettings

        self.settings = settings or QwenProviderSettings.from_env()
        self.api_key = self.settings.require_api_key()
        if conversation_client is None:
            import dashscope

            conversation_client = dashscope.MultiModalConversation
        self.conversation_client = conversation_client

    def extract_image_text(self, image_path: Path) -> OcrResult:
        from imperial_rag.providers import build_qwen_ocr_message, parse_qwen_ocr_response

        response = self.conversation_client.call(
            api_key=self.api_key,
            model=self.settings.vision_model,
            messages=[build_qwen_ocr_message(image_path, self.settings)],
            ocr_options={"task": self.settings.ocr_task},
        )
        return OcrResult(
            text=parse_qwen_ocr_response(response),
            method=f"dashscope:{self.settings.vision_model}",
        )


class LegacyOpenAIOcrClient:
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self._model_name = model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from langchain_openai import ChatOpenAI

            self._model = ChatOpenAI(model=self._model_name, temperature=0)
        return self._model

    def extract_image_text(self, image_path: Path) -> OcrResult:
        image_bytes = image_path.read_bytes()
        encoded = base64.b64encode(image_bytes).decode("ascii")
        mime_type, _ = mimetypes.guess_type(image_path.name)
        mime_type = mime_type or "image/jpeg"
        response = self.model.invoke(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all visible Russian and English text verbatim. Do not summarize.",
                        },
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
                    ],
                }
            ]
        )
        return OcrResult(text=str(response.content).strip(), method="legacy_openai_vision")


OcrClient = QwenOcrClient
```

Keep `OcrResult` and `OcrCache` unchanged.

- [ ] **Step 4: Update OCR gates**

In `src/imperial_rag/pipeline.py`, replace `_build_ocr_client` and `_ocr_appears_configured` with:

```python
def _build_ocr_client(enable_ocr: bool) -> Any | None:
    if not enable_ocr or not _ocr_appears_configured():
        return None
    from imperial_rag.ocr import OcrClient

    return OcrClient()


def _ocr_appears_configured() -> bool:
    from imperial_rag.providers import dashscope_configured

    return dashscope_configured()
```

In `scripts/ingest.py`, replace `_ocr_appears_configured` with:

```python
def _ocr_appears_configured() -> bool:
    from imperial_rag.providers import dashscope_configured

    return dashscope_configured()
```

- [ ] **Step 5: Run OCR/extraction tests and commit**

Run:

```bash
uv run python -m pytest tests/test_providers.py tests/test_extraction.py tests/test_scripts.py -q
git add src/imperial_rag/ocr.py src/imperial_rag/pipeline.py scripts/ingest.py tests/test_providers.py tests/test_scripts.py
git commit -m "feat: switch ocr client to qwen"
```

Expected: OCR tests pass, extraction tests keep using fake OCR clients, and no test needs a live DashScope call.

## Task 4: Qwen Embeddings And Vector Metadata Guards

**Files:**
- Modify: `src/imperial_rag/indexing.py`
- Modify: `src/imperial_rag/pipeline.py`
- Modify: `tests/test_indexing.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add failing indexing metadata tests**

Add this import near the top of `tests/test_indexing.py`:

```python
import pytest
```

Append to `tests/test_indexing.py`:

```python
def test_create_qdrant_vector_store_uses_qwen_embeddings_by_default(monkeypatch, tmp_path: Path) -> None:
    created = {}

    class FakeClient:
        def __init__(self, url):
            created["url"] = url

    class FakeVectorStore:
        def __init__(self, client, collection_name, embedding):
            created["client"] = client
            created["collection_name"] = collection_name
            created["embedding"] = embedding

    fake_embeddings = object()
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setattr("imperial_rag.indexing.QdrantClient", FakeClient)
    monkeypatch.setattr("imperial_rag.indexing.QdrantVectorStore", FakeVectorStore)
    monkeypatch.setattr("imperial_rag.indexing.create_embeddings", lambda: fake_embeddings)

    settings = Settings(workspace_root=tmp_path, qdrant_url="http://127.0.0.1:6333", qdrant_collection="test")
    store = create_qdrant_vector_store(settings)

    assert isinstance(store, FakeVectorStore)
    assert created["embedding"] is fake_embeddings


def test_index_vector_documents_records_qwen_vector_metadata(monkeypatch, tmp_path: Path) -> None:
    from imperial_rag.providers import read_vector_metadata

    class FakeVectorStore:
        def add_documents(self, documents, ids):
            return ids

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    settings = Settings(workspace_root=tmp_path)
    docs = [Document(page_content="one", metadata={"citation_id": "file1:body:0"})]

    index_vector_documents(docs, settings=settings, vector_store=FakeVectorStore())

    metadata = read_vector_metadata(settings)
    assert metadata is not None
    assert metadata.provider == "dashscope"
    assert metadata.embedding_model == "text-embedding-v4"
    assert metadata.embedding_dimensions == 2048


def test_create_qdrant_vector_store_rejects_mismatched_vector_metadata(monkeypatch, tmp_path: Path) -> None:
    from imperial_rag.providers import VectorProviderMetadata, VectorProviderMismatchError, write_vector_metadata

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    settings = Settings(workspace_root=tmp_path)
    write_vector_metadata(
        settings,
        VectorProviderMetadata(
            provider="openai",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
            distance="cosine",
        ),
    )

    with pytest.raises(VectorProviderMismatchError):
        create_qdrant_vector_store(settings)
```

Append to `tests/test_pipeline.py`:

```python
def test_run_ingestion_records_embedding_model_when_vector_indexed(tmp_path, monkeypatch):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.txt").write_text("Регламент возврата брака.", encoding="utf-8")
    _install_fake_dependencies(monkeypatch)

    class FakeVectorStore:
        def add_documents(self, documents, ids):
            return ids

    summary = run_ingestion(settings=FakeSettings(tmp_path), enable_ocr=False, index_vectors=True)

    assert summary.vector_indexed is True
    assert FakeManifestStore.last is not None
    assert FakeManifestStore.last.index_updates[0]["embedding_model"] == "text-embedding-v4:2048"
```

In `_install_fake_dependencies`, change fake indexing module setup to expose metadata helpers:

```python
    indexing = ModuleType("imperial_rag.indexing")
    indexing.KeywordIndex = FakeKeywordIndex
    indexing.create_qdrant_vector_store = lambda settings: SimpleNamespace(add_documents=lambda documents, ids: ids)
    indexing.index_vector_documents = lambda documents, settings=None, vector_store=None: [doc.metadata["chunk_id"] for doc in documents]
    indexing.index_documents = lambda vector_store, documents: [doc.metadata["chunk_id"] for doc in documents]
    indexing.embedding_model_identifier = lambda: "text-embedding-v4:2048"
```

- [ ] **Step 2: Run indexing tests and verify failure**

Run:

```bash
uv run python -m pytest tests/test_indexing.py tests/test_pipeline.py -q
```

Expected: FAIL because `create_embeddings`, provider metadata recording, and fake dependency changes are not wired.

- [ ] **Step 3: Wire Qwen embeddings and metadata in indexing**

Modify imports in `src/imperial_rag/indexing.py`:

```python
from imperial_rag.providers import (
    create_embeddings,
    ensure_vector_metadata_compatible,
    QwenProviderSettings,
    write_vector_metadata,
)
```

Remove `from langchain_openai import OpenAIEmbeddings`.

Replace `create_qdrant_vector_store` with:

```python
def create_qdrant_vector_store(settings: Settings, embeddings: object | None = None) -> QdrantVectorStore:
    if embeddings is None:
        ensure_vector_metadata_compatible(settings)
        embeddings = create_embeddings()
    client = QdrantClient(url=settings.qdrant_url)
    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
    )
```

Add:

```python
def embedding_model_identifier(provider_settings: QwenProviderSettings | None = None) -> str:
    resolved = provider_settings or QwenProviderSettings.from_env()
    dimensions = resolved.embedding_dimensions
    return resolved.embedding_model if dimensions is None else f"{resolved.embedding_model}:{dimensions}"
```

Update `index_vector_documents` after `add_documents` succeeds:

```python
    added_ids = list(vector_store.add_documents(documents=documents, ids=resolved_ids))
    if settings is not None:
        write_vector_metadata(settings, QwenProviderSettings.from_env().vector_metadata())
    return added_ids
```

- [ ] **Step 4: Wire pipeline vector indexing through `index_vector_documents`**

In `src/imperial_rag/pipeline.py`, update `_load_dependencies` imports:

```python
from imperial_rag.indexing import KeywordIndex, create_qdrant_vector_store, embedding_model_identifier, index_vector_documents
```

Update dependency dict:

```python
        "create_qdrant_vector_store": create_qdrant_vector_store,
        "embedding_model_identifier": embedding_model_identifier,
        "index_vector_documents": index_vector_documents,
```

Replace `_build_vector_store`:

```python
def _build_vector_store(settings: Any, index_vectors: bool) -> Any | None:
    if not index_vectors:
        return None
    from imperial_rag.indexing import create_qdrant_vector_store

    return create_qdrant_vector_store(settings)
```

Replace `_index_with_vector_store` signature and body:

```python
def _index_with_vector_store(index_vector_documents: Any, settings: Any, vector_store: Any, chunks: list[Any]) -> bool:
    if not chunks:
        return True
    index_vector_documents(chunks, settings=settings, vector_store=vector_store)
    return True
```

Update the call site:

```python
    vector_indexed = (
        _index_with_vector_store(deps["index_vector_documents"], settings, vector_store, chunks)
        if vector_store is not None
        else False
    )
```

Update `_update_index_status` signature to accept `embedding_model: str | None`, pass it from the call site as:

```python
            embedding_model=deps["embedding_model_identifier"]() if vector_indexed else None,
```

Inside `manifest_store.update_index_status`, pass:

```python
        embedding_model=embedding_model,
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
uv run python -m pytest tests/test_indexing.py tests/test_pipeline.py tests/test_pipeline_integration.py -q
git add src/imperial_rag/indexing.py src/imperial_rag/pipeline.py tests/test_indexing.py tests/test_pipeline.py
git commit -m "feat: guard qwen vector metadata"
```

Expected: indexing tests pass, fake pipeline tests pass, and real pipeline tests still pass without vector indexing.

## Task 5: Runtime Chat And Semantic Search Gating

**Files:**
- Modify: `src/imperial_rag/runtime.py`
- Modify: `src/imperial_rag/workflows.py`
- Modify: `tests/test_runtime.py`

- [ ] **Step 1: Add failing runtime tests**

Append to `tests/test_runtime.py`:

```python
def test_semantic_search_enabled_uses_dashscope_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    from imperial_rag.runtime import _semantic_search_enabled

    assert _semantic_search_enabled() is False

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    assert _semantic_search_enabled() is True


def test_build_query_dependencies_skips_vector_search_on_metadata_mismatch(monkeypatch, tmp_path):
    calls = {}

    class FakeKeywordIndex:
        def __init__(self, db_path):
            calls["keyword_db_path"] = db_path

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setattr("imperial_rag.runtime.KeywordIndex", FakeKeywordIndex)
    monkeypatch.setattr("imperial_rag.runtime.vector_metadata_matches_config", lambda settings: False)

    dependencies = create_runtime(Settings(workspace_root=tmp_path)).dependencies
    if dependencies is None:
        dependencies = __import__("imperial_rag.runtime", fromlist=["build_query_dependencies"]).build_query_dependencies(
            Settings(workspace_root=tmp_path)
        )

    assert getattr(dependencies.vector_search, "provider_mismatch", False) is True
    assert calls["keyword_db_path"] == tmp_path / ".imperial_rag" / "keyword.sqlite3"


def test_runtime_uses_qwen_chat_model_by_default(monkeypatch, tmp_path):
    fake_chat_model = object()
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setattr("imperial_rag.runtime.create_chat_model", lambda: fake_chat_model)
    monkeypatch.setattr("imperial_rag.runtime.KeywordIndex", lambda db_path: object())
    monkeypatch.setattr("imperial_rag.runtime._semantic_search_enabled", lambda: False)

    deps = __import__("imperial_rag.runtime", fromlist=["build_query_dependencies"]).build_query_dependencies(
        Settings(workspace_root=tmp_path)
    )

    assert deps.chat_model is fake_chat_model
```

- [ ] **Step 2: Run runtime tests and verify failure**

Run:

```bash
uv run python -m pytest tests/test_runtime.py -q
```

Expected: FAIL because runtime still checks OpenAI/Azure keys and uses `_LazyChatModel`.

- [ ] **Step 3: Update runtime**

Modify imports in `src/imperial_rag/runtime.py`:

```python
from imperial_rag.providers import (
    create_chat_model,
    dashscope_configured,
    vector_metadata_matches_config,
)
```

Remove `_LazyChatModel`.

Add:

```python
class _ProviderMismatchVectorSearch:
    provider_mismatch = True

    def similarity_search(self, query: str, k: int):
        return []

    def max_marginal_relevance_search(self, query: str, k: int, fetch_k: int, lambda_mult: float):
        return []
```

Change `QueryDependencies`:

```python
@dataclass(frozen=True)
class QueryDependencies:
    vector_search: object
    keyword_search: object
    chat_model: object
```

Replace `build_query_dependencies` body:

```python
def build_query_dependencies(settings: Settings) -> QueryDependencies:
    vector_search: object
    if _semantic_search_enabled() and vector_metadata_matches_config(settings):
        try:
            vector_search = make_qdrant_store(settings.qdrant_url, settings.qdrant_collection)
        except Exception:
            vector_search = _NoopVectorSearch()
    elif _semantic_search_enabled():
        vector_search = _ProviderMismatchVectorSearch()
    else:
        vector_search = _NoopVectorSearch()
    return QueryDependencies(
        vector_search=vector_search,
        keyword_search=KeywordIndex(settings.keyword_db_path),
        chat_model=create_chat_model(),
    )
```

Replace `_semantic_search_enabled`:

```python
def _semantic_search_enabled() -> bool:
    return dashscope_configured()
```

- [ ] **Step 4: Teach retrieval diagnostics about provider mismatch**

In `src/imperial_rag/retrieval.py`, inside `HybridRetriever.retrieve`, before calling `_vector_docs`, add:

```python
        if getattr(self.vector_search, "provider_mismatch", False):
            vector_status = "provider_mismatch"
            fallbacks.append("vector_provider_mismatch")
        else:
            try:
                vector_docs = self._vector_docs(query)
            except Exception:
                vector_status = "unavailable"
                fallbacks.append("vector_search_failed")
                vector_docs = []
```

Remove the old surrounding `try` block for `vector_docs` so the variable is not assigned twice.

- [ ] **Step 5: Remove default top-level OpenAI import in workflows**

In `src/imperial_rag/workflows.py`, delete:

```python
from langchain_openai import ChatOpenAI
```

Add:

```python
def _legacy_openai_chat_model():
    from imperial_rag.providers import QwenProviderSettings

    if not QwenProviderSettings.from_env().allow_legacy_openai:
        raise RuntimeError("Legacy OpenAI chat is disabled. Use Qwen provider defaults or set IMPERIAL_RAG_ALLOW_LEGACY_OPENAI=true.")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-4.1-mini", temperature=0)
```

Replace:

```python
resolved_model = model or ChatOpenAI(model="gpt-4.1-mini", temperature=0)
```

with:

```python
resolved_model = model or _legacy_openai_chat_model()
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
uv run python -m pytest tests/test_runtime.py tests/test_retrieval.py tests/test_workflows.py -q
git add src/imperial_rag/runtime.py src/imperial_rag/retrieval.py src/imperial_rag/workflows.py tests/test_runtime.py
git commit -m "feat: use qwen runtime defaults"
```

Expected: runtime uses DashScope readiness, provider mismatch is visible as diagnostics, and default workflow imports no OpenAI chat model.

## Task 6: DashScope Reranker With Deterministic Fallback

**Files:**
- Modify: `src/imperial_rag/retrieval.py`
- Modify: `tests/test_retrieval.py`
- Modify: `.env.example`

- [ ] **Step 1: Add failing reranker tests**

Append to `tests/test_retrieval.py`:

```python
def test_retrieval_settings_defaults_to_dashscope_reranker(monkeypatch):
    monkeypatch.delenv("IMPERIAL_RAG_PRIMARY_RERANKER", raising=False)
    monkeypatch.delenv("IMPERIAL_RAG_FALLBACK_RERANKER", raising=False)
    monkeypatch.delenv("IMPERIAL_RAG_QWEN_RERANK_MODEL", raising=False)

    settings = RetrievalSettings.from_env()

    assert settings.primary_reranker == "dashscope:qwen3-rerank"
    assert settings.fallback_reranker == "fallback:deterministic"


def test_reranker_uses_dashscope_when_key_is_configured(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    docs = [
        Document(page_content="first", metadata={"citation_id": "first"}),
        Document(page_content="second", metadata={"citation_id": "second"}),
    ]
    diagnostics = {"fallbacks": []}

    class FakeCompressor:
        def compress_documents(self, documents, query):
            assert query == "question"
            return [documents[1]]

    monkeypatch.setattr("imperial_rag.retrieval.create_reranker", lambda top_n: FakeCompressor())

    reranked = Reranker(settings=RetrievalSettings(rerank_top_n=1)).rerank("question", docs, diagnostics)

    assert [doc.metadata["citation_id"] for doc in reranked] == ["second"]
    assert diagnostics["reranker"] == "dashscope:qwen3-rerank"
    assert diagnostics["reranked_candidates"] == 1


def test_reranker_falls_back_without_dashscope_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    docs = [
        Document(page_content="Общие правила склада.", metadata={"citation_id": "warehouse"}),
        Document(page_content="Порядок возврата брака.", metadata={"citation_id": "return", "_keyword_rank": 0}),
    ]
    diagnostics = {"fallbacks": []}

    reranked = Reranker(settings=RetrievalSettings(rerank_top_n=1)).rerank("возврат брака", docs, diagnostics)

    assert [doc.metadata["citation_id"] for doc in reranked] == ["return"]
    assert diagnostics["reranker"] == "fallback:deterministic"
    assert "reranker_missing_dashscope_api_key" in diagnostics["fallbacks"]
```

- [ ] **Step 2: Run reranker tests and verify failure**

Run:

```bash
uv run python -m pytest tests/test_retrieval.py -q
```

Expected: FAIL because defaults still point to Cohere and `Reranker` checks `COHERE_API_KEY`.

- [ ] **Step 3: Update reranker settings and implementation**

Modify `src/imperial_rag/retrieval.py` imports:

```python
from imperial_rag.providers import create_reranker, dashscope_configured
```

Change `RetrievalSettings` defaults:

```python
    primary_reranker: str = "dashscope:qwen3-rerank"
    fallback_reranker: str = "fallback:deterministic"
```

In `RetrievalSettings.from_env`, set primary reranker from Qwen-specific env first:

```python
            primary_reranker=_env_str(
                "IMPERIAL_RAG_PRIMARY_RERANKER",
                f"dashscope:{_env_str('IMPERIAL_RAG_QWEN_RERANK_MODEL', 'qwen3-rerank')}",
            ),
            fallback_reranker=_env_str("IMPERIAL_RAG_FALLBACK_RERANKER", cls.fallback_reranker),
```

Replace the Cohere branch in `Reranker.rerank`:

```python
        if not dashscope_configured():
            diagnostics.setdefault("fallbacks", []).append("reranker_missing_dashscope_api_key")
            return self._fallback_rerank(query, candidates, diagnostics)

        try:
            reranked = self._dashscope_rerank(query, candidates)
        except Exception:
            diagnostics.setdefault("fallbacks", []).append(f"reranker_failed:{self.settings.primary_reranker}")
            return self._fallback_rerank(query, candidates, diagnostics)

        diagnostics["reranker"] = self.settings.primary_reranker
        backfilled = self._backfill(query, reranked, candidates)
        diagnostics["reranked_candidates"] = len(backfilled)
        return backfilled
```

Replace `_cohere_rerank` with:

```python
    def _dashscope_rerank(self, query: str, documents: list[Document]) -> list[Document]:
        compressor = create_reranker(top_n=self.settings.rerank_top_n)
        return list(compressor.compress_documents(documents=documents, query=query))
```

Remove `_cohere_model_name` if no longer used.

- [ ] **Step 4: Update `.env.example` retrieval defaults**

Change:

```dotenv
IMPERIAL_RAG_PRIMARY_RERANKER=cohere:rerank-v3.5
IMPERIAL_RAG_FALLBACK_RERANKER=cohere:rerank-multilingual-v3.0
```

to:

```dotenv
IMPERIAL_RAG_PRIMARY_RERANKER=dashscope:qwen3-rerank
IMPERIAL_RAG_FALLBACK_RERANKER=fallback:deterministic
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
uv run python -m pytest tests/test_retrieval.py tests/test_runtime.py -q
git add src/imperial_rag/retrieval.py tests/test_retrieval.py .env.example
git commit -m "feat: use dashscope reranking"
```

Expected: reranker defaults and fallback diagnostics are Qwen/DashScope-specific.

## Task 7: Scripts, Documentation, And Final Provider Cleanup

**Files:**
- Modify: `scripts/ingest.py`
- Modify: `README.md`
- Modify: `tests/test_scripts.py`
- Modify: `tests/test_dependencies.py`

- [ ] **Step 1: Add failing script key-gate test**

Append to `tests/test_scripts.py`:

```python
def test_ingest_index_vectors_requires_dashscope_key(monkeypatch, tmp_path):
    module = _load_script("scripts/ingest.py", "ingest_script_vector_gate")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    class FakeSettings:
        qdrant_url = "http://127.0.0.1:6333"
        qdrant_collection = "imperial_chunks_qwen"

    with pytest.raises(SystemExit) as exc:
        module._build_vector_store(FakeSettings(), index_vectors=True)

    assert exc.value.code == 2
```

Add `import pytest` at the top of `tests/test_scripts.py`.

- [ ] **Step 2: Run script test and verify failure**

Run:

```bash
uv run python -m pytest tests/test_scripts.py -q
```

Expected: FAIL because `_build_vector_store` does not fail fast without `DASHSCOPE_API_KEY`.

- [ ] **Step 3: Update ingest script vector gate**

In `scripts/ingest.py`, replace `_build_vector_store`:

```python
def _build_vector_store(settings: Any, index_vectors: bool) -> Any | None:
    if not index_vectors:
        return None
    from imperial_rag.providers import dashscope_configured

    if not dashscope_configured():
        raise SystemExit("DASHSCOPE_API_KEY is required when --index-vectors is used.")
    from imperial_rag.indexing import create_qdrant_vector_store

    return create_qdrant_vector_store(settings)
```

- [ ] **Step 4: Update dependency tests for default OpenAI/Cohere removal**

In `tests/test_dependencies.py`, replace `test_project_includes_cohere_reranking_dependency` with:

```python
def test_default_provider_dependencies_are_qwen_dashscope():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {_normalize_dependency_name(dependency) for dependency in pyproject["project"]["dependencies"]}

    assert "dashscope" in dependencies
    assert "langchain-qwq" in dependencies
```

Do not remove `langchain-openai` or `langchain-cohere` in this task unless `rg -n "langchain_openai|langchain_cohere" src tests scripts` shows no opt-in legacy import remains. If imports remain only behind `IMPERIAL_RAG_ALLOW_LEGACY_*`, keep the packages for this migration and defer removal to a follow-up dependency cleanup.

- [ ] **Step 5: Update README provider setup**

In `README.md`, replace the provider sentence under Quickstart:

```markdown
Fill in local secrets such as `DASHSCOPE_API_KEY` in `.env` or your shell environment. Do not commit real keys.
```

Replace the Qdrant defaults section:

```markdown
Defaults:

- URL: `http://localhost:6333`
- collection: `imperial_chunks_qwen`
- provider metadata: `.imperial_rag/vector_provider.json`
```

Add after the vector indexing command:

```markdown
Changing embedding providers or dimensions requires a clean vector collection. For the Qwen migration, use `QDRANT_COLLECTION=imperial_chunks_qwen` or recreate the old collection before reindexing. The runtime disables semantic search when local vector metadata does not match the configured Qwen embedding model and dimensions.
```

- [ ] **Step 6: Run targeted tests and commit**

Run:

```bash
uv run python -m pytest tests/test_scripts.py tests/test_dependencies.py -q
git add scripts/ingest.py tests/test_scripts.py tests/test_dependencies.py README.md
git commit -m "docs: document qwen provider operations"
```

Expected: script tests and dependency tests pass, and docs explain the new default provider and vector collection policy.

## Task 8: Final Verification And Operator Commands

**Files:**
- Modify: `docs/superpowers/plans/2026-06-04-qwen-model-migration.md` only if implementation discoveries require plan corrections before execution continues.

- [ ] **Step 1: Scan for forbidden default imports**

Run:

```bash
rg -n "from langchain_openai import ChatOpenAI|OpenAIEmbeddings|from langchain_cohere import CohereRerank|COHERE_API_KEY|OPENAI_API_KEY|AZURE_OPENAI_API_KEY" src scripts tests .env.example README.md
```

Expected: any OpenAI/Cohere hits are either legacy opt-in code guarded by `IMPERIAL_RAG_ALLOW_LEGACY_*`, tests proving legacy behavior is opt-in, or historical docs that have been updated. No default runtime/indexing/OCR/rerank path should require OpenAI or Cohere keys.

- [ ] **Step 2: Run full offline test suite**

Run:

```bash
uv run python -m pytest -q
```

Expected: PASS with no live DashScope calls and no live Qdrant requirement.

- [ ] **Step 3: Verify import-level provider availability**

Run:

```bash
uv run python - <<'PY'
import dashscope
from langchain_qwq import ChatQwen
from imperial_rag.providers import QwenProviderSettings

print("dashscope", bool(dashscope.TextEmbedding), bool(dashscope.TextReRank), bool(dashscope.MultiModalConversation))
print("chatqwen", ChatQwen)
print(QwenProviderSettings.from_env().chat_model)
PY
```

Expected: output includes `dashscope True True True`, a `ChatQwen` class, and `qwen3.7-max`.

- [ ] **Step 4: Verify keyword-only runtime degrades safely without key**

Run:

```bash
DASHSCOPE_API_KEY= uv run python - <<'PY'
from imperial_rag.config import Settings
from imperial_rag.runtime import build_query_dependencies

deps = build_query_dependencies(Settings())
print(type(deps.vector_search).__name__)
print(type(deps.keyword_search).__name__)
PY
```

Expected: vector search is a no-op class and keyword search is `KeywordIndex`.

- [ ] **Step 5: Document operator vector rebuild commands**

Use these commands after Qdrant is running and `DASHSCOPE_API_KEY` is configured:

```bash
./scripts/start_qdrant.sh
QDRANT_COLLECTION=imperial_chunks_qwen uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial --index-vectors
QDRANT_COLLECTION=imperial_chunks_qwen uv run python scripts/query.py "question text"
```

Expected: ingestion prints `vector_indexed=True`; query returns an answer or the strict refusal text with no provider mismatch diagnostic.

- [ ] **Step 6: Commit final cleanup**

Run:

```bash
git status --short
git diff --check
git add pyproject.toml uv.lock .env.example README.md src/imperial_rag tests scripts
git commit -m "feat: migrate defaults to hosted qwen"
```

Expected: commit succeeds only if there are remaining migration files not already committed by previous tasks. If `git status --short` is clean, do not create an empty commit.

## Self-Review Checklist

- Spec coverage: Tasks 1-2 cover dependencies, provider config, endpoint config, and model factories. Task 3 covers Qwen OCR payloads and parsing. Task 4 covers Qwen embeddings, vector metadata, and clean reindex guards. Task 5 covers runtime chat and semantic search gating. Task 6 covers DashScope reranking and deterministic fallback. Task 7 covers scripts/docs. Task 8 covers acceptance verification and operator commands.
- Placeholder scan: this plan intentionally avoids placeholder markers and names concrete files, functions, tests, commands, and expected outcomes.
- Type consistency: provider settings, vector metadata, OCR result, embeddings, reranker, and runtime dependency names are introduced before later tasks reference them.
