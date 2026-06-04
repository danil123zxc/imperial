from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


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


def test_ingest_ocr_gate_uses_dashscope_key(monkeypatch):
    module = _load_script("scripts/ingest.py", "ingest_script_ocr_gate")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-legacy-key")

    assert module._ocr_appears_configured() is False

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    assert module._ocr_appears_configured() is True


def test_ingest_vector_store_requires_dashscope_key_before_importing_qdrant(monkeypatch, capsys):
    module = _load_script("scripts/ingest.py", "ingest_script_vector_gate")
    providers = types.ModuleType("imperial_rag.providers")
    providers.dashscope_configured = lambda: False
    indexing = types.ModuleType("imperial_rag.indexing")
    indexing.make_qdrant_store = lambda qdrant_url, collection_name: pytest.fail(
        "Qdrant builder should not run without key"
    )
    indexing.create_qdrant_vector_store = lambda settings: pytest.fail("Qdrant builder should not run without key")

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "imperial_rag.providers", providers)
    monkeypatch.setitem(sys.modules, "imperial_rag.indexing", indexing)

    with pytest.raises(SystemExit) as exc_info:
        settings = types.SimpleNamespace(qdrant_url="http://localhost:6333", qdrant_collection="test")
        module._build_vector_store(settings, index_vectors=True)

    assert exc_info.value.code == 2
    assert "DASHSCOPE_API_KEY is required when --index-vectors is used." in capsys.readouterr().err


def test_ingest_vector_store_builds_qdrant_after_dashscope_gate(monkeypatch):
    module = _load_script("scripts/ingest.py", "ingest_script_vector_build")
    providers = types.ModuleType("imperial_rag.providers")
    providers.dashscope_configured = lambda: True
    indexing = types.ModuleType("imperial_rag.indexing")
    vector_store = object()
    created_with = []
    indexing.make_qdrant_store = lambda qdrant_url, collection_name: pytest.fail("legacy builder should not run")
    indexing.create_qdrant_vector_store = lambda settings: created_with.append(settings) or vector_store

    monkeypatch.setitem(sys.modules, "imperial_rag.providers", providers)
    monkeypatch.setitem(sys.modules, "imperial_rag.indexing", indexing)

    settings = types.SimpleNamespace(qdrant_url="http://localhost:6333", qdrant_collection="test")

    assert module._build_vector_store(settings, index_vectors=True) is vector_store
    assert created_with == [settings]


def test_ingest_vector_store_disabled_does_not_require_dashscope_key(monkeypatch):
    module = _load_script("scripts/ingest.py", "ingest_script_vector_disabled")
    providers = types.ModuleType("imperial_rag.providers")
    providers.dashscope_configured = lambda: pytest.fail("DashScope gate should not run")
    indexing = types.ModuleType("imperial_rag.indexing")
    indexing.make_qdrant_store = lambda qdrant_url, collection_name: pytest.fail("Qdrant builder should not run")
    indexing.create_qdrant_vector_store = lambda settings: pytest.fail("Qdrant builder should not run")

    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "imperial_rag.providers", providers)
    monkeypatch.setitem(sys.modules, "imperial_rag.indexing", indexing)

    assert module._build_vector_store(object(), index_vectors=False) is None


def _load_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
