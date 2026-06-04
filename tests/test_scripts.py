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


def test_ingest_ocr_gate_uses_dashscope_key(monkeypatch):
    module = _load_script("scripts/ingest.py", "ingest_script_ocr_gate")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-legacy-key")

    assert module._ocr_appears_configured() is False

    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    assert module._ocr_appears_configured() is True


def _load_script(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
