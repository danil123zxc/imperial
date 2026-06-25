from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_all_evals_script_imports_and_defines_main():
    module = _load_all_evals_runner()

    assert hasattr(module, "main")


def test_all_evals_default_creates_phoenix_experiment_with_faithfulness(monkeypatch):
    module = _load_all_evals_runner()
    settings = SimpleNamespace(
        phoenix_client_endpoint="http://localhost:6006",
        phoenix_project_name="imperial-rag",
    )
    examples = [{"question": "Что делать?", "expected_behavior": "cite_answer"}]
    captured: dict[str, object] = {}

    monkeypatch.setattr(module.phoenix_eval, "_load_project_env", lambda workspace_root: captured.setdefault("env", workspace_root))
    monkeypatch.setattr(module.phoenix_eval, "_build_settings", lambda workspace_root: settings)
    monkeypatch.setattr(module.phoenix_eval, "load_questions", lambda path: examples)
    monkeypatch.setattr(module.phoenix_eval, "_configure_tracing", lambda settings, enabled: captured.setdefault("tracing", (settings, enabled)))
    monkeypatch.setattr(module, "_assert_phoenix_reachable", lambda endpoint: captured.setdefault("preflight", endpoint))

    def fake_run_phoenix_experiment(**kwargs):
        captured["experiment"] = kwargs

    monkeypatch.setattr(module.phoenix_eval, "run_phoenix_experiment", fake_run_phoenix_experiment)

    module.main([])

    assert captured["env"] is None
    assert captured["preflight"] == "http://localhost:6006"
    assert captured["tracing"] == (settings, True)
    assert captured["experiment"] == {
        "examples": examples,
        "settings": settings,
        "dataset_name": "imperial-rag-gold-questions",
        "experiment_name": "imperial-rag-all-evals",
        "ragas_metric_names": ["faithfulness", "answer_relevancy"],
        "concurrency": module.phoenix_eval.DEFAULT_PHOENIX_CONCURRENCY,
    }


def test_all_evals_can_create_deterministic_only_phoenix_experiment(monkeypatch):
    module = _load_all_evals_runner()
    settings = SimpleNamespace(
        phoenix_client_endpoint="http://localhost:6006",
        phoenix_project_name="imperial-rag",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(module.phoenix_eval, "_load_project_env", lambda workspace_root: None)
    monkeypatch.setattr(module.phoenix_eval, "_build_settings", lambda workspace_root: settings)
    monkeypatch.setattr(module.phoenix_eval, "load_questions", lambda path: [])
    monkeypatch.setattr(module.phoenix_eval, "_configure_tracing", lambda settings, enabled: None)
    monkeypatch.setattr(module, "_assert_phoenix_reachable", lambda endpoint: None)
    monkeypatch.setattr(module.phoenix_eval, "run_phoenix_experiment", lambda **kwargs: captured.update(kwargs))

    module.main(["--ragas-metrics", "none"])

    assert captured["ragas_metric_names"] == []
    assert captured["concurrency"] == module.phoenix_eval.DEFAULT_PHOENIX_CONCURRENCY


def test_all_evals_forwards_id_context_recall_metric(monkeypatch):
    module = _load_all_evals_runner()
    settings = SimpleNamespace(
        phoenix_client_endpoint="http://localhost:6006",
        phoenix_project_name="imperial-rag",
    )
    examples = [
        {
            "question": "Что делать?",
            "expected_behavior": "cite_answer",
            "reference_context_ids": ["file-a"],
        }
    ]
    captured: dict[str, object] = {}

    monkeypatch.setattr(module.phoenix_eval, "_load_project_env", lambda workspace_root: None)
    monkeypatch.setattr(module.phoenix_eval, "_build_settings", lambda workspace_root: settings)
    monkeypatch.setattr(module.phoenix_eval, "load_questions", lambda path: examples)
    monkeypatch.setattr(module.phoenix_eval, "_configure_tracing", lambda settings, enabled: None)
    monkeypatch.setattr(module, "_assert_phoenix_reachable", lambda endpoint: None)
    monkeypatch.setattr(module.phoenix_eval, "run_phoenix_experiment", lambda **kwargs: captured.update(kwargs))

    module.main(["--ragas-metrics", "id_context_recall", "--concurrency", "5"])

    assert captured["examples"] == examples
    assert captured["ragas_metric_names"] == ["id_context_recall"]
    assert captured["concurrency"] == 5


def test_all_evals_preflight_fails_with_phoenix_start_hint(monkeypatch):
    module = _load_all_evals_runner()

    def broken_urlopen(url: str, timeout: float):
        raise OSError("connection refused")

    monkeypatch.setattr(module.request, "urlopen", broken_urlopen)

    with pytest.raises(SystemExit) as exc_info:
        module._assert_phoenix_reachable("http://localhost:6006")

    message = str(exc_info.value)
    assert "Phoenix is not reachable at http://localhost:6006" in message
    assert "docker compose up -d phoenix" in message


def _load_all_evals_runner():
    spec = importlib.util.spec_from_file_location("run_all_evals_for_test", Path("scripts/run_all_evals.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
