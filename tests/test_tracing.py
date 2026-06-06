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
    monkeypatch.delenv("IMPERIAL_RAG_TRACING_ENABLED", raising=False)
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
            "verbose": False,
        }
    ]


def test_configure_phoenix_tracing_rejects_changed_key_after_configuration(monkeypatch, tmp_path: Path) -> None:
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
    settings_a = Settings(
        workspace_root=tmp_path,
        phoenix_project_name="project-a",
        phoenix_collector_endpoint="http://localhost:6006/v1/traces",
    )
    settings_b = Settings(
        workspace_root=tmp_path,
        phoenix_project_name="project-b",
        phoenix_collector_endpoint="http://localhost:7007/v1/traces",
    )

    assert configure_phoenix_tracing(settings_a, enabled=True) is provider
    with pytest.raises(RuntimeError, match="Phoenix tracing is already configured"):
        configure_phoenix_tracing(settings_b, enabled=True)
    assert calls == [
        {
            "project_name": "project-a",
            "endpoint": "http://localhost:6006/v1/traces",
            "auto_instrument": True,
            "verbose": False,
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
    monkeypatch.setattr("imperial_rag.tracing._collector_endpoint_reachable", lambda endpoint: True)

    assert configure_phoenix_tracing(Settings(workspace_root=tmp_path), enabled=None) is provider


def test_env_enabled_phoenix_tracing_skips_when_collector_is_unreachable(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel = types.ModuleType("phoenix.otel")
    fake_otel.register = lambda **kwargs: pytest.fail("Phoenix should not register when env-enabled endpoint is down")
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.otel", fake_otel)
    monkeypatch.setenv("PHOENIX_TRACING_ENABLED", "true")
    monkeypatch.setattr("imperial_rag.tracing._collector_endpoint_reachable", lambda endpoint: False)
    settings = Settings(
        workspace_root=tmp_path,
        phoenix_collector_endpoint="http://localhost:6006/v1/traces",
    )

    assert configure_phoenix_tracing(settings, enabled=None) is None


def test_configure_phoenix_tracing_errors_clearly_when_dependency_missing(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    monkeypatch.setitem(sys.modules, "phoenix.otel", None)
    settings = Settings(workspace_root=tmp_path)

    with pytest.raises(RuntimeError, match="Phoenix tracing dependencies are missing"):
        configure_phoenix_tracing(settings, enabled=True)
