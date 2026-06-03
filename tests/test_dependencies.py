from __future__ import annotations

import re
import tomllib
from pathlib import Path


def _normalize_dependency_name(dependency: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", dependency)
    if match is None:
        raise AssertionError(f"Could not parse dependency name from {dependency!r}")
    return re.sub(r"[-_.]+", "-", match.group(1)).lower()


def test_project_uses_phoenix_dependencies_instead_of_legacy_tracing_package():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {_normalize_dependency_name(dependency) for dependency in pyproject["project"]["dependencies"]}
    legacy_tracing_package = "lang" + "smith"

    assert legacy_tracing_package not in dependencies
    assert "arize-phoenix-client" in dependencies
    assert "arize-phoenix-otel" in dependencies
    assert "openinference-instrumentation-langchain" in dependencies
    assert "openinference-instrumentation-openai" in dependencies


def test_project_includes_cohere_reranking_dependency():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {_normalize_dependency_name(dependency) for dependency in pyproject["project"]["dependencies"]}

    assert "langchain-cohere" in dependencies
