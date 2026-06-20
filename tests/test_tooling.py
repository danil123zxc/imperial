from __future__ import annotations

import stat
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_dev_tooling_includes_lint_type_and_coverage() -> None:
    pyproject = _pyproject()
    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]

    assert any(dependency.startswith("ruff") for dependency in dev_dependencies)
    assert any(dependency.startswith("mypy") for dependency in dev_dependencies)
    assert any(dependency.startswith("pytest-cov") for dependency in dev_dependencies)
    assert "mypy" in pyproject["tool"]


def test_local_check_script_runs_offline_quality_gate() -> None:
    script_path = ROOT / "scripts" / "check.sh"
    script = script_path.read_text(encoding="utf-8")

    assert script_path.stat().st_mode & stat.S_IXUSR
    assert "uv run --extra dev ruff check ." in script
    assert "uv run --extra dev mypy src/imperial_rag" in script
    assert "uv run --extra dev python -m pytest --cov=imperial_rag -q" in script
    assert "git diff --check" in script
