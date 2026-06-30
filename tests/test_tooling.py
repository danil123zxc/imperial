from __future__ import annotations

import stat
import tomllib
import json
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
    assert "GITHUB_BASE_REF" in script
    assert 'git diff --check "origin/${GITHUB_BASE_REF}...HEAD"' in script


def test_gitignore_keeps_docs_trackable_except_generated_reports() -> None:
    ignore_path = ROOT / ".gitignore"
    ignore_lines = {
        line.strip()
        for line in ignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }

    assert "docs/" not in ignore_lines
    assert "docs/superpowers/reports/" in ignore_lines


def test_pyright_contract_targets_repo_sources_without_broad_ignores() -> None:
    config_path = ROOT / "pyrightconfig.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["include"] == ["src", "scripts", "tests"]
    assert config["extraPaths"] == ["src"]
    assert config["pythonVersion"] == "3.12"
    assert config["pythonPlatform"] == "Darwin"
    assert config["venvPath"] == "."
    assert config["venv"] == ".venv"
    assert "ignore" not in config


def test_pyright_summary_script_is_non_blocking_tooling() -> None:
    script_path = ROOT / "scripts" / "summarize_pyright.py"
    script = script_path.read_text(encoding="utf-8")
    check_script = (ROOT / "scripts" / "check.sh").read_text(encoding="utf-8")

    assert script_path.exists()
    assert "generalDiagnostics" in script
    assert "summary" in script
    assert "summarize_pyright.py" not in check_script
