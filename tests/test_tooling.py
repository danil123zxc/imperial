from __future__ import annotations

import json
import re
import stat
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MANUAL_LOOP_SKILLS = {
    "daily-triage": ".codex/skills/loop-triage/SKILL.md",
    "ci-sweeper-manual": ".codex/skills/ci-sweeper-manual/SKILL.md",
    "eval-regression-check": ".codex/skills/eval-regression-check/SKILL.md",
    "ingestion-promotion-review": ".codex/skills/ingestion-promotion-review/SKILL.md",
}


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _markdown_loop_statuses(file_name: str, status_column: int) -> dict[str, str]:
    statuses = {}
    for line in (ROOT / file_name).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("| `"):
            continue
        columns = [column.strip() for column in stripped.strip("|").split("|")]
        statuses[columns[0].strip("`")] = columns[status_column]
    return statuses


def _registry_block(loop_id: str) -> str:
    registry = (ROOT / "patterns" / "registry.yaml").read_text(encoding="utf-8")
    match = re.search(rf"  - id: {re.escape(loop_id)}\n(?:(?!  - id: ).*\n?)*", registry)
    assert match is not None
    return match.group(0)


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


def test_manual_l1_loop_scaffold_is_active_and_report_only() -> None:
    loop_statuses = _markdown_loop_statuses("LOOP.md", status_column=2)
    state_statuses = _markdown_loop_statuses("STATE.md", status_column=1)

    for loop_id, skill_path in ACTIVE_MANUAL_LOOP_SKILLS.items():
        block = _registry_block(loop_id)

        assert loop_statuses[loop_id] == "Active L1 report-only"
        assert state_statuses[loop_id] == "Active"
        assert (ROOT / skill_path).exists()
        assert "    status: active\n" in block
        assert "    level: L1\n" in block
        assert "    provider_backed: false\n" in block
        assert "    max_subagents_per_run: 0\n" in block
        assert "      - STATE.md\n" in block
        assert "      - loop-run-log.md\n" in block
        assert "    budget:\n" in block
        if loop_id != "daily-triage":
            assert f"    skill_path: {skill_path}\n" in block

    assert loop_statuses["post-merge-cleanup"] == "Candidate L1 report-only"
    assert state_statuses["post-merge-cleanup"] == "Candidate"
