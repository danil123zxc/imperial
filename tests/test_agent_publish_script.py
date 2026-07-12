from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PUBLISH_SCRIPT = PROJECT_ROOT / "scripts" / "publish_agent_pr.sh"


def _run(
    *args: str,
    cwd: Path,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        check=check,
        capture_output=True,
        env=env,
        text=True,
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run("git", *args, cwd=repo)


def _create_repo(tmp_path: Path) -> Path:
    remote = tmp_path / "remote.git"
    repo = tmp_path / "repo"
    _run("git", "init", "--bare", str(remote), cwd=tmp_path)
    _run("git", "init", "-b", "main", str(repo), cwd=tmp_path)
    _git(repo, "config", "user.name", "Agent Test")
    _git(repo, "config", "user.email", "agent@example.test")
    _git(repo, "config", "commit.gpgsign", "false")

    scripts = repo / "scripts"
    scripts.mkdir()
    shutil.copy2(PUBLISH_SCRIPT, scripts / "publish_agent_pr.sh")
    check_script = scripts / "check.sh"
    expected_root = shlex.quote(str(repo))
    check_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'[[ "${{IMPERIAL_RAG_WORKSPACE_ROOT:-}}" == {expected_root} ]]\n',
        encoding="utf-8",
    )
    check_script.chmod(0o755)
    (repo / "README.md").write_text("# Test repo\n", encoding="utf-8")
    _git(repo, "add", "README.md", "scripts/check.sh", "scripts/publish_agent_pr.sh")
    _git(repo, "commit", "-m", "chore: initialize test repo")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")
    return repo


def test_dry_run_accepts_clean_fresh_codex_branch(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)
    worktree = tmp_path / "agent-worktree"
    _git(
        repo,
        "worktree",
        "add",
        "-b",
        "codex/test-publish",
        str(worktree),
        "origin/main",
    )
    (worktree / "feature.txt").write_text("verified\n", encoding="utf-8")
    _git(worktree, "add", "feature.txt")
    _git(worktree, "commit", "-m", "feat: add verified change")

    result = _run(
        "bash",
        "scripts/publish_agent_pr.sh",
        "--verifier-approved",
        "--dry-run",
        cwd=worktree,
    )

    assert "Dry run successful" in result.stdout


def test_refuses_protected_branch(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)

    result = _run(
        "bash",
        "scripts/publish_agent_pr.sh",
        "--verifier-approved",
        "--dry-run",
        cwd=repo,
        check=False,
    )

    assert result.returncode != 0
    assert "protected branch 'main'" in result.stderr


def test_refuses_denylisted_path(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)
    _git(repo, "checkout", "-b", "codex/test-denylist")
    (repo / ".env").write_text("SECRET=redacted\n", encoding="utf-8")
    _git(repo, "add", ".env")
    _git(repo, "commit", "-m", "test: add denied path")

    result = _run(
        "bash",
        "scripts/publish_agent_pr.sh",
        "--verifier-approved",
        "--dry-run",
        cwd=repo,
        check=False,
    )

    assert result.returncode != 0
    assert "denylisted path changed: .env" in result.stderr


def test_requires_verifier_approval_marker(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)

    result = _run(
        "bash",
        "scripts/publish_agent_pr.sh",
        "--dry-run",
        cwd=repo,
        check=False,
    )

    assert result.returncode != 0
    assert "independent verifier APPROVE is required" in result.stderr


def test_base_is_not_operator_configurable(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)

    result = _run(
        "bash",
        "scripts/publish_agent_pr.sh",
        "--verifier-approved",
        "--base",
        "dev",
        cwd=repo,
        check=False,
    )

    assert result.returncode != 0
    assert "unknown argument: --base" in result.stderr


def test_refuses_branch_with_closed_or_merged_pr(tmp_path: Path) -> None:
    repo = _create_repo(tmp_path)
    worktree = tmp_path / "closed-pr-worktree"
    _git(
        repo,
        "worktree",
        "add",
        "-b",
        "codex/reused-branch",
        str(worktree),
        "origin/main",
    )
    (worktree / "feature.txt").write_text("verified\n", encoding="utf-8")
    _git(worktree, "add", "feature.txt")
    _git(worktree, "commit", "-m", "feat: add verified change")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"$*\" == *\"--state closed\"* ]]; then\n"
        "  printf '7\\tMERGED\\thttps://example.test/pr/7\\n'\n"
        "fi\n",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = _run(
        "bash",
        "scripts/publish_agent_pr.sh",
        "--verifier-approved",
        cwd=worktree,
        check=False,
        env=env,
    )

    assert result.returncode != 0
    assert "branch already has a closed or merged PR" in result.stderr
