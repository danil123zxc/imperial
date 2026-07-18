from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SCRIPT = ROOT / "scripts" / "deploy_server.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
OLD_SHA = "1" * 40
NEW_SHA = "2" * 40
NEWER_SHA = "3" * 40


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _fake_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    deploy_root = tmp_path / "deploy"
    state_dir = tmp_path / "state"
    fake_bin = tmp_path / "bin"
    fake_state = tmp_path / "fake-state"
    deploy_root.mkdir()
    state_dir.mkdir()
    fake_bin.mkdir()
    fake_state.mkdir()
    (deploy_root / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
    (fake_state / "current_sha").write_text(f"{OLD_SHA}\n", encoding="utf-8")
    (fake_state / "running_sha").write_text(f"{OLD_SHA}\n", encoding="utf-8")

    _write_executable(
        fake_bin / "git",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'git %s\n' "$*" >> "$FAKE_STATE_DIR/commands.log"
if [[ "${1:-}" == "-C" ]]; then
  shift 2
fi
case "${1:-}" in
  status)
    [[ "${FAKE_GIT_DIRTY:-0}" == "1" ]] && printf ' M tracked.txt\n'
    ;;
  fetch)
    ;;
  rev-parse)
    case "${2:-}" in
      HEAD) cat "$FAKE_STATE_DIR/current_sha" ;;
      origin/main*) printf '%s\n' "$FAKE_REMOTE_SHA" ;;
      *) exit 1 ;;
    esac
    ;;
  cat-file)
    ;;
  checkout)
    printf '%s\n' "$3" > "$FAKE_STATE_DIR/current_sha"
    ;;
  *)
    exit 1
    ;;
esac
""",
    )
    _write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -euo pipefail
printf 'docker %s\n' "$*" >> "$FAKE_STATE_DIR/commands.log"
checkout_sha=$(tr -d '\n' < "$FAKE_STATE_DIR/current_sha")
running_sha=$(tr -d '\n' < "$FAKE_STATE_DIR/running_sha")
if [[ "${1:-}" == "compose" ]]; then
  case "${2:-}" in
    build)
      if [[ "${FAKE_BUILD_FAIL_SHA:-}" == "$checkout_sha" ]]; then
        exit 1
      fi
      ;;
    up)
      if [[ "${FAKE_UP_FAIL_SHA:-}" == "$checkout_sha" ]]; then
        exit 1
      fi
      printf '%s\n' "$checkout_sha" > "$FAKE_STATE_DIR/running_sha"
      touch "$FAKE_STATE_DIR/container_started"
      ;;
    ps)
      if [[ "${FAKE_NO_CONTAINER:-0}" == "1" && ! -f "$FAKE_STATE_DIR/container_started" ]]; then
        exit 0
      fi
      printf 'fake-app-container\n'
      ;;
    logs)
      printf 'PRIVATE CONTAINER LOG CONTENT\n'
      ;;
    *)
      exit 1
      ;;
  esac
elif [[ "${1:-}" == "inspect" ]]; then
  if [[ "${FAKE_HEALTH_FAIL_ALL:-0}" == "1" || "${FAKE_UNHEALTHY_SHA:-}" == "$running_sha" ]]; then
    printf 'unhealthy\n'
  else
    printf 'healthy\n'
  fi
else
  exit 1
fi
exit 0
""",
    )
    _write_executable(
        fake_bin / "curl",
        """#!/usr/bin/env bash
set -euo pipefail
running_sha=$(tr -d '\n' < "$FAKE_STATE_DIR/running_sha")
if [[ "${FAKE_HEALTH_FAIL_ALL:-0}" == "1" || "${FAKE_UNHEALTHY_SHA:-}" == "$running_sha" ]]; then
  exit 22
fi
""",
    )
    _write_executable(
        fake_bin / "flock",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ "${FAKE_FLOCK_FAIL:-0}" == "1" ]]; then
  exit 1
fi
exit 0
""",
    )

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "FAKE_REMOTE_SHA": NEW_SHA,
            "FAKE_STATE_DIR": str(fake_state),
            "IMPERIAL_DEPLOY_TEST_MODE": "1",
            "IMPERIAL_DEPLOY_ROOT": str(deploy_root),
            "IMPERIAL_DEPLOY_STATE_DIR": str(state_dir),
            "IMPERIAL_DEPLOY_HEALTH_ATTEMPTS": "1",
            "IMPERIAL_DEPLOY_HEALTH_INTERVAL": "0",
            "SSH_ORIGINAL_COMMAND": f"deploy {NEW_SHA}",
        }
    )
    return env, deploy_root, state_dir, fake_state


def _run(
    env: dict[str, str],
    *args: str,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("bash", str(DEPLOY_SCRIPT), *args),
        check=check,
        capture_output=True,
        env=env,
        text=True,
    )


def _current_sha(fake_state: Path) -> str:
    return (fake_state / "current_sha").read_text(encoding="utf-8").strip()


def _commands(fake_state: Path) -> str:
    path = fake_state / "commands.log"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_rejects_malformed_and_unauthorized_forced_commands(tmp_path: Path) -> None:
    env, _, _, fake_state = _fake_environment(tmp_path)

    env["SSH_ORIGINAL_COMMAND"] = "rollback"
    result = _run(env)
    assert result.returncode != 0
    assert "unauthorized deployment command" in result.stderr

    env["SSH_ORIGINAL_COMMAND"] = f"deploy {NEW_SHA} extra"
    result = _run(env)
    assert result.returncode != 0
    assert "unauthorized deployment command" in result.stderr
    assert _commands(fake_state) == ""


def test_superseded_commit_is_a_successful_no_op(tmp_path: Path) -> None:
    env, _, state_dir, fake_state = _fake_environment(tmp_path)
    env["FAKE_REMOTE_SHA"] = NEWER_SHA

    result = _run(env)

    assert result.returncode == 0
    assert "superseded" in result.stdout
    assert _current_sha(fake_state) == OLD_SHA
    assert "git checkout" not in _commands(fake_state)
    assert "result=superseded" in (state_dir / "deployments.log").read_text(encoding="utf-8")


def test_dirty_deployment_worktree_is_rejected(tmp_path: Path) -> None:
    env, _, _, fake_state = _fake_environment(tmp_path)
    env["FAKE_GIT_DIRTY"] = "1"

    result = _run(env)

    assert result.returncode != 0
    assert "tracked changes" in result.stderr
    assert "git fetch" not in _commands(fake_state)


def test_successful_deploy_replaces_only_the_app(tmp_path: Path) -> None:
    env, _, state_dir, fake_state = _fake_environment(tmp_path)

    result = _run(env)

    assert result.returncode == 0
    assert "Production deployment is healthy." in result.stdout
    assert _current_sha(fake_state) == NEW_SHA
    commands = _commands(fake_state)
    assert "docker compose build app" in commands
    assert "docker compose up -d --no-deps app" in commands
    assert "docker compose down" not in commands
    assert "ingest" not in commands
    assert (state_dir / "last_good_sha").read_text(encoding="utf-8").strip() == NEW_SHA
    assert (state_dir / "previous_good_sha").read_text(encoding="utf-8").strip() == OLD_SHA
    assert "result=healthy" in (state_dir / "deployments.log").read_text(encoding="utf-8")


def test_interrupted_checkout_redeploys_and_preserves_the_last_healthy_rollback(tmp_path: Path) -> None:
    env, _, state_dir, fake_state = _fake_environment(tmp_path)
    (fake_state / "current_sha").write_text(f"{NEW_SHA}\n", encoding="utf-8")
    (state_dir / "last_good_sha").write_text(f"{OLD_SHA}\n", encoding="utf-8")

    result = _run(env)

    assert result.returncode == 0
    commands = _commands(fake_state)
    assert "docker compose build app" in commands
    assert "docker compose up -d --no-deps app" in commands
    assert (fake_state / "running_sha").read_text(encoding="utf-8").strip() == NEW_SHA
    assert (state_dir / "previous_good_sha").read_text(encoding="utf-8").strip() == OLD_SHA


def test_stopped_current_application_is_rebuilt_instead_of_reported_healthy(tmp_path: Path) -> None:
    env, _, state_dir, fake_state = _fake_environment(tmp_path)
    (fake_state / "current_sha").write_text(f"{NEW_SHA}\n", encoding="utf-8")
    (fake_state / "running_sha").write_text(f"{NEW_SHA}\n", encoding="utf-8")
    (state_dir / "last_good_sha").write_text(f"{NEW_SHA}\n", encoding="utf-8")
    (state_dir / "previous_good_sha").write_text(f"{OLD_SHA}\n", encoding="utf-8")
    env["FAKE_NO_CONTAINER"] = "1"

    result = _run(env)

    assert result.returncode == 0
    assert "already runs" not in result.stdout
    assert "rebuilding it" in result.stdout
    assert "docker compose build app" in _commands(fake_state)
    assert (state_dir / "previous_good_sha").read_text(encoding="utf-8").strip() == OLD_SHA


def test_failed_current_sha_repair_rolls_back_to_the_older_healthy_sha(tmp_path: Path) -> None:
    env, _, state_dir, fake_state = _fake_environment(tmp_path)
    (fake_state / "current_sha").write_text(f"{NEW_SHA}\n", encoding="utf-8")
    (fake_state / "running_sha").write_text(f"{NEW_SHA}\n", encoding="utf-8")
    (state_dir / "last_good_sha").write_text(f"{NEW_SHA}\n", encoding="utf-8")
    (state_dir / "previous_good_sha").write_text(f"{OLD_SHA}\n", encoding="utf-8")
    env["FAKE_UNHEALTHY_SHA"] = NEW_SHA

    result = _run(env)

    assert result.returncode != 0
    assert "production was restored" in result.stderr
    assert _current_sha(fake_state) == OLD_SHA
    assert (fake_state / "running_sha").read_text(encoding="utf-8").strip() == OLD_SHA
    assert (state_dir / "last_good_sha").read_text(encoding="utf-8").strip() == OLD_SHA


def test_build_failure_restores_checkout_without_replacing_app(tmp_path: Path) -> None:
    env, _, state_dir, fake_state = _fake_environment(tmp_path)
    env["FAKE_BUILD_FAIL_SHA"] = NEW_SHA

    result = _run(env)

    assert result.returncode != 0
    assert "running container was not replaced" in result.stderr
    assert _current_sha(fake_state) == OLD_SHA
    commands = _commands(fake_state)
    assert commands.count("docker compose build app") == 1
    assert "docker compose up" not in commands
    assert "result=failed" in (state_dir / "deployments.log").read_text(encoding="utf-8")


def test_health_failure_rolls_back_and_keeps_private_logs_local(tmp_path: Path) -> None:
    env, _, state_dir, fake_state = _fake_environment(tmp_path)
    env["FAKE_UNHEALTHY_SHA"] = NEW_SHA

    result = _run(env)

    assert result.returncode != 0
    assert "production was restored" in result.stderr
    assert "PRIVATE CONTAINER LOG CONTENT" not in result.stdout
    assert "PRIVATE CONTAINER LOG CONTENT" not in result.stderr
    assert _current_sha(fake_state) == OLD_SHA
    commands = _commands(fake_state)
    assert commands.count("docker compose build app") == 2
    assert commands.count("docker compose up -d --no-deps app") == 2
    failure_log = state_dir / f"failure-{NEW_SHA}.log"
    assert "PRIVATE CONTAINER LOG CONTENT" in failure_log.read_text(encoding="utf-8")
    assert failure_log.stat().st_mode & 0o777 == 0o600
    assert "result=restored" in (state_dir / "deployments.log").read_text(encoding="utf-8")


def test_start_failure_rolls_back_to_the_previous_app(tmp_path: Path) -> None:
    env, _, state_dir, fake_state = _fake_environment(tmp_path)
    env["FAKE_UP_FAIL_SHA"] = NEW_SHA

    result = _run(env)

    assert result.returncode != 0
    assert "application start failed; production was restored" in result.stderr
    assert _current_sha(fake_state) == OLD_SHA
    commands = _commands(fake_state)
    assert commands.count("docker compose build app") == 2
    assert commands.count("docker compose up -d --no-deps app") == 2
    assert "result=restored" in (state_dir / "deployments.log").read_text(encoding="utf-8")


def test_failed_rollback_is_reported_without_exposing_private_logs(tmp_path: Path) -> None:
    env, _, state_dir, fake_state = _fake_environment(tmp_path)
    env["FAKE_HEALTH_FAIL_ALL"] = "1"

    result = _run(env)

    assert result.returncode != 0
    assert "automatic rollback did not recover health" in result.stderr
    assert "PRIVATE CONTAINER LOG CONTENT" not in result.stdout + result.stderr
    assert _current_sha(fake_state) == OLD_SHA
    audit = (state_dir / "deployments.log").read_text(encoding="utf-8")
    assert "phase=rollback result=failed_health" in audit


def test_operator_can_roll_back_to_the_recorded_previous_commit(tmp_path: Path) -> None:
    env, _, state_dir, fake_state = _fake_environment(tmp_path)
    env.pop("SSH_ORIGINAL_COMMAND")
    (fake_state / "current_sha").write_text(f"{NEW_SHA}\n", encoding="utf-8")
    (state_dir / "last_good_sha").write_text(f"{NEW_SHA}\n", encoding="utf-8")
    (state_dir / "previous_good_sha").write_text(f"{OLD_SHA}\n", encoding="utf-8")

    result = _run(env, "rollback")

    assert result.returncode == 0
    assert _current_sha(fake_state) == OLD_SHA
    assert (state_dir / "last_good_sha").read_text(encoding="utf-8").strip() == OLD_SHA
    assert (state_dir / "previous_good_sha").read_text(encoding="utf-8").strip() == NEW_SHA
    assert "git fetch" not in _commands(fake_state)


def test_exclusive_lock_rejects_a_second_deployment(tmp_path: Path) -> None:
    env, _, _, fake_state = _fake_environment(tmp_path)
    env["FAKE_FLOCK_FAIL"] = "1"

    result = _run(env)

    assert result.returncode != 0
    assert "another deployment is already in progress" in result.stderr
    assert _commands(fake_state) == ""


def test_workflow_deploys_only_green_main_pushes_over_tailscale() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "needs: quality" in workflow
    assert "github.event_name == 'push'" in workflow
    assert "github.ref == 'refs/heads/main'" in workflow
    assert "name: production" in workflow
    assert "group: production" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "id-token: write" in workflow
    assert "tailscale/github-action@306e68a486fd2350f2bfc3b19fcd143891a4a2d8" in workflow
    assert "oauth-client-id: ${{ secrets.TS_OAUTH_CLIENT_ID }}" in workflow
    assert "audience: ${{ secrets.TS_AUDIENCE }}" in workflow
    assert "tags: tag:github-ci" in workflow
    assert "DEPLOY_SSH_KEY" in workflow
    assert "DEPLOY_KNOWN_HOSTS" in workflow
    assert '"$DEPLOY_USER@$DEPLOY_HOST" \\' in workflow
    assert '"deploy $GITHUB_SHA"' in workflow
    assert "workflow_run:" not in workflow
