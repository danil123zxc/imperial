#!/usr/bin/env bash
set -euo pipefail

DEFAULT_DEPLOY_ROOT=/home/server1/imperial-deploy
DEFAULT_STATE_DIR=/home/server1/.local/state/imperial-deploy
DEFAULT_HEALTH_URL=http://127.0.0.1:8501/_stcore/health

if [[ "${IMPERIAL_DEPLOY_TEST_MODE:-0}" == "1" ]]; then
  deploy_root=${IMPERIAL_DEPLOY_ROOT:-$DEFAULT_DEPLOY_ROOT}
  state_dir=${IMPERIAL_DEPLOY_STATE_DIR:-$DEFAULT_STATE_DIR}
  health_url=${IMPERIAL_DEPLOY_HEALTH_URL:-$DEFAULT_HEALTH_URL}
  health_attempts=${IMPERIAL_DEPLOY_HEALTH_ATTEMPTS:-90}
  health_interval=${IMPERIAL_DEPLOY_HEALTH_INTERVAL:-2}
else
  deploy_root=$DEFAULT_DEPLOY_ROOT
  state_dir=$DEFAULT_STATE_DIR
  health_url=$DEFAULT_HEALTH_URL
  health_attempts=90
  health_interval=2
fi

audit_log="$state_dir/deployments.log"
last_good_file="$state_dir/last_good_sha"
previous_good_file="$state_dir/previous_good_sha"

usage() {
  printf 'Usage: %s deploy <commit-sha>\n' "$0"
  printf '       %s rollback\n' "$0"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '%s\n' "$*"
}

audit() {
  local sha=$1
  local phase=$2
  local result=$3
  printf '%s sha=%s phase=%s result=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$sha" "$phase" "$result" >> "$audit_log"
}

validate_sha() {
  [[ "$1" =~ ^[0-9a-f]{40}$ ]] || die "commit SHA must be exactly 40 lowercase hexadecimal characters"
}

compose() {
  docker compose "$@"
}

app_is_healthy() {
  local attempt
  local container_id
  local health_status

  for ((attempt = 1; attempt <= health_attempts; attempt++)); do
    container_id=$(compose ps -q app 2>/dev/null || true)
    if [[ -n "$container_id" ]]; then
      health_status=$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$container_id" 2>/dev/null || true)
      if [[ "$health_status" == "healthy" ]] && curl -fsS --max-time 3 "$health_url" >/dev/null 2>&1; then
        return 0
      fi
    fi
    sleep "$health_interval"
  done
  return 1
}

capture_private_failure_logs() {
  local sha=$1
  local failure_log="$state_dir/failure-$sha.log"
  compose logs --no-color --tail=200 app > "$failure_log" 2>&1 || true
  chmod 600 "$failure_log"
}

checkout_commit() {
  git checkout --detach "$1" >/dev/null
}

restore_after_failed_build() {
  local previous_sha=$1
  checkout_commit "$previous_sha"
}

rollback_running_app() {
  local failed_sha=$1
  local previous_sha=$2

  capture_private_failure_logs "$failed_sha"
  checkout_commit "$previous_sha"
  if ! compose build app >/dev/null; then
    audit "$failed_sha" rollback failed_build
    return 1
  fi
  if ! compose up -d --no-deps app >/dev/null; then
    audit "$failed_sha" rollback failed_start
    return 1
  fi
  if ! app_is_healthy; then
    capture_private_failure_logs "$previous_sha"
    audit "$failed_sha" rollback failed_health
    return 1
  fi
  printf '%s\n' "$previous_sha" > "$last_good_file"
  chmod 600 "$last_good_file"
  audit "$failed_sha" rollback restored
  return 0
}

deploy_commit() {
  local target_sha=$1
  local require_remote_head=$2
  local checkout_sha
  local last_good_sha=""
  local previous_good_sha=""
  local previous_sha
  local remote_sha

  validate_sha "$target_sha"
  [[ -d "$deploy_root" ]] || die "deployment worktree is missing"
  [[ -f "$deploy_root/compose.yaml" ]] || die "deployment compose file is missing"
  cd "$deploy_root"

  [[ -z "$(git status --porcelain --untracked-files=no)" ]] || die "deployment worktree has tracked changes"
  checkout_sha=$(git rev-parse HEAD)
  validate_sha "$checkout_sha"
  previous_sha=$checkout_sha

  if [[ -f "$last_good_file" ]]; then
    last_good_sha=$(tr -d '\n' < "$last_good_file")
    validate_sha "$last_good_sha"
    git cat-file -e "$last_good_sha^{commit}" 2>/dev/null || die "recorded healthy commit is unavailable"
    previous_sha=$last_good_sha
  fi

  if [[ "$require_remote_head" == "1" ]]; then
    info "Fetching production branch..."
    git fetch --no-tags origin main >/dev/null
    remote_sha=$(git rev-parse 'origin/main^{commit}')
    validate_sha "$remote_sha"
    if [[ "$target_sha" != "$remote_sha" ]]; then
      audit "$target_sha" deploy superseded
      info "Deployment superseded by a newer main commit."
      return 0
    fi
  fi

  git cat-file -e "$target_sha^{commit}" 2>/dev/null || die "requested commit is unavailable"
  if [[ "$target_sha" == "$last_good_sha" && -f "$previous_good_file" ]]; then
    previous_good_sha=$(tr -d '\n' < "$previous_good_file")
    validate_sha "$previous_good_sha"
    git cat-file -e "$previous_good_sha^{commit}" 2>/dev/null || die "recorded previous healthy commit is unavailable"
    previous_sha=$previous_good_sha
  fi
  if [[ "$target_sha" == "$checkout_sha" && "$target_sha" == "$last_good_sha" ]]; then
    if app_is_healthy; then
      audit "$target_sha" deploy already_current
      info "Production already runs the requested commit and is healthy."
      return 0
    fi
    info "The requested commit is checked out but the application is unhealthy; rebuilding it."
  fi

  audit "$target_sha" checkout started
  checkout_commit "$target_sha"

  info "Building the application image..."
  if ! compose build app >/dev/null; then
    restore_after_failed_build "$checkout_sha"
    audit "$target_sha" build failed
    die "application image build failed; the running container was not replaced"
  fi

  info "Replacing the application container..."
  if ! compose up -d --no-deps app >/dev/null; then
    if rollback_running_app "$target_sha" "$previous_sha"; then
      die "application start failed; production was restored to the previous commit"
    fi
    die "application start failed and automatic rollback did not recover health"
  fi

  info "Waiting for application health..."
  if ! app_is_healthy; then
    if rollback_running_app "$target_sha" "$previous_sha"; then
      die "application health check failed; production was restored to the previous commit"
    fi
    die "application health check failed and automatic rollback did not recover health"
  fi

  printf '%s\n' "$previous_sha" > "$previous_good_file"
  printf '%s\n' "$target_sha" > "$last_good_file"
  chmod 600 "$previous_good_file" "$last_good_file"
  audit "$target_sha" deploy healthy
  info "Production deployment is healthy."
}

manual_rollback() {
  local current_sha
  local rollback_sha

  [[ -z "${SSH_ORIGINAL_COMMAND:-}" ]] || die "rollback is available only to an interactive operator key"
  [[ -f "$previous_good_file" ]] || die "no previous healthy commit is recorded"
  rollback_sha=$(tr -d '\n' < "$previous_good_file")
  validate_sha "$rollback_sha"
  current_sha=$(git -C "$deploy_root" rev-parse HEAD)
  validate_sha "$current_sha"

  deploy_commit "$rollback_sha" 0
  printf '%s\n' "$current_sha" > "$previous_good_file"
  printf '%s\n' "$rollback_sha" > "$last_good_file"
  chmod 600 "$previous_good_file" "$last_good_file"
}

mkdir -p "$state_dir"
chmod 700 "$state_dir"
touch "$audit_log"
chmod 600 "$audit_log"

exec 9> "$state_dir/deploy.lock"
flock -n 9 || die "another deployment is already in progress"

if [[ -n "${SSH_ORIGINAL_COMMAND:-}" ]]; then
  [[ $# -eq 0 ]] || die "forced deployment command does not accept local arguments"
  read -r remote_action remote_sha extra <<< "$SSH_ORIGINAL_COMMAND"
  [[ "$remote_action" == "deploy" && -n "${remote_sha:-}" && -z "${extra:-}" ]] || die "unauthorized deployment command"
  deploy_commit "$remote_sha" 1
  exit 0
fi

case "${1:-}" in
  deploy)
    [[ $# -eq 2 ]] || { usage >&2; exit 1; }
    deploy_commit "$2" 1
    ;;
  rollback)
    [[ $# -eq 1 ]] || { usage >&2; exit 1; }
    manual_rollback
    ;;
  -h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac
