#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${IMPERIAL_RAG_WORKSPACE_ROOT:-}" ]]; then
  repo_root=$(git rev-parse --show-toplevel)
  common_git_dir=$(git rev-parse --git-common-dir)
  if [[ "$common_git_dir" = /* ]]; then
    canonical_workspace_root=$(cd "$common_git_dir/.." && pwd -P)
  else
    canonical_workspace_root=$(cd "$repo_root/$common_git_dir/.." && pwd -P)
  fi
  export IMPERIAL_RAG_WORKSPACE_ROOT="$canonical_workspace_root"
fi

uv run --extra dev ruff check .
uv run --extra dev mypy src/imperial_rag
uv run --extra dev python -m pytest --cov=imperial_rag -q
if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
  git fetch --no-tags origin "refs/heads/${GITHUB_BASE_REF}:refs/remotes/origin/${GITHUB_BASE_REF}"
  git diff --check "origin/${GITHUB_BASE_REF}...HEAD"
else
  git diff --check
fi
