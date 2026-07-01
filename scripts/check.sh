#!/usr/bin/env bash
set -euo pipefail

uv run --extra dev ruff check .
uv run --extra dev mypy src/imperial_rag
uv run --extra dev python -m pytest --cov=imperial_rag -q
if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
  git fetch --no-tags origin "refs/heads/${GITHUB_BASE_REF}:refs/remotes/origin/${GITHUB_BASE_REF}"
  git diff --check "origin/${GITHUB_BASE_REF}...HEAD"
else
  git diff --check
fi
