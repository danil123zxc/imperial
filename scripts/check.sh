#!/usr/bin/env bash
set -euo pipefail

uv run --extra dev ruff check .
uv run --extra dev mypy src/imperial_rag
uv run --extra dev python -m pytest --cov=imperial_rag -q
git diff --check
