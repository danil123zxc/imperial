from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _dockerignore_entries() -> set[str]:
    lines = _read(".dockerignore").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}


def test_dockerignore_excludes_private_and_generated_data() -> None:
    entries = _dockerignore_entries()

    assert ".env" in entries
    assert ".env.*" in entries
    assert "documents/" in entries
    assert ".imperial_rag/" in entries
    assert ".git/" in entries
    assert "__pycache__/" in entries
    assert ".pytest_cache/" in entries
    assert ".venv/" in entries


def test_dockerfile_builds_uv_streamlit_runtime() -> None:
    dockerfile = _read("Dockerfile")

    assert "FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim" in dockerfile
    assert "WORKDIR /app" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert "COPY pyproject.toml uv.lock ./" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert "COPY scripts ./scripts" in dockerfile
    assert '"streamlit", "run", "src/imperial_rag/web_app.py"' in dockerfile
    assert '"--server.address", "0.0.0.0"' in dockerfile
    assert '"--server.port", "8501"' in dockerfile
