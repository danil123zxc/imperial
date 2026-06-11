from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _dockerignore_entries() -> set[str]:
    lines = _read(".dockerignore").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.strip().startswith("#")}


def _service_block(compose: str, service_name: str) -> str:
    lines = compose.splitlines()
    start = lines.index(f"  {service_name}:")
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.startswith("  ") and not line.startswith("    ") and line.endswith(":"):
            end = index
            break
        if line and not line.startswith(" "):
            end = index
            break
    return "\n".join(lines[start:end])


def test_dockerignore_excludes_private_and_generated_data() -> None:
    entries = _dockerignore_entries()

    assert ".env" in entries
    assert ".env.*" in entries
    assert "documents/" in entries
    assert ".imperial_rag/" in entries
    assert ".git/" in entries
    assert "__pycache__/" in entries
    assert "**/__pycache__/" in entries
    assert "**/*.py[cod]" in entries
    assert ".pytest_cache/" in entries
    assert ".venv/" in entries


def test_dockerfile_builds_uv_streamlit_runtime() -> None:
    dockerfile = _read("Dockerfile")

    assert "FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim" in dockerfile
    assert "FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:" in dockerfile
    assert "WORKDIR /app" in dockerfile
    assert "IMPERIAL_RAG_WORKSPACE_ROOT=/app" in dockerfile
    assert "PYTHONPATH=/app/src" in dockerfile
    assert "uv sync --frozen --no-dev --no-cache" in dockerfile
    assert "COPY pyproject.toml uv.lock ./" in dockerfile
    assert "COPY src ./src" in dockerfile
    assert "COPY scripts ./scripts" in dockerfile
    assert "EXPOSE 8501" in dockerfile
    assert '"streamlit", "run", "src/imperial_rag/web_app.py"' in dockerfile
    assert '"--server.address", "0.0.0.0"' in dockerfile
    assert '"--server.port", "8501"' in dockerfile
    assert '"--server.headless", "true"' in dockerfile


def test_compose_defines_private_app_and_ingest_services() -> None:
    compose = _read("compose.yaml")
    app = _service_block(compose, "app")
    ingest = _service_block(compose, "ingest")
    phoenix = _service_block(compose, "phoenix")
    qdrant = _service_block(compose, "qdrant")

    required_snippets = [
        "x-imperial-app-base:",
        "app:",
        "ingest:",
        "path: .env",
        "required: false",
        'profiles: ["ingest"]',
        '"127.0.0.1:8501:8501"',
        '"127.0.0.1:6006:6006"',
        '"127.0.0.1:4317:4317"',
        '"127.0.0.1:6333:6333"',
        "QDRANT_URL: http://qdrant:6333",
        "PHOENIX_CLIENT_ENDPOINT: http://phoenix:6006",
        "PHOENIX_COLLECTOR_ENDPOINT: http://phoenix:6006/v1/traces",
        "./documents:/app/documents:ro",
        "./.imperial_rag:/app/.imperial_rag",
        "scripts/ingest.py",
        "--index-vectors",
    ]

    for snippet in required_snippets:
        assert snippet in compose

    assert "ports:" not in ingest
    assert '"127.0.0.1:8501:8501"' in app
    assert '"127.0.0.1:6333:6333"' in qdrant
    assert '"127.0.0.1:6006:6006"' in phoenix
    assert '"127.0.0.1:4317:4317"' in phoenix
