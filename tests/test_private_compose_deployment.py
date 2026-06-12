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
    elasticsearch = _service_block(compose, "elasticsearch")
    kibana = _service_block(compose, "kibana")

    required_snippets = [
        "x-imperial-app-base:",
        "app:",
        "ingest:",
        "elasticsearch:",
        "kibana:",
        "path: .env",
        "required: false",
        'profiles: ["ingest"]',
        '"127.0.0.1:8501:8501"',
        '"127.0.0.1:6006:6006"',
        '"127.0.0.1:4317:4317"',
        '"127.0.0.1:6333:6333"',
        '"127.0.0.1:9200:9200"',
        '"127.0.0.1:5601:5601"',
        "QDRANT_URL: http://qdrant:6333",
        "ELASTICSEARCH_URL: http://elasticsearch:9200",
        "ELASTICSEARCH_INDEX: imperial_keyword_chunks",
        "PHOENIX_CLIENT_ENDPOINT: http://phoenix:6006",
        "PHOENIX_COLLECTOR_ENDPOINT: http://phoenix:6006/v1/traces",
        "docker.elastic.co/elasticsearch/elasticsearch:8.19.15",
        "docker.elastic.co/kibana/kibana:8.19.15",
        "discovery.type: single-node",
        'xpack.security.enabled: "false"',
        'xpack.security.http.ssl.enabled: "false"',
        "SERVER_NAME: kibana",
        "ELASTICSEARCH_HOSTS: '[\"http://elasticsearch:9200\"]'",
        "ES_JAVA_OPTS: -Xms512m -Xmx512m",
        "./documents:/app/documents:ro",
        "./.imperial_rag:/app/.imperial_rag",
        "elasticsearch_data:/usr/share/elasticsearch/data",
        "  elasticsearch_data:\n    driver: local",
        "scripts/ingest.py",
        "--index-vectors",
    ]

    for snippet in required_snippets:
        assert snippet in compose

    assert compose.count("condition: service_healthy") >= 3
    assert "ports:" not in ingest
    assert '"127.0.0.1:8501:8501"' in app
    assert '"127.0.0.1:6333:6333"' in qdrant
    assert '"127.0.0.1:9200:9200"' in elasticsearch
    assert '"127.0.0.1:6006:6006"' in phoenix
    assert '"127.0.0.1:4317:4317"' in phoenix
    assert '"127.0.0.1:5601:5601"' in kibana
    assert "http://elasticsearch:9200" in kibana


def test_compose_pins_phoenix_and_qdrant_images() -> None:
    compose = _read("compose.yaml")
    phoenix = _service_block(compose, "phoenix")
    qdrant = _service_block(compose, "qdrant")

    assert "image: arizephoenix/phoenix:latest\n" not in compose
    assert "image: qdrant/qdrant:latest\n" not in compose
    assert "image: arizephoenix/phoenix:" in phoenix
    assert "image: qdrant/qdrant:" in qdrant
    assert "@sha256:" in phoenix
    assert "@sha256:" in qdrant


def test_env_example_documents_compose_overrides() -> None:
    env_example = _read(".env.example")
    lines = set(env_example.splitlines())

    assert "Compose container overrides" in env_example
    assert "# IMPERIAL_RAG_WORKSPACE_ROOT=/app" in lines
    assert "# ELASTICSEARCH_URL=http://elasticsearch:9200" in lines
    assert "# QDRANT_URL=http://qdrant:6333" in lines
    assert "# PHOENIX_CLIENT_ENDPOINT=http://phoenix:6006" in lines
    assert "# PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006/v1/traces" in lines
    assert "IMPERIAL_RAG_WORKSPACE_ROOT=/app" not in lines
    assert "ELASTICSEARCH_URL=http://elasticsearch:9200" not in lines
    assert "QDRANT_URL=http://qdrant:6333" not in lines
    assert "PHOENIX_CLIENT_ENDPOINT=http://phoenix:6006" not in lines
    assert "PHOENIX_COLLECTOR_ENDPOINT=http://phoenix:6006/v1/traces" not in lines


def test_readme_documents_private_compose_deployment() -> None:
    readme = _read("README.md")

    assert "## Private Compose Deployment" in readme
    assert "docker compose up -d elasticsearch qdrant phoenix app kibana" in readme
    assert "docker compose --profile ingest up ingest" in readme
    assert "http://127.0.0.1:8501/_stcore/health" in readme
    assert "http://127.0.0.1:9200" in readme
    assert "http://127.0.0.1:5601" in readme
