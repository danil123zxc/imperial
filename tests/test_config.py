from pathlib import Path

from imperial_rag.config import Settings


def test_settings_defaults_to_workspace_documents():
    settings = Settings()

    assert settings.workspace_root == Path("/Users/danil/Public/imperial")
    assert settings.documents_root == Path("/Users/danil/Public/imperial/documents")
    assert settings.processed_root == Path("/Users/danil/Public/imperial/.imperial_rag")
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "imperial_chunks_qwen"
    assert settings.phoenix_project_name == "imperial-rag"
    assert settings.phoenix_collector_endpoint == "http://localhost:6006/v1/traces"
    assert settings.phoenix_client_endpoint == "http://localhost:6006"
    assert settings.manifest_db_path == Path("/Users/danil/Public/imperial/.imperial_rag/manifest.sqlite3")
    assert settings.keyword_db_path == Path("/Users/danil/Public/imperial/.imperial_rag/keyword.sqlite3")
    assert settings.extraction_root == Path("/Users/danil/Public/imperial/.imperial_rag/extracted")


def test_settings_reads_environment_overrides_including_qdrant_collection(monkeypatch, tmp_path):
    monkeypatch.setenv("IMPERIAL_RAG_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_chunks")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "test-project")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix.internal:6006/v1/traces")
    monkeypatch.setenv("PHOENIX_CLIENT_ENDPOINT", "http://phoenix.internal:6006")

    settings = Settings()

    assert settings.workspace_root == tmp_path
    assert settings.documents_root == tmp_path / "documents"
    assert settings.processed_root == tmp_path / ".imperial_rag"
    assert settings.qdrant_url == "http://127.0.0.1:6333"
    assert settings.qdrant_collection == "test_chunks"
    assert settings.phoenix_project_name == "test-project"
    assert settings.phoenix_collector_endpoint == "http://phoenix.internal:6006/v1/traces"
    assert settings.phoenix_client_endpoint == "http://phoenix.internal:6006"
