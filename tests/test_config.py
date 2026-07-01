import os
from pathlib import Path

from imperial_rag.config import Settings


def test_settings_defaults_to_workspace_documents():
    settings = Settings()

    assert settings.workspace_root == Path("/Users/danil/Public/imperial")
    assert settings.documents_root == Path("/Users/danil/Public/imperial/documents")
    assert settings.processed_root == Path("/Users/danil/Public/imperial/.imperial_rag")
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.qdrant_collection == "imperial_chunks_qwen"
    assert settings.elasticsearch_url == "http://localhost:9200"
    assert settings.elasticsearch_index == "imperial_keyword_chunks"
    assert settings.phoenix_project_name == "imperial-rag"
    assert settings.phoenix_collector_endpoint == "http://localhost:6006/v1/traces"
    assert settings.phoenix_client_endpoint == "http://localhost:6006"
    assert settings.log_level == "INFO"
    assert settings.log_format == "json"
    assert settings.manifest_db_path == Path("/Users/danil/Public/imperial/.imperial_rag/manifest.sqlite3")
    assert settings.auth_db_path == Path("/Users/danil/Public/imperial/.imperial_rag/auth.sqlite3")
    assert settings.extraction_root == Path("/Users/danil/Public/imperial/.imperial_rag/extracted")


def test_settings_reads_environment_overrides_including_qdrant_collection(monkeypatch, tmp_path):
    monkeypatch.setenv("IMPERIAL_RAG_WORKSPACE_ROOT", str(tmp_path))
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6333")
    monkeypatch.setenv("QDRANT_COLLECTION", "test_chunks")
    monkeypatch.setenv("ELASTICSEARCH_URL", "http://127.0.0.1:9200")
    monkeypatch.setenv("ELASTICSEARCH_INDEX", "test_keyword_chunks")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "test-project")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix.internal:6006/v1/traces")
    monkeypatch.setenv("PHOENIX_CLIENT_ENDPOINT", "http://phoenix.internal:6006")
    monkeypatch.setenv("IMPERIAL_RAG_LOG_LEVEL", "debug")
    monkeypatch.setenv("IMPERIAL_RAG_LOG_FORMAT", "plain")

    settings = Settings()

    assert settings.workspace_root == tmp_path
    assert settings.documents_root == tmp_path / "documents"
    assert settings.processed_root == tmp_path / ".imperial_rag"
    assert settings.qdrant_url == "http://127.0.0.1:6333"
    assert settings.qdrant_collection == "test_chunks"
    assert settings.elasticsearch_url == "http://127.0.0.1:9200"
    assert settings.elasticsearch_index == "test_keyword_chunks"
    assert settings.phoenix_project_name == "test-project"
    assert settings.phoenix_collector_endpoint == "http://phoenix.internal:6006/v1/traces"
    assert settings.phoenix_client_endpoint == "http://phoenix.internal:6006"
    assert settings.log_level == "DEBUG"
    assert settings.log_format == "plain"


def test_settings_allows_extraction_root_override(tmp_path):
    shadow_root = tmp_path / ".imperial_rag" / "extracted-shadow-v2"

    settings = Settings(workspace_root=tmp_path, extraction_root_override=shadow_root)

    assert settings.processed_root == tmp_path / ".imperial_rag"
    assert settings.extraction_root == shadow_root
    assert settings.manifest_db_path == shadow_root / "manifest.sqlite3"


def test_settings_allows_explicit_manifest_db_override(tmp_path):
    shadow_root = tmp_path / ".imperial_rag" / "extracted-shadow-v2"
    manifest_path = tmp_path / ".imperial_rag" / "manifest-shadow-v2.sqlite3"

    settings = Settings(
        workspace_root=tmp_path,
        extraction_root_override=shadow_root,
        manifest_db_path_override=manifest_path,
    )

    assert settings.extraction_root == shadow_root
    assert settings.manifest_db_path == manifest_path


def test_load_project_env_reads_workspace_dotenv_without_overriding_exported_values(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "DASHSCOPE_API_KEY=dotenv-dashscope-key",
                "QDRANT_COLLECTION=dotenv_chunks",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("QDRANT_COLLECTION", "exported_chunks")

    from imperial_rag.env import load_project_env

    assert load_project_env(tmp_path) is True
    assert os.environ["DASHSCOPE_API_KEY"] == "dotenv-dashscope-key"
    assert os.environ["QDRANT_COLLECTION"] == "exported_chunks"
