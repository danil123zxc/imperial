from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_WORKSPACE_ROOT = Path("/Users/danil/Public/imperial")


def _log_level_from_env() -> str:
    raw = os.environ.get("IMPERIAL_RAG_LOG_LEVEL", "INFO").strip().upper()
    return raw or "INFO"


def _log_format_from_env() -> str:
    raw = os.environ.get("IMPERIAL_RAG_LOG_FORMAT", "json").strip().casefold()
    return raw if raw == "json" else "json"


@dataclass(frozen=True)
class Settings:
    workspace_root: Path = field(
        default_factory=lambda: Path(os.environ.get("IMPERIAL_RAG_WORKSPACE_ROOT", DEFAULT_WORKSPACE_ROOT))
    )
    qdrant_url: str = field(default_factory=lambda: os.environ.get("QDRANT_URL", "http://localhost:6333"))
    qdrant_collection: str = field(default_factory=lambda: os.environ.get("QDRANT_COLLECTION", "imperial_chunks_qwen"))
    elasticsearch_url: str = field(default_factory=lambda: os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"))
    elasticsearch_index: str = field(
        default_factory=lambda: os.environ.get("ELASTICSEARCH_INDEX", "imperial_keyword_chunks")
    )
    phoenix_project_name: str = field(default_factory=lambda: os.environ.get("PHOENIX_PROJECT_NAME", "imperial-rag"))
    phoenix_collector_endpoint: str = field(
        default_factory=lambda: os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
    )
    phoenix_client_endpoint: str = field(
        default_factory=lambda: os.environ.get("PHOENIX_CLIENT_ENDPOINT", "http://localhost:6006")
    )
    log_level: str = field(default_factory=_log_level_from_env)
    log_format: str = field(default_factory=_log_format_from_env)

    @property
    def documents_root(self) -> Path:
        return self.workspace_root / "documents"

    @property
    def processed_root(self) -> Path:
        return self.workspace_root / ".imperial_rag"

    @property
    def manifest_db_path(self) -> Path:
        return self.processed_root / "manifest.sqlite3"

    @property
    def keyword_db_path(self) -> Path:
        return self.processed_root / "keyword.sqlite3"

    @property
    def extraction_root(self) -> Path:
        return self.processed_root / "extracted"
