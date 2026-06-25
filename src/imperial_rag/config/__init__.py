from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _log_level_from_env() -> str:
    raw = env_str("IMPERIAL_RAG_LOG_LEVEL", "INFO").strip().upper()
    return raw or "INFO"


def _log_format_from_env() -> str:
    raw = env_str("IMPERIAL_RAG_LOG_FORMAT", "json").strip().casefold()
    return raw if raw in {"json", "plain"} else "json"


def env_str(name: str, default: str, *, strip: bool = True) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip() if strip else raw


def env_optional_str(name: str, *, strip: bool = True) -> str | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip() if strip else raw


def env_int(name: str, default: int, *, minimum: int | None = None, invalid: str = "raise") -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        if invalid == "default":
            value = default
        else:
            raise
    if minimum is not None:
        return max(value, minimum)
    return value


def env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def env_float(name: str, default: float, *, invalid: str = "raise") -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        if invalid == "default":
            return default
        raise


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().casefold() in TRUE_ENV_VALUES


def env_optional_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().casefold() in TRUE_ENV_VALUES


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
    extraction_root_override: Path | None = None
    baseline_extraction_root: Path | None = None

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
    def auth_db_path(self) -> Path:
        return self.processed_root / "auth.sqlite3"

    @property
    def chat_history_db_path(self) -> Path:
        return self.processed_root / "chat_history.sqlite3"

    @property
    def extraction_root(self) -> Path:
        if self.extraction_root_override is not None:
            return self.extraction_root_override
        return self.processed_root / "extracted"
