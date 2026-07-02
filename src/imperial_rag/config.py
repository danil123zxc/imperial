from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field, field_validator
from pydantic_core import PydanticUseDefault
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(populate_by_name=True, frozen=True, extra="ignore")

    workspace_root: Path = Field(
        DEFAULT_WORKSPACE_ROOT,
        validation_alias=AliasChoices("IMPERIAL_RAG_WORKSPACE_ROOT", "workspace_root"),
    )
    qdrant_url: str = Field("http://localhost:6333", validation_alias=AliasChoices("QDRANT_URL", "qdrant_url"))
    qdrant_collection: str = Field(
        "imperial_chunks_qwen",
        validation_alias=AliasChoices("QDRANT_COLLECTION", "qdrant_collection"),
    )
    elasticsearch_url: str = Field(
        "http://localhost:9200",
        validation_alias=AliasChoices("ELASTICSEARCH_URL", "elasticsearch_url"),
    )
    elasticsearch_index: str = Field(
        "imperial_keyword_chunks",
        validation_alias=AliasChoices("ELASTICSEARCH_INDEX", "elasticsearch_index"),
    )
    phoenix_project_name: str = Field(
        "imperial-rag",
        validation_alias=AliasChoices("PHOENIX_PROJECT_NAME", "phoenix_project_name"),
    )
    phoenix_collector_endpoint: str = Field(
        "http://localhost:6006/v1/traces",
        validation_alias=AliasChoices("PHOENIX_COLLECTOR_ENDPOINT", "phoenix_collector_endpoint"),
    )
    phoenix_client_endpoint: str = Field(
        "http://localhost:6006",
        validation_alias=AliasChoices("PHOENIX_CLIENT_ENDPOINT", "phoenix_client_endpoint"),
    )
    log_level: str = Field("INFO", validation_alias=AliasChoices("IMPERIAL_RAG_LOG_LEVEL", "log_level"))
    log_format: str = Field("json", validation_alias=AliasChoices("IMPERIAL_RAG_LOG_FORMAT", "log_format"))
    extraction_root_override: Path | None = None
    baseline_extraction_root: Path | None = None
    manifest_db_path_override: Path | None = None
    recreate_qdrant_collection: bool = False

    @field_validator("*", mode="before")
    @classmethod
    def _blank_env_uses_default(cls, value: Any) -> Any:
        if isinstance(value, str) and value.strip() == "":
            raise PydanticUseDefault()
        return value

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return value.strip().upper() or "INFO"

    @field_validator("log_format")
    @classmethod
    def _normalize_log_format(cls, value: str) -> str:
        normalized = value.strip().casefold()
        return normalized if normalized in {"json", "plain"} else "json"

    @property
    def documents_root(self) -> Path:
        return self.workspace_root / "documents"

    @property
    def processed_root(self) -> Path:
        return self.workspace_root / ".imperial_rag"

    @property
    def manifest_db_path(self) -> Path:
        if self.manifest_db_path_override is not None:
            return self.manifest_db_path_override
        if self.extraction_root_override is not None:
            return self.extraction_root / "manifest.sqlite3"
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
