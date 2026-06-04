from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_DASHSCOPE_COMPAT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_CHAT_MODEL = "qwen3.7-max"
DEFAULT_QWEN_VISION_MODEL = "qwen-vl-ocr-2025-11-20"
DEFAULT_QWEN_OCR_TASK = "multi_lan"
DEFAULT_QWEN_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_QWEN_EMBEDDING_DIMENSIONS = 2048
DEFAULT_QWEN_RERANK_MODEL = "qwen3-rerank"
VECTOR_PROVIDER = "dashscope"


class MissingDashScopeKeyError(RuntimeError):
    pass


class VectorProviderMismatchError(RuntimeError):
    pass


class DashScopeProviderError(RuntimeError):
    pass


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _env_optional_str(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip()


def _env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _env_optional_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class QwenProviderSettings:
    api_key: str | None
    region: str = "beijing"
    base_url: str = DEFAULT_DASHSCOPE_BASE_URL
    compat_base_url: str = DEFAULT_DASHSCOPE_COMPAT_BASE_URL
    chat_model: str = DEFAULT_QWEN_CHAT_MODEL
    vision_model: str = DEFAULT_QWEN_VISION_MODEL
    ocr_task: str = DEFAULT_QWEN_OCR_TASK
    ocr_min_pixels: int | None = None
    ocr_max_pixels: int | None = None
    ocr_enable_rotate: bool | None = None
    embedding_model: str = DEFAULT_QWEN_EMBEDDING_MODEL
    embedding_dimensions: int | None = DEFAULT_QWEN_EMBEDDING_DIMENSIONS
    rerank_model: str = DEFAULT_QWEN_RERANK_MODEL
    allow_legacy_openai: bool = False
    allow_legacy_cohere: bool = False

    @classmethod
    def from_env(cls) -> "QwenProviderSettings":
        return cls(
            api_key=_env_optional_str("DASHSCOPE_API_KEY"),
            region=_env_str("IMPERIAL_RAG_DASHSCOPE_REGION", "beijing"),
            base_url=_env_str("IMPERIAL_RAG_DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL),
            compat_base_url=_env_str("IMPERIAL_RAG_DASHSCOPE_COMPAT_BASE_URL", DEFAULT_DASHSCOPE_COMPAT_BASE_URL),
            chat_model=_env_str("IMPERIAL_RAG_QWEN_CHAT_MODEL", DEFAULT_QWEN_CHAT_MODEL),
            vision_model=_env_str("IMPERIAL_RAG_QWEN_VISION_MODEL", DEFAULT_QWEN_VISION_MODEL),
            ocr_task=_env_str("IMPERIAL_RAG_QWEN_OCR_TASK", DEFAULT_QWEN_OCR_TASK),
            ocr_min_pixels=_env_optional_int("IMPERIAL_RAG_QWEN_OCR_MIN_PIXELS"),
            ocr_max_pixels=_env_optional_int("IMPERIAL_RAG_QWEN_OCR_MAX_PIXELS"),
            ocr_enable_rotate=_env_optional_bool("IMPERIAL_RAG_QWEN_OCR_ENABLE_ROTATE"),
            embedding_model=_env_str("IMPERIAL_RAG_QWEN_EMBEDDING_MODEL", DEFAULT_QWEN_EMBEDDING_MODEL),
            embedding_dimensions=_env_optional_int("IMPERIAL_RAG_QWEN_EMBEDDING_DIMENSIONS")
            if os.environ.get("IMPERIAL_RAG_QWEN_EMBEDDING_DIMENSIONS", "").strip()
            else DEFAULT_QWEN_EMBEDDING_DIMENSIONS,
            rerank_model=_env_str("IMPERIAL_RAG_QWEN_RERANK_MODEL", DEFAULT_QWEN_RERANK_MODEL),
            allow_legacy_openai=_env_bool("IMPERIAL_RAG_ALLOW_LEGACY_OPENAI"),
            allow_legacy_cohere=_env_bool("IMPERIAL_RAG_ALLOW_LEGACY_COHERE"),
        )

    def require_api_key(self) -> str:
        if not self.api_key:
            raise MissingDashScopeKeyError("DASHSCOPE_API_KEY is required for hosted Qwen/DashScope behavior.")
        return self.api_key

    def vector_metadata(self) -> "VectorProviderMetadata":
        return VectorProviderMetadata(
            provider=VECTOR_PROVIDER,
            embedding_model=self.embedding_model,
            embedding_dimensions=self.embedding_dimensions,
            distance="cosine",
        )


@dataclass(frozen=True)
class VectorProviderMetadata:
    provider: str
    embedding_model: str
    embedding_dimensions: int | None
    distance: str = "cosine"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VectorProviderMetadata":
        return cls(
            provider=str(payload["provider"]),
            embedding_model=str(payload["embedding_model"]),
            embedding_dimensions=payload.get("embedding_dimensions"),
            distance=str(payload.get("distance", "cosine")),
        )


def dashscope_configured(settings: QwenProviderSettings | None = None) -> bool:
    resolved = settings or QwenProviderSettings.from_env()
    return bool(resolved.api_key)


def configure_dashscope_sdk(settings: QwenProviderSettings | None = None) -> None:
    resolved = settings or QwenProviderSettings.from_env()
    api_key = resolved.require_api_key()
    import dashscope

    dashscope.api_key = api_key
    dashscope.base_http_api_url = resolved.base_url


def vector_metadata_path(settings: Any) -> Path:
    return Path(settings.processed_root) / "vector_provider.json"


def read_vector_metadata(settings: Any) -> VectorProviderMetadata | None:
    path = vector_metadata_path(settings)
    if not path.exists():
        return None
    return VectorProviderMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_vector_metadata(settings: Any, metadata: VectorProviderMetadata) -> None:
    path = vector_metadata_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def vector_metadata_matches_config(settings: Any, provider_settings: QwenProviderSettings | None = None) -> bool:
    existing = read_vector_metadata(settings)
    if existing is None:
        return False
    expected = (provider_settings or QwenProviderSettings.from_env()).vector_metadata()
    return existing == expected


def ensure_vector_metadata_compatible(settings: Any, provider_settings: QwenProviderSettings | None = None) -> None:
    existing = read_vector_metadata(settings)
    if existing is None:
        return
    expected = (provider_settings or QwenProviderSettings.from_env()).vector_metadata()
    if existing != expected:
        raise VectorProviderMismatchError(
            "Qdrant vector provider metadata mismatch: "
            f"existing={existing.to_dict()} expected={expected.to_dict()}"
        )


def _import_chat_qwen():
    from langchain_qwq import ChatQwen

    return ChatQwen


def _import_dashscope_rerank():
    from langchain_community.document_compressors.dashscope_rerank import DashScopeRerank

    return DashScopeRerank


def create_chat_model(settings: QwenProviderSettings | None = None) -> Any:
    resolved = settings or QwenProviderSettings.from_env()
    resolved.require_api_key()
    chat_cls = _import_chat_qwen()
    return chat_cls(model=resolved.chat_model, temperature=0)


def create_reranker(top_n: int, settings: QwenProviderSettings | None = None) -> Any:
    resolved = settings or QwenProviderSettings.from_env()
    api_key = resolved.require_api_key()
    reranker_cls = _import_dashscope_rerank()
    return reranker_cls(model=resolved.rerank_model, top_n=top_n, api_key=api_key)


class DashScopeTextEmbeddings(Embeddings):
    def __init__(self, settings: QwenProviderSettings | None = None, client: Any | None = None) -> None:
        self.settings = settings or QwenProviderSettings.from_env()
        self.api_key = self.settings.require_api_key()
        self.model = self.settings.embedding_model
        self.dimensions = self.settings.embedding_dimensions
        if client is None:
            import dashscope

            client = dashscope.TextEmbedding
        self.client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, text_type="document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], text_type="query")[0]

    def _embed(self, texts: list[str], text_type: str) -> list[list[float]]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "text_type": text_type,
            "api_key": self.api_key,
        }
        if self.dimensions is not None:
            kwargs["dimension"] = self.dimensions
        response = self.client.call(**kwargs)
        status_code = _response_get(response, "status_code")
        if status_code != 200:
            code = _response_get(response, "code") or "dashscope_error"
            message = _sanitize_provider_message(
                _response_get(response, "message") or "DashScope request failed",
                self.api_key,
            )
            raise DashScopeProviderError(
                f"DashScope embedding failed: status_code={status_code} code={code} message={message}"
            )
        output = _response_get(response, "output") or {}
        embeddings = _response_get(output, "embeddings") or []
        return [list(_response_get(item, "embedding")) for item in embeddings]


def create_embeddings(settings: QwenProviderSettings | None = None) -> Embeddings:
    resolved = settings or QwenProviderSettings.from_env()
    if resolved.embedding_dimensions is not None:
        return DashScopeTextEmbeddings(settings=resolved)
    from langchain_community.embeddings.dashscope import DashScopeEmbeddings

    return DashScopeEmbeddings(model=resolved.embedding_model, dashscope_api_key=resolved.require_api_key())


def _sanitize_provider_message(message: str, api_key: str | None) -> str:
    sanitized = str(message)
    if api_key:
        sanitized = sanitized.replace(api_key, "[redacted]")
    return sanitized


def build_qwen_ocr_message(image_path: Path, settings: QwenProviderSettings | None = None) -> dict[str, Any]:
    resolved = settings or QwenProviderSettings.from_env()
    mime_type, _ = mimetypes.guess_type(image_path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    image_payload: dict[str, Any] = {"image": f"data:{mime_type};base64,{encoded}"}
    if resolved.ocr_min_pixels is not None:
        image_payload["min_pixels"] = resolved.ocr_min_pixels
    if resolved.ocr_max_pixels is not None:
        image_payload["max_pixels"] = resolved.ocr_max_pixels
    if resolved.ocr_enable_rotate is not None:
        image_payload["enable_rotate"] = resolved.ocr_enable_rotate
    return {"role": "user", "content": [image_payload]}


def parse_qwen_ocr_response(response: Any) -> str:
    output = _response_get(response, "output")
    choices = _response_get(output, "choices") or []
    if not choices:
        return ""
    message = _response_get(choices[0], "message")
    content = _response_get(message, "content") or []
    if isinstance(content, str):
        return content.strip()
    text_parts: list[str] = []
    for item in content:
        text = _response_get(item, "text")
        if text:
            text_parts.append(str(text))
    return "\n".join(part.strip() for part in text_parts if part.strip()).strip()


def _response_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)
