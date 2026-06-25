# ruff: noqa: F405
from __future__ import annotations

from imperial_rag.integrations.dashscope import *  # noqa: F403

__all__ = [
    "DASHSCOPE_EMBEDDING_BATCH_SIZE",
    "DEFAULT_DASHSCOPE_BASE_URL",
    "DEFAULT_DASHSCOPE_COMPAT_BASE_URL",
    "DEFAULT_QWEN_CHAT_MODEL",
    "DEFAULT_QWEN_EMBEDDING_DIMENSIONS",
    "DEFAULT_QWEN_EMBEDDING_MODEL",
    "DEFAULT_QWEN_OCR_TASK",
    "DEFAULT_QWEN_RERANK_MODEL",
    "DEFAULT_QWEN_VISION_MODEL",
    "DashScopeProviderError",
    "DashScopeTextEmbeddings",
    "MissingDashScopeKeyError",
    "QwenProviderSettings",
    "VECTOR_PROVIDER",
    "VectorProviderMetadata",
    "VectorProviderMismatchError",
    "build_qwen_ocr_message",
    "configure_dashscope_sdk",
    "create_chat_model",
    "create_embeddings",
    "create_reranker",
    "dashscope_configured",
    "ensure_vector_metadata_compatible",
    "parse_qwen_ocr_response",
    "read_vector_metadata",
    "vector_metadata_matches_config",
    "vector_metadata_path",
    "write_vector_metadata",
]
