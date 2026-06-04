from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from imperial_rag.config import Settings


def clear_provider_env(monkeypatch):
    for name in (
        "DASHSCOPE_API_KEY",
        "IMPERIAL_RAG_DASHSCOPE_REGION",
        "IMPERIAL_RAG_DASHSCOPE_BASE_URL",
        "IMPERIAL_RAG_DASHSCOPE_COMPAT_BASE_URL",
        "IMPERIAL_RAG_QWEN_CHAT_MODEL",
        "IMPERIAL_RAG_QWEN_VISION_MODEL",
        "IMPERIAL_RAG_QWEN_OCR_TASK",
        "IMPERIAL_RAG_QWEN_OCR_MIN_PIXELS",
        "IMPERIAL_RAG_QWEN_OCR_MAX_PIXELS",
        "IMPERIAL_RAG_QWEN_OCR_ENABLE_ROTATE",
        "IMPERIAL_RAG_QWEN_EMBEDDING_MODEL",
        "IMPERIAL_RAG_QWEN_EMBEDDING_DIMENSIONS",
        "IMPERIAL_RAG_QWEN_RERANK_MODEL",
        "IMPERIAL_RAG_ALLOW_LEGACY_OPENAI",
        "IMPERIAL_RAG_ALLOW_LEGACY_COHERE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_qwen_provider_settings_defaults(monkeypatch):
    clear_provider_env(monkeypatch)

    from imperial_rag.providers import QwenProviderSettings

    settings = QwenProviderSettings.from_env()

    assert settings.api_key is None
    assert settings.region == "beijing"
    assert settings.base_url == "https://dashscope.aliyuncs.com/api/v1"
    assert settings.compat_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert settings.chat_model == "qwen3.7-max"
    assert settings.vision_model == "qwen-vl-ocr-2025-11-20"
    assert settings.ocr_task == "multi_lan"
    assert settings.ocr_min_pixels is None
    assert settings.ocr_max_pixels is None
    assert settings.ocr_enable_rotate is None
    assert settings.embedding_model == "text-embedding-v4"
    assert settings.embedding_dimensions == 2048
    assert settings.rerank_model == "qwen3-rerank"
    assert settings.allow_legacy_openai is False
    assert settings.allow_legacy_cohere is False


def test_qwen_provider_settings_read_environment(monkeypatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setenv("IMPERIAL_RAG_DASHSCOPE_REGION", "singapore")
    monkeypatch.setenv("IMPERIAL_RAG_DASHSCOPE_BASE_URL", "https://workspace.ap-southeast-1.maas.aliyuncs.com/api/v1")
    monkeypatch.setenv("IMPERIAL_RAG_DASHSCOPE_COMPAT_BASE_URL", "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_CHAT_MODEL", "qwen-test-chat")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_VISION_MODEL", "qwen-test-ocr")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_OCR_TASK", "text_recognition")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_OCR_MIN_PIXELS", "3072")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_OCR_MAX_PIXELS", "8388608")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_OCR_ENABLE_ROTATE", "true")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_EMBEDDING_MODEL", "text-embedding-v4")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_RERANK_MODEL", "qwen-test-rerank")
    monkeypatch.setenv("IMPERIAL_RAG_ALLOW_LEGACY_OPENAI", "1")
    monkeypatch.setenv("IMPERIAL_RAG_ALLOW_LEGACY_COHERE", "yes")

    from imperial_rag.providers import QwenProviderSettings

    settings = QwenProviderSettings.from_env()

    assert settings.api_key == "dashscope-test-key"
    assert settings.region == "singapore"
    assert settings.base_url == "https://workspace.ap-southeast-1.maas.aliyuncs.com/api/v1"
    assert settings.compat_base_url == "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    assert settings.chat_model == "qwen-test-chat"
    assert settings.vision_model == "qwen-test-ocr"
    assert settings.ocr_task == "text_recognition"
    assert settings.ocr_min_pixels == 3072
    assert settings.ocr_max_pixels == 8388608
    assert settings.ocr_enable_rotate is True
    assert settings.embedding_dimensions == 1024
    assert settings.rerank_model == "qwen-test-rerank"
    assert settings.allow_legacy_openai is True
    assert settings.allow_legacy_cohere is True


def test_dashscope_configured_requires_key(monkeypatch):
    clear_provider_env(monkeypatch)

    from imperial_rag.providers import dashscope_configured

    assert dashscope_configured() is False
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    assert dashscope_configured() is True
