from __future__ import annotations

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


def test_qwen_provider_vector_metadata_defaults():
    from imperial_rag.providers import QwenProviderSettings

    metadata = QwenProviderSettings(api_key=None).vector_metadata()

    assert metadata.provider == "dashscope"
    assert metadata.embedding_model == "text-embedding-v4"
    assert metadata.embedding_dimensions == 2048
    assert metadata.distance == "cosine"


def test_vector_metadata_write_read_round_trip(tmp_path):
    from imperial_rag.providers import QwenProviderSettings, read_vector_metadata, write_vector_metadata

    settings = Settings(workspace_root=tmp_path)
    metadata = QwenProviderSettings(api_key=None).vector_metadata()

    write_vector_metadata(settings, metadata)

    assert read_vector_metadata(settings) == metadata


def test_missing_vector_metadata_returns_none_and_does_not_match_config(tmp_path):
    from imperial_rag.providers import read_vector_metadata, vector_metadata_matches_config

    settings = Settings(workspace_root=tmp_path)

    assert read_vector_metadata(settings) is None
    assert vector_metadata_matches_config(settings) is False


def test_vector_metadata_matches_config(tmp_path):
    from imperial_rag.providers import QwenProviderSettings, vector_metadata_matches_config, write_vector_metadata

    settings = Settings(workspace_root=tmp_path)
    provider_settings = QwenProviderSettings(api_key=None)

    write_vector_metadata(settings, provider_settings.vector_metadata())

    assert vector_metadata_matches_config(settings, provider_settings) is True


def test_vector_metadata_mismatch_raises_without_dashscope_key_value(tmp_path):
    from imperial_rag.providers import (
        QwenProviderSettings,
        VectorProviderMismatchError,
        ensure_vector_metadata_compatible,
        write_vector_metadata,
    )

    settings = Settings(workspace_root=tmp_path)
    write_vector_metadata(settings, QwenProviderSettings(api_key=None).vector_metadata())

    provider_settings = QwenProviderSettings(api_key="dashscope-secret-key", embedding_dimensions=1024)
    with pytest.raises(VectorProviderMismatchError) as exc_info:
        ensure_vector_metadata_compatible(settings, provider_settings)

    message = str(exc_info.value)
    assert "dashscope-secret-key" not in message
    assert "text-embedding-v4" in message


def test_qwen_chat_factory_uses_chatqwen(monkeypatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")

    created = {}

    class FakeChatQwen:
        def __init__(self, model, temperature):
            created["model"] = model
            created["temperature"] = temperature

    import imperial_rag.providers as providers

    monkeypatch.setattr(providers, "_import_chat_qwen", lambda: FakeChatQwen)

    model = providers.create_chat_model()

    assert isinstance(model, FakeChatQwen)
    assert created == {"model": "qwen3.7-max", "temperature": 0}


def test_qwen_embedding_factory_uses_dimension_aware_wrapper(monkeypatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_EMBEDDING_DIMENSIONS", "2048")

    from imperial_rag.providers import DashScopeTextEmbeddings, create_embeddings

    embeddings = create_embeddings()

    assert isinstance(embeddings, DashScopeTextEmbeddings)
    assert embeddings.model == "text-embedding-v4"
    assert embeddings.dimensions == 2048


def test_qwen_reranker_factory_uses_dashscope_rerank(monkeypatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    created = {}

    class FakeDashScopeRerank:
        def __init__(self, model, top_n, api_key):
            created["model"] = model
            created["top_n"] = top_n
            created["api_key"] = api_key

    import imperial_rag.providers as providers

    monkeypatch.setattr(providers, "_import_dashscope_rerank", lambda: FakeDashScopeRerank)

    reranker = providers.create_reranker(top_n=7)

    assert isinstance(reranker, FakeDashScopeRerank)
    assert created == {"model": "qwen3-rerank", "top_n": 7, "api_key": "dashscope-test-key"}


def test_dashscope_text_embeddings_call_sdk_with_dimensions(monkeypatch):
    calls = []

    class FakeTextEmbedding:
        @staticmethod
        def call(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={"embeddings": [{"embedding": [1.0, 2.0]}, {"embedding": [3.0, 4.0]}]},
            )

    from imperial_rag.providers import DashScopeTextEmbeddings, QwenProviderSettings

    settings = QwenProviderSettings(api_key="key", embedding_dimensions=2048)
    embeddings = DashScopeTextEmbeddings(settings=settings, client=FakeTextEmbedding)

    assert embeddings.embed_documents(["a", "b"]) == [[1.0, 2.0], [3.0, 4.0]]
    assert calls == [
        {
            "model": "text-embedding-v4",
            "input": ["a", "b"],
            "text_type": "document",
            "dimension": 2048,
            "api_key": "key",
        }
    ]


def test_dashscope_text_embeddings_raise_clean_error_without_secret():
    class FakeTextEmbedding:
        @staticmethod
        def call(**kwargs):
            return SimpleNamespace(status_code=401, code="InvalidApiKey", message="bad key sk-secret")

    from imperial_rag.providers import DashScopeProviderError, DashScopeTextEmbeddings, QwenProviderSettings

    embeddings = DashScopeTextEmbeddings(
        settings=QwenProviderSettings(api_key="sk-secret"),
        client=FakeTextEmbedding,
    )

    with pytest.raises(DashScopeProviderError) as exc:
        embeddings.embed_query("question")

    assert "sk-secret" not in str(exc.value)
    assert "InvalidApiKey" in str(exc.value)


def test_build_qwen_ocr_message_includes_base64_and_options(tmp_path, monkeypatch):
    clear_provider_env(monkeypatch)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-test-key")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_OCR_MIN_PIXELS", "3072")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_OCR_MAX_PIXELS", "8388608")
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_OCR_ENABLE_ROTATE", "false")
    image_path = tmp_path / "scan.jpg"
    image_path.write_bytes(b"fake-image")

    from imperial_rag.providers import build_qwen_ocr_message, QwenProviderSettings

    message = build_qwen_ocr_message(image_path, QwenProviderSettings.from_env())

    content = message["content"][0]
    assert message["role"] == "user"
    assert content["image"].startswith("data:image/jpeg;base64,")
    assert content["min_pixels"] == 3072
    assert content["max_pixels"] == 8388608
    assert content["enable_rotate"] is False


def test_parse_qwen_ocr_response_extracts_text():
    from imperial_rag.providers import parse_qwen_ocr_response

    response = {
        "output": {
            "choices": [
                {
                    "message": SimpleNamespace(
                        content=[
                            {"text": " Распознанный текст "},
                        ]
                    )
                }
            ]
        }
    }

    assert parse_qwen_ocr_response(response) == "Распознанный текст"
