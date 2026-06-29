from __future__ import annotations

import importlib

import pytest

# Legacy top-level redirect packages that were removed in favour of importing
# the canonical domain modules directly.
REMOVED_SHIM_MODULES = (
    "imperial_rag.auth",
    "imperial_rag.chunking",
    "imperial_rag.extraction",
    "imperial_rag.manifest",
    "imperial_rag.ocr",
    "imperial_rag.pipeline",
    "imperial_rag.keyword",
    "imperial_rag.elasticsearch_keyword",
    "imperial_rag.runtime",
    "imperial_rag.workflows",
    "imperial_rag.providers",
    "imperial_rag.tracing",
    "imperial_rag.ragas_eval",
    "imperial_rag.web_app",
)


@pytest.mark.parametrize("module_name", REMOVED_SHIM_MODULES)
def test_legacy_shim_modules_are_gone(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_flattened_single_file_modules_import():
    for module_name in (
        "imperial_rag.cli",
        "imperial_rag.config",
        "imperial_rag.document_ids",
        "imperial_rag.env",
    ):
        assert importlib.import_module(module_name) is not None


def test_lifecycle_import_paths_are_available():
    from imperial_rag.answering.runtime import Runtime, create_runtime
    from imperial_rag.answering.strict import REFUSAL_TEXT, build_strict_messages
    from imperial_rag.answering.workflow import build_query_workflow
    from imperial_rag.app.auth import AuthStore, AuthenticationStatus
    from imperial_rag.app.web import APP_TITLE, build_status_summary
    from imperial_rag.evals.ragas import DEFAULT_RAGAS_METRICS, parse_ragas_metric_names
    from imperial_rag.indexing.health import qdrant_health, qdrant_is_healthy
    from imperial_rag.indexing.vector import create_qdrant_vector_store, stable_chunk_id
    from imperial_rag.ingestion.chunking import build_chunks
    from imperial_rag.ingestion.extraction import ExtractionResult, extract_file
    from imperial_rag.ingestion.manifest import FileRecord, FileStatus, ManifestStore
    from imperial_rag.ingestion.ocr import OcrCache, OcrResult
    from imperial_rag.ingestion.pipeline import IngestionSummary, run_ingestion
    from imperial_rag.ingestion.workflow import build_ingestion_workflow
    from imperial_rag.integrations.dashscope import QwenProviderSettings, create_chat_model
    from imperial_rag.observability.logging import configure_observability, log_event
    from imperial_rag.observability.phoenix import configure_phoenix_tracing, trace_pipeline_step
    from imperial_rag.retrieval.elasticsearch import ElasticsearchKeywordIndex
    from imperial_rag.retrieval.lexical import build_elasticsearch_token_query, searchable_document_text
    from imperial_rag.retrieval.service import RetrievalService, RetrievalSettings

    assert APP_TITLE == "Imperial RAG"
    assert REFUSAL_TEXT
    assert parse_ragas_metric_names("") == list(DEFAULT_RAGAS_METRICS)
    assert AuthenticationStatus.AUTHENTICATED.value == "authenticated"
    assert FileStatus.INDEXED.value == "indexed"
    assert AuthStore.__name__ == "AuthStore"
    assert ElasticsearchKeywordIndex.__name__ == "ElasticsearchKeywordIndex"
    assert ExtractionResult.__name__ == "ExtractionResult"
    assert FileRecord.__name__ == "FileRecord"
    assert IngestionSummary.__name__ == "IngestionSummary"
    assert ManifestStore.__name__ == "ManifestStore"
    assert OcrCache.__name__ == "OcrCache"
    assert OcrResult.__name__ == "OcrResult"
    assert QwenProviderSettings.__name__ == "QwenProviderSettings"
    assert RetrievalService.__name__ == "RetrievalService"
    assert RetrievalSettings.__name__ == "RetrievalSettings"
    assert Runtime.__name__ == "Runtime"
    assert callable(build_chunks)
    assert callable(build_elasticsearch_token_query)
    assert callable(build_ingestion_workflow)
    assert callable(build_query_workflow)
    assert callable(build_status_summary)
    assert callable(build_strict_messages)
    assert callable(configure_observability)
    assert callable(configure_phoenix_tracing)
    assert callable(create_chat_model)
    assert callable(create_qdrant_vector_store)
    assert callable(create_runtime)
    assert callable(extract_file)
    assert callable(log_event)
    assert callable(parse_ragas_metric_names)
    assert callable(qdrant_health)
    assert callable(qdrant_is_healthy)
    assert callable(run_ingestion)
    assert callable(searchable_document_text)
    assert callable(stable_chunk_id)
    assert callable(trace_pipeline_step)
