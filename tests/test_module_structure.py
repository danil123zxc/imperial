from __future__ import annotations


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
    assert DEFAULT_RAGAS_METRICS
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


def test_old_import_paths_remain_compatible_with_lifecycle_modules():
    import imperial_rag.auth as old_auth
    import imperial_rag.chunking as old_chunking
    import imperial_rag.elasticsearch_keyword as old_elasticsearch_keyword
    import imperial_rag.extraction as old_extraction
    import imperial_rag.keyword as old_keyword
    import imperial_rag.manifest as old_manifest
    import imperial_rag.ocr as old_ocr
    import imperial_rag.pipeline as old_pipeline
    import imperial_rag.providers as old_providers
    import imperial_rag.ragas_eval as old_ragas_eval
    import imperial_rag.runtime as old_runtime
    import imperial_rag.tracing as old_tracing
    import imperial_rag.web_app as old_web_app
    import imperial_rag.workflows as old_workflows
    from imperial_rag.answering import build_strict_messages as old_build_strict_messages
    from imperial_rag.indexing import create_qdrant_vector_store as old_create_qdrant_vector_store
    from imperial_rag.indexing import stable_chunk_id as old_stable_chunk_id
    from imperial_rag.observability import configure_observability as old_configure_observability
    from imperial_rag.retrieval import RetrievalService as old_retrieval_service

    from imperial_rag.answering.strict import build_strict_messages
    from imperial_rag.answering.runtime import Runtime, create_runtime
    from imperial_rag.answering.workflow import build_ingestion_workflow, build_query_workflow
    from imperial_rag.app.auth import AuthStore
    from imperial_rag.app.web import APP_TITLE
    from imperial_rag.evals.ragas import parse_ragas_metric_names
    from imperial_rag.indexing.vector import create_qdrant_vector_store, stable_chunk_id
    from imperial_rag.ingestion.chunking import build_chunks
    from imperial_rag.ingestion.extraction import extract_file
    from imperial_rag.ingestion.manifest import ManifestStore
    from imperial_rag.ingestion.ocr import OcrCache
    from imperial_rag.ingestion.pipeline import run_ingestion
    from imperial_rag.integrations.dashscope import QwenProviderSettings
    from imperial_rag.observability.logging import configure_observability
    from imperial_rag.observability.phoenix import configure_phoenix_tracing
    from imperial_rag.retrieval.elasticsearch import ElasticsearchKeywordIndex
    from imperial_rag.retrieval.lexical import searchable_document_text
    from imperial_rag.retrieval.service import RetrievalService

    assert old_auth.AuthStore is AuthStore
    assert old_build_strict_messages is build_strict_messages
    assert old_chunking.build_chunks is build_chunks
    assert old_configure_observability is configure_observability
    assert old_create_qdrant_vector_store is create_qdrant_vector_store
    assert old_elasticsearch_keyword.ElasticsearchKeywordIndex is ElasticsearchKeywordIndex
    assert old_extraction.extract_file is extract_file
    assert old_keyword.searchable_document_text is searchable_document_text
    assert old_manifest.ManifestStore is ManifestStore
    assert old_ocr.OcrCache is OcrCache
    assert old_pipeline.run_ingestion is run_ingestion
    assert old_providers.QwenProviderSettings is QwenProviderSettings
    assert old_ragas_eval.parse_ragas_metric_names is parse_ragas_metric_names
    assert old_retrieval_service is RetrievalService
    assert old_runtime.Runtime is Runtime
    assert old_runtime.create_runtime is create_runtime
    assert old_stable_chunk_id is stable_chunk_id
    assert old_tracing.configure_phoenix_tracing is configure_phoenix_tracing
    assert old_web_app.APP_TITLE == APP_TITLE
    assert old_workflows.build_ingestion_workflow is build_ingestion_workflow
    assert old_workflows.build_query_workflow is build_query_workflow
