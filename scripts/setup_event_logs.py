from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


EVENT_POLICY_NAME = "imperial-rag-events-delete-30d"
EVAL_POLICY_NAME = "imperial-rag-eval-summaries-delete-90d"
EVENT_TEMPLATE_NAME = "imperial-rag-events-template-v1"
EVAL_TEMPLATE_NAME = "imperial-rag-eval-summaries-template-v1"


def main(argv: list[str] | None = None) -> None:
    _ensure_src_on_path()
    parser = argparse.ArgumentParser(description="Set up local Elasticsearch data streams for Imperial event logs.")
    parser.add_argument("--workspace-root", type=Path)
    args = parser.parse_args(argv)

    from imperial_rag.cli import build_settings, load_project_environment
    from imperial_rag.observability.eventlog import DEFAULT_EVAL_DATA_STREAM, DEFAULT_EVENT_DATA_STREAM

    load_project_environment(args.workspace_root)
    settings = build_settings(args.workspace_root)
    event_stream = os.environ.get("IMPERIAL_RAG_EVENTLOG_ELASTICSEARCH_DATA_STREAM", DEFAULT_EVENT_DATA_STREAM).strip()
    eval_stream = os.environ.get("IMPERIAL_RAG_EVENTLOG_EVAL_DATA_STREAM", DEFAULT_EVAL_DATA_STREAM).strip()

    try:
        from elasticsearch import Elasticsearch
    except ImportError as exc:
        raise SystemExit("Elasticsearch Python client is not installed.") from exc

    client = Elasticsearch(settings.elasticsearch_url)
    setup_event_log_streams(client, event_stream=event_stream, eval_stream=eval_stream)
    print(f"event_data_stream={event_stream}")
    print(f"eval_data_stream={eval_stream}")


def setup_event_log_streams(client: Any, *, event_stream: str, eval_stream: str) -> None:
    _put_delete_policy(client, EVENT_POLICY_NAME, days=30)
    _put_delete_policy(client, EVAL_POLICY_NAME, days=90)
    _put_template(client, EVENT_TEMPLATE_NAME, stream_name=event_stream, policy_name=EVENT_POLICY_NAME)
    _put_template(client, EVAL_TEMPLATE_NAME, stream_name=eval_stream, policy_name=EVAL_POLICY_NAME)
    _ensure_data_stream(client, event_stream)
    _ensure_data_stream(client, eval_stream)


def _put_delete_policy(client: Any, name: str, *, days: int) -> None:
    client.ilm.put_lifecycle(
        name=name,
        policy={
            "phases": {
                "delete": {
                    "min_age": f"{days}d",
                    "actions": {"delete": {}},
                }
            }
        },
    )


def _put_template(client: Any, name: str, *, stream_name: str, policy_name: str) -> None:
    client.indices.put_index_template(
        name=name,
        index_patterns=[stream_name],
        data_stream={},
        priority=500,
        template={
            "settings": {
                "index.lifecycle.name": policy_name,
                "number_of_replicas": 0,
            },
            "mappings": {
                "dynamic": "strict",
                "properties": _event_mappings(),
            },
        },
    )


def _ensure_data_stream(client: Any, stream_name: str) -> None:
    exists = False
    try:
        result = client.indices.exists_data_stream(name=stream_name)
        exists = bool(result)
    except Exception:
        exists = False
    if exists:
        return
    try:
        client.indices.create_data_stream(name=stream_name)
    except Exception as exc:
        if "resource_already_exists" not in str(exc):
            raise


def _event_mappings() -> dict[str, dict[str, Any]]:
    keyword_fields = (
        "app_version",
        "component",
        "dependency",
        "dependency_status",
        "environment",
        "error_code",
        "error_type",
        "event",
        "git_sha",
        "image_digest",
        "image_tag",
        "keyword_search_status",
        "level",
        "model_error_type",
        "operation",
        "phoenix_session_id",
        "phoenix_trace_id",
        "ragas_metrics",
        "request_id",
        "reranker",
        "schema_version",
        "service",
        "session_id",
        "status",
        "user_hash",
        "vector_search_status",
    )
    long_fields = (
        "chunk_count",
        "duration_ms",
        "example_count",
        "failed_files",
        "fallback_count",
        "final_evidence",
        "indexed_files",
        "keyword_candidates",
        "keyword_indexed",
        "manifest_only_files",
        "merged_candidates",
        "no_text_files",
        "passed_count",
        "rerank_input_candidates",
        "reranked_candidates",
        "skipped_count",
        "total_files",
        "unsupported_files",
        "vector_candidates",
        "vector_indexed",
    )
    boolean_fields = (
        "citations_valid",
        "enable_ocr",
        "eventlog_elasticsearch_enabled",
        "index_vectors",
        "phoenix_mode",
        "trace_enabled",
        "wrote_output",
    )
    mappings: dict[str, dict[str, Any]] = {"@timestamp": {"type": "date"}}
    mappings.update({field: {"type": "keyword"} for field in keyword_fields})
    mappings.update({field: {"type": "long"} for field in long_fields})
    mappings.update({field: {"type": "boolean"} for field in boolean_fields})
    return mappings


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


if __name__ == "__main__":
    main()
