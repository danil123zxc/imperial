from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any


def main(argv: list[str] | None = None) -> None:
    _ensure_src_on_path()
    parser = argparse.ArgumentParser(description="Ingest the Imperial document corpus.")
    parser.add_argument("--workspace-root", type=Path, help="Workspace root containing documents/.")
    parser.add_argument("--enable-ocr", action="store_true", help="Use the configured paid OCR client.")
    parser.add_argument("--index-vectors", action="store_true", help="Index chunks into the configured vector store.")
    parser.add_argument("--trace-phoenix", action="store_true", help="Send this run's traces to configured Phoenix.")
    parser.add_argument("--trace-session-id", help="Phoenix session.id for grouping traces.")
    args = parser.parse_args(argv)

    _load_project_env(args.workspace_root)
    settings = _build_settings(args.workspace_root)
    _configure_observability(settings)
    _configure_tracing(settings, args.trace_phoenix)
    trace_session_id = _trace_session_id(args.trace_session_id)
    started_at = perf_counter()
    try:
        with _trace_context(trace_session_id):
            summary = _run(settings=settings, enable_ocr=args.enable_ocr, index_vectors=args.index_vectors)
    except (Exception, SystemExit) as exc:
        _log_failure(
            "ingest",
            exc,
            started_at,
            enable_ocr=args.enable_ocr,
            index_vectors=args.index_vectors,
        )
        raise
    _log_ingest_completion(summary, started_at, enable_ocr=args.enable_ocr, index_vectors=args.index_vectors)
    print_summary(summary)


def print_summary(summary: Any) -> None:
    fields = (
        ("scanned_files", "total_files"),
        ("indexed_files", "indexed_files"),
        ("manifest_only_files", "manifest_only_files"),
        ("no_text_files", "no_text_files"),
        ("unsupported_files", "unsupported_files"),
        ("failed_files", "failed_files"),
        ("chunks", "chunk_count"),
        ("keyword_indexed", "keyword_indexed"),
        ("vector_indexed", "vector_indexed"),
    )
    for label, attr in fields:
        print(f"{label}={_summary_value(summary, attr, default=0)}")


def _configure_tracing(settings: Any, trace_phoenix: bool) -> None:
    from imperial_rag.tracing import configure_phoenix_tracing

    configure_phoenix_tracing(settings, enabled=True if trace_phoenix else None)


def _trace_context(session_id: str):
    from imperial_rag.tracing import phoenix_trace_context

    return phoenix_trace_context(session_id)


def _trace_session_id(explicit: str | None) -> str:
    if explicit is not None and explicit.strip():
        return explicit.strip()
    env_value = os.environ.get("IMPERIAL_RAG_TRACE_SESSION_ID", "").strip()
    if env_value:
        return env_value
    return f"cli_{uuid.uuid4()}"


def _configure_observability(settings: Any) -> None:
    from imperial_rag.observability import configure_observability

    configure_observability(settings)


def _log_ingest_completion(summary: Any, started_at: float, *, enable_ocr: bool, index_vectors: bool) -> None:
    from imperial_rag.observability import log_event

    fields = _summary_log_fields(summary)
    failed_files = _int_value(fields.get("failed_files"))
    log_event(
        "imperial_rag.ingest",
        level="error" if failed_files else "info",
        operation="ingest",
        status="failed_files" if failed_files else "success",
        component="cli",
        duration_ms=_duration_ms(started_at),
        enable_ocr=enable_ocr,
        index_vectors=index_vectors,
        **fields,
    )


def _log_failure(operation: str, exc: BaseException, started_at: float, **fields: Any) -> None:
    from imperial_rag.observability import log_failure

    log_failure(operation, exc, component="cli", duration_ms=_duration_ms(started_at), **fields)


def _run(settings: Any, enable_ocr: bool, index_vectors: bool) -> Any:
    workflow = _build_ingestion_workflow()
    if workflow is not None:
        ocr_client = _build_ocr_client(enable_ocr)
        vector_store = _build_vector_store(settings, index_vectors)
        result = workflow.invoke(
            {
                "settings": settings,
                "enable_ocr": enable_ocr,
                "index_vectors": index_vectors,
                "ocr_client": ocr_client,
                "vector_store": vector_store,
            }
        )
        if isinstance(result, dict) and "summary" in result:
            return result["summary"]
        return result

    from imperial_rag.pipeline import run_ingestion

    return run_ingestion(settings=settings, enable_ocr=enable_ocr, index_vectors=index_vectors)


def _build_ingestion_workflow() -> Any | None:
    try:
        from imperial_rag.workflows import build_ingestion_workflow
    except (ImportError, AttributeError):
        return None
    return build_ingestion_workflow()


def _build_ocr_client(enable_ocr: bool) -> Any | None:
    if not enable_ocr or not _ocr_appears_configured():
        return None
    try:
        from imperial_rag.ocr import OcrClient
    except ImportError:
        return None
    return OcrClient()


def _build_vector_store(settings: Any, index_vectors: bool) -> Any | None:
    if not index_vectors:
        return None
    from imperial_rag.providers import dashscope_configured

    if not dashscope_configured():
        print("DASHSCOPE_API_KEY is required when --index-vectors is used.", file=sys.stderr)
        raise SystemExit(2)

    from imperial_rag.indexing import create_qdrant_vector_store

    return create_qdrant_vector_store(settings)


def _ocr_appears_configured() -> bool:
    from imperial_rag.providers import dashscope_configured

    return dashscope_configured()


def _build_settings(workspace_root: Path | None) -> Any:
    from imperial_rag.config import Settings

    if workspace_root is None:
        return Settings()
    try:
        return Settings(workspace_root=workspace_root)
    except TypeError:
        os.environ["IMPERIAL_RAG_WORKSPACE_ROOT"] = str(workspace_root)
        return Settings()


def _load_project_env(workspace_root: Path | None) -> None:
    from imperial_rag.env import load_project_env

    load_project_env(workspace_root)


def _summary_value(summary: Any, attr: str, default: Any = "") -> Any:
    if isinstance(summary, dict):
        return summary.get(attr, default)
    return getattr(summary, attr, default)


def _summary_log_fields(summary: Any) -> dict[str, Any]:
    return {
        "total_files": _summary_value(summary, "total_files", 0),
        "indexed_files": _summary_value(summary, "indexed_files", 0),
        "failed_files": _summary_value(summary, "failed_files", 0),
        "chunk_count": _summary_value(summary, "chunk_count", 0),
        "keyword_indexed": _summary_value(summary, "keyword_indexed", 0),
        "vector_indexed": _summary_value(summary, "vector_indexed", 0),
    }


def _duration_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


if __name__ == "__main__":
    main()
