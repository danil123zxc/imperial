from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> None:
    _ensure_src_on_path()
    parser = argparse.ArgumentParser(description="Ingest the Imperial document corpus.")
    parser.add_argument("--workspace-root", type=Path, help="Workspace root containing documents/.")
    parser.add_argument("--enable-ocr", action="store_true", help="Use the configured paid OCR client.")
    parser.add_argument("--index-vectors", action="store_true", help="Index chunks into the configured vector store.")
    parser.add_argument("--trace-phoenix", action="store_true", help="Send this run's traces to configured Phoenix.")
    args = parser.parse_args(argv)

    settings = _build_settings(args.workspace_root)
    _configure_tracing(settings, args.trace_phoenix)
    summary = _run(settings=settings, enable_ocr=args.enable_ocr, index_vectors=args.index_vectors)
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
    from imperial_rag.indexing import make_qdrant_store

    return make_qdrant_store(settings.qdrant_url, settings.qdrant_collection)


def _ocr_appears_configured() -> bool:
    return any(
        os.environ.get(name)
        for name in (
            "OPENAI_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "IMPERIAL_RAG_OCR_API_KEY",
        )
    )


def _build_settings(workspace_root: Path | None) -> Any:
    from imperial_rag.config import Settings

    if workspace_root is None:
        return Settings()
    try:
        return Settings(workspace_root=workspace_root)
    except TypeError:
        os.environ["IMPERIAL_RAG_WORKSPACE_ROOT"] = str(workspace_root)
        return Settings()


def _summary_value(summary: Any, attr: str, default: Any = "") -> Any:
    if isinstance(summary, dict):
        return summary.get(attr, default)
    return getattr(summary, attr, default)


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


if __name__ == "__main__":
    main()
