from __future__ import annotations

import inspect
import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from imperial_rag.tracing import trace_pipeline_step


@dataclass(frozen=True)
class IngestionSummary:
    total_files: int
    indexed_files: int
    manifest_only_files: int
    no_text_files: int
    unsupported_files: int
    failed_files: int
    chunk_count: int
    keyword_indexed: bool
    vector_indexed: bool
    extraction_root: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_ingestion(
    settings: Any | None = None,
    enable_ocr: bool = False,
    index_vectors: bool = False,
) -> IngestionSummary:
    """Run the concrete local ingestion pipeline."""

    deps = _load_dependencies()
    resolved_settings = settings or deps["Settings"]()
    return _run(
        deps=deps,
        settings=resolved_settings,
        ocr_client=_build_ocr_client(enable_ocr),
        vector_store=_build_vector_store(resolved_settings, index_vectors),
        index_vectors=index_vectors,
    )


def ingest_corpus(
    settings: Any | None = None,
    ocr_client: Any | None = None,
    vector_store: Any | None = None,
) -> IngestionSummary:
    """Compatibility wrapper for the plan's earlier pipeline name."""

    deps = _load_dependencies()
    resolved_settings = settings or deps["Settings"]()
    return _run(
        deps=deps,
        settings=resolved_settings,
        ocr_client=ocr_client,
        vector_store=vector_store,
        index_vectors=vector_store is not None,
    )


def _run(
    deps: dict[str, Any],
    settings: Any,
    ocr_client: Any | None,
    vector_store: Any | None,
    index_vectors: bool,
) -> IngestionSummary:
    with trace_pipeline_step(
        "ingest.corpus",
        "corpus",
        attributes={
            "ingest.workspace_root": str(getattr(settings, "workspace_root", "")),
            "ingest.index_vectors": index_vectors,
            "ingest.ocr_enabled": ocr_client is not None,
        },
    ) as corpus_span:
        extraction_root = Path(settings.extraction_root)
        extraction_root.mkdir(parents=True, exist_ok=True)

        with trace_pipeline_step(
            "ingest.scan_files",
            "corpus",
            attributes={"ingest.documents_root": str(settings.documents_root)},
        ) as scan_span:
            records = deps["assign_duplicate_groups"](deps["scan_files"](Path(settings.documents_root)))
            scan_span.set_output({"total_files": len(records)})

        with ExitStack() as resources:
            manifest_store = resources.enter_context(deps["ManifestStore"](Path(settings.manifest_db_path)))
            manifest_store.replace_records(records)
            ocr_cache = _build_ocr_cache(extraction_root) if ocr_client is not None else None
            if ocr_cache is not None:
                ocr_cache = resources.enter_context(ocr_cache)

            extracted_documents: list[Any] = []
            status_by_file: dict[str, Any] = {}
            method_by_file: dict[str, str | None] = {}
            error_by_file: dict[str, str | None] = {}

            with trace_pipeline_step(
                "ingest.extract_files",
                "corpus",
                attributes={"ingest.total_files": len(records), "ingest.ocr_enabled": ocr_client is not None},
            ) as extract_span:
                for record in records:
                    file_id = str(record.file_id)
                    try:
                        result = _extract_record(
                            extract_file=deps["extract_file"],
                            record=record,
                            ocr_client=ocr_client,
                            ocr_cache=ocr_cache,
                            artifact_root=extraction_root / "artifacts",
                        )
                    except Exception as exc:  # pragma: no cover - integration protection
                        status_by_file[file_id] = deps["FileStatus"].FAILED
                        method_by_file[file_id] = _planned_extraction_method(record)
                        error_by_file[file_id] = str(exc)
                        manifest_store.update_status(
                            file_id=file_id,
                            status=deps["FileStatus"].FAILED,
                            extraction_method=method_by_file[file_id],
                            error_message=str(exc),
                            chunk_count=0,
                        )
                        continue

                    documents = list(getattr(result, "documents", []) or [])
                    status_by_file[file_id] = result.status
                    method_by_file[file_id] = getattr(result, "extraction_method", None)
                    error_by_file[file_id] = getattr(result, "message", None) or None
                    if documents:
                        _write_extracted_artifact(extraction_root, record, result)
                        extracted_documents.extend(documents)
                extract_span.set_output(
                    _extraction_trace_output(
                        extracted_documents=extracted_documents,
                        status_by_file=status_by_file,
                        error_by_file=error_by_file,
                        failed_status=deps["FileStatus"].FAILED,
                    )
                )

            retrieval_settings = deps["RetrievalSettings"].from_env()
            with trace_pipeline_step(
                "ingest.build_chunks",
                "corpus",
                attributes={
                    "ingest.document_count": len(extracted_documents),
                    "ingest.chunk_size": retrieval_settings.chunk_size,
                    "ingest.chunk_overlap": retrieval_settings.chunk_overlap,
                },
            ) as chunk_span:
                chunks = list(
                    deps["build_chunks"](
                        extracted_documents,
                        chunk_size=retrieval_settings.chunk_size,
                        chunk_overlap=retrieval_settings.chunk_overlap,
                    )
                )
                _write_chunks(extraction_root, chunks)
                chunk_span.set_output(
                    {
                        "document_count": len(extracted_documents),
                        "chunk_count": len(chunks),
                        "chunk_size": retrieval_settings.chunk_size,
                        "chunk_overlap": retrieval_settings.chunk_overlap,
                    }
                )

            with trace_pipeline_step(
                "ingest.keyword_index",
                "corpus",
                attributes={"ingest.chunk_count": len(chunks)},
            ) as keyword_span:
                keyword_indexed = _replace_keyword_index(deps["KeywordSearchIndex"], settings, chunks)
                keyword_span.set_output({"chunk_count": len(chunks), "indexed": keyword_indexed})

            vector_indexed = False
            if vector_store is not None:
                with trace_pipeline_step(
                    "ingest.vector_index",
                    "corpus",
                    attributes={"ingest.chunk_count": len(chunks)},
                ) as vector_span:
                    vector_indexed = _index_with_vector_store(
                        deps["index_vector_documents"], settings, vector_store, chunks
                    )
                    vector_span.set_output({"chunk_count": len(chunks), "indexed": vector_indexed})

            chunk_count_by_file = _count_chunks_by_file(chunks)

            for record in records:
                file_id = str(record.file_id)
                status = status_by_file.get(file_id, deps["FileStatus"].PENDING)
                chunk_count = chunk_count_by_file.get(file_id, 0)
                manifest_store.update_status(
                    file_id=file_id,
                    status=status,
                    extraction_method=method_by_file.get(file_id),
                    error_message=error_by_file.get(file_id),
                    chunk_count=chunk_count,
                )
                _update_index_status(
                    deps=deps,
                    manifest_store=manifest_store,
                    file_id=file_id,
                    status=status,
                    chunk_count=chunk_count,
                    keyword_indexed=keyword_indexed,
                    index_vectors=index_vectors,
                    vector_indexed=vector_indexed,
                    embedding_model=deps["embedding_model_identifier"]() if vector_indexed else None,
                )

            status_counts = Counter(_status_value(status) for status in status_by_file.values())
            summary = IngestionSummary(
                total_files=len(records),
                indexed_files=status_counts[_status_value(deps["FileStatus"].INDEXED)],
                manifest_only_files=status_counts[_status_value(deps["FileStatus"].MANIFEST_ONLY)],
                no_text_files=status_counts[_status_value(deps["FileStatus"].NO_TEXT)],
                unsupported_files=status_counts[_status_value(deps["FileStatus"].UNSUPPORTED)],
                failed_files=status_counts[_status_value(deps["FileStatus"].FAILED)],
                chunk_count=len(chunks),
                keyword_indexed=keyword_indexed,
                vector_indexed=vector_indexed,
                extraction_root=str(extraction_root),
            )
            corpus_span.set_output(_summary_trace_output(summary))
            return summary


def _load_dependencies() -> dict[str, Any]:
    from imperial_rag.chunking import build_chunks
    from imperial_rag.config import Settings
    from imperial_rag.elasticsearch_keyword import ElasticsearchKeywordIndex
    from imperial_rag.extraction import extract_file
    from imperial_rag.indexing import (
        create_qdrant_vector_store,
        embedding_model_identifier,
        index_vector_documents,
    )
    from imperial_rag.manifest import (
        FileStatus,
        IndexStatus,
        ManifestStore,
        assign_duplicate_groups,
        scan_files,
    )
    from imperial_rag.retrieval import RetrievalSettings

    return {
        "Settings": Settings,
        "RetrievalSettings": RetrievalSettings,
        "build_chunks": build_chunks,
        "extract_file": extract_file,
        "KeywordSearchIndex": ElasticsearchKeywordIndex,
        "create_qdrant_vector_store": create_qdrant_vector_store,
        "embedding_model_identifier": embedding_model_identifier,
        "index_vector_documents": index_vector_documents,
        "FileStatus": FileStatus,
        "IndexStatus": IndexStatus,
        "ManifestStore": ManifestStore,
        "assign_duplicate_groups": assign_duplicate_groups,
        "scan_files": scan_files,
    }


def _summary_trace_output(summary: IngestionSummary) -> dict[str, Any]:
    return {
        "total_files": summary.total_files,
        "indexed_files": summary.indexed_files,
        "manifest_only_files": summary.manifest_only_files,
        "no_text_files": summary.no_text_files,
        "unsupported_files": summary.unsupported_files,
        "failed_files": summary.failed_files,
        "chunk_count": summary.chunk_count,
        "keyword_indexed": summary.keyword_indexed,
        "vector_indexed": summary.vector_indexed,
    }


def _extraction_trace_output(
    *,
    extracted_documents: list[Any],
    status_by_file: dict[str, Any],
    error_by_file: dict[str, str | None],
    failed_status: Any,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "document_count": len(extracted_documents),
        "status_counts": dict(sorted(Counter(_status_value(status) for status in status_by_file.values()).items())),
    }
    failed_status_value = _status_value(failed_status)
    failed_files = [
        {"file_id": file_id, "message": str(error_by_file.get(file_id) or "")}
        for file_id, status in status_by_file.items()
        if _status_value(status) == failed_status_value
    ][:10]
    output["failed_file_count"] = sum(
        1 for status in status_by_file.values() if _status_value(status) == failed_status_value
    )
    if failed_files:
        output["failed_files"] = failed_files
    return output


def _build_ocr_client(enable_ocr: bool) -> Any | None:
    if not enable_ocr or not _ocr_appears_configured():
        return None
    from imperial_rag.ocr import OcrClient

    return OcrClient()


def _build_ocr_cache(extraction_root: Path) -> Any | None:
    try:
        from imperial_rag.ocr import OcrCache
    except ImportError:
        return None
    return OcrCache(extraction_root / "ocr-cache")


def _build_vector_store(settings: Any, index_vectors: bool) -> Any | None:
    if not index_vectors:
        return None
    from imperial_rag.indexing import create_qdrant_vector_store

    return create_qdrant_vector_store(settings)


def _ocr_appears_configured() -> bool:
    from imperial_rag.providers import dashscope_configured

    return dashscope_configured()


def _extract_record(
    extract_file: Any,
    record: Any,
    ocr_client: Any | None,
    ocr_cache: Any | None,
    artifact_root: Path,
) -> Any:
    signature = inspect.signature(extract_file)
    kwargs: dict[str, Any] = {}
    if "ocr_client" in signature.parameters:
        kwargs["ocr_client"] = ocr_client
    if "ocr_cache" in signature.parameters:
        kwargs["ocr_cache"] = ocr_cache
    if "artifact_root" in signature.parameters:
        kwargs["artifact_root"] = artifact_root
    return extract_file(record, **kwargs)


def _write_extracted_artifact(extraction_root: Path, record: Any, result: Any) -> None:
    target = _safe_artifact_path(extraction_root / "documents", f"{record.file_id}.json")
    payload = {
        "file_id": str(record.file_id),
        "relative_path": Path(record.relative_path).as_posix(),
        "status": _status_value(result.status),
        "extraction_method": getattr(result, "extraction_method", None),
        "documents": [
            {
                "page_content": str(document.page_content),
                "metadata": dict(document.metadata),
            }
            for document in getattr(result, "documents", [])
        ],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_chunks(extraction_root: Path, chunks: list[Any]) -> None:
    chunks_path = _safe_artifact_path(extraction_root, "chunks.jsonl")
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(
                json.dumps(
                    {
                        "page_content": str(chunk.page_content),
                        "metadata": dict(chunk.metadata),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _safe_artifact_path(root: Path, filename: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = (root / filename).resolve()
    resolved_root = root.resolve()
    if path.parent != resolved_root:
        raise ValueError(f"unsafe artifact path: {filename}")
    return path


def _count_chunks_by_file(chunks: list[Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for chunk in chunks:
        file_id = chunk.metadata.get("file_id")
        if file_id is not None:
            counts[str(file_id)] += 1
    return counts


def _replace_keyword_index(keyword_index_cls: Any, settings: Any, chunks: list[Any]) -> bool:
    keyword_index = keyword_index_cls(settings)
    keyword_index.replace_all(chunks)
    return True


def _index_with_vector_store(index_vector_documents: Any, settings: Any, vector_store: Any, chunks: list[Any]) -> bool:
    if not chunks:
        return True
    index_vector_documents(chunks, settings=settings, vector_store=vector_store)
    return True


def _update_index_status(
    deps: dict[str, Any],
    manifest_store: Any,
    file_id: str,
    status: Any,
    chunk_count: int,
    keyword_indexed: bool,
    index_vectors: bool,
    vector_indexed: bool,
    embedding_model: str | None,
) -> None:
    index_status = deps["IndexStatus"]
    file_status = deps["FileStatus"]
    if status == file_status.INDEXED and chunk_count > 0:
        keyword_status = index_status.INDEXED if keyword_indexed else index_status.FAILED
        if index_vectors:
            vector_status = index_status.INDEXED if vector_indexed else index_status.FAILED
        else:
            vector_status = index_status.SKIPPED
    else:
        keyword_status = index_status.SKIPPED
        vector_status = index_status.SKIPPED
    indexed_embedding_model = embedding_model if vector_status == index_status.INDEXED else None
    manifest_store.update_index_status(
        file_id=file_id,
        keyword_index_status=keyword_status,
        vector_index_status=vector_status,
        embedding_model=indexed_embedding_model,
    )


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status))


def _planned_extraction_method(record: Any) -> str | None:
    extension = str(getattr(record, "extension", "")).casefold()
    return {
        ".docx": "python_docx",
        ".pdf": "pymupdf",
        ".jpg": "image_ocr",
        ".jpeg": "image_ocr",
        ".png": "image_ocr",
        ".tif": "image_ocr",
        ".tiff": "image_ocr",
        ".xlsx": "openpyxl",
        ".rtf": "striprtf",
        ".doc": "legacy_doc_unsupported",
        ".rar": None,
        ".zip": None,
        ".7z": None,
    }.get(extension)
