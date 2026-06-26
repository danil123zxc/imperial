from __future__ import annotations

import inspect
import json
import os
import uuid
import hashlib
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from imperial_rag.ingestion.ledger import write_corpus_ledger
from imperial_rag.observability.phoenix import imperial_trace_attributes, trace_lineage_attributes, trace_pipeline_step


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


def _new_ingest_run_id() -> str:
    explicit = os.environ.get("IMPERIAL_RAG_INGEST_RUN_ID", "").strip()
    if explicit:
        return explicit
    return f"ingest_{uuid.uuid4().hex}"


def _ingest_trace_attributes(step: str, ingest_run_id: str, attributes: dict[str, Any] | None = None) -> dict[str, Any]:
    return imperial_trace_attributes(
        "ingest",
        step,
        {
            "imperial.ingest_run_id": ingest_run_id,
            **dict(attributes or {}),
        },
    )


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
    ingest_run_id = _new_ingest_run_id()
    with trace_pipeline_step(
        "ingest.corpus",
        "corpus",
        attributes=_ingest_trace_attributes(
            "corpus",
            ingest_run_id,
            {
                "ingest.workspace_root": str(getattr(settings, "workspace_root", "")),
                "ingest.index_vectors": index_vectors,
                "ingest.ocr_enabled": ocr_client is not None,
            },
        ),
    ) as corpus_span:
        extraction_root = Path(settings.extraction_root)
        extraction_root.mkdir(parents=True, exist_ok=True)

        with trace_pipeline_step(
            "ingest.scan_files",
            "corpus",
            attributes=_ingest_trace_attributes(
                "scan_files",
                ingest_run_id,
                {"ingest.documents_root": str(settings.documents_root)},
            ),
        ) as scan_span:
            records = deps["assign_duplicate_groups"](deps["scan_files"](Path(settings.documents_root)))
            scan_span.set_output(_scan_trace_output(records))

        with ExitStack() as resources:
            manifest_store = resources.enter_context(deps["ManifestStore"](_manifest_db_path(settings, extraction_root)))
            manifest_store.replace_records(records)
            ocr_cache = _build_ocr_cache(extraction_root) if ocr_client is not None else None
            if ocr_cache is not None:
                ocr_cache = resources.enter_context(ocr_cache)

            extracted_documents: list[Any] = []
            status_by_file: dict[str, Any] = {}
            method_by_file: dict[str, str | None] = {}
            error_by_file: dict[str, str | None] = {}
            source_document_count_by_file: dict[str, int] = {}
            ocr_document_count_by_file: dict[str, int] = {}

            with trace_pipeline_step(
                "ingest.extract_files",
                "corpus",
                attributes=_ingest_trace_attributes(
                    "extract_files",
                    ingest_run_id,
                    {"ingest.total_files": len(records), "ingest.ocr_enabled": ocr_client is not None},
                ),
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
                        source_document_count_by_file[file_id] = 0
                        ocr_document_count_by_file[file_id] = 0
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
                    source_document_count_by_file[file_id] = len(documents)
                    ocr_document_count_by_file[file_id] = _ocr_document_count(documents)
                    if documents:
                        _write_extracted_artifact(extraction_root, record, result)
                        extracted_documents.extend(documents)
                extract_span.set_output(
                    _extraction_trace_output(
                        extracted_documents=extracted_documents,
                        status_by_file=status_by_file,
                        method_by_file=method_by_file,
                        error_by_file=error_by_file,
                        failed_status=deps["FileStatus"].FAILED,
                    )
                )

            retrieval_settings = deps["RetrievalSettings"].from_env()
            with trace_pipeline_step(
                "ingest.build_chunks",
                "corpus",
                attributes=_ingest_trace_attributes(
                    "build_chunks",
                    ingest_run_id,
                    {
                        "ingest.document_count": len(extracted_documents),
                        "ingest.chunk_size": retrieval_settings.chunk_size,
                        "ingest.chunk_overlap": retrieval_settings.chunk_overlap,
                    },
                ),
            ) as chunk_span:
                chunks = list(
                    deps["build_chunks"](
                        extracted_documents,
                        chunk_size=retrieval_settings.chunk_size,
                        chunk_overlap=retrieval_settings.chunk_overlap,
                    )
                )
                baseline_root = Path(getattr(settings, "baseline_extraction_root", None) or extraction_root)
                previous_chunk_rows = _read_existing_chunks(baseline_root / "chunks.jsonl")
                _write_chunks(extraction_root, chunks)
                _write_old_to_new_id_map(extraction_root, previous_chunk_rows, chunks)
                corpus_version = _corpus_version(chunks)
                chunk_span.set_attribute("imperial.corpus_version", corpus_version)
                chunk_span.set_output(
                    _chunk_trace_output(
                        chunks=chunks,
                        document_count=len(extracted_documents),
                        chunk_size=retrieval_settings.chunk_size,
                        chunk_overlap=retrieval_settings.chunk_overlap,
                        corpus_version=corpus_version,
                    )
                )

            with trace_pipeline_step(
                "ingest.keyword_index",
                "corpus",
                attributes=_ingest_trace_attributes(
                    "keyword_index",
                    ingest_run_id,
                    {
                        "imperial.corpus_version": corpus_version,
                        "imperial.keyword_index": getattr(settings, "elasticsearch_index", None),
                        "ingest.chunk_count": len(chunks),
                    },
                ),
            ) as keyword_span:
                keyword_indexed = _replace_keyword_index(deps["KeywordSearchIndex"], settings, chunks)
                keyword_span.set_output(_keyword_index_trace_output(settings, chunks, keyword_indexed))

            vector_indexed = False
            vector_added_ids: list[str] = []
            embedding_model = deps["embedding_model_identifier"]() if vector_store is not None else None
            if vector_store is not None:
                with trace_pipeline_step(
                    "ingest.vector_index",
                    "corpus",
                    attributes=_ingest_trace_attributes(
                        "vector_index",
                        ingest_run_id,
                        {
                            "imperial.corpus_version": corpus_version,
                            "imperial.embedding_model": embedding_model,
                            "imperial.qdrant_collection": getattr(settings, "qdrant_collection", None),
                            "ingest.chunk_count": len(chunks),
                        },
                    ),
                ) as vector_span:
                    with trace_lineage_attributes(
                        {
                            "imperial.ingest_run_id": ingest_run_id,
                            "imperial.corpus_version": corpus_version,
                            "imperial.embedding_model": embedding_model,
                            "imperial.qdrant_collection": getattr(settings, "qdrant_collection", None),
                        }
                    ):
                        vector_indexed, vector_added_ids = _index_with_vector_store(
                            deps["index_vector_documents"], settings, vector_store, chunks
                        )
                    vector_span.set_output(
                        _vector_index_trace_output(settings, chunks, vector_indexed, vector_added_ids, embedding_model)
                    )

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
                    embedding_model=embedding_model if vector_indexed else None,
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
            write_corpus_ledger(
                extraction_root,
                records,
                status_by_file=status_by_file,
                method_by_file=method_by_file,
                error_by_file=error_by_file,
                source_document_count_by_file=source_document_count_by_file,
                ocr_document_count_by_file=ocr_document_count_by_file,
                chunks=chunks,
                keyword_indexed=keyword_indexed,
                vector_indexed=vector_indexed,
                embedding_model=embedding_model if vector_indexed else None,
            )
            index_version = _index_version(
                corpus_version=corpus_version,
                settings=settings,
                keyword_indexed=keyword_indexed,
                vector_indexed=vector_indexed,
                embedding_model=embedding_model if vector_indexed else None,
            )
            lineage = _index_lineage_payload(
                ingest_run_id=ingest_run_id,
                corpus_version=corpus_version,
                index_version=index_version,
                settings=settings,
                keyword_indexed=keyword_indexed,
                vector_indexed=vector_indexed,
                embedding_model=embedding_model if vector_indexed else None,
            )
            _write_index_lineage(extraction_root, lineage)
            corpus_span.set_attribute("imperial.corpus_version", corpus_version)
            corpus_span.set_attribute("imperial.index_version", index_version)
            corpus_span.set_output(
                _summary_trace_output(
                    summary,
                    corpus_version=corpus_version,
                    index_version=index_version,
                    settings=settings,
                    embedding_model=embedding_model if vector_indexed else None,
                )
            )
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


def _ocr_document_count(documents: list[Any]) -> int:
    return sum(1 for document in documents if _has_metadata_value(document, "ocr_method"))


def _has_metadata_value(document: Any, key: str) -> bool:
    metadata = dict(getattr(document, "metadata", {}) or {})
    value = metadata.get(key)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _manifest_db_path(settings: Any, extraction_root: Path) -> Path:
    override = getattr(settings, "manifest_db_path_override", None)
    if override is not None:
        return Path(override)
    if getattr(settings, "extraction_root_override", None) is not None:
        return extraction_root / "manifest.sqlite3"
    return Path(settings.manifest_db_path)


def _scan_trace_output(records: list[Any]) -> dict[str, Any]:
    extensions = [_record_extension(record) for record in records]
    unsupported = sum(1 for extension in extensions if extension in {".doc", ".rar", ".zip", ".7z"})
    duplicate_groups = {
        str(group_id)
        for record in records
        if (group_id := getattr(record, "duplicate_group_id", None)) not in (None, "")
        and int(getattr(record, "duplicate_group_size", 0) or 0) > 1
    }
    return {
        "total_files": len(records),
        "supported_files": len(records) - unsupported,
        "unsupported_files": unsupported,
        "extension_counts": dict(sorted(Counter(extensions).items())),
        "duplicate_group_count": len(duplicate_groups),
    }


def _summary_trace_output(
    summary: IngestionSummary,
    *,
    corpus_version: str | None = None,
    index_version: str | None = None,
    settings: Any | None = None,
    embedding_model: str | None = None,
) -> dict[str, Any]:
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
        **({} if corpus_version is None else {"corpus_version": corpus_version}),
        **({} if index_version is None else {"index_version": index_version}),
        **({} if settings is None else {"keyword_index": getattr(settings, "elasticsearch_index", None)}),
        **(
            {}
            if settings is None or not summary.vector_indexed
            else {"qdrant_collection": getattr(settings, "qdrant_collection", None)}
        ),
        **({} if embedding_model is None else {"embedding_model": embedding_model}),
    }


def _extraction_trace_output(
    *,
    extracted_documents: list[Any],
    status_by_file: dict[str, Any],
    method_by_file: dict[str, str | None],
    error_by_file: dict[str, str | None],
    failed_status: Any,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "document_count": len(extracted_documents),
        "status_counts": dict(sorted(Counter(_status_value(status) for status in status_by_file.values()).items())),
        "extraction_methods": dict(
            sorted(Counter(method for method in method_by_file.values() if method).items())
        ),
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


def _chunk_trace_output(
    *,
    chunks: list[Any],
    document_count: int,
    chunk_size: int,
    chunk_overlap: int,
    corpus_version: str,
) -> dict[str, Any]:
    chunk_hashes = [_chunk_content_hash(chunk) for chunk in chunks]
    return {
        "document_count": document_count,
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "corpus_version": corpus_version,
        "chunk_hashes": {
            "count": len(chunk_hashes),
            "top": chunk_hashes[:10],
        },
    }


def _keyword_index_trace_output(settings: Any, chunks: list[Any], indexed: bool) -> dict[str, Any]:
    return {
        "chunk_count": len(chunks),
        "indexed": indexed,
        "elasticsearch_index": getattr(settings, "elasticsearch_index", None),
        "indexed_count": len(chunks) if indexed else 0,
        "replace_all_success": indexed,
    }


def _vector_index_trace_output(
    settings: Any,
    chunks: list[Any],
    indexed: bool,
    added_ids: list[str],
    embedding_model: str | None,
) -> dict[str, Any]:
    return {
        "chunk_count": len(chunks),
        "indexed": indexed,
        "qdrant_collection": getattr(settings, "qdrant_collection", None),
        "added_id_count": len(added_ids),
        "embedding_model": embedding_model,
        "embedding_dimensions": _embedding_dimensions_from_identifier(embedding_model),
    }


def _corpus_version(chunks: list[Any]) -> str:
    payload = "\n".join(_chunk_signature(chunk) for chunk in chunks)
    return f"corpus_sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _index_version(
    *,
    corpus_version: str,
    settings: Any,
    keyword_indexed: bool,
    vector_indexed: bool,
    embedding_model: str | None,
) -> str:
    payload = {
        "corpus_version": corpus_version,
        "keyword_index": getattr(settings, "elasticsearch_index", None),
        "keyword_indexed": keyword_indexed,
        "qdrant_collection": getattr(settings, "qdrant_collection", None) if vector_indexed else None,
        "vector_indexed": vector_indexed,
        "embedding_model": embedding_model,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return f"index_sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def _index_lineage_payload(
    *,
    ingest_run_id: str,
    corpus_version: str,
    index_version: str,
    settings: Any,
    keyword_indexed: bool,
    vector_indexed: bool,
    embedding_model: str | None,
) -> dict[str, Any]:
    return {
        "ingest_run_id": ingest_run_id,
        "corpus_version": corpus_version,
        "index_version": index_version,
        "keyword_index": getattr(settings, "elasticsearch_index", None),
        "qdrant_collection": getattr(settings, "qdrant_collection", None) if vector_indexed else None,
        "embedding_model": embedding_model,
        "keyword_indexed": keyword_indexed,
        "vector_indexed": vector_indexed,
    }


def _write_index_lineage(extraction_root: Path, payload: dict[str, Any]) -> None:
    target = _safe_artifact_path(extraction_root, "index-lineage.json")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _chunk_signature(chunk: Any) -> str:
    payload = {
        "page_content": str(getattr(chunk, "page_content", "")),
        "metadata": dict(getattr(chunk, "metadata", {}) or {}),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _chunk_content_hash(chunk: Any) -> str:
    content = str(getattr(chunk, "page_content", ""))
    return f"content_sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"


def _record_extension(record: Any) -> str:
    extension = str(getattr(record, "extension", "") or "").casefold()
    if extension:
        return extension
    path = getattr(record, "relative_path", None) or getattr(record, "absolute_path", None) or getattr(record, "filename", "")
    return Path(str(path)).suffix.casefold()


def _embedding_dimensions_from_identifier(identifier: str | None) -> int | None:
    if not identifier or ":" not in identifier:
        return None
    raw = identifier.rsplit(":", 1)[-1]
    try:
        return int(raw)
    except ValueError:
        return None


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

    if getattr(settings, "recreate_qdrant_collection", False):
        from imperial_rag.indexing import reset_qdrant_collection

        reset_qdrant_collection(settings)
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


def _read_existing_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_old_to_new_id_map(extraction_root: Path, old_rows: list[dict[str, Any]], chunks: list[Any]) -> None:
    new_by_file: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        metadata = dict(getattr(chunk, "metadata", {}) or {})
        file_id = metadata.get("file_id")
        if file_id is not None:
            new_by_file.setdefault(str(file_id), []).append(metadata)

    rows: list[dict[str, Any]] = []
    for row in old_rows:
        old = dict(row.get("metadata") or {})
        file_id = str(old.get("file_id") or "")
        new_candidates = new_by_file.get(file_id, [])
        new = new_candidates.pop(0) if new_candidates else {}
        rows.append(
            {
                "file_id": file_id,
                "old_chunk_id": old.get("chunk_id"),
                "old_citation_id": old.get("citation_id"),
                "new_chunk_id": new.get("chunk_id"),
                "new_citation_id": new.get("citation_id"),
                "source_locator": new.get("source_locator"),
                "status": "mapped" if new.get("chunk_id") else "unmapped",
            }
        )

    for file_id, remaining in sorted(new_by_file.items()):
        for new in remaining:
            rows.append(
                {
                    "file_id": file_id,
                    "old_chunk_id": None,
                    "old_citation_id": None,
                    "new_chunk_id": new.get("chunk_id"),
                    "new_citation_id": new.get("citation_id"),
                    "source_locator": new.get("source_locator"),
                    "status": "new_only",
                }
            )

    payload = {"schema_version": "old-to-new-id-map-v1", "rows": rows}
    target = _safe_artifact_path(extraction_root, "old-to-new-id-map.json")
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


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


def _index_with_vector_store(index_vector_documents: Any, settings: Any, vector_store: Any, chunks: list[Any]) -> tuple[bool, list[str]]:
    if not chunks:
        return True, []
    added_ids = index_vector_documents(chunks, settings=settings, vector_store=vector_store)
    return True, [str(added_id) for added_id in (added_ids or [])]


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
