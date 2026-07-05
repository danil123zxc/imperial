from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from imperial_rag.jsonl import write_jsonl as _write_jsonl


LEDGER_SCHEMA_VERSION = "corpus-ledger-v1"
LEDGER_SUMMARY_SCHEMA_VERSION = "corpus-ledger-summary-v1"
REQUIRED_CHUNK_METADATA = ("chunk_id", "citation_id", "source_locator", "body_start_index")


def write_corpus_ledger(
    extraction_root: Path,
    records: Iterable[Any],
    *,
    status_by_file: Mapping[str, Any],
    method_by_file: Mapping[str, str | None],
    error_by_file: Mapping[str, str | None],
    source_document_count_by_file: Mapping[str, int],
    ocr_document_count_by_file: Mapping[str, int],
    chunks: Iterable[Any],
    keyword_indexed: bool,
    vector_indexed: bool,
    embedding_model: str | None,
) -> dict[str, Any]:
    chunk_stats = _chunk_stats_by_file(chunks)
    rows = [
        _ledger_row(
            record,
            status_by_file=status_by_file,
            method_by_file=method_by_file,
            error_by_file=error_by_file,
            source_document_count_by_file=source_document_count_by_file,
            ocr_document_count_by_file=ocr_document_count_by_file,
            chunk_stats=chunk_stats,
            keyword_indexed=keyword_indexed,
            vector_indexed=vector_indexed,
            embedding_model=embedding_model,
        )
        for record in records
    ]
    _write_jsonl(extraction_root / "corpus-ledger.jsonl", rows)
    summary = _ledger_summary(rows)
    _write_json(extraction_root / "corpus-ledger-summary.json", summary)
    return summary


def _ledger_row(
    record: Any,
    *,
    status_by_file: Mapping[str, Any],
    method_by_file: Mapping[str, str | None],
    error_by_file: Mapping[str, str | None],
    source_document_count_by_file: Mapping[str, int],
    ocr_document_count_by_file: Mapping[str, int],
    chunk_stats: Mapping[str, dict[str, int]],
    keyword_indexed: bool,
    vector_indexed: bool,
    embedding_model: str | None,
) -> dict[str, Any]:
    file_id = str(getattr(record, "file_id", ""))
    status = _status_value(status_by_file.get(file_id, getattr(record, "status", "pending")))
    stats = chunk_stats.get(file_id, {})
    chunk_count = int(stats.get("chunk_count", 0))
    locator_coverage = _coverage(stats.get("locator_count", 0), chunk_count)
    required_metadata_coverage = _coverage(stats.get("required_metadata_count", 0), chunk_count)
    duplicate_group_id = _optional_str(getattr(record, "duplicate_group_id", None))
    failure_reason = _optional_str(error_by_file.get(file_id))
    row = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "file_id": file_id,
        "relative_path": _path_posix(getattr(record, "relative_path", "")),
        "filename": str(getattr(record, "filename", "") or ""),
        "extension": _extension(record),
        "size_bytes": int(getattr(record, "size_bytes", 0) or 0),
        "sha256": str(getattr(record, "sha256", "") or ""),
        "modified_ns": int(getattr(record, "modified_ns", 0) or 0),
        "parent_folder": _path_posix(getattr(record, "parent_folder", "")),
        "inferred_category": str(getattr(record, "inferred_category", "") or ""),
        "status": status,
        "extraction_method": method_by_file.get(file_id),
        "failure_reason": failure_reason,
        "duplicate_group_id": duplicate_group_id,
        "duplicate_action": "duplicate_group_member" if duplicate_group_id else "unique",
        "source_document_count": int(source_document_count_by_file.get(file_id, 0) or 0),
        "ocr_document_count": int(ocr_document_count_by_file.get(file_id, 0) or 0),
        "chunk_count": chunk_count,
        "locator_coverage": locator_coverage,
        "required_metadata_coverage": required_metadata_coverage,
        "index_inclusion_reason": _index_inclusion_reason(status, chunk_count, failure_reason),
        "keyword_index_status": _keyword_index_status(status, chunk_count, keyword_indexed),
        "vector_index_status": _vector_index_status(status, chunk_count, vector_indexed),
        "embedding_model": embedding_model if vector_indexed and chunk_count > 0 and status == "indexed" else None,
    }
    return row


def _chunk_stats_by_file(chunks: Iterable[Any]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for chunk in chunks:
        metadata = dict(getattr(chunk, "metadata", {}) or {})
        file_id = metadata.get("file_id")
        if file_id is None:
            continue
        file_stats = stats.setdefault(
            str(file_id),
            {
                "chunk_count": 0,
                "locator_count": 0,
                "required_metadata_count": 0,
            },
        )
        file_stats["chunk_count"] += 1
        if _has_value(metadata.get("source_locator")):
            file_stats["locator_count"] += 1
        if all(_has_value(metadata.get(key)) for key in REQUIRED_CHUNK_METADATA):
            file_stats["required_metadata_count"] += 1
    return stats


def _ledger_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_chunks = sum(int(row["chunk_count"]) for row in rows)
    located_chunks = sum(round(float(row["locator_coverage"]) * int(row["chunk_count"])) for row in rows)
    required_metadata_chunks = sum(
        round(float(row["required_metadata_coverage"]) * int(row["chunk_count"])) for row in rows
    )
    duplicate_groups = {
        str(row["duplicate_group_id"])
        for row in rows
        if row.get("duplicate_group_id")
    }
    status_counts = Counter(str(row["status"]) for row in rows)
    return {
        "schema_version": LEDGER_SUMMARY_SCHEMA_VERSION,
        "total_files": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "indexed_files": status_counts.get("indexed", 0),
        "manifest_only_files": status_counts.get("manifest_only", 0),
        "no_text_files": status_counts.get("no_text", 0),
        "unsupported_files": status_counts.get("unsupported", 0),
        "failed_files": status_counts.get("failed", 0),
        "chunk_count": total_chunks,
        "source_document_count": sum(int(row["source_document_count"]) for row in rows),
        "ocr_document_count": sum(int(row["ocr_document_count"]) for row in rows),
        "duplicate_group_count": len(duplicate_groups),
        "locator_coverage": _coverage(located_chunks, total_chunks),
        "required_metadata_coverage": _coverage(required_metadata_chunks, total_chunks),
    }


def _index_inclusion_reason(status: str, chunk_count: int, failure_reason: str | None) -> str:
    if status == "indexed" and chunk_count > 0:
        return "indexable"
    if status == "indexed":
        return "indexed_without_chunks"
    if status == "manifest_only":
        return "manifest_only"
    if status == "no_text":
        return "no_text"
    if status == "unsupported":
        return "unsupported"
    if status == "failed":
        return failure_reason or "extraction_failed"
    return status or "unknown"


def _keyword_index_status(status: str, chunk_count: int, keyword_indexed: bool) -> str:
    if status == "indexed" and chunk_count > 0:
        return "indexed" if keyword_indexed else "failed"
    return "skipped"


def _vector_index_status(status: str, chunk_count: int, vector_indexed: bool) -> str:
    if status == "indexed" and chunk_count > 0:
        return "indexed" if vector_indexed else "skipped"
    return "skipped"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _coverage(count: int | float | None, total: int | float | None) -> float:
    denominator = float(total or 0)
    if denominator <= 0:
        return 0.0
    return round(float(count or 0) / denominator, 4)


def _status_value(status: Any) -> str:
    return str(getattr(status, "value", status))


def _path_posix(value: Any) -> str:
    if value is None:
        return ""
    return Path(str(value)).as_posix()


def _extension(record: Any) -> str:
    extension = str(getattr(record, "extension", "") or "")
    if extension:
        return extension
    return Path(_path_posix(getattr(record, "relative_path", ""))).suffix


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True
