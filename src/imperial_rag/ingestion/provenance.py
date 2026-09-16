"""Exact coordinates in immutable extracted source units, independent of chunks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from imperial_rag.serialization import stable_json_dumps


def digest(value: Any) -> str:
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def source_identity(document: Document) -> dict[str, str]:
    from imperial_rag.ingestion.chunking import _source_locator

    metadata = document.metadata
    text_hash = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
    identity = [metadata.get("file_id") or metadata.get("relative_path") or "unknown",
                metadata.get("file_hash"), metadata.get("source_type"), _source_locator(metadata), text_hash]
    return {"source_id": digest(identity), "text_sha256": text_hash}


def document_payload(document: Document) -> dict[str, Any]:
    return {"page_content": document.page_content, "metadata": dict(document.metadata)}


def freeze_sources(documents: list[Document], path: Path) -> dict[str, Any]:
    rows = sorted((document_payload(doc) | source_identity(doc) for doc in documents),
                  key=lambda row: row["source_id"])
    if not rows or len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("Snapshot must contain nonempty, uniquely identified source units")
    payload = {"schema_version": "imperial-source-snapshot-v1", "sources": rows}
    snapshot = payload | {"snapshot_hash": digest(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of an annotated snapshot.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(snapshot, stream, ensure_ascii=False, indent=2)
    return snapshot


def freeze_extracted_sources(documents_root: Path, authority_path: Path, output: Path) -> dict[str, Any]:
    from imperial_rag.ingestion.authority import apply_authority_and_exact_deduplication, load_authority_catalog

    documents: list[Document] = []
    for path in sorted(documents_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents.extend(Document(**row) for row in payload["documents"])
    retained = apply_authority_and_exact_deduplication(documents, load_authority_catalog(authority_path))
    return freeze_sources(retained, output)


def load_snapshot(path: Path) -> dict[str, Any]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in snapshot.items() if key != "snapshot_hash"}
    if payload.get("schema_version") != "imperial-source-snapshot-v1" or digest(payload) != snapshot.get("snapshot_hash"):
        raise ValueError("Invalid or modified source snapshot")
    sources = snapshot["sources"]
    if not sources or len({row["source_id"] for row in sources}) != len(sources):
        raise ValueError("Snapshot source identities must be unique")
    for row in sources:
        if source_identity(Document(page_content=row["page_content"], metadata=row["metadata"])) != {
            key: row[key] for key in ("source_id", "text_sha256")
        }:
            raise ValueError("Snapshot source identity mismatch")
    return snapshot


def validate_span(span: dict[str, Any], sources: dict[str, dict[str, Any]]) -> str:
    source = sources.get(span.get("source_id", ""))
    if source is None or source["text_sha256"] != span.get("text_sha256"):
        raise ValueError("Unknown source or stale text version")
    start, end = span.get("start"), span.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source["page_content"]):
        raise ValueError("Source offsets must satisfy 0 <= start < end <= source length")
    return source["page_content"][start:end]


def validate_chunk(document: Document, sources: dict[str, dict[str, Any]]) -> None:
    spans = document.metadata.get("source_spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("Retrieved chunk is missing source mappings; rebuild the shadow index")
    for span in spans:
        text = validate_span(span, sources)
        start, end = span.get("chunk_start"), span.get("chunk_end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(document.page_content):
            raise ValueError("Invalid chunk offsets")
        if document.page_content[start:end] != text:
            raise ValueError("Chunk source mapping does not match its delivered text")
