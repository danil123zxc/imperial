from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document


_STATUS_PRIORITY = {"active": 3, "draft": 2, "archived": 1}


def load_authority_catalog(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("documents", payload) if isinstance(payload, dict) else payload
    catalog: dict[str, dict[str, Any]] = {}
    if isinstance(rows, dict):
        iterator: Iterable[tuple[str, Any]] = rows.items()
    elif isinstance(rows, list):
        iterator = ((str(row.get("relative_path") or ""), row) for row in rows if isinstance(row, dict))
    else:
        raise ValueError(f"authority catalog must contain a mapping or document list: {path}")
    for relative_path, raw in iterator:
        normalized = Path(relative_path).as_posix().strip()
        if normalized:
            catalog[normalized] = dict(raw)
    return catalog


def apply_authority_and_exact_deduplication(
    documents: list[Document],
    catalog: dict[str, dict[str, Any]],
) -> list[Document]:
    by_hash: dict[str, list[Document]] = defaultdict(list)
    without_hash: list[Document] = []
    for document in documents:
        file_hash = str((document.metadata or {}).get("file_hash") or "").strip()
        (by_hash[file_hash] if file_hash else without_hash).append(document)

    retained = list(without_hash)
    for file_hash, group_documents in sorted(by_hash.items()):
        paths = sorted(
            {
                Path(str(document.metadata.get("relative_path") or "unknown")).as_posix()
                for document in group_documents
            }
        )
        canonical_path = max(paths, key=lambda path: _authority_sort_key(path, catalog.get(path, {})))
        for document in group_documents:
            path = Path(str(document.metadata.get("relative_path") or "unknown")).as_posix()
            if path != canonical_path:
                continue
            metadata = _authority_metadata(document.metadata, catalog.get(path, {}))
            metadata.update(
                {
                    "canonical_source_path": canonical_path,
                    "provenance_paths": paths,
                    "exact_duplicate_count": len(paths),
                    "exact_duplicate_hash": file_hash,
                }
            )
            retained.append(Document(page_content=document.page_content, metadata=metadata))
    return retained


def _authority_sort_key(relative_path: str, entry: dict[str, Any]) -> tuple[int, int, str]:
    status = str(entry.get("status") or "active").casefold()
    rank = int(entry.get("authoritative_rank") or 0)
    # max() chooses the highest status/rank and the lexicographically last path;
    # negate path preference by comparing a stable inverted representation.
    inverted_path = "".join(chr(0x10FFFF - ord(char)) for char in relative_path)
    return (_STATUS_PRIORITY.get(status, 0), rank, inverted_path)


def _authority_metadata(metadata: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(metadata)
    enriched.update(
        {
            "department": entry.get("department") or metadata.get("inferred_category") or "",
            "document_type": entry.get("document_type") or metadata.get("file_extension") or "unknown",
            "authority_status": entry.get("status") or "active",
            "effective_from": entry.get("effective_from"),
            "effective_to": entry.get("effective_to"),
            "document_owner": entry.get("owner"),
            "authoritative_rank": int(entry.get("authoritative_rank") or 0),
            "supersedes": list(entry.get("supersedes") or []),
            "version_group": entry.get("version_group"),
        }
    )
    return enriched
