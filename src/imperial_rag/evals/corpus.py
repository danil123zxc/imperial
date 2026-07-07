"""Shared chunk-level corpus model for eval auditing and evidence packets.

Eval gold IDs resolve with chunk_id-first precedence (chunk_id, then
citation_id, then file_id for chunks that carry neither). Retrieval spans
prefer citation_id (see retrieval/identity.py); eval matching deliberately
differs because gold rows pin the stable chunk_id emitted by ingestion.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from imperial_rag.jsonl import iter_jsonl


# Metadata fields that identify a retrieved chunk, in precedence order.
CHUNK_ID_METADATA_FIELDS = ("chunk_id", "citation_id")


@dataclass
class CorpusChunk:
    file_id: str
    chunk_id: str = ""
    citation_id: str = ""
    relative_path: str = ""
    file_name: str = ""
    file_path: str = ""
    parent_folder: str = ""
    source_type: str = ""
    chunk_index: int = 0
    page_number: str = ""
    section_heading: str = ""
    source_locator: str = ""
    text: str = ""

    @property
    def reference_id(self) -> str:
        return self.chunk_id or self.citation_id or self.file_id

    @cached_property
    def normalized_search_text(self) -> str:
        return normalize_text(
            "\n".join(
                part
                for part in [
                    self.relative_path,
                    self.file_name,
                    self.file_path,
                    self.parent_folder,
                    self.source_type,
                    self.page_number,
                    self.section_heading,
                    self.source_locator,
                    self.text,
                ]
                if part
            )
        )

    def to_packet(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "chunk_id": self.chunk_id,
            "citation_id": self.citation_id,
            "relative_path": self.relative_path,
            "file_name": self.file_name,
            "chunk_index": self.chunk_index,
            "text": self.text,
        }


@dataclass
class ChunkCorpus:
    chunks_by_reference_id: dict[str, CorpusChunk] = field(default_factory=dict)

    def resolve(self, context_id: str) -> CorpusChunk | None:
        return self.chunks_by_reference_id.get(str(context_id or "").strip())


def load_chunk_corpus(path: Path) -> ChunkCorpus:
    corpus = ChunkCorpus()
    for chunk in iter_corpus_chunks(path):
        corpus.chunks_by_reference_id.setdefault(chunk.reference_id, chunk)
    return corpus


def iter_corpus_chunks(path: Path) -> Iterator[CorpusChunk]:
    if not path.exists():
        return
    for payload in iter_jsonl(path):
        chunk = chunk_from_payload(payload)
        if chunk is not None:
            yield chunk


def chunk_from_payload(payload: Mapping[str, Any]) -> CorpusChunk | None:
    metadata = dict(payload.get("metadata") or {})
    file_id = str(metadata.get("file_id") or payload.get("file_id") or "").strip()
    if not file_id:
        return None
    return CorpusChunk(
        file_id=file_id,
        chunk_id=str(metadata.get("chunk_id") or payload.get("chunk_id") or "").strip(),
        citation_id=str(metadata.get("citation_id") or payload.get("citation_id") or "").strip(),
        relative_path=str(metadata.get("relative_path") or "").strip(),
        file_name=str(metadata.get("file_name") or "").strip(),
        file_path=str(metadata.get("file_path") or "").strip(),
        parent_folder=str(metadata.get("parent_folder") or "").strip(),
        source_type=str(metadata.get("source_type") or "").strip(),
        chunk_index=_int_value(metadata.get("chunk_index") or payload.get("chunk_index")),
        page_number=str(metadata.get("page_number") or "").strip(),
        section_heading=str(metadata.get("section_heading") or "").strip(),
        source_locator=str(metadata.get("source_locator") or "").strip(),
        text=str(payload.get("page_content") or payload.get("text") or "").strip(),
    )


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(normalized.split())


def unique_nonempty(values: Iterable[Any]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            resolved.append(text)
    return resolved


def clean_string_list(value: Any, *, allow_scalar: bool = False) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not allow_scalar or value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def clean_context_ids(values: Any) -> list[str]:
    if values is None:
        raw_values: Iterable[Any] = ()
    elif isinstance(values, (str, bytes)) or isinstance(values, Mapping) or not isinstance(values, Iterable):
        raw_values = (values,)
    else:
        raw_values = values
    return unique_nonempty(raw_values)


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
