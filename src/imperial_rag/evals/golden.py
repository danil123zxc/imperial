from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class EvidenceChunk:
    file_id: str
    chunk_id: str
    citation_id: str
    relative_path: str
    file_name: str
    chunk_index: int
    text: str

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


@dataclass(frozen=True)
class EvidenceCorpus:
    chunks_by_file_id: dict[str, list[EvidenceChunk]]

    def chunks_for(self, file_id: str) -> list[EvidenceChunk]:
        return list(self.chunks_by_file_id.get(file_id, []))

    def resolves(self, file_id: str) -> bool:
        return bool(self.chunks_by_file_id.get(file_id))


def load_evidence_corpus(path: Path) -> EvidenceCorpus:
    chunks_by_file_id: dict[str, list[EvidenceChunk]] = {}
    if not path.exists():
        return EvidenceCorpus(chunks_by_file_id={})

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        chunk = _evidence_chunk_from_payload(payload)
        if chunk is None:
            continue
        chunks_by_file_id.setdefault(chunk.file_id, []).append(chunk)

    for chunks in chunks_by_file_id.values():
        chunks.sort(key=lambda chunk: (chunk.chunk_index, chunk.chunk_id, chunk.citation_id))
    return EvidenceCorpus(chunks_by_file_id=chunks_by_file_id)


def build_evidence_packets(
    rows: Iterable[Mapping[str, Any]],
    *,
    corpus: EvidenceCorpus,
    audit_rows: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    audit_by_id = {str(row.get("id") or ""): dict(row) for row in audit_rows}
    return [_evidence_packet(row, corpus=corpus, audit=audit_by_id.get(str(row.get("id") or ""))) for row in rows]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) for row in rows)
    path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")


def _evidence_packet(
    row: Mapping[str, Any],
    *,
    corpus: EvidenceCorpus,
    audit: dict[str, Any] | None,
) -> dict[str, Any]:
    context_ids = _clean_list(row.get("reference_context_ids"))
    resolved_ids = [context_id for context_id in context_ids if corpus.resolves(context_id)]
    unresolved_ids = [context_id for context_id in context_ids if context_id not in resolved_ids]
    evidence = [
        chunk.to_packet()
        for context_id in context_ids
        for chunk in corpus.chunks_for(context_id)
    ]
    status = _gold_status(row, audit=audit, resolved_ids=resolved_ids, unresolved_ids=unresolved_ids)

    return {
        "id": str(row.get("id") or ""),
        "suite": str(row.get("suite") or ""),
        "tags": _clean_list(row.get("tags")),
        "lane": _packet_lane(row, audit),
        "expected_behavior": str(row.get("expected_behavior") or ""),
        "question": str(row.get("question") or ""),
        "expected_source_hints": _clean_list(row.get("expected_source_hints")),
        "current_reference_answer": str(row.get("reference_answer") or ""),
        "candidate_reference_answer": "",
        "current_reference_context_ids": context_ids,
        "resolved_reference_context_ids": resolved_ids,
        "unresolved_reference_context_ids": unresolved_ids,
        "gold_status": status,
        "evidence": evidence,
        "audit": audit or {},
        "review_notes": _review_notes(row, status=status, unresolved_ids=unresolved_ids),
    }


def _evidence_chunk_from_payload(payload: Mapping[str, Any]) -> EvidenceChunk | None:
    metadata = dict(payload.get("metadata") or {})
    file_id = str(metadata.get("file_id") or payload.get("file_id") or "").strip()
    if not file_id:
        return None
    text = str(payload.get("page_content") or payload.get("text") or "").strip()
    chunk_index = _int_value(metadata.get("chunk_index") or payload.get("chunk_index"))
    return EvidenceChunk(
        file_id=file_id,
        chunk_id=str(metadata.get("chunk_id") or payload.get("chunk_id") or "").strip(),
        citation_id=str(metadata.get("citation_id") or payload.get("citation_id") or "").strip(),
        relative_path=str(metadata.get("relative_path") or "").strip(),
        file_name=str(metadata.get("file_name") or "").strip(),
        chunk_index=chunk_index,
        text=text,
    )


def _gold_status(
    row: Mapping[str, Any],
    *,
    audit: Mapping[str, Any] | None,
    resolved_ids: list[str],
    unresolved_ids: list[str],
) -> str:
    if str(row.get("quarantine_reason") or "").strip() or str((audit or {}).get("action") or "") == "quarantine":
        return "quarantined"
    if row.get("expected_behavior") == "refuse_if_not_found":
        return "refusal_boundary"
    if unresolved_ids:
        return "unresolved_gold_context"
    if not resolved_ids:
        return "needs_gold_context"
    if str((audit or {}).get("action") or "") in {"needs_ingestion", "rewrite"}:
        return str((audit or {}).get("action"))
    return "ready_for_review"


def _review_notes(row: Mapping[str, Any], *, status: str, unresolved_ids: list[str]) -> list[str]:
    if status == "refusal_boundary":
        return ["Keep this as an out-of-corpus refusal. Do not add reference_context_ids or corpus facts."]
    if status == "quarantined":
        reason = str(row.get("quarantine_reason") or "").strip()
        notes = ["Do not promote this row as normal gold coverage until quarantine_reason is resolved."]
        if reason:
            notes.append(f"quarantine_reason: {reason}")
        return notes
    if status == "unresolved_gold_context":
        return [f"Resolve or replace missing reference_context_ids before promotion: {', '.join(unresolved_ids)}."]
    if status == "needs_gold_context":
        return ["Add resolving reference_context_ids before treating this as normal gold coverage."]
    if row.get("expected_behavior") == "surface_conflict":
        return [
            "Draft or revise reference_answer only from the resolved evidence chunks.",
            "Explicitly state the version conflict or uncertainty and map each side to chunk_id values.",
        ]
    return [
        "Draft or revise reference_answer only from the resolved evidence chunks.",
        "Every substantive claim must map back to one or more chunk_id values.",
    ]


def _packet_lane(row: Mapping[str, Any], audit: Mapping[str, Any] | None) -> str:
    explicit_lane = str(row.get("lane") or "").strip()
    if explicit_lane:
        return explicit_lane
    return str((audit or {}).get("lane") or "")


def _clean_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _int_value(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
