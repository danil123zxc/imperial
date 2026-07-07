from __future__ import annotations

from typing import Any, Iterable, Mapping

from imperial_rag.evals.corpus import ChunkCorpus as EvidenceCorpus
from imperial_rag.evals.corpus import clean_string_list
from imperial_rag.evals.corpus import load_chunk_corpus as load_evidence_corpus
from imperial_rag.jsonl import write_jsonl

__all__ = ["EvidenceCorpus", "build_evidence_packets", "load_evidence_corpus", "write_jsonl"]


def build_evidence_packets(
    rows: Iterable[Mapping[str, Any]],
    *,
    corpus: EvidenceCorpus,
    audit_rows: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    audit_by_id = {str(row.get("id") or ""): dict(row) for row in audit_rows}
    return [_evidence_packet(row, corpus=corpus, audit=audit_by_id.get(str(row.get("id") or ""))) for row in rows]


def _evidence_packet(
    row: Mapping[str, Any],
    *,
    corpus: EvidenceCorpus,
    audit: dict[str, Any] | None,
) -> dict[str, Any]:
    context_ids = clean_string_list(row.get("reference_context_ids"))
    resolved_ids: list[str] = []
    unresolved_ids: list[str] = []
    evidence: list[dict[str, Any]] = []
    for context_id in context_ids:
        chunk = corpus.resolve(context_id)
        if chunk is None:
            unresolved_ids.append(context_id)
        else:
            resolved_ids.append(context_id)
            evidence.append(chunk.to_packet())
    status = _gold_status(row, audit=audit, resolved_ids=resolved_ids, unresolved_ids=unresolved_ids)

    return {
        "id": str(row.get("id") or ""),
        "suite": str(row.get("suite") or ""),
        "tags": clean_string_list(row.get("tags")),
        "lane": _packet_lane(row, audit),
        "expected_behavior": str(row.get("expected_behavior") or ""),
        "question": str(row.get("question") or ""),
        "expected_source_hints": clean_string_list(row.get("expected_source_hints")),
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
