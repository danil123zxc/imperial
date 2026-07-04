from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from imperial_rag.evals.corpus import (
    ChunkCorpus,
    CorpusChunk,
    iter_corpus_chunks,
    normalize_text,
    unique_nonempty,
)


VALID_EXPECTED_BEHAVIORS = {"cite_answer", "refuse_if_not_found", "surface_conflict"}
SUPPORTED_PHOENIX_METRICS = {"faithfulness", "answer_relevancy", "id_context_recall"}
VALID_LANES = {
    "indexed_answerability",
    "conflict_version_behavior",
    "refusal_out_of_corpus_behavior",
    "known_missing_document_coverage",
}
LANES_BY_BEHAVIOR = {
    "cite_answer": "indexed_answerability",
    "surface_conflict": "conflict_version_behavior",
}
# Single source of truth for the audit-row schema: _audit_row must emit exactly
# these keys, validate_eval_contract requires them, and the markdown table
# renders them minus the wide review-only fields.
AUDIT_ROW_COLUMNS = (
    "id",
    "expected_behavior",
    "lane",
    "current_reference_context_ids",
    "resolved_chunk_ids",
    "file_only_reference_context_ids",
    "unresolved_reference_context_ids",
    "resolved_indexed_file_ids",
    "candidate_chunk_ids",
    "candidate_file_ids",
    "candidate_locators",
    "source_path",
    "indexed_status",
    "reference_answer_quality",
    "expected_source_hints_quality",
    "action",
    "quarantine_reason",
    "backlog_category",
    "notes",
)
MARKDOWN_EXCLUDED_COLUMNS = {"candidate_locators", "notes"}
REQUIRED_AUDIT_KEYS = set(AUDIT_ROW_COLUMNS)


@dataclass
class CorpusDocument:
    file_id: str
    relative_path: str = ""
    file_name: str = ""
    file_path: str = ""
    parent_folder: str = ""
    chunk_count: int = 0


@dataclass
class CorpusIndex:
    documents: dict[str, CorpusDocument]
    chunks: ChunkCorpus = field(default_factory=ChunkCorpus)

    def resolve(self, file_id: str) -> CorpusDocument | None:
        return self.documents.get(file_id)

    def resolve_chunk(self, context_id: str) -> CorpusChunk | None:
        return self.chunks.resolve(context_id)

    def candidate_chunks(self, hints: Iterable[str], *, limit: int = 8) -> list[CorpusChunk]:
        normalized_hints = _normalized_hints(hints)
        if not normalized_hints:
            return []
        matched = [
            chunk
            for chunk in self.chunks.chunks_by_reference_id.values()
            if _normalized_match_score(chunk.normalized_search_text, normalized_hints) > 0
        ]
        return _ranked_by_hint_score(matched, normalized_hints)[:limit]

    def chunks_for_files(self, file_ids: Iterable[str], hints: Iterable[str], *, limit: int = 8) -> list[CorpusChunk]:
        wanted = {str(file_id or "").strip() for file_id in file_ids}
        chunks = [chunk for chunk in self.chunks.chunks_by_reference_id.values() if chunk.file_id in wanted]
        return _ranked_by_hint_score(chunks, _normalized_hints(hints))[:limit]


def load_corpus_index(chunks_path: Path) -> CorpusIndex:
    index = CorpusIndex(documents={})
    for chunk in iter_corpus_chunks(chunks_path):
        document = index.documents.setdefault(chunk.file_id, CorpusDocument(file_id=chunk.file_id))
        document.chunk_count += 1
        for field_name in ("relative_path", "file_name", "file_path", "parent_folder"):
            value = getattr(chunk, field_name)
            if value and not getattr(document, field_name):
                setattr(document, field_name, value)
        index.chunks.chunks_by_reference_id.setdefault(chunk.reference_id, chunk)
    return index


def load_question_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_eval_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    corpus_index: CorpusIndex,
    documents_root: Path,
) -> list[dict[str, Any]]:
    source_paths = _discover_source_paths(documents_root)
    return [_audit_row(row, corpus_index=corpus_index, source_paths=source_paths) for row in rows]


def validate_eval_contract(
    audit_rows: Iterable[Mapping[str, Any]],
    *,
    phoenix_metric_names: Iterable[str] = (),
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for metric_name in phoenix_metric_names:
        metric = str(metric_name).strip()
        if metric and metric not in SUPPORTED_PHOENIX_METRICS:
            findings.append(
                {
                    "severity": "error",
                    "row_id": None,
                    "code": "unsupported_phoenix_metric",
                    "message": f"Phoenix evaluator path does not support {metric}",
                }
            )

    for row in audit_rows:
        row_id = str(row.get("id") or "").strip() or None
        missing_keys = sorted(REQUIRED_AUDIT_KEYS - set(row))
        if missing_keys:
            findings.append(
                {
                    "severity": "error",
                    "row_id": row_id,
                    "code": "missing_audit_columns",
                    "message": f"Audit row is missing required columns: {', '.join(missing_keys)}",
                }
            )
        if not row_id:
            findings.append(
                {"severity": "error", "row_id": None, "code": "missing_id", "message": "Eval row is missing id"}
            )
        elif row_id in seen_ids:
            findings.append(
                {
                    "severity": "error",
                    "row_id": row_id,
                    "code": "duplicate_id",
                    "message": f"Duplicate eval row id {row_id}",
                }
            )
        else:
            seen_ids.add(row_id)

        behavior = row.get("expected_behavior")
        if behavior not in VALID_EXPECTED_BEHAVIORS:
            findings.append(
                {
                    "severity": "error",
                    "row_id": row_id,
                    "code": "invalid_expected_behavior",
                    "message": f"Invalid expected_behavior {behavior!r}",
                }
            )

        current_ids = _clean_list(row.get("current_reference_context_ids"))
        resolved_ids = _clean_list(row.get("resolved_chunk_ids"))
        file_only_ids = _clean_list(row.get("file_only_reference_context_ids"))
        unresolved_ids = _clean_list(row.get("unresolved_reference_context_ids"))
        if file_only_ids:
            findings.append(
                {
                    "severity": "error",
                    "row_id": row_id,
                    "code": "file_only_reference_context_ids",
                    "message": "reference_context_ids must identify chunks, not files",
                }
            )
        if unresolved_ids:
            findings.append(
                {
                    "severity": "error",
                    "row_id": row_id,
                    "code": "unresolved_reference_context_ids",
                    "message": "reference_context_ids must resolve against the extracted corpus index",
                }
            )

        lane = row.get("lane")
        action = row.get("action")
        if lane not in VALID_LANES:
            findings.append(
                {
                    "severity": "error",
                    "row_id": row_id,
                    "code": "invalid_lane",
                    "message": f"Invalid eval lane {lane!r}",
                }
            )
        elif behavior == "cite_answer" and lane not in {"indexed_answerability", "known_missing_document_coverage"}:
            findings.append(
                {
                    "severity": "error",
                    "row_id": row_id,
                    "code": "lane_expected_behavior_mismatch",
                    "message": "cite_answer rows must use indexed_answerability or known_missing_document_coverage lanes",
                }
            )
        elif behavior == "surface_conflict" and lane != "conflict_version_behavior":
            findings.append(
                {
                    "severity": "error",
                    "row_id": row_id,
                    "code": "lane_expected_behavior_mismatch",
                    "message": "surface_conflict rows must use conflict_version_behavior lane",
                }
            )
        elif behavior == "refuse_if_not_found" and lane not in {
            "refusal_out_of_corpus_behavior",
            "known_missing_document_coverage",
        }:
            findings.append(
                {
                    "severity": "error",
                    "row_id": row_id,
                    "code": "lane_expected_behavior_mismatch",
                    "message": "refuse_if_not_found rows must use refusal_out_of_corpus_behavior or known_missing_document_coverage lanes",
                }
            )
        if lane == "indexed_answerability" and action not in {"quarantine", "needs_ingestion"} and not resolved_ids:
            findings.append(
                {
                    "severity": "error",
                    "row_id": row_id,
                    "code": "missing_required_reference_context_ids",
                    "message": "indexed_answerability rows require at least one resolving reference_context_id unless quarantined",
                }
            )
        if lane == "conflict_version_behavior" and action not in {"quarantine", "needs_ingestion"} and len(resolved_ids) < 2:
            findings.append(
                {
                    "severity": "error",
                    "row_id": row_id,
                    "code": "missing_required_reference_context_ids",
                    "message": "conflict_version_behavior rows require at least two resolving reference_context_ids unless quarantined",
                }
            )
        if lane == "refusal_out_of_corpus_behavior" and current_ids:
            findings.append(
                {
                    "severity": "warning",
                    "row_id": row_id,
                    "code": "unexpected_reference_context_ids",
                    "message": "out-of-corpus refusal rows should not carry reference_context_ids",
                }
            )
        if action == "quarantine" and not str(row.get("quarantine_reason") or "").strip():
            findings.append(
                {
                    "severity": "error",
                    "row_id": row_id,
                    "code": "missing_quarantine_reason",
                    "message": "quarantine rows require a quarantine_reason",
                }
            )
    return findings


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) for row in rows)
    path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")


def write_markdown_table(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    columns = [column for column in AUDIT_ROW_COLUMNS if column not in MARKDOWN_EXCLUDED_COLUMNS]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_markdown_value(row.get(column)) for column in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _audit_row(
    row: Mapping[str, Any],
    *,
    corpus_index: CorpusIndex,
    source_paths: list[Path],
) -> dict[str, Any]:
    row_id = str(row.get("id") or "").strip()
    behavior = str(row.get("expected_behavior") or "").strip()
    hints = [str(hint).strip() for hint in row.get("expected_source_hints") or [] if str(hint).strip()]
    current_ids = _clean_list(row.get("reference_context_ids"))
    resolved_chunks: list[CorpusChunk] = []
    file_only_ids: list[str] = []
    unresolved_ids: list[str] = []
    for context_id in current_ids:
        chunk = corpus_index.resolve_chunk(context_id)
        if chunk is not None:
            resolved_chunks.append(chunk)
        elif corpus_index.resolve(context_id) is not None:
            file_only_ids.append(context_id)
        else:
            unresolved_ids.append(context_id)
    resolved_chunk_ids = unique_nonempty(chunk.reference_id for chunk in resolved_chunks)
    if file_only_ids:
        candidate_chunks = corpus_index.chunks_for_files(file_only_ids, hints)
    else:
        candidate_chunks = corpus_index.candidate_chunks(hints)
    source_path = _best_source_path(hints, source_paths)
    candidate_ids = _candidate_file_ids(candidate_chunks, hints=hints, source_path=source_path)
    lane = _lane_for(row)
    answer_quality = _reference_answer_quality(str(row.get("reference_answer") or ""))
    hints_quality = _expected_source_hints_quality(
        hints,
        resolved_chunks=resolved_chunks,
        candidate_ids=candidate_ids,
        source_path=source_path,
    )
    indexed_status = _indexed_status(
        lane=lane,
        current_ids=current_ids,
        file_only_ids=file_only_ids,
        unresolved_ids=unresolved_ids,
        candidate_ids=candidate_ids,
        hints_quality=hints_quality,
        source_path=source_path,
    )
    row_quarantine_reason = str(row.get("quarantine_reason") or "").strip()
    action, quarantine_reason, backlog_category, notes = _row_action(
        lane=lane,
        current_ids=current_ids,
        resolved_ids=resolved_chunk_ids,
        file_only_ids=file_only_ids,
        unresolved_ids=unresolved_ids,
        candidate_ids=candidate_ids,
        source_path=source_path,
        answer_quality=answer_quality,
        hints_quality=hints_quality,
        row_quarantine_reason=row_quarantine_reason,
    )

    return {
        "id": row_id,
        "expected_behavior": behavior,
        "lane": lane,
        "current_reference_context_ids": current_ids,
        "resolved_chunk_ids": resolved_chunk_ids,
        "file_only_reference_context_ids": file_only_ids,
        "unresolved_reference_context_ids": unresolved_ids,
        "resolved_indexed_file_ids": unique_nonempty(chunk.file_id for chunk in resolved_chunks),
        "candidate_chunk_ids": unique_nonempty(chunk.reference_id for chunk in candidate_chunks),
        "candidate_file_ids": candidate_ids,
        "candidate_locators": [_chunk_locator(chunk) for chunk in candidate_chunks],
        "source_path": _relative_source_path(source_path),
        "indexed_status": indexed_status,
        "reference_answer_quality": answer_quality,
        "expected_source_hints_quality": hints_quality,
        "action": action,
        "quarantine_reason": quarantine_reason,
        "backlog_category": backlog_category,
        "notes": notes,
    }


def _lane_for(row: Mapping[str, Any]) -> str:
    explicit_lane = str(row.get("lane") or "").strip()
    if explicit_lane:
        return explicit_lane
    behavior = row.get("expected_behavior")
    if behavior == "refuse_if_not_found":
        tags = {str(tag).casefold() for tag in row.get("tags") or []}
        if tags & {"known_missing_doc", "known-missing-doc", "missing_document", "needs_ingestion"}:
            return "known_missing_document_coverage"
        return "refusal_out_of_corpus_behavior"
    return LANES_BY_BEHAVIOR.get(str(behavior), "unknown")


def _indexed_status(
    *,
    lane: str,
    current_ids: list[str],
    file_only_ids: list[str],
    unresolved_ids: list[str],
    candidate_ids: list[str],
    hints_quality: str,
    source_path: Path | None,
) -> str:
    if lane == "refusal_out_of_corpus_behavior":
        return "out_of_corpus"
    if file_only_ids:
        return "file_only_gold_ids"
    if unresolved_ids:
        return "unresolved_gold_ids"
    if current_ids and hints_quality == "hit":
        return "indexed"
    if source_path is not None and hints_quality == "source_path_only":
        return "source_exists_not_indexed"
    if candidate_ids:
        return "candidate_indexed"
    if not current_ids:
        return "missing_gold_ids"
    return "indexed"


def _row_action(
    *,
    lane: str,
    current_ids: list[str],
    resolved_ids: list[str],
    file_only_ids: list[str],
    unresolved_ids: list[str],
    candidate_ids: list[str],
    source_path: Path | None,
    answer_quality: str,
    hints_quality: str,
    row_quarantine_reason: str = "",
) -> tuple[str, str, str, list[str]]:
    notes: list[str] = []
    if row_quarantine_reason:
        notes.append("row is quarantined by explicit dataset metadata")
        return "quarantine", row_quarantine_reason, "row_contract", notes
    if lane == "refusal_out_of_corpus_behavior" and not current_ids:
        return "keep", "", "none", notes

    if file_only_ids:
        notes.append("reference_context_ids resolve only as file IDs; migrate them to chunk IDs")
        return "rewrite", "file_only_reference_context_ids", "chunk_id_migration", notes

    if unresolved_ids:
        notes.append("one or more reference_context_ids do not resolve against the extracted corpus")
        return "needs_ingestion", "unresolved_reference_context_ids", "missing_indexed_source", notes

    if current_ids and hints_quality in {"source_path_only", "missing"}:
        notes.append(
            "reference_context_ids resolve, but the resolved indexed files do not contain the expected source hints"
        )
        if source_path is not None:
            return "needs_ingestion", "gold_ids_do_not_match_hints", "missing_indexed_source", notes
        return "quarantine", "gold_ids_do_not_match_hints", "row_contract", notes

    if lane == "conflict_version_behavior" and len(resolved_ids) < 2:
        if candidate_ids:
            notes.append("candidate indexed files exist but gold IDs are not backfilled")
            return "rewrite", "", "gold_id_backfill", notes
        if source_path is not None:
            return "needs_ingestion", "missing_conflict_gold_ids", "missing_indexed_source", notes
        return "quarantine", "missing_conflict_gold_ids", "row_contract", notes

    if lane == "indexed_answerability" and not resolved_ids:
        if candidate_ids:
            notes.append("candidate indexed files exist but gold IDs are not backfilled")
            return "rewrite", "", "gold_id_backfill", notes
        if source_path is not None:
            return "needs_ingestion", "missing_gold_id_for_existing_source", "missing_indexed_source", notes
        return "quarantine", "missing_gold_id", "row_contract", notes

    if answer_quality == "generic_meta_reference":
        notes.append("reference_answer describes desired behavior instead of concrete evidence")
        return "rewrite", "", "answer_key_rewrite", notes

    return "keep", "", "none", notes


def _expected_source_hints_quality(
    hints: list[str],
    *,
    resolved_chunks: list[CorpusChunk],
    candidate_ids: list[str],
    source_path: Path | None,
) -> str:
    normalized_hints = _normalized_hints(hints)
    if not normalized_hints:
        return "not_required"
    resolved_text = " ".join(chunk.normalized_search_text for chunk in resolved_chunks)
    resolved_score = _normalized_match_score(resolved_text, normalized_hints) if resolved_text else 0
    source_score = (
        _normalized_match_score(normalize_text(str(source_path)), normalized_hints) if source_path is not None else 0
    )
    if source_score > resolved_score:
        return "source_path_only"
    if resolved_score > 0:
        return "hit"
    if candidate_ids:
        return "candidate_only"
    if source_path is not None:
        return "source_path_only"
    return "missing"


def _reference_answer_quality(answer: str) -> str:
    normalized = normalize_text(answer)
    if not normalized:
        return "missing"
    generic_markers = (
        "ответ должен",
        "ответ должно",
        "должен ссылаться",
        "должна ссылаться",
        "должны быть взяты",
        "должен явно",
    )
    if any(marker in normalized for marker in generic_markers):
        return "generic_meta_reference"
    return "evidence_shaped"


def _discover_source_paths(documents_root: Path) -> list[Path]:
    if not documents_root.exists():
        return []
    return sorted(path for path in documents_root.rglob("*") if path.is_file() and not _is_ignored_source_path(path))


def _is_ignored_source_path(path: Path) -> bool:
    name = path.name
    return name.startswith("~$") or name.startswith(".~lock.") or name in {".DS_Store", "Thumbs.db"}


def _best_source_path(hints: list[str], source_paths: list[Path]) -> Path | None:
    normalized_hints = _normalized_hints(hints)
    if not normalized_hints:
        return None
    scored: list[tuple[int, str, Path]] = []
    for path in source_paths:
        score = _normalized_match_score(normalize_text(str(path)), normalized_hints)
        if score > 0:
            scored.append((score, str(path), path))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2]


def _candidate_file_ids(
    candidate_chunks: list[CorpusChunk],
    *,
    hints: list[str],
    source_path: Path | None,
) -> list[str]:
    candidate_ids = unique_nonempty(chunk.file_id for chunk in candidate_chunks)
    normalized_hints = _normalized_hints(hints)
    if not candidate_ids or source_path is None or not normalized_hints:
        return candidate_ids
    source_score = _normalized_match_score(normalize_text(str(source_path)), normalized_hints)
    candidate_score = max(
        _normalized_match_score(chunk.normalized_search_text, normalized_hints) for chunk in candidate_chunks
    )
    if source_score > candidate_score:
        return []
    return candidate_ids


def _relative_source_path(path: Path | None) -> str:
    if path is None:
        return ""
    parts = list(path.parts)
    if "documents" in parts:
        index = parts.index("documents")
        return str(Path(*parts[index + 1 :]))
    return str(path)


def _chunk_locator(chunk: CorpusChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.reference_id,
        "citation_id": chunk.citation_id,
        "file_id": chunk.file_id,
        "relative_path": chunk.relative_path,
        "file_name": chunk.file_name,
        "chunk_index": chunk.chunk_index,
        "page_number": chunk.page_number,
        "section_heading": chunk.section_heading,
        "source_locator": chunk.source_locator,
        "evidence_quote": " ".join(chunk.text.split())[:300],
    }


def _clean_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalized_hints(hints: Iterable[str]) -> list[str]:
    return [normalize_text(hint) for hint in hints if normalize_text(hint)]


def _normalized_match_score(normalized_value: str, normalized_hints: list[str]) -> int:
    return sum(1 for hint in normalized_hints if hint in normalized_value)


def _ranked_by_hint_score(chunks: list[CorpusChunk], normalized_hints: list[str]) -> list[CorpusChunk]:
    return sorted(
        chunks,
        key=lambda chunk: (
            -_normalized_match_score(chunk.normalized_search_text, normalized_hints),
            chunk.relative_path or chunk.file_name,
            chunk.reference_id,
        ),
    )


def _markdown_value(value: Any) -> str:
    if isinstance(value, list):
        raw = ", ".join(str(item) for item in value)
    else:
        raw = str(value or "")
    return raw.replace("|", "\\|").replace("\n", " ")
