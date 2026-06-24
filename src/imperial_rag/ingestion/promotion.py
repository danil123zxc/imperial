from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PromotionGateResult:
    passed: bool
    errors: list[str]
    summary: dict[str, int | float]


def check_promotion_gates(
    baseline_root: Path,
    shadow_root: Path,
    *,
    questions_path: Path,
    min_locator_coverage: float = 0.95,
) -> PromotionGateResult:
    errors: list[str] = []
    if baseline_root.resolve() == shadow_root.resolve():
        errors.append("baseline and shadow roots must be different")

    baseline_rows = _read_jsonl_required(baseline_root / "corpus-ledger.jsonl", errors)
    shadow_rows = _read_jsonl_required(shadow_root / "corpus-ledger.jsonl", errors)
    baseline_chunks = _read_jsonl_required(baseline_root / "chunks.jsonl", errors)
    id_map = _read_json_required(shadow_root / "old-to-new-id-map.json", errors)
    reviewed_drops = _read_optional_json(shadow_root / "reviewed-drops.json", default={"rows": []})
    questions = _read_jsonl_required(questions_path, errors)

    baseline_ids = {str(row.get("file_id")) for row in baseline_rows if row.get("file_id") is not None}
    shadow_ids = {str(row.get("file_id")) for row in shadow_rows if row.get("file_id") is not None}
    if not shadow_ids.issuperset(baseline_ids):
        for file_id in sorted(baseline_ids - shadow_ids):
            errors.append(f"baseline file missing from shadow ledger: {file_id}")

    baseline_indexed = sum(1 for row in baseline_rows if row.get("status") == "indexed")
    shadow_indexed = sum(1 for row in shadow_rows if row.get("status") == "indexed")
    if shadow_indexed < baseline_indexed:
        errors.append(f"shadow indexed file count regressed: {shadow_indexed} < {baseline_indexed}")

    shadow_chunk_count = sum(int(row.get("chunk_count") or 0) for row in shadow_rows)
    if shadow_chunk_count == 0:
        errors.append("shadow chunk count is zero")

    locator_rows = [row for row in shadow_rows if int(row.get("chunk_count") or 0) > 0]
    locator_coverage = _mean(float(row.get("locator_coverage") or 0.0) for row in locator_rows)
    if locator_rows and locator_coverage < min_locator_coverage:
        errors.append(f"shadow locator coverage below gate: {locator_coverage} < {min_locator_coverage}")

    baseline_chunk_ids = _metadata_values(baseline_chunks, "chunk_id")
    baseline_citation_ids = _metadata_values(baseline_chunks, "citation_id")
    mapped_old_chunk_ids = {
        str(row.get("old_chunk_id"))
        for row in id_map.get("rows", [])
        if row.get("old_chunk_id") and row.get("new_chunk_id")
    }
    mapped_old_citation_ids = {
        str(row.get("old_citation_id"))
        for row in id_map.get("rows", [])
        if row.get("old_citation_id") and row.get("new_citation_id")
    }
    reviewed_drop_chunk_ids, reviewed_drop_citation_ids = _reviewed_drop_ids(reviewed_drops, errors)
    unmapped_old_chunk_ids = baseline_chunk_ids - mapped_old_chunk_ids - reviewed_drop_chunk_ids
    for old_chunk_id in sorted(unmapped_old_chunk_ids):
        errors.append(f"old chunk has no replacement or reviewed drop: {old_chunk_id}")

    for context_id in _reference_context_ids(questions):
        if context_id in shadow_ids:
            continue
        if context_id in baseline_chunk_ids:
            if context_id not in mapped_old_chunk_ids and context_id not in reviewed_drop_chunk_ids:
                errors.append(f"gold old chunk ID has no replacement or reviewed drop: {context_id}")
            continue
        if context_id in baseline_citation_ids:
            if context_id not in mapped_old_citation_ids and context_id not in reviewed_drop_citation_ids:
                errors.append(f"gold old citation ID has no replacement or reviewed drop: {context_id}")
            continue
        errors.append(f"gold reference_context_id missing from shadow ledger: {context_id}")

    summary = {
        "baseline_files": len(baseline_rows),
        "shadow_files": len(shadow_rows),
        "baseline_indexed_files": baseline_indexed,
        "shadow_indexed_files": shadow_indexed,
        "baseline_chunk_ids": len(baseline_chunk_ids),
        "shadow_chunk_count": shadow_chunk_count,
        "shadow_locator_coverage": locator_coverage,
        "mapped_old_chunk_ids": len(mapped_old_chunk_ids),
        "reviewed_drop_chunk_ids": len(reviewed_drop_chunk_ids),
        "unmapped_old_chunk_ids": len(unmapped_old_chunk_ids),
    }
    return PromotionGateResult(passed=not errors, errors=errors, summary=summary)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_jsonl_required(path: Path, errors: list[str]) -> list[dict]:
    if not path.exists():
        errors.append(f"required artifact missing: {path}")
        return []
    try:
        return _read_jsonl(path)
    except json.JSONDecodeError as exc:
        errors.append(f"required artifact is not valid JSONL: {path}: {exc}")
        return []


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_required(path: Path, errors: list[str]) -> dict:
    if not path.exists():
        errors.append(f"required artifact missing: {path}")
        return {"rows": []}
    try:
        return _read_json(path)
    except json.JSONDecodeError as exc:
        errors.append(f"required artifact is not valid JSON: {path}: {exc}")
        return {"rows": []}


def _read_optional_json(path: Path, *, default: dict) -> dict:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_values(rows: list[dict], key: str) -> set[str]:
    values: set[str] = set()
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        value = metadata.get(key)
        if value is not None and str(value).strip():
            values.add(str(value))
    return values


def _reviewed_drop_ids(payload: dict, errors: list[str]) -> tuple[set[str], set[str]]:
    required = {"old_chunk_id", "reason", "reviewed_by", "reviewed_at", "rollback_impact"}
    chunk_ids: set[str] = set()
    citation_ids: set[str] = set()
    for index, row in enumerate(payload.get("rows", [])):
        missing = sorted(field for field in required if not str(row.get(field) or "").strip())
        if missing:
            errors.append(f"reviewed drop row {index} missing fields: {', '.join(missing)}")
            continue
        chunk_ids.add(str(row["old_chunk_id"]))
        if row.get("old_citation_id"):
            citation_ids.add(str(row["old_citation_id"]))
    return chunk_ids, citation_ids


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(sum(materialized) / len(materialized), 4)


def _reference_context_ids(rows: list[dict]) -> list[str]:
    ids: list[str] = []
    for row in rows:
        for context_id in row.get("reference_context_ids") or []:
            value = str(context_id).strip()
            if value:
                ids.append(value)
    return ids
