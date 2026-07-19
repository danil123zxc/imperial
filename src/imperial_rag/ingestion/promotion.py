from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imperial_rag.jsonl import read_jsonl


@dataclass(frozen=True)
class PromotionGateResult:
    passed: bool
    errors: list[str]
    summary: dict[str, Any]


def check_promotion_gates(
    baseline_root: Path,
    shadow_root: Path,
    *,
    questions_path: Path,
    min_locator_coverage: float = 0.95,
    expected_keyword_index: str | None = None,
    expected_qdrant_collection: str | None = None,
) -> PromotionGateResult:
    errors: list[str] = []
    if baseline_root.resolve() == shadow_root.resolve():
        errors.append("baseline and shadow roots must be different")

    baseline_rows = _read_jsonl_required(baseline_root / "corpus-ledger.jsonl", errors)
    shadow_rows = _read_jsonl_required(shadow_root / "corpus-ledger.jsonl", errors)
    baseline_chunks = _read_jsonl_required(baseline_root / "chunks.jsonl", errors)
    shadow_chunks = _read_jsonl_required(shadow_root / "chunks.jsonl", errors)
    id_map = _read_json_artifact(shadow_root / "old-to-new-id-map.json", errors, default={"rows": []}, required=True)
    shadow_lineage = _read_json_artifact(shadow_root / "index-lineage.json", errors, default={"rows": []}, required=True)
    reviewed_drops = _read_json_artifact(shadow_root / "reviewed-drops.json", errors, default={"rows": []}, required=False)
    questions = _read_jsonl_required(questions_path, errors)
    _check_shadow_lineage(
        shadow_lineage,
        errors,
        expected_keyword_index=expected_keyword_index,
        expected_qdrant_collection=expected_qdrant_collection,
    )

    baseline_ids = {str(row.get("file_id")) for row in baseline_rows if row.get("file_id") is not None}
    shadow_ids = {str(row.get("file_id")) for row in shadow_rows if row.get("file_id") is not None}
    errors.extend(f"baseline file missing from shadow ledger: {file_id}" for file_id in sorted(baseline_ids - shadow_ids))
    shadow_status_by_id = {str(row.get("file_id")): str(row.get("status") or "") for row in shadow_rows}
    for row in baseline_rows:
        file_id = str(row.get("file_id") or "")
        if row.get("status") == "indexed" and shadow_status_by_id.get(file_id) in {"failed", "no_text"}:
            errors.append(
                f"previously indexed file regressed to {shadow_status_by_id[file_id]} without approval: {file_id}"
            )

    baseline_indexed = sum(1 for row in baseline_rows if row.get("status") == "indexed")
    shadow_indexed = sum(1 for row in shadow_rows if row.get("status") == "indexed")
    if shadow_indexed < baseline_indexed:
        errors.append(f"shadow indexed file count regressed: {shadow_indexed} < {baseline_indexed}")

    shadow_chunk_count = sum(int(row.get("chunk_count") or 0) for row in shadow_rows)
    if shadow_chunk_count == 0:
        errors.append("shadow chunk count is zero")

    strict_integrity = "chunk_count" in shadow_lineage
    if strict_integrity:
        _check_chunk_integrity(shadow_chunks, shadow_rows, shadow_lineage, errors)
        _check_ocr_integrity(shadow_chunks, shadow_lineage, errors)

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
    if strict_integrity:
        _check_id_map_targets(id_map, shadow_chunks, errors)
    unmapped_old_chunk_ids = baseline_chunk_ids - mapped_old_chunk_ids - reviewed_drop_chunk_ids
    errors.extend(
        f"old chunk has no replacement or reviewed drop: {old_chunk_id}"
        for old_chunk_id in sorted(unmapped_old_chunk_ids)
    )

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
        "shadow_chunk_rows": len(shadow_chunks),
        "shadow_locator_coverage": locator_coverage,
        "mapped_old_chunk_ids": len(mapped_old_chunk_ids),
        "reviewed_drop_chunk_ids": len(reviewed_drop_chunk_ids),
        "unmapped_old_chunk_ids": len(unmapped_old_chunk_ids),
        "shadow_index_version": shadow_lineage.get("index_version"),
        "shadow_keyword_index": shadow_lineage.get("keyword_index"),
        "shadow_qdrant_collection": shadow_lineage.get("qdrant_collection"),
    }
    return PromotionGateResult(passed=not errors, errors=errors, summary=summary)


def _check_chunk_integrity(
    chunks: list[dict],
    ledger_rows: list[dict],
    lineage: dict[str, Any],
    errors: list[str],
) -> None:
    ledger_count = sum(int(row.get("chunk_count") or 0) for row in ledger_rows)
    chunk_ids = _metadata_value_list(chunks, "chunk_id")
    citation_ids = _metadata_value_list(chunks, "citation_id")
    locators = [
        f"{(row.get('metadata') or {}).get('file_id')}:{(row.get('metadata') or {}).get('source_locator')}"
        for row in chunks
    ]
    expected_counts = {
        "ledger chunk count": ledger_count,
        "lineage chunk count": int(lineage.get("chunk_count") or 0),
        "lineage keyword document count": int(lineage.get("keyword_document_count") or 0),
    }
    if lineage.get("vector_indexed") is True:
        expected_counts["lineage vector document count"] = int(lineage.get("vector_document_count") or 0)
    for label, count in expected_counts.items():
        if count != len(chunks):
            errors.append(f"{label} mismatch: {count} != {len(chunks)}")
    _check_unique_complete("chunk_id", chunk_ids, len(chunks), errors)
    _check_unique_complete("citation_id", citation_ids, len(chunks), errors)
    _check_unique_complete("source_locator", locators, len(chunks), errors)


def _check_unique_complete(label: str, values: list[str], row_count: int, errors: list[str]) -> None:
    if len(values) != row_count:
        errors.append(f"shadow {label} coverage mismatch: {len(values)} != {row_count}")
    if len(set(values)) != len(values):
        errors.append(f"shadow {label} values are not unique")


def _check_ocr_integrity(chunks: list[dict], lineage: dict[str, Any], errors: list[str]) -> None:
    expected_model = str(lineage.get("ocr_model") or "")
    expected_schema = str(lineage.get("ocr_recipe_schema") or "")
    for index, row in enumerate(chunks):
        metadata = dict(row.get("metadata") or {})
        method = str(metadata.get("ocr_method") or "")
        if not method:
            continue
        if expected_model and expected_model not in method:
            errors.append(f"OCR chunk {index} model mismatch: {method} does not contain {expected_model}")
        if expected_schema and str(metadata.get("ocr_recipe_schema") or "") != expected_schema:
            errors.append(f"OCR chunk {index} recipe schema mismatch")
        if not str(metadata.get("ocr_recipe_hash") or "").startswith("sha256:"):
            errors.append(f"OCR chunk {index} missing recipe hash")
        text = str(row.get("page_content") or "").casefold()
        if "no visible text" in text or "no readable text" in text or "текст не обнаружен" in text:
            errors.append(f"OCR chunk {index} contains canned no-text output")


def _check_id_map_targets(payload: dict, chunks: list[dict], errors: list[str]) -> None:
    targets = {
        str((row.get("metadata") or {}).get("chunk_id")): row
        for row in chunks
        if (row.get("metadata") or {}).get("chunk_id")
    }
    for index, mapping in enumerate(payload.get("rows", [])):
        target_id = mapping.get("new_chunk_id")
        if not target_id:
            continue
        target = targets.get(str(target_id))
        if target is None:
            errors.append(f"ID map row {index} target does not exist: {target_id}")
            continue
        expected_hash = str(mapping.get("new_content_sha256") or "")
        if expected_hash:
            import hashlib

            actual_hash = hashlib.sha256(str(target.get("page_content") or "").encode("utf-8")).hexdigest()
            if actual_hash != expected_hash:
                errors.append(f"ID map row {index} target content hash mismatch")


def _metadata_value_list(rows: list[dict], key: str) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = (row.get("metadata") or {}).get(key)
        if value is not None and str(value).strip():
            values.append(str(value))
    return values


def _check_shadow_lineage(
    lineage: dict,
    errors: list[str],
    *,
    expected_keyword_index: str | None,
    expected_qdrant_collection: str | None,
) -> None:
    if not lineage:
        return
    errors.extend(
        f"shadow lineage missing field: {field}"
        for field in ("ingest_run_id", "corpus_version", "index_version", "keyword_index")
        if not str(lineage.get(field) or "").strip()
    )
    if lineage.get("keyword_indexed") is not True:
        errors.append("shadow lineage was not keyword indexed")
    if expected_keyword_index is not None and str(lineage.get("keyword_index") or "") != expected_keyword_index:
        errors.append(
            f"shadow lineage keyword index mismatch: {lineage.get('keyword_index')} != {expected_keyword_index}"
        )
    if expected_qdrant_collection is not None:
        if lineage.get("vector_indexed") is not True:
            errors.append("shadow lineage was not vector indexed")
        if str(lineage.get("qdrant_collection") or "") != expected_qdrant_collection:
            errors.append(
                "shadow lineage Qdrant collection mismatch: "
                f"{lineage.get('qdrant_collection')} != {expected_qdrant_collection}"
            )


def _read_jsonl_required(path: Path, errors: list[str]) -> list[dict]:
    if not path.exists():
        errors.append(f"required artifact missing: {path}")
        return []
    try:
        return read_jsonl(path)
    except json.JSONDecodeError as exc:
        errors.append(f"required artifact is not valid JSONL: {path}: {exc}")
        return []


def _read_json_artifact(path: Path, errors: list[str], *, default: dict[str, Any], required: bool) -> dict[str, Any]:
    if not path.exists():
        if required:
            errors.append(f"required artifact missing: {path}")
        return dict(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        artifact_type = "required" if required else "optional"
        errors.append(f"{artifact_type} artifact is not valid JSON: {path}: {exc}")
        return dict(default)


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
