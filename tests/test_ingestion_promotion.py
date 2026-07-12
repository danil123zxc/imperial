from __future__ import annotations

import json
from pathlib import Path

from imperial_rag.ingestion.promotion import PromotionGateResult, check_promotion_gates


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_chunks(path: Path, rows: list[dict]) -> None:
    _write_jsonl(path, [{"page_content": row.get("page_content", "text"), "metadata": row["metadata"]} for row in rows])


def _write_lineage(
    path: Path,
    *,
    keyword_index: str = "keyword-shadow",
    qdrant_collection: str = "qdrant-shadow",
) -> None:
    path.write_text(
        json.dumps(
            {
                "ingest_run_id": "ingest-test",
                "corpus_version": "corpus_sha256:test",
                "index_version": "index_sha256:test",
                "keyword_index": keyword_index,
                "qdrant_collection": qdrant_collection,
                "embedding_model": "text-embedding-v4:2048",
                "keyword_indexed": True,
                "vector_indexed": True,
            }
        ),
        encoding="utf-8",
    )


def test_check_promotion_gates_accepts_improved_shadow(tmp_path):
    baseline = tmp_path / "baseline"
    shadow = tmp_path / "shadow"
    baseline.mkdir()
    shadow.mkdir()
    _write_jsonl(
        baseline / "corpus-ledger.jsonl",
        [
            {
                "file_id": "file-a",
                "status": "indexed",
                "chunk_count": 1,
                "locator_coverage": 0.0,
                "index_inclusion_reason": "indexable",
            }
        ],
    )
    _write_jsonl(
        shadow / "corpus-ledger.jsonl",
        [
            {
                "file_id": "file-a",
                "status": "indexed",
                "chunk_count": 2,
                "locator_coverage": 1.0,
                "index_inclusion_reason": "indexable",
            }
        ],
    )
    _write_chunks(
        baseline / "chunks.jsonl",
        [{"metadata": {"file_id": "file-a", "chunk_id": "old", "citation_id": "old-citation"}}],
    )
    _write_chunks(
        shadow / "chunks.jsonl",
        [{"metadata": {"file_id": "file-a", "chunk_id": "new", "citation_id": "new-citation"}}],
    )
    (shadow / "old-to-new-id-map.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "file_id": "file-a",
                        "old_chunk_id": "old",
                        "old_citation_id": "old-citation",
                        "new_chunk_id": "new",
                        "new_citation_id": "new-citation",
                        "status": "mapped",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_lineage(shadow / "index-lineage.json")
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(questions, [{"id": "q1", "reference_context_ids": ["file-a"]}])

    result = check_promotion_gates(
        baseline,
        shadow,
        questions_path=questions,
        expected_keyword_index="keyword-shadow",
        expected_qdrant_collection="qdrant-shadow",
    )

    assert isinstance(result, PromotionGateResult)
    assert result.passed is True
    assert result.errors == []
    assert result.summary["shadow_index_version"] == "index_sha256:test"


def test_check_promotion_gates_rejects_missing_shadow_lineage(tmp_path):
    baseline = tmp_path / "baseline"
    shadow = tmp_path / "shadow"
    baseline.mkdir()
    shadow.mkdir()
    _write_jsonl(baseline / "corpus-ledger.jsonl", [{"file_id": "file-a", "status": "indexed", "chunk_count": 1}])
    _write_jsonl(
        shadow / "corpus-ledger.jsonl",
        [{"file_id": "file-a", "status": "indexed", "chunk_count": 1, "locator_coverage": 1.0}],
    )
    _write_chunks(baseline / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "old-1"}}])
    _write_chunks(shadow / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "new-1"}}])
    (shadow / "old-to-new-id-map.json").write_text(
        json.dumps({"rows": [{"file_id": "file-a", "old_chunk_id": "old-1", "new_chunk_id": "new-1"}]}),
        encoding="utf-8",
    )
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(questions, [])

    result = check_promotion_gates(baseline, shadow, questions_path=questions)

    assert result.passed is False
    assert any("required artifact missing" in error and "index-lineage.json" in error for error in result.errors)


def test_check_promotion_gates_rejects_unexpected_shadow_lineage_targets(tmp_path):
    baseline = tmp_path / "baseline"
    shadow = tmp_path / "shadow"
    baseline.mkdir()
    shadow.mkdir()
    _write_jsonl(baseline / "corpus-ledger.jsonl", [{"file_id": "file-a", "status": "indexed", "chunk_count": 1}])
    _write_jsonl(
        shadow / "corpus-ledger.jsonl",
        [{"file_id": "file-a", "status": "indexed", "chunk_count": 1, "locator_coverage": 1.0}],
    )
    _write_chunks(baseline / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "old-1"}}])
    _write_chunks(shadow / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "new-1"}}])
    (shadow / "old-to-new-id-map.json").write_text(
        json.dumps({"rows": [{"file_id": "file-a", "old_chunk_id": "old-1", "new_chunk_id": "new-1"}]}),
        encoding="utf-8",
    )
    _write_lineage(shadow / "index-lineage.json", keyword_index="wrong-keyword", qdrant_collection="wrong-qdrant")
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(questions, [])

    result = check_promotion_gates(
        baseline,
        shadow,
        questions_path=questions,
        expected_keyword_index="keyword-shadow",
        expected_qdrant_collection="qdrant-shadow",
    )

    assert result.passed is False
    assert "shadow lineage keyword index mismatch: wrong-keyword != keyword-shadow" in result.errors
    assert "shadow lineage Qdrant collection mismatch: wrong-qdrant != qdrant-shadow" in result.errors


def test_check_promotion_gates_rejects_missing_gold_reference_id(tmp_path):
    baseline = tmp_path / "baseline"
    shadow = tmp_path / "shadow"
    baseline.mkdir()
    shadow.mkdir()
    _write_jsonl(baseline / "corpus-ledger.jsonl", [{"file_id": "file-a", "status": "indexed", "chunk_count": 1}])
    _write_jsonl(shadow / "corpus-ledger.jsonl", [{"file_id": "file-b", "status": "indexed", "chunk_count": 1}])
    _write_chunks(baseline / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "old"}}])
    _write_chunks(shadow / "chunks.jsonl", [{"metadata": {"file_id": "file-b", "chunk_id": "new"}}])
    (shadow / "old-to-new-id-map.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(questions, [{"id": "q1", "reference_context_ids": ["file-a"]}])

    result = check_promotion_gates(baseline, shadow, questions_path=questions)

    assert result.passed is False
    assert "gold reference_context_id missing from shadow ledger: file-a" in result.errors


def test_check_promotion_gates_rejects_self_comparison(tmp_path):
    root = tmp_path / "same"
    root.mkdir()
    _write_jsonl(root / "corpus-ledger.jsonl", [{"file_id": "file-a", "status": "indexed", "chunk_count": 1}])
    _write_chunks(root / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "old"}}])
    (root / "old-to-new-id-map.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(questions, [])

    result = check_promotion_gates(root, root, questions_path=questions)

    assert result.passed is False
    assert "baseline and shadow roots must be different" in result.errors


def test_check_promotion_gates_reports_missing_required_artifacts(tmp_path):
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(questions, [])

    result = check_promotion_gates(tmp_path / "baseline", tmp_path / "shadow", questions_path=questions)

    assert result.passed is False
    assert any("required artifact missing" in error for error in result.errors)


def test_check_promotion_gates_rejects_partial_old_to_new_id_map(tmp_path):
    baseline = tmp_path / "baseline"
    shadow = tmp_path / "shadow"
    baseline.mkdir()
    shadow.mkdir()
    _write_jsonl(baseline / "corpus-ledger.jsonl", [{"file_id": "file-a", "status": "indexed", "chunk_count": 2}])
    _write_jsonl(
        shadow / "corpus-ledger.jsonl",
        [{"file_id": "file-a", "status": "indexed", "chunk_count": 1, "locator_coverage": 1.0}],
    )
    _write_chunks(
        baseline / "chunks.jsonl",
        [
            {"metadata": {"file_id": "file-a", "chunk_id": "old-1"}},
            {"metadata": {"file_id": "file-a", "chunk_id": "old-2"}},
        ],
    )
    _write_chunks(shadow / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "new-1"}}])
    (shadow / "old-to-new-id-map.json").write_text(
        json.dumps({"rows": [{"file_id": "file-a", "old_chunk_id": "old-1", "new_chunk_id": "new-1", "status": "mapped"}]}),
        encoding="utf-8",
    )
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(questions, [])

    result = check_promotion_gates(baseline, shadow, questions_path=questions)

    assert result.passed is False
    assert "old chunk has no replacement or reviewed drop: old-2" in result.errors


def test_check_promotion_gates_accepts_reviewed_drop(tmp_path):
    baseline = tmp_path / "baseline"
    shadow = tmp_path / "shadow"
    baseline.mkdir()
    shadow.mkdir()
    _write_jsonl(baseline / "corpus-ledger.jsonl", [{"file_id": "file-a", "status": "indexed", "chunk_count": 1}])
    _write_jsonl(
        shadow / "corpus-ledger.jsonl",
        [{"file_id": "file-a", "status": "indexed", "chunk_count": 1, "locator_coverage": 1.0}],
    )
    _write_chunks(baseline / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "old-1"}}])
    _write_chunks(shadow / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "new-1"}}])
    (shadow / "old-to-new-id-map.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    (shadow / "reviewed-drops.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "old_chunk_id": "old-1",
                        "reason": "duplicate content merged into new-1",
                        "reviewed_by": "migration-review",
                        "reviewed_at": "2026-06-25T00:00:00Z",
                        "rollback_impact": "citation redirects to merged chunk",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_lineage(shadow / "index-lineage.json")
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(questions, [])

    result = check_promotion_gates(baseline, shadow, questions_path=questions)

    assert result.passed is True
    assert result.errors == []


def test_check_promotion_gates_reports_invalid_optional_reviewed_drops(tmp_path):
    baseline = tmp_path / "baseline"
    shadow = tmp_path / "shadow"
    baseline.mkdir()
    shadow.mkdir()
    _write_jsonl(baseline / "corpus-ledger.jsonl", [{"file_id": "file-a", "status": "indexed", "chunk_count": 1}])
    _write_jsonl(
        shadow / "corpus-ledger.jsonl",
        [{"file_id": "file-a", "status": "indexed", "chunk_count": 1, "locator_coverage": 1.0}],
    )
    _write_chunks(baseline / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "old-1"}}])
    _write_chunks(shadow / "chunks.jsonl", [{"metadata": {"file_id": "file-a", "chunk_id": "new-1"}}])
    (shadow / "old-to-new-id-map.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    (shadow / "reviewed-drops.json").write_text("{", encoding="utf-8")
    _write_lineage(shadow / "index-lineage.json")
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(questions, [])

    result = check_promotion_gates(baseline, shadow, questions_path=questions)

    assert result.passed is False
    assert any(
        "optional artifact is not valid JSON" in error and "reviewed-drops.json" in error
        for error in result.errors
    )


def test_check_promotion_gates_enforces_strict_chunk_and_index_parity(tmp_path):
    baseline = tmp_path / "baseline"
    shadow = tmp_path / "shadow"
    baseline.mkdir()
    shadow.mkdir()
    _write_jsonl(baseline / "corpus-ledger.jsonl", [{"file_id": "file-a", "status": "indexed", "chunk_count": 1}])
    _write_jsonl(
        shadow / "corpus-ledger.jsonl",
        [{"file_id": "file-a", "status": "indexed", "chunk_count": 1, "locator_coverage": 1.0}],
    )
    _write_chunks(
        baseline / "chunks.jsonl",
        [{"page_content": "same", "metadata": {"file_id": "file-a", "chunk_id": "old"}}],
    )
    _write_chunks(
        shadow / "chunks.jsonl",
        [
            {
                "page_content": "same",
                "metadata": {
                    "file_id": "file-a",
                    "chunk_id": "new",
                    "citation_id": "citation-new",
                    "source_locator": "body:1:chunk:0",
                },
            }
        ],
    )
    (shadow / "old-to-new-id-map.json").write_text(
        json.dumps({"rows": [{"old_chunk_id": "old", "new_chunk_id": "new"}]}),
        encoding="utf-8",
    )
    lineage = {
        "ingest_run_id": "run",
        "corpus_version": "corpus",
        "index_version": "index",
        "keyword_index": "keyword-shadow",
        "qdrant_collection": "qdrant-shadow",
        "keyword_indexed": True,
        "vector_indexed": True,
        "chunk_count": 1,
        "keyword_document_count": 2,
        "vector_document_count": 1,
    }
    (shadow / "index-lineage.json").write_text(json.dumps(lineage), encoding="utf-8")
    questions = tmp_path / "questions.jsonl"
    _write_jsonl(questions, [])

    result = check_promotion_gates(baseline, shadow, questions_path=questions)

    assert result.passed is False
    assert "lineage keyword document count mismatch: 2 != 1" in result.errors
