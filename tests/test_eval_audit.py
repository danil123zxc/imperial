from __future__ import annotations

import json
import importlib.util
from pathlib import Path


def test_audit_detects_gold_id_that_points_away_from_existing_source(tmp_path):
    from imperial_rag.evals.audit import audit_eval_rows, load_corpus_index

    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks_path,
        [
            _chunk(
                "timesheets",
                relative_path="11. РЕГЛАМЕНТЫ/ПРИКАЗ О табелях и мотивациях.pdf",
                file_name="ПРИКАЗ О табелях и мотивациях.pdf",
                text="Правила оформления табелей и мотивационных листов при отсутствии на рабочем месте.",
            )
        ],
    )
    documents_root = tmp_path / "documents"
    source_path = documents_root / "11. РЕГЛАМЕНТЫ" / "1. БЛАНКИ" / "Акт об отсутствии на рабочем месте.doc"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("source placeholder", encoding="utf-8")

    audit = audit_eval_rows(
        [
            {
                "id": "imperial-cite-003",
                "suite": "imperial_gold_core",
                "tags": ["hr", "absence", "gold_context"],
                "question": "Что оформить при отсутствии сотрудника?",
                "expected_behavior": "cite_answer",
                "expected_source_hints": ["Акт об отсутствии", "рабочем месте"],
                "reference_context_ids": ["timesheets"],
                "reference_answer": "При отсутствии сотрудника нужно оформить акт об отсутствии.",
            }
        ],
        corpus_index=load_corpus_index(chunks_path),
        documents_root=documents_root,
    )

    assert audit == [
        {
            "id": "imperial-cite-003",
            "expected_behavior": "cite_answer",
            "lane": "indexed_answerability",
            "current_reference_context_ids": ["timesheets"],
            "resolved_chunk_ids": [],
            "file_only_reference_context_ids": ["timesheets"],
            "unresolved_reference_context_ids": [],
            "resolved_indexed_file_ids": [],
            "candidate_chunk_ids": ["timesheets:chunk-0"],
            "candidate_file_ids": [],
            "candidate_locators": [
                {
                    "chunk_id": "timesheets:chunk-0",
                    "citation_id": "ПРИКАЗ О табелях и мотивациях.pdf#chunk-0",
                    "file_id": "timesheets",
                    "relative_path": "11. РЕГЛАМЕНТЫ/ПРИКАЗ О табелях и мотивациях.pdf",
                    "file_name": "ПРИКАЗ О табелях и мотивациях.pdf",
                    "chunk_index": 0,
                    "page_number": "",
                    "section_heading": "",
                    "source_locator": "",
                    "evidence_quote": (
                        "Правила оформления табелей и мотивационных листов при отсутствии на рабочем месте."
                    ),
                },
            ],
            "source_path": "11. РЕГЛАМЕНТЫ/1. БЛАНКИ/Акт об отсутствии на рабочем месте.doc",
            "indexed_status": "file_only_gold_ids",
            "reference_answer_quality": "evidence_shaped",
            "expected_source_hints_quality": "source_path_only",
            "action": "rewrite",
            "quarantine_reason": "file_only_reference_context_ids",
            "backlog_category": "chunk_id_migration",
            "notes": ["reference_context_ids resolve only as file IDs; migrate them to chunk IDs"],
        }
    ]


def test_audit_routes_conflict_and_out_of_corpus_refusal_rows(tmp_path):
    from imperial_rag.evals.audit import audit_eval_rows, load_corpus_index

    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks_path,
        [
            _chunk(
                "warehouse-v1",
                relative_path="11. РЕГЛАМЕНТЫ/РЕГЛАМЕНТ СКЛАДА/НОВЫЙ РЕГЛАМЕНТ СКЛАДА.docx",
                file_name="НОВЫЙ РЕГЛАМЕНТ СКЛАДА.docx",
                text="Версия регламента склада про приемку.",
            ),
            _chunk(
                "warehouse-v2",
                relative_path="11. РЕГЛАМЕНТЫ/РЕГЛАМЕНТ СКЛАДА/НОВЫЙ РЕГЛАМЕНТ СКЛАДА ЧТО ЗАМЕНИТЬ.docx",
                file_name="НОВЫЙ РЕГЛАМЕНТ СКЛАДА ЧТО ЗАМЕНИТЬ.docx",
                text="Другая версия регламента склада.",
            ),
        ],
    )

    audit = audit_eval_rows(
        [
            {
                "id": "imperial-conflict-001",
                "suite": "imperial_gold_core",
                "tags": ["warehouse", "conflict"],
                "question": "Какая версия регламента склада действует?",
                "expected_behavior": "surface_conflict",
                "expected_source_hints": ["РЕГЛАМЕНТ СКЛАДА"],
                "reference_context_ids": None,
                "reference_answer": "Ответ должен явно показать конфликт версий регламента склада.",
            },
            {
                "id": "imperial-refuse-001",
                "suite": "imperial_gold_core",
                "tags": ["out_of_corpus", "science"],
                "question": "Какая температура плавления вольфрама?",
                "expected_behavior": "refuse_if_not_found",
                "expected_source_hints": [],
                "reference_context_ids": None,
                "reference_answer": "В проиндексированных документах Imperial нет ответа о температуре плавления.",
            },
        ],
        corpus_index=load_corpus_index(chunks_path),
        documents_root=tmp_path / "documents",
    )

    by_id = {row["id"]: row for row in audit}
    assert by_id["imperial-conflict-001"]["lane"] == "conflict_version_behavior"
    assert set(by_id["imperial-conflict-001"]["candidate_file_ids"]) == {"warehouse-v1", "warehouse-v2"}
    assert by_id["imperial-conflict-001"]["indexed_status"] == "candidate_indexed"
    assert by_id["imperial-conflict-001"]["reference_answer_quality"] == "generic_meta_reference"
    assert by_id["imperial-conflict-001"]["action"] == "rewrite"
    assert by_id["imperial-conflict-001"]["backlog_category"] == "gold_id_backfill"

    assert by_id["imperial-refuse-001"]["lane"] == "refusal_out_of_corpus_behavior"
    assert by_id["imperial-refuse-001"]["indexed_status"] == "out_of_corpus"
    assert by_id["imperial-refuse-001"]["expected_source_hints_quality"] == "not_required"
    assert by_id["imperial-refuse-001"]["action"] == "keep"
    assert by_id["imperial-refuse-001"]["backlog_category"] == "none"


def test_eval_contract_validator_reports_missing_gold_ids_and_unsupported_metrics(tmp_path):
    from imperial_rag.evals.audit import audit_eval_rows, load_corpus_index, validate_eval_contract

    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks_path,
        [
            _chunk(
                "warehouse-v1",
                relative_path="РЕГЛАМЕНТ СКЛАДА/НОВЫЙ РЕГЛАМЕНТ СКЛАДА.docx",
                file_name="НОВЫЙ РЕГЛАМЕНТ СКЛАДА.docx",
                text="Версия регламента склада.",
            )
        ],
    )
    audit = audit_eval_rows(
        [
            {
                "id": "imperial-conflict-001",
                "suite": "imperial_gold_core",
                "tags": ["warehouse", "conflict"],
                "question": "Какая версия регламента склада действует?",
                "expected_behavior": "surface_conflict",
                "expected_source_hints": ["РЕГЛАМЕНТ СКЛАДА"],
                "reference_context_ids": [],
                "reference_answer": "Ответ должен показать конфликт.",
            }
        ],
        corpus_index=load_corpus_index(chunks_path),
        documents_root=tmp_path / "documents",
    )

    findings = validate_eval_contract(audit, phoenix_metric_names=["faithfulness", "factual_correctness"])

    assert {
        "severity": "error",
        "row_id": "imperial-conflict-001",
        "code": "missing_required_reference_context_ids",
        "message": "conflict_version_behavior rows require at least two resolving reference_context_ids unless quarantined",
    } in findings
    assert {
        "severity": "error",
        "row_id": None,
        "code": "unsupported_phoenix_metric",
        "message": "Phoenix evaluator path does not support factual_correctness",
    } in findings


def test_eval_contract_validator_rejects_file_only_references(tmp_path):
    from imperial_rag.evals.audit import audit_eval_rows, load_corpus_index, validate_eval_contract

    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks_path,
        [
            _chunk(
                "file-a",
                chunk_id="chunk-a",
                citation_id="source.docx#body:chunk-0",
                relative_path="documents/source.docx",
                file_name="source.docx",
                text="Регламент источника.",
            )
        ],
    )

    audit = audit_eval_rows(
        [
            {
                "id": "imperial-cite-001",
                "suite": "imperial_gold_core",
                "tags": ["returns"],
                "question": "Как оформить возврат?",
                "expected_behavior": "cite_answer",
                "expected_source_hints": ["Регламент источника"],
                "reference_context_ids": ["file-a"],
                "reference_answer": "Возврат оформляется по регламенту.",
            }
        ],
        corpus_index=load_corpus_index(chunks_path),
        documents_root=tmp_path / "documents",
    )

    assert audit[0]["resolved_chunk_ids"] == []
    assert audit[0]["file_only_reference_context_ids"] == ["file-a"]
    assert audit[0]["candidate_chunk_ids"] == ["chunk-a"]
    findings = validate_eval_contract(audit)
    assert {
        "severity": "error",
        "row_id": "imperial-cite-001",
        "code": "file_only_reference_context_ids",
        "message": "reference_context_ids must identify chunks, not files",
    } in findings


def test_eval_contract_validator_rejects_invalid_or_mismatched_lanes(tmp_path):
    from imperial_rag.evals.audit import audit_eval_rows, load_corpus_index, validate_eval_contract

    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks_path,
        [
            _chunk(
                "file-a",
                relative_path="documents/source.docx",
                file_name="source.docx",
                text="Регламент источника.",
            )
        ],
    )
    audit = audit_eval_rows(
        [
            {
                "id": "imperial-cite-001",
                "suite": "imperial_gold_core",
                "tags": ["returns"],
                "lane": "refusal_out_of_corpus_behavior",
                "question": "Как оформить возврат?",
                "expected_behavior": "cite_answer",
                "expected_source_hints": ["source"],
                "reference_context_ids": ["file-a"],
                "reference_answer": "Возврат оформляется по регламенту.",
            },
            {
                "id": "imperial-cite-002",
                "suite": "imperial_gold_core",
                "tags": ["returns"],
                "lane": "not_a_lane",
                "question": "Как оформить возврат?",
                "expected_behavior": "cite_answer",
                "expected_source_hints": ["source"],
                "reference_context_ids": ["file-a"],
                "reference_answer": "Возврат оформляется по регламенту.",
            },
        ],
        corpus_index=load_corpus_index(chunks_path),
        documents_root=tmp_path / "documents",
    )

    findings = validate_eval_contract(audit)

    assert {
        "severity": "error",
        "row_id": "imperial-cite-001",
        "code": "lane_expected_behavior_mismatch",
        "message": "cite_answer rows must use indexed_answerability or known_missing_document_coverage lanes",
    } in findings
    assert {
        "severity": "error",
        "row_id": "imperial-cite-002",
        "code": "invalid_lane",
        "message": "Invalid eval lane 'not_a_lane'",
    } in findings


def test_row_level_quarantine_allows_known_bad_gold_contract(tmp_path):
    from imperial_rag.evals.audit import audit_eval_rows, load_corpus_index, validate_eval_contract

    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks_path,
        [
            _chunk(
                "timesheets",
                relative_path="11. РЕГЛАМЕНТЫ/ПРИКАЗ О табелях и мотивациях.pdf",
                file_name="ПРИКАЗ О табелях и мотивациях.pdf",
                text="Правила оформления табелей.",
            )
        ],
    )
    documents_root = tmp_path / "documents"
    source_path = documents_root / "11. РЕГЛАМЕНТЫ" / "1. БЛАНКИ" / "Акт об отсутствии на рабочем месте.doc"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("source placeholder", encoding="utf-8")

    audit = audit_eval_rows(
        [
            {
                "id": "imperial-cite-003",
                "suite": "imperial_gold_core",
                "tags": ["hr", "absence", "gold_context"],
                "lane": "indexed_answerability",
                "question": "Что оформить при отсутствии сотрудника?",
                "expected_behavior": "cite_answer",
                "expected_source_hints": ["Акт об отсутствии", "рабочем месте"],
                "reference_context_ids": [],
                "reference_answer": "При отсутствии сотрудника нужно оформить акт об отсутствии.",
                "quarantine_reason": "source_document_exists_but_is_not_indexed",
            }
        ],
        corpus_index=load_corpus_index(chunks_path),
        documents_root=documents_root,
    )

    assert audit[0]["action"] == "quarantine"
    assert audit[0]["quarantine_reason"] == "source_document_exists_but_is_not_indexed"
    assert validate_eval_contract(audit) == []


def test_audit_cli_writes_artifacts_and_findings(tmp_path):
    module = _load_audit_script()
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "row-audit.jsonl"
    markdown_path = tmp_path / "row-audit.md"
    findings_path = tmp_path / "row-audit-findings.jsonl"
    _write_jsonl(
        questions_path,
        [
            {
                "id": "imperial-conflict-001",
                "suite": "imperial_gold_core",
                "tags": ["warehouse", "conflict"],
                "question": "Какая версия регламента склада действует?",
                "expected_behavior": "surface_conflict",
                "expected_source_hints": ["РЕГЛАМЕНТ СКЛАДА"],
                "reference_context_ids": [],
                "reference_answer": "Ответ должен показать конфликт.",
            }
        ],
    )
    _write_jsonl(
        chunks_path,
        [
            _chunk(
                "warehouse-v1",
                relative_path="РЕГЛАМЕНТ СКЛАДА/НОВЫЙ РЕГЛАМЕНТ СКЛАДА.docx",
                file_name="НОВЫЙ РЕГЛАМЕНТ СКЛАДА.docx",
                text="Версия регламента склада.",
            )
        ],
    )

    exit_code = module.main(
        [
            "--questions-path",
            str(questions_path),
            "--chunks-path",
            str(chunks_path),
            "--documents-root",
            str(tmp_path / "documents"),
            "--output-path",
            str(output_path),
            "--markdown-path",
            str(markdown_path),
            "--findings-path",
            str(findings_path),
            "--phoenix-metrics",
            "faithfulness,factual_correctness",
        ]
    )

    assert exit_code == 0
    audit_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    findings = [json.loads(line) for line in findings_path.read_text(encoding="utf-8").splitlines()]
    assert audit_rows[0]["id"] == "imperial-conflict-001"
    assert audit_rows[0]["lane"] == "conflict_version_behavior"
    assert "| id | expected_behavior | lane |" in markdown_path.read_text(encoding="utf-8")
    assert {finding["code"] for finding in findings} == {
        "missing_required_reference_context_ids",
        "unsupported_phoenix_metric",
    }


def _chunk(
    file_id: str,
    *,
    relative_path: str,
    file_name: str,
    text: str,
    chunk_id: str | None = None,
    citation_id: str | None = None,
) -> dict[str, object]:
    return {
        "page_content": text,
        "metadata": {
            "file_id": file_id,
            "chunk_id": chunk_id or f"{file_id}:chunk-0",
            "citation_id": citation_id or f"{file_name}#chunk-0",
            "relative_path": relative_path,
            "file_name": file_name,
            "file_path": f"/private/documents/{relative_path}",
            "parent_folder": str(Path(relative_path).parent),
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _load_audit_script():
    spec = importlib.util.spec_from_file_location("audit_eval_rows_for_test", Path("scripts/audit_eval_rows.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
