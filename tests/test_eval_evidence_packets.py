from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_build_evidence_packets_resolves_gold_context_chunks(tmp_path):
    from imperial_rag.evals.golden import build_evidence_packets, load_evidence_corpus

    chunks_path = tmp_path / "chunks.jsonl"
    _write_jsonl(
        chunks_path,
        [
            _chunk(
                "return-reg",
                chunk_id="return-reg:body::0:abc",
                citation_id="returns.docx#body:chunk-0",
                relative_path="11/returns.docx",
                file_name="returns.docx",
                text="Срок принятия решения о возврате товара компанией - 7 календарных дней.",
            ),
            _chunk(
                "return-reg",
                chunk_id="return-reg:body::1:def",
                citation_id="returns.docx#body:chunk-1",
                relative_path="11/returns.docx",
                file_name="returns.docx",
                text="Решение принимается совместно начальником склада и супервайзером.",
                chunk_index=1,
            ),
        ],
    )

    packets = build_evidence_packets(
        [
            {
                "id": "imperial-cite-001",
                "suite": "imperial_gold_core",
                "tags": ["returns"],
                "lane": "indexed_answerability",
                "question": "Как оформить возврат?",
                "expected_behavior": "cite_answer",
                "expected_source_hints": ["returns"],
                "reference_context_ids": ["return-reg:body::1:def"],
                "reference_answer": "Возврат оформляется по регламенту.",
            }
        ],
        corpus=load_evidence_corpus(chunks_path),
        audit_rows=[
            {
                "id": "imperial-cite-001",
                "lane": "indexed_answerability",
                "action": "keep",
                "quarantine_reason": "",
                "notes": [],
            }
        ],
    )

    assert packets == [
        {
            "id": "imperial-cite-001",
            "suite": "imperial_gold_core",
            "tags": ["returns"],
            "lane": "indexed_answerability",
            "expected_behavior": "cite_answer",
            "question": "Как оформить возврат?",
            "expected_source_hints": ["returns"],
            "current_reference_answer": "Возврат оформляется по регламенту.",
            "candidate_reference_answer": "",
            "current_reference_context_ids": ["return-reg:body::1:def"],
            "resolved_reference_context_ids": ["return-reg:body::1:def"],
            "unresolved_reference_context_ids": [],
            "gold_status": "ready_for_review",
            "evidence": [
                {
                    "file_id": "return-reg",
                    "chunk_id": "return-reg:body::1:def",
                    "citation_id": "returns.docx#body:chunk-1",
                    "relative_path": "11/returns.docx",
                    "file_name": "returns.docx",
                    "chunk_index": 1,
                    "text": "Решение принимается совместно начальником склада и супервайзером.",
                },
            ],
            "audit": {
                "id": "imperial-cite-001",
                "lane": "indexed_answerability",
                "action": "keep",
                "quarantine_reason": "",
                "notes": [],
            },
            "review_notes": [
                "Draft or revise reference_answer only from the resolved evidence chunks.",
                "Every substantive claim must map back to one or more chunk_id values.",
            ],
        }
    ]


def test_build_evidence_packets_keeps_refusal_rows_without_evidence(tmp_path):
    from imperial_rag.evals.golden import EvidenceCorpus, build_evidence_packets

    packets = build_evidence_packets(
        [
            {
                "id": "imperial-refuse-001",
                "suite": "imperial_gold_core",
                "tags": ["out_of_corpus"],
                "lane": "refusal_out_of_corpus_behavior",
                "question": "Какую температуру плавления имеет вольфрам?",
                "expected_behavior": "refuse_if_not_found",
                "expected_source_hints": [],
                "reference_answer": "В документах Imperial нет ответа.",
            }
        ],
        corpus=EvidenceCorpus(),
    )

    assert packets[0]["gold_status"] == "refusal_boundary"
    assert packets[0]["current_reference_context_ids"] == []
    assert packets[0]["evidence"] == []
    assert packets[0]["review_notes"] == [
        "Keep this as an out-of-corpus refusal. Do not add reference_context_ids or corpus facts."
    ]


def test_build_evidence_packets_marks_quarantined_rows_non_promotable(tmp_path):
    from imperial_rag.evals.golden import EvidenceCorpus, build_evidence_packets

    packets = build_evidence_packets(
        [
            {
                "id": "imperial-cite-003",
                "suite": "imperial_gold_core",
                "tags": ["hr", "quarantined"],
                "lane": "indexed_answerability",
                "question": "Что делать при отсутствии сотрудника?",
                "expected_behavior": "cite_answer",
                "expected_source_hints": ["Акт об отсутствии"],
                "reference_context_ids": [],
                "reference_answer": "Нужно оформить акт.",
                "quarantine_reason": "source_document_exists_but_is_not_indexed",
            }
        ],
        corpus=EvidenceCorpus(),
    )

    assert packets[0]["gold_status"] == "quarantined"
    assert packets[0]["review_notes"] == [
        "Do not promote this row as normal gold coverage until quarantine_reason is resolved.",
        "quarantine_reason: source_document_exists_but_is_not_indexed",
    ]


def test_generate_eval_evidence_packets_cli_writes_review_artifacts(tmp_path):
    module = _load_packet_script()
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "packets.jsonl"
    markdown_path = tmp_path / "packets.md"
    _write_jsonl(
        questions_path,
        [
            {
                "id": "imperial-cite-001",
                "suite": "imperial_gold_core",
                "tags": ["returns"],
                "lane": "indexed_answerability",
                "question": "Как оформить возврат?",
                "expected_behavior": "cite_answer",
                "expected_source_hints": ["returns"],
                "reference_context_ids": ["return-reg:body::0:abc"],
                "reference_answer": "Возврат оформляется по регламенту.",
            }
        ],
    )
    _write_jsonl(
        chunks_path,
        [
            _chunk(
                "return-reg",
                chunk_id="return-reg:body::0:abc",
                citation_id="returns.docx#body:chunk-0",
                relative_path="11/returns.docx",
                file_name="returns.docx",
                text="Срок принятия решения о возврате товара компанией - 7 календарных дней.",
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
        ]
    )

    packets = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    markdown = markdown_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert packets[0]["gold_status"] == "ready_for_review"
    assert packets[0]["audit"]["action"] == "keep"
    assert "## imperial-cite-001" in markdown
    assert "return-reg:body::0:abc" in markdown
    assert "Возврат оформляется по регламенту." in markdown


def test_generate_eval_evidence_packets_strict_fails_on_missing_non_quarantined_gold_context(tmp_path):
    module = _load_packet_script()
    questions_path = tmp_path / "questions.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    output_path = tmp_path / "packets.jsonl"
    _write_jsonl(
        questions_path,
        [
            {
                "id": "imperial-cite-001",
                "suite": "imperial_gold_core",
                "tags": ["returns"],
                "lane": "indexed_answerability",
                "question": "Как оформить возврат?",
                "expected_behavior": "cite_answer",
                "expected_source_hints": ["returns"],
                "reference_answer": "Возврат оформляется по регламенту.",
            }
        ],
    )
    _write_jsonl(chunks_path, [])

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
            "--strict",
        ]
    )

    packets = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert exit_code == 1
    assert packets[0]["gold_status"] == "quarantined"


def _chunk(
    file_id: str,
    *,
    chunk_id: str,
    citation_id: str,
    relative_path: str,
    file_name: str,
    text: str,
    chunk_index: int = 0,
) -> dict[str, object]:
    return {
        "page_content": text,
        "metadata": {
            "file_id": file_id,
            "chunk_id": chunk_id,
            "citation_id": citation_id,
            "relative_path": relative_path,
            "file_name": file_name,
            "file_path": f"/private/documents/{relative_path}",
            "parent_folder": str(Path(relative_path).parent),
            "chunk_index": chunk_index,
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _load_packet_script():
    spec = importlib.util.spec_from_file_location(
        "generate_eval_evidence_packets_for_test",
        Path("scripts/generate_eval_evidence_packets.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
