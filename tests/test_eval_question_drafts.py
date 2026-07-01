from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def test_build_question_drafts_marks_rows_review_required():
    module = _load_draft_runner()
    chunks = [
        {
            "page_content": "Регламент описывает возврат брака из магазина.",
            "metadata": {
                "file_name": "РЕГЛАМЕНТ О БРАКЕ.docx",
                "relative_path": "11/РЕГЛАМЕНТ О БРАКЕ.docx",
                "citation_id": "11/РЕГЛАМЕНТ О БРАКЕ.docx#body:chunk-0",
            },
        }
    ]

    drafts = module.build_question_drafts(chunks, limit=1)

    assert drafts == [
        {
            "question": "Какие правила описаны в документе РЕГЛАМЕНТ О БРАКЕ?",
            "expected_behavior": "draft_review_required",
            "expected_source_hints": ["РЕГЛАМЕНТ О БРАКЕ.docx", "11/РЕГЛАМЕНТ О БРАКЕ.docx"],
            "reference_answer": "",
            "review_status": "draft",
            "draft_source_citation_id": "11/РЕГЛАМЕНТ О БРАКЕ.docx#body:chunk-0",
            "generated_by": "imperial_eval_question_drafts_v1",
        }
    ]


def test_write_question_drafts_uses_jsonl_without_touching_gold_file(tmp_path):
    module = _load_draft_runner()
    output_path = tmp_path / "drafts.jsonl"
    gold_path = tmp_path / "questions.jsonl"
    gold_path.write_text('{"question":"existing"}\n', encoding="utf-8")

    module.write_jsonl(
        output_path,
        [
            {
                "question": "Какие правила описаны в документе Регламент?",
                "expected_behavior": "draft_review_required",
                "expected_source_hints": ["Регламент"],
                "reference_answer": "",
                "review_status": "draft",
            }
        ],
    )

    assert json.loads(output_path.read_text(encoding="utf-8"))["review_status"] == "draft"
    assert gold_path.read_text(encoding="utf-8") == '{"question":"existing"}\n'


def _load_draft_runner():
    spec = importlib.util.spec_from_file_location(
        "draft_eval_questions_for_test",
        Path("scripts/draft_eval_questions.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
