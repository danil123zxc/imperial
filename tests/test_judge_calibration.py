from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_calibration_rows_prepare_factual_correctness_inputs():
    module = _load_calibration_script()

    rows = module.prepare_calibration_rows(
        [
            {
                "id": "judge-calibration-001",
                "question": "Как оформить возврат?",
                "reference_answer": "Нужно оформить документы.",
                "candidate_answer": "Оформить документы.",
                "human_label": "correct",
                "expected_behavior": "cite_answer",
                "lane": "indexed_answerability",
            }
        ]
    )

    assert rows == [
        {
            "id": "judge-calibration-001",
            "user_input": "Как оформить возврат?",
            "response": "Оформить документы.",
            "reference": "Нужно оформить документы.",
            "human_label": "correct",
            "expected_behavior": "cite_answer",
            "lane": "indexed_answerability",
        }
    ]


def test_calibration_summary_reports_accuracy_confusion_and_separation():
    module = _load_calibration_script()
    rows = [
        {"id": "row-1", "human_label": "correct"},
        {"id": "row-2", "human_label": "incorrect"},
        {"id": "row-3", "human_label": "incorrect", "lane": "conflict_version_behavior"},
    ]
    metric_records = [
        {"factual_correctness(mode=f1)": 0.9, "explanation": "matches"},
        {"factual_correctness": 0.2, "explanation": "contradicts"},
        {"factual_correctness": 0.7, "explanation": "too lenient"},
    ]

    result = module.summarize_calibration(
        rows,
        metric_records,
        metric_name="factual_correctness",
        score_cutoff=0.5,
        pass_threshold=0.8,
        judge_model="qwen-test",
        judge_config={"metric": "factual_correctness"},
        run_timestamp="2026-06-25T00:00:00+00:00",
    )

    assert result["summary"] | {"score_separation": 0.45} == {
        "metric": "factual_correctness",
        "judge_model": "qwen-test",
        "judge_config": {"metric": "factual_correctness"},
        "run_timestamp": "2026-06-25T00:00:00+00:00",
        "row_count": 3,
        "score_cutoff": 0.5,
        "pass_threshold": 0.8,
        "accuracy": 2 / 3,
        "passed": False,
        "confusion_matrix": {
            "true_correct_pred_correct": 1,
            "true_correct_pred_incorrect": 0,
            "true_incorrect_pred_correct": 1,
            "true_incorrect_pred_incorrect": 1,
        },
        "mean_correct_score": 0.9,
        "mean_incorrect_score": 0.44999999999999996,
        "score_separation": 0.45,
    }
    assert result["summary"]["score_separation"] == pytest.approx(0.45)
    assert result["rows"][2]["predicted_label"] == "correct"
    assert result["rows"][2]["matches_human_label"] is False
    assert result["rows"][2]["lane"] == "conflict_version_behavior"


def test_calibration_cli_uses_ragas_factual_correctness_and_writes_artifact(tmp_path, monkeypatch):
    module = _load_calibration_script()
    calibration_path = tmp_path / "calibration.jsonl"
    output_path = tmp_path / "calibration-result.json"
    _write_jsonl(
        calibration_path,
        [
            {
                "id": "judge-calibration-001",
                "suite": "russian_judge_calibration",
                "locked": True,
                "expected_behavior": "cite_answer",
                "question": "Как оформить возврат?",
                "reference_answer": "Нужно оформить документы.",
                "candidate_answer": "Оформить документы.",
                "human_label": "correct",
                "lane": "indexed_answerability",
            }
        ],
    )
    captured: dict[str, object] = {}

    def fake_evaluate_ragas_rows(rows, metric_names, evaluator_llm=None):
        captured["rows"] = rows
        captured["metric_names"] = metric_names
        captured["evaluator_llm"] = evaluator_llm
        return [{"factual_correctness": 0.95, "explanation": "grounded"}]

    monkeypatch.setattr(module, "evaluate_ragas_rows", fake_evaluate_ragas_rows)
    monkeypatch.setattr(
        module,
        "resolve_judge_settings",
        lambda judge_model=None: SimpleNamespace(chat_model=judge_model or "qwen-test"),
    )

    def fake_build_evaluator_llm(settings):
        captured["judge_settings"] = settings
        return "pinned-judge-llm"

    monkeypatch.setattr(module, "build_evaluator_llm", fake_build_evaluator_llm)
    monkeypatch.setattr(module, "utc_timestamp", lambda: "2026-06-25T00:00:00+00:00")

    exit_code = module.main(
        [
            "--calibration-path",
            str(calibration_path),
            "--output-path",
            str(output_path),
            "--pass-threshold",
            "0.8",
            "--score-cutoff",
            "0.5",
            "--judge-model",
            "qwen3.7-plus",
        ]
    )

    assert exit_code == 0
    assert captured["metric_names"] == ["factual_correctness"]
    assert captured["judge_settings"].chat_model == "qwen3.7-plus"
    assert captured["evaluator_llm"] == "pinned-judge-llm"
    assert captured["rows"] == [
        {
            "id": "judge-calibration-001",
            "user_input": "Как оформить возврат?",
            "response": "Оформить документы.",
            "reference": "Нужно оформить документы.",
            "human_label": "correct",
            "expected_behavior": "cite_answer",
            "lane": "indexed_answerability",
        }
    ]
    artifact = json.loads(output_path.read_text(encoding="utf-8"))
    assert artifact["summary"]["passed"] is True
    assert artifact["summary"]["judge_model"] == "qwen3.7-plus"
    assert artifact["summary"]["judge_config"]["judge_model"] == "qwen3.7-plus"
    assert artifact["rows"][0]["score"] == 0.95


def _load_calibration_script():
    spec = importlib.util.spec_from_file_location(
        "run_judge_calibration_for_test",
        Path("scripts/run_judge_calibration.py"),
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
