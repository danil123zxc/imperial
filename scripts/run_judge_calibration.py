from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from _bootstrap import ensure_src_on_path as _ensure_src_on_path

_ensure_src_on_path(__file__)

from imperial_rag.cli import load_project_environment as _load_project_env  # noqa: E402
from imperial_rag.evals.ragas_runner import (  # noqa: E402
    build_evaluator_llm,
    evaluate_ragas_rows,
    result_records,
)
from imperial_rag.jsonl import read_jsonl  # noqa: E402


DEFAULT_CALIBRATION_PATH = Path("evals/russian_judge_calibration.jsonl")
DEFAULT_METRIC_NAME = "factual_correctness"
HUMAN_LABELS = {"correct", "incorrect"}


def load_calibration_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def prepare_calibration_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id") or f"row-{index}")
        human_label = str(row.get("human_label") or "").strip()
        if human_label not in HUMAN_LABELS:
            raise ValueError(f"{row_id}: human_label must be one of {sorted(HUMAN_LABELS)}")
        required = {
            "question": row.get("question"),
            "reference_answer": row.get("reference_answer"),
            "candidate_answer": row.get("candidate_answer"),
        }
        missing = [name for name, value in required.items() if not str(value or "").strip()]
        if missing:
            raise ValueError(f"{row_id}: missing required fields: {', '.join(missing)}")
        prepared_row = {
            "id": row_id,
            "user_input": str(row["question"]).strip(),
            "response": str(row["candidate_answer"]).strip(),
            "reference": str(row["reference_answer"]).strip(),
            "human_label": human_label,
            "expected_behavior": str(row.get("expected_behavior") or "").strip(),
        }
        if row.get("lane"):
            prepared_row["lane"] = str(row["lane"]).strip()
        prepared.append(prepared_row)
    return prepared


def summarize_calibration(
    rows: Sequence[Mapping[str, Any]],
    metric_records: Sequence[Mapping[str, Any]],
    *,
    metric_name: str,
    score_cutoff: float,
    pass_threshold: float,
    judge_model: str,
    judge_config: Mapping[str, Any],
    run_timestamp: str,
) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    confusion = {
        "true_correct_pred_correct": 0,
        "true_correct_pred_incorrect": 0,
        "true_incorrect_pred_correct": 0,
        "true_incorrect_pred_incorrect": 0,
    }
    correct_scores: list[float] = []
    incorrect_scores: list[float] = []

    for index, row in enumerate(rows):
        record = metric_records[index] if index < len(metric_records) else {}
        score = _metric_score(record, metric_name)
        human_label = str(row.get("human_label") or "")
        predicted_label = "correct" if score is not None and score >= score_cutoff else "incorrect"
        matches = human_label == predicted_label
        if human_label == "correct":
            correct_scores.append(float(score)) if score is not None else None
            key = "true_correct_pred_correct" if predicted_label == "correct" else "true_correct_pred_incorrect"
        else:
            incorrect_scores.append(float(score)) if score is not None else None
            key = "true_incorrect_pred_correct" if predicted_label == "correct" else "true_incorrect_pred_incorrect"
        confusion[key] += 1
        output_rows.append(
            {
                "id": row.get("id"),
                "lane": row.get("lane"),
                "human_label": human_label,
                "score": score,
                "predicted_label": predicted_label,
                "matches_human_label": matches,
                "explanation": record.get("explanation") or record.get(f"{metric_name}_explanation"),
            }
        )

    row_count = len(rows)
    matched_count = sum(1 for row in output_rows if row["matches_human_label"])
    accuracy = matched_count / row_count if row_count else 0.0
    true_positive_rate = _safe_divide(
        confusion["true_correct_pred_correct"],
        confusion["true_correct_pred_correct"] + confusion["true_correct_pred_incorrect"],
    )
    true_negative_rate = _safe_divide(
        confusion["true_incorrect_pred_incorrect"],
        confusion["true_incorrect_pred_correct"] + confusion["true_incorrect_pred_incorrect"],
    )
    false_positive_rate = _safe_divide(
        confusion["true_incorrect_pred_correct"],
        confusion["true_incorrect_pred_correct"] + confusion["true_incorrect_pred_incorrect"],
    )
    false_negative_rate = _safe_divide(
        confusion["true_correct_pred_incorrect"],
        confusion["true_correct_pred_correct"] + confusion["true_correct_pred_incorrect"],
    )
    balanced_accuracy = (
        (true_positive_rate + true_negative_rate) / 2
        if true_positive_rate is not None and true_negative_rate is not None
        else None
    )
    cohen_kappa = _cohen_kappa(confusion, row_count=row_count, accuracy=accuracy)
    mean_correct_score = mean(correct_scores) if correct_scores else None
    mean_incorrect_score = mean(incorrect_scores) if incorrect_scores else None
    score_separation = (
        mean_correct_score - mean_incorrect_score
        if mean_correct_score is not None and mean_incorrect_score is not None
        else None
    )
    return {
        "summary": {
            "metric": metric_name,
            "judge_model": judge_model,
            "judge_config": dict(judge_config),
            "run_timestamp": run_timestamp,
            "row_count": row_count,
            "score_cutoff": score_cutoff,
            "pass_threshold": pass_threshold,
            "accuracy": accuracy,
            "true_positive_rate": true_positive_rate,
            "true_negative_rate": true_negative_rate,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "balanced_accuracy": balanced_accuracy,
            "cohen_kappa": cohen_kappa,
            "passed": accuracy >= pass_threshold,
            "confusion_matrix": confusion,
            "mean_correct_score": mean_correct_score,
            "mean_incorrect_score": mean_incorrect_score,
            "score_separation": score_separation,
        },
        "rows": output_rows,
    }


def resolve_judge_settings(judge_model: str | None = None) -> Any:
    from dataclasses import replace
    from imperial_rag.integrations.dashscope import QwenProviderSettings

    settings = QwenProviderSettings.from_env()
    if judge_model:
        return replace(settings, chat_model=judge_model)
    return settings


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the Russian judge used for Ragas factual correctness.")
    parser.add_argument("--calibration-path", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--metric", default=DEFAULT_METRIC_NAME)
    parser.add_argument("--judge-model", help="Qwen/DashScope chat model to use for the calibration judge.")
    parser.add_argument("--score-cutoff", type=float, default=0.5)
    parser.add_argument("--pass-threshold", type=float, default=0.85)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if accuracy is below the pass threshold.")
    args = parser.parse_args(argv)

    _load_project_env(args.workspace_root)
    raw_rows = load_calibration_rows(args.calibration_path)
    prepared_rows = prepare_calibration_rows(raw_rows)
    judge_settings = resolve_judge_settings(args.judge_model)
    evaluator_llm = build_evaluator_llm(judge_settings)
    metric_records = result_records(
        evaluate_ragas_rows(prepared_rows, [args.metric], evaluator_llm=evaluator_llm)
    )
    result = summarize_calibration(
        prepared_rows,
        metric_records,
        metric_name=args.metric,
        score_cutoff=args.score_cutoff,
        pass_threshold=args.pass_threshold,
        judge_model=judge_settings.chat_model,
        judge_config={
            "metric": args.metric,
            "judge_model": judge_settings.chat_model,
            "score_cutoff": args.score_cutoff,
            "pass_threshold": args.pass_threshold,
        },
        run_timestamp=utc_timestamp(),
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = result["summary"]
    print(f"calibration_rows={summary['row_count']}")
    print(f"calibration_metric={summary['metric']}")
    print(f"calibration_accuracy={summary['accuracy']:.3f}")
    print(f"calibration_passed={str(summary['passed']).lower()}")
    print(f"calibration_output={args.output_path}")
    return 1 if args.strict and not summary["passed"] else 0


def _metric_score(record: Mapping[str, Any], metric_name: str) -> float | None:
    values: list[Any] = []
    if metric_name in record:
        values.append(record[metric_name])
    values.extend(value for key, value in record.items() if key.startswith(f"{metric_name}("))
    if "score" in record:
        values.append(record["score"])

    for value in values:
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            return score
    return None


def _safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _cohen_kappa(confusion: Mapping[str, int], *, row_count: int, accuracy: float) -> float | None:
    if row_count == 0:
        return None
    true_correct = confusion["true_correct_pred_correct"] + confusion["true_correct_pred_incorrect"]
    true_incorrect = confusion["true_incorrect_pred_correct"] + confusion["true_incorrect_pred_incorrect"]
    predicted_correct = confusion["true_correct_pred_correct"] + confusion["true_incorrect_pred_correct"]
    predicted_incorrect = confusion["true_correct_pred_incorrect"] + confusion["true_incorrect_pred_incorrect"]
    expected_agreement = (
        (true_correct / row_count) * (predicted_correct / row_count)
        + (true_incorrect / row_count) * (predicted_incorrect / row_count)
    )
    if expected_agreement == 1:
        return 1.0
    return (accuracy - expected_agreement) / (1 - expected_agreement)


if __name__ == "__main__":
    raise SystemExit(main())
