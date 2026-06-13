from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_phoenix_eval import DEFAULT_QUESTIONS_PATH, build_runtime, load_questions, run_target
from imperial_rag.ragas_eval import (
    DEFAULT_RAGAS_METRICS,
    REFERENCE_REQUIRED_RAGAS_METRICS,
    SUPPORTED_RAGAS_METRICS,
    evaluate_id_context_recall_rows,
    evaluate_faithfulness_rows,
    faithfulness_row_from_run_output,
    parse_ragas_metric_names,
    retrieved_context_ids_from_output,
    retrieved_contexts_from_output,
    validate_ragas_metric_requirements,
)


DEFAULT_METRICS = DEFAULT_RAGAS_METRICS
REFERENCE_REQUIRED_METRICS = REFERENCE_REQUIRED_RAGAS_METRICS
SUPPORTED_METRICS = SUPPORTED_RAGAS_METRICS


class PreparedRagasRows:
    def __init__(self, rows: list[dict[str, Any]], skipped: int) -> None:
        self.rows = rows
        self.skipped = skipped


def build_ragas_rows(examples: list[dict[str, Any]], runtime: Any | None = None) -> PreparedRagasRows:
    resolved_runtime = runtime or build_runtime()
    rows: list[dict[str, Any]] = []
    skipped = 0
    for example in examples:
        if example.get("expected_behavior") == "refuse_if_not_found":
            skipped += 1
            continue
        outputs = run_target({"question": example["question"]}, runtime=resolved_runtime)
        row = faithfulness_row_from_run_output({"question": example["question"]}, outputs)
        if row is None:
            skipped += 1
            continue
        row["expected_behavior"] = example.get("expected_behavior")
        row["expected_source_hints"] = example.get("expected_source_hints", [])
        row["retrieved_context_ids"] = retrieved_context_ids_from_output(outputs)
        if example.get("reference_answer"):
            row["reference"] = example["reference_answer"]
        if "reference_context_ids" in example:
            row["reference_context_ids"] = [
                str(context_id).strip() for context_id in example.get("reference_context_ids") or []
            ]
        rows.append(row)
    return PreparedRagasRows(rows=rows, skipped=skipped)


def parse_metric_names(raw_metrics: str | None) -> list[str]:
    return parse_ragas_metric_names(raw_metrics, default=DEFAULT_METRICS)


def validate_metric_requirements(metric_names: list[str], rows: list[dict[str, Any]]) -> None:
    validate_ragas_metric_requirements(
        metric_names,
        rows,
        reference_key="reference",
        row_label_key="user_input",
    )


def evaluate_ragas_rows(
    rows: list[dict[str, Any]],
    metric_names: list[str],
    evaluate_fn: Callable[..., Any] | None = None,
) -> Any:
    validate_metric_requirements(metric_names, rows)
    sidecar_records: list[dict[str, Any]] | None = None
    reference_metric_names = [name for name in metric_names if name not in {"faithfulness", "id_context_recall"}]
    if "faithfulness" in metric_names:
        sidecar_records = evaluate_faithfulness_rows(rows)
    if "id_context_recall" in metric_names:
        id_context_recall_records = evaluate_id_context_recall_rows(rows)
        sidecar_records = (
            _merge_records_by_position(sidecar_records, id_context_recall_records)
            if sidecar_records is not None
            else id_context_recall_records
        )
    if not reference_metric_names:
        return sidecar_records or []

    evaluator_llm = build_evaluator_llm()
    dataset = build_ragas_dataset(rows)
    metrics = build_ragas_metrics(reference_metric_names, evaluator_llm)
    resolved_evaluate = evaluate_fn or _import_ragas_evaluate()
    reference_result = resolved_evaluate(dataset=dataset, metrics=metrics)
    if sidecar_records is not None:
        return _merge_records_by_position(sidecar_records, result_records(reference_result))
    return reference_result


def build_ragas_dataset(rows: list[dict[str, Any]]) -> Any:
    _install_ragas_langchain_community_compat()
    try:
        import ragas
    except ImportError as exc:
        raise SystemExit("Ragas is not installed; run `uv sync --extra dev`.") from exc

    EvaluationDataset = getattr(ragas, "EvaluationDataset", None)
    if EvaluationDataset is not None:
        return EvaluationDataset.from_list(rows)

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit("Ragas dataset fallback requires the `datasets` package.") from exc
    return Dataset.from_list(rows)


def build_ragas_metrics(metric_names: list[str], evaluator_llm: Any) -> list[Any]:
    unsupported_here = sorted(set(metric_names) & {"faithfulness", "id_context_recall"})
    if unsupported_here:
        raise SystemExit(
            "Ragas Faithfulness and ID context recall are evaluated through imperial_rag.ragas_eval."
        )

    _install_ragas_langchain_community_compat()
    try:
        from ragas.metrics.collections import ContextRecall, FactualCorrectness
    except ImportError as exc:
        raise SystemExit("Ragas collections metrics are not installed; run `uv sync --extra dev`.") from exc

    factories: dict[str, Callable[..., Any]] = {
        "context_recall": ContextRecall,
        "factual_correctness": FactualCorrectness,
    }
    return [factories[name](llm=evaluator_llm) for name in metric_names]


def build_evaluator_llm(provider_settings: Any | None = None) -> Any:
    from imperial_rag.providers import MissingDashScopeKeyError, QwenProviderSettings

    settings = provider_settings or QwenProviderSettings.from_env()

    try:
        api_key = settings.require_api_key()
    except MissingDashScopeKeyError as exc:
        raise SystemExit("DASHSCOPE_API_KEY is required to run Ragas evaluator metrics.") from exc

    AsyncOpenAI = _import_async_openai()
    llm_factory = _import_llm_factory()
    client = AsyncOpenAI(api_key=api_key, base_url=settings.compat_base_url)
    return llm_factory(settings.chat_model, client=client, provider="openai")


def result_records(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [dict(row) for row in result]
    if hasattr(result, "to_pandas"):
        return result.to_pandas().to_dict(orient="records")
    scores = getattr(result, "scores", None)
    if scores is not None:
        return list(scores)
    if isinstance(result, dict) and "scores" in result:
        return list(result["scores"])
    return [{"result": str(result)}]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    _ensure_src_on_path()
    parser = argparse.ArgumentParser(description="Run Imperial RAG quality evaluations with Ragas.")
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--output-path", type=Path)
    args = parser.parse_args(argv)

    _load_project_env(args.workspace_root)
    settings = _build_settings(args.workspace_root)
    _configure_observability(settings)
    started_at = perf_counter()
    try:
        examples = load_questions(args.questions_path)
        metric_names = parse_metric_names(args.metrics)
        prepared = build_ragas_rows(examples, runtime=build_runtime(settings=settings))
        if not prepared.rows:
            raise SystemExit("No supported Ragas rows were prepared from the eval examples.")

        result = evaluate_ragas_rows(prepared.rows, metric_names)
        records = result_records(result)
        if args.output_path:
            write_jsonl(args.output_path, records)
        else:
            for record in records:
                print(json.dumps(record, ensure_ascii=False, sort_keys=True))
        print(f"ragas_examples={len(prepared.rows)}")
        print(f"ragas_skipped={prepared.skipped}")
        print(f"ragas_metrics={','.join(metric_names)}")
        _log_completion(
            started_at,
            example_count=len(prepared.rows),
            skipped_count=prepared.skipped,
            ragas_metrics=",".join(metric_names),
            wrote_output=bool(args.output_path),
        )
    except (Exception, SystemExit) as exc:
        _log_failure("ragas_eval", exc, started_at, ragas_metrics=args.metrics)
        raise


def _retrieved_contexts(outputs: dict[str, Any]) -> list[str]:
    return retrieved_contexts_from_output(outputs)


def _configure_observability(settings: Any) -> None:
    from imperial_rag.observability import configure_observability

    configure_observability(settings)


def _log_completion(started_at: float, **fields: Any) -> None:
    from imperial_rag.observability import log_event

    log_event(
        "imperial_rag.ragas_eval",
        operation="ragas_eval",
        status="success",
        component="cli",
        duration_ms=_duration_ms(started_at),
        **fields,
    )


def _log_failure(operation: str, exc: BaseException, started_at: float, **fields: Any) -> None:
    from imperial_rag.observability import log_failure

    log_failure(operation, exc, component="cli", duration_ms=_duration_ms(started_at), **fields)


def _duration_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)


def _merge_records_by_position(
    primary_records: list[dict[str, Any]],
    secondary_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not secondary_records:
        return primary_records
    merged: list[dict[str, Any]] = []
    for index, primary in enumerate(primary_records):
        secondary = secondary_records[index] if index < len(secondary_records) else {}
        merged.append({**primary, **secondary})
    return merged


def _build_settings(workspace_root: Path | None) -> Any:
    from imperial_rag.config import Settings

    if workspace_root is None:
        return Settings()
    return Settings(workspace_root=workspace_root)


def _load_project_env(workspace_root: Path | None) -> None:
    from imperial_rag.env import load_project_env

    load_project_env(workspace_root)


def _import_async_openai() -> Any:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("OpenAI client is not installed; run `uv sync --extra dev`.") from exc
    return AsyncOpenAI


def _import_llm_factory() -> Callable[..., Any]:
    _install_ragas_langchain_community_compat()
    try:
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise SystemExit("Ragas LLM factory is not installed; run `uv sync --extra dev`.") from exc
    return llm_factory


def _import_ragas_evaluate() -> Callable[..., Any]:
    _install_ragas_langchain_community_compat()
    try:
        from ragas import evaluate
    except ImportError as exc:
        raise SystemExit("Ragas is not installed; run `uv sync --extra dev`.") from exc
    return evaluate


def _install_ragas_langchain_community_compat() -> None:
    module_name = "langchain_community.chat_models.vertexai"
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`langchain-community` is being sunset.*",
                category=DeprecationWarning,
            )
            importlib.import_module(module_name)
        return
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

    module = types.ModuleType(module_name)

    class ChatVertexAI:
        pass

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


if __name__ == "__main__":
    main()
