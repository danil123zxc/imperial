from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import sys
import types
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import anyio


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_phoenix_eval import DEFAULT_QUESTIONS_PATH, build_runtime, load_questions, positive_int, run_target
from imperial_rag.ragas_eval import (
    DEFAULT_RAGAS_CONCURRENCY,
    DEFAULT_RAGAS_METRICS,
    REFERENCE_REQUIRED_RAGAS_METRICS,
    SUPPORTED_RAGAS_METRICS,
    answer_relevancy_row_from_run_output,
    evaluation_dataset_from_rows,
    evaluate_answer_relevancy_rows_async,
    evaluate_id_context_recall_rows_async,
    evaluate_faithfulness_rows_async,
    parse_ragas_metric_names,
    retrieved_context_ids_from_output,
    retrieved_contexts_from_output,
    validate_ragas_metric_requirements,
)


DEFAULT_METRICS = DEFAULT_RAGAS_METRICS
DEFAULT_CONCURRENCY = DEFAULT_RAGAS_CONCURRENCY
REFERENCE_REQUIRED_METRICS = REFERENCE_REQUIRED_RAGAS_METRICS
SUPPORTED_METRICS = SUPPORTED_RAGAS_METRICS
SIDECAR_METRIC_ORDER = ("faithfulness", "answer_relevancy", "id_context_recall")


class PreparedRagasRows:
    def __init__(self, rows: list[dict[str, Any]], skipped: int) -> None:
        self.rows = rows
        self.skipped = skipped


def build_ragas_rows(examples: list[dict[str, Any]], runtime: Any | None = None) -> PreparedRagasRows:
    resolved_runtime = runtime or build_runtime()
    rows: list[dict[str, Any]] = []
    skipped = 0
    for example in examples:
        if example.get("expected_behavior") != "cite_answer":
            skipped += 1
            continue
        outputs = run_target({"question": example["question"]}, runtime=resolved_runtime)
        row = answer_relevancy_row_from_run_output({"question": example["question"]}, outputs)
        if row is None:
            skipped += 1
            continue
        if "id" in example:
            row["id"] = example["id"]
        if "suite" in example:
            row["suite"] = example["suite"]
        if "tags" in example:
            row["tags"] = list(example.get("tags") or [])
        retrieved_contexts = retrieved_contexts_from_output(outputs)
        if retrieved_contexts:
            row["retrieved_contexts"] = retrieved_contexts
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
    evaluator_llm: Any | None = None,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    batch_size: int | None = None,
) -> Any:
    return _run_async(
        evaluate_ragas_rows_async(
            rows,
            metric_names,
            evaluate_fn=evaluate_fn,
            evaluator_llm=evaluator_llm,
            concurrency=concurrency,
            batch_size=batch_size,
        )
    )


async def evaluate_ragas_rows_async(
    rows: list[dict[str, Any]],
    metric_names: list[str],
    evaluate_fn: Callable[..., Any] | None = None,
    evaluator_llm: Any | None = None,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    batch_size: int | None = None,
) -> Any:
    concurrency = positive_int(concurrency)
    validate_metric_requirements(metric_names, rows)
    sidecar_records: list[dict[str, Any]] | None = None
    reference_metric_names = [
        name for name in metric_names if name not in SIDECAR_METRIC_ORDER
    ]
    sidecar_metric_names = [name for name in SIDECAR_METRIC_ORDER if name in metric_names]
    if sidecar_metric_names:
        sidecar_results = await _evaluate_sidecar_metrics_concurrently(
            rows,
            sidecar_metric_names,
            concurrency=concurrency,
        )
        for metric_name in sidecar_metric_names:
            metric_records = sidecar_results[metric_name]
            sidecar_records = (
                _merge_records_by_position(sidecar_records, metric_records)
                if sidecar_records is not None
                else metric_records
            )
    if not reference_metric_names:
        return sidecar_records or []

    evaluator_llm = evaluator_llm or build_evaluator_llm()
    dataset = build_ragas_dataset(rows)
    metrics = build_ragas_metrics(reference_metric_names, evaluator_llm)
    resolved_evaluate = evaluate_fn or _import_ragas_aevaluate()
    evaluate_kwargs: dict[str, Any] = {"dataset": dataset, "metrics": metrics}
    if batch_size is not None:
        evaluate_kwargs["batch_size"] = batch_size
    # TODO: migrate this reference-metric path to Ragas @experiment once the repo
    # has an equivalent async adapter.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*aevaluate.*deprecated.*",
            category=DeprecationWarning,
        )
        reference_result = await _resolve_awaitable(resolved_evaluate(**evaluate_kwargs))
    reference_records = _merge_records_by_position(_row_metadata_records(rows), result_records(reference_result))
    if sidecar_records is not None:
        return _merge_records_by_position(sidecar_records, reference_records)
    return reference_records


async def _evaluate_sidecar_metrics_concurrently(
    rows: list[dict[str, Any]],
    metric_names: list[str],
    *,
    concurrency: int,
) -> dict[str, list[dict[str, Any]]]:
    async def evaluate(metric_name: str) -> list[dict[str, Any]]:
        if metric_name == "faithfulness":
            return await evaluate_faithfulness_rows_async(rows, concurrency=concurrency)
        if metric_name == "answer_relevancy":
            return await evaluate_answer_relevancy_rows_async(rows, concurrency=concurrency)
        if metric_name == "id_context_recall":
            return await evaluate_id_context_recall_rows_async(rows, concurrency=concurrency)
        raise AssertionError(f"Unsupported sidecar metric: {metric_name}")

    results = await asyncio.gather(*(evaluate(metric_name) for metric_name in metric_names))
    return dict(zip(metric_names, results, strict=True))


def build_ragas_dataset(rows: list[dict[str, Any]]) -> Any:
    return evaluation_dataset_from_rows(rows)


def build_ragas_metrics(metric_names: list[str], evaluator_llm: Any) -> list[Any]:
    unsupported_here = sorted(set(metric_names) & {"faithfulness", "answer_relevancy", "id_context_recall"})
    if unsupported_here:
        raise SystemExit(
            "Ragas Faithfulness, Answer Relevancy, and ID context recall are evaluated through imperial_rag.ragas_eval."
        )

    _install_ragas_langchain_community_compat()
    try:
        from ragas.metrics._context_recall import LLMContextRecall
        from ragas.metrics._factual_correctness import FactualCorrectness
        from ragas.metrics.base import Metric
    except ImportError as exc:
        raise SystemExit("Ragas reference metrics are not installed; run `uv sync --extra dev`.") from exc

    factories: dict[str, Callable[..., Any]] = {
        # Ragas 0.4.3 exposes collections metrics too, but ragas.evaluate()
        # still validates the legacy Metric hierarchy.
        "context_recall": LLMContextRecall,
        "factual_correctness": FactualCorrectness,
    }
    metrics = [factories[name](llm=evaluator_llm) for name in metric_names]
    invalid_metrics = [type(metric).__name__ for metric in metrics if not isinstance(metric, Metric)]
    if invalid_metrics:
        raise TypeError(
            "Ragas reference metrics must be initialized ragas.metrics.base.Metric instances: "
            + ", ".join(invalid_metrics)
        )
    return metrics


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
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        default=DEFAULT_CONCURRENCY,
        help="Maximum concurrent rows per selected Ragas sidecar metric.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Optional batch size passed to ragas.aevaluate for reference metrics.",
    )
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

        result = evaluate_ragas_rows(
            prepared.rows,
            metric_names,
            concurrency=args.concurrency,
            batch_size=args.batch_size,
        )
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
    from imperial_rag.cli import configure_observability

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
    from imperial_rag.cli import log_failure

    log_failure(operation, exc, started_at, **fields)


def _duration_ms(started_at: float) -> int:
    from imperial_rag.cli import duration_ms

    return duration_ms(started_at)


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


def _row_metadata_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata_keys = ("id", "suite", "lane", "tags", "expected_behavior")
    records: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {}
        for key in metadata_keys:
            if key not in row:
                continue
            value = row[key]
            record[key] = list(value) if key == "tags" and isinstance(value, list) else value
        records.append(record)
    return records


def _build_settings(workspace_root: Path | None) -> Any:
    from imperial_rag.cli import build_settings

    return build_settings(workspace_root)


def _load_project_env(workspace_root: Path | None) -> None:
    from imperial_rag.cli import load_project_environment

    load_project_environment(workspace_root)


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


def _import_ragas_aevaluate() -> Callable[..., Any]:
    _install_ragas_langchain_community_compat()
    try:
        from ragas import aevaluate
    except ImportError as exc:
        raise SystemExit("Ragas async evaluation is not installed; run `uv sync --extra dev`.") from exc
    return aevaluate


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

    setattr(module, "ChatVertexAI", ChatVertexAI)
    sys.modules[module_name] = module


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _run_async(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return anyio.run(_await_result, awaitable)
    if hasattr(awaitable, "close"):
        awaitable.close()
    raise RuntimeError("Cannot synchronously run async Ragas evaluation inside a running event loop.")


async def _resolve_awaitable(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


async def _await_result(awaitable: Any) -> Any:
    return await awaitable


if __name__ == "__main__":
    main()
