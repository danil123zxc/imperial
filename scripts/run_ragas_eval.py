from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_phoenix_eval import DEFAULT_QUESTIONS_PATH, build_runtime, load_questions, run_target


DEFAULT_METRICS = ("faithfulness",)
REFERENCE_REQUIRED_METRICS = {"context_recall", "factual_correctness"}
SUPPORTED_METRICS = DEFAULT_METRICS + ("context_recall", "factual_correctness")


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
        contexts = _retrieved_contexts(outputs)
        response = str(outputs.get("answer", "")).strip()
        if not contexts or not response:
            skipped += 1
            continue
        row = {
            "user_input": example["question"],
            "response": response,
            "retrieved_contexts": contexts,
            "expected_behavior": example.get("expected_behavior"),
            "expected_source_hints": example.get("expected_source_hints", []),
        }
        if example.get("reference_answer"):
            row["reference"] = example["reference_answer"]
        rows.append(row)
    return PreparedRagasRows(rows=rows, skipped=skipped)


def parse_metric_names(raw_metrics: str | None) -> list[str]:
    if not raw_metrics:
        return list(DEFAULT_METRICS)
    names = [name.strip().casefold().replace("-", "_") for name in raw_metrics.split(",") if name.strip()]
    unsupported = sorted(set(names) - set(SUPPORTED_METRICS))
    if unsupported:
        supported = ", ".join(SUPPORTED_METRICS)
        raise SystemExit(f"Unsupported Ragas metrics: {', '.join(unsupported)}. Supported metrics: {supported}.")
    return names


def validate_metric_requirements(metric_names: list[str], rows: list[dict[str, Any]]) -> None:
    reference_metrics = sorted(set(metric_names) & REFERENCE_REQUIRED_METRICS)
    if not reference_metrics:
        return
    missing_reference = [row["user_input"] for row in rows if not row.get("reference")]
    if missing_reference:
        joined_metrics = ", ".join(reference_metrics)
        raise SystemExit(
            f"Ragas metrics {joined_metrics} require reference_answer in evals/questions.jsonl. "
            f"Missing reference_answer for {len(missing_reference)} prepared rows."
        )


def evaluate_ragas_rows(
    rows: list[dict[str, Any]],
    metric_names: list[str],
    evaluate_fn: Callable[..., Any] | None = None,
) -> Any:
    validate_metric_requirements(metric_names, rows)
    dataset = build_ragas_dataset(rows)
    metrics = build_ragas_metrics(metric_names)
    evaluator_llm = build_evaluator_llm()
    resolved_evaluate = evaluate_fn or _import_ragas_evaluate()
    return resolved_evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm)


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


def build_ragas_metrics(metric_names: list[str]) -> list[Any]:
    _install_ragas_langchain_community_compat()
    try:
        from ragas.metrics._context_recall import LLMContextRecall
        from ragas.metrics._factual_correctness import FactualCorrectness
        from ragas.metrics._faithfulness import Faithfulness
    except ImportError as exc:
        try:
            from ragas.metrics import Faithfulness, FactualCorrectness, LLMContextRecall
        except ImportError:
            raise SystemExit("Ragas metrics are not installed; run `uv sync --extra dev`.") from exc

    factories: dict[str, Callable[[], Any]] = {
        "faithfulness": Faithfulness,
        "context_recall": LLMContextRecall,
        "factual_correctness": FactualCorrectness,
    }
    return [factories[name]() for name in metric_names]


def build_evaluator_llm() -> Any:
    _install_ragas_langchain_community_compat()
    try:
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise SystemExit("Ragas LLM wrappers are not installed; run `uv sync --extra dev`.") from exc

    from imperial_rag.providers import MissingDashScopeKeyError, create_chat_model

    try:
        model = create_chat_model()
    except MissingDashScopeKeyError as exc:
        raise SystemExit("DASHSCOPE_API_KEY is required to run Ragas evaluator metrics.") from exc
    return LangchainLLMWrapper(model)


def result_records(result: Any) -> list[dict[str, Any]]:
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

    settings = _build_settings(args.workspace_root)
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


def _retrieved_contexts(outputs: dict[str, Any]) -> list[str]:
    contexts: list[str] = []
    for document in outputs.get("documents", []) or []:
        text = str(document.get("page_content", "")).strip()
        if text:
            contexts.append(text)
    return contexts


def _build_settings(workspace_root: Path | None) -> Any:
    from imperial_rag.config import Settings

    if workspace_root is None:
        return Settings()
    return Settings(workspace_root=workspace_root)


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
