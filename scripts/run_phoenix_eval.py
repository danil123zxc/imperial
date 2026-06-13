from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any


DEFAULT_QUESTIONS_PATH = Path("evals/questions.jsonl")
DEFAULT_EXPERIMENT_NAME = "imperial-rag-citation-grounding"
DEFAULT_RETRIEVAL_METRIC_K = 5
REFUSAL_FALLBACKS = (
    "I could not find this clearly in the indexed documents.",
    "не удалось найти",
    "не найдено",
    "нет в проиндексированных документах",
)
_RAGAS_FAITHFULNESS_SCORER: Any | None = None


def load_questions(path: Path = DEFAULT_QUESTIONS_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not payload.get("question"):
            raise ValueError(f"missing question on line {line_number}")
        rows.append(payload)
    return rows


def target(inputs: dict[str, Any]) -> dict[str, Any]:
    return run_target(inputs)


def run_target(inputs: dict[str, Any], runtime: Any | None = None) -> dict[str, Any]:
    question = str(inputs["question"])
    resolved_runtime = runtime or build_runtime()
    result = _coerce_result(resolved_runtime.query(question))
    evidence = result.get("evidence", []) or result.get("documents", [])
    return {
        "answer": str(result.get("answer", "")),
        "citations": list(result.get("citations") or result.get("sources") or []),
        "sources": list(result.get("sources") or result.get("citations") or []),
        "documents": [_document_payload(document) for document in evidence],
        "retrieval": dict(result.get("retrieval") or {}),
    }


def build_runtime(settings: Any | None = None) -> Any:
    try:
        from imperial_rag.runtime import create_runtime
    except (ImportError, AttributeError):
        create_runtime = None
    if create_runtime is not None:
        return create_runtime(settings) if settings is not None else create_runtime()

    try:
        from imperial_rag.runtime import Runtime
    except (ImportError, AttributeError):
        Runtime = None

    if Runtime is not None:
        return Runtime(settings=settings) if settings is not None else Runtime()

    from imperial_rag.runtime import build_live_query_workflow

    workflow = build_live_query_workflow(settings) if settings is not None else build_live_query_workflow()

    class WorkflowRuntime:
        def query(self, question: str) -> dict[str, Any]:
            return _coerce_result(workflow.invoke({"question": question}))

    return WorkflowRuntime()


def citation_behavior(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = (reference_outputs or inputs).get("expected_behavior")
    answer = str(outputs.get("answer", ""))
    citations = outputs.get("citations") or outputs.get("sources") or []

    if expected == "refuse_if_not_found":
        score = _looks_like_refusal(answer) and not citations
    elif expected == "cite_answer":
        score = bool(citations) and not _looks_like_refusal(answer)
    elif expected == "surface_conflict":
        score = bool(citations) and _mentions_conflict(answer)
    else:
        score = False
    return {"key": "citation_behavior", "score": bool(score)}


def source_hint_behavior(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference = reference_outputs or inputs
    hints = [str(hint).casefold() for hint in reference.get("expected_source_hints", [])]
    if not hints:
        return {"key": "source_hint_behavior", "score": True}
    sources = outputs.get("sources") or []
    citations = outputs.get("citations") or []
    haystack = "\n".join(
        [
            *(str(source) for source in [*sources, *citations]),
            *(_document_search_text(document) for document in outputs.get("documents", []) or []),
        ]
    ).casefold()
    return {"key": "source_hint_behavior", "score": any(hint in haystack for hint in hints)}


def phoenix_citation_behavior(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> bool:
    return bool(citation_behavior(input or {}, output, expected)["score"])


def phoenix_source_hint_behavior(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> bool:
    return bool(source_hint_behavior(input or {}, output, expected)["score"])


def phoenix_retrieval_relevance(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return retrieval_relevance_metrics(input or {}, output or {}, expected)


def phoenix_ragas_faithfulness(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from imperial_rag.ragas_eval import score_faithfulness_for_phoenix

    return score_faithfulness_for_phoenix(
        input=input or {},
        output=output or {},
        scorer=_get_ragas_faithfulness_scorer(),
    )


def run_local_eval(examples: list[dict[str, Any]], runtime: Any | None = None) -> list[dict[str, Any]]:
    resolved_runtime = runtime or build_runtime()
    rows: list[dict[str, Any]] = []
    for example in examples:
        inputs = {"question": example["question"]}
        reference_outputs = {
            "expected_behavior": example["expected_behavior"],
            "expected_source_hints": example.get("expected_source_hints", []),
        }
        outputs = run_target(inputs, runtime=resolved_runtime)
        retrieval_metrics = retrieval_relevance_metrics(inputs, outputs, reference_outputs)
        retrieval_metadata = retrieval_metrics.get("metadata", {})
        rows.append(
            {
                "question": example["question"],
                "citation_behavior": citation_behavior(inputs, outputs, reference_outputs)["score"],
                "source_hint_behavior": source_hint_behavior(inputs, outputs, reference_outputs)["score"],
                f"retrieval_hit_at_{DEFAULT_RETRIEVAL_METRIC_K}": retrieval_metadata.get(
                    f"hit_at_{DEFAULT_RETRIEVAL_METRIC_K}"
                ),
                f"retrieval_precision_at_{DEFAULT_RETRIEVAL_METRIC_K}": retrieval_metadata.get(
                    f"precision_at_{DEFAULT_RETRIEVAL_METRIC_K}"
                ),
                f"retrieval_ndcg_at_{DEFAULT_RETRIEVAL_METRIC_K}": retrieval_metadata.get(
                    f"ndcg_at_{DEFAULT_RETRIEVAL_METRIC_K}"
                ),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    _ensure_src_on_path()
    parser = argparse.ArgumentParser(description="Run Imperial RAG citation/refusal evaluations.")
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--use-phoenix", action="store_true", help="Store dataset and experiment results in Phoenix.")
    parser.add_argument("--trace-phoenix", action="store_true", help="Send this run's traces to configured Phoenix.")
    parser.add_argument(
        "--ragas-metrics",
        default="faithfulness",
        help="Comma-separated Ragas metrics to attach in Phoenix mode, or 'none'.",
    )
    args = parser.parse_args(argv)

    _load_project_env(args.workspace_root)
    settings = _build_settings(args.workspace_root)
    _configure_observability(settings)
    started_at = perf_counter()
    try:
        if args.trace_phoenix or args.use_phoenix:
            _configure_tracing(settings, enabled=True)
        examples = load_questions(args.questions_path)
        metric_names = parse_phoenix_ragas_metrics(args.ragas_metrics)

        if args.use_phoenix:
            run_phoenix_experiment(
                examples=examples,
                settings=settings,
                dataset_name=args.dataset_name or f"{settings.phoenix_project_name}-gold-questions",
                experiment_name=args.experiment_name,
                ragas_metric_names=metric_names,
            )
            _log_eval_completion(
                started_at,
                operation="phoenix_eval",
                status="success",
                example_count=len(examples),
                phoenix_mode=True,
                ragas_metrics=",".join(metric_names),
            )
            return

        rows = run_local_eval(examples, runtime=build_runtime(settings=settings))
        passed = sum(1 for row in rows if row["citation_behavior"] and row["source_hint_behavior"])
        print(f"local_eval_examples={len(rows)}")
        print(f"local_eval_passed={passed}")
        _log_eval_completion(
            started_at,
            operation="phoenix_eval",
            status="success",
            example_count=len(rows),
            passed_count=passed,
            phoenix_mode=False,
            ragas_metrics=",".join(metric_names),
        )
    except (Exception, SystemExit) as exc:
        _log_failure(
            "phoenix_eval",
            exc,
            started_at,
            phoenix_mode=args.use_phoenix,
            ragas_metrics=args.ragas_metrics,
        )
        raise


def run_phoenix_experiment(
    examples: list[dict[str, Any]],
    settings: Any,
    dataset_name: str,
    experiment_name: str,
    ragas_metric_names: list[str] | None = None,
) -> None:
    resolved_ragas_metric_names = list(ragas_metric_names if ragas_metric_names is not None else ["faithfulness"])
    _validate_phoenix_ragas_metric_requirements(resolved_ragas_metric_names, examples)
    evaluators = _phoenix_evaluators(resolved_ragas_metric_names)

    try:
        from phoenix.client import Client
    except ImportError as exc:
        raise SystemExit("Phoenix client is not installed; install arize-phoenix-client.") from exc

    if "faithfulness" in resolved_ragas_metric_names:
        _get_ragas_faithfulness_scorer()
    client = Client(base_url=settings.phoenix_client_endpoint)
    inputs, outputs, metadata = _to_phoenix_dataset_rows(examples)
    dataset = client.datasets.create_dataset(
        name=dataset_name,
        dataset_description="Imperial RAG gold questions loaded from evals/questions.jsonl.",
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
    )
    runtime = build_runtime(settings=settings)

    def bound_target(inputs: dict[str, Any]) -> dict[str, Any]:
        return run_target(inputs, runtime=runtime)

    experiment = client.experiments.run_experiment(
        dataset=dataset,
        task=bound_target,
        evaluators=evaluators,
        experiment_name=experiment_name,
        experiment_description=_phoenix_experiment_description(resolved_ragas_metric_names),
    )
    print(f"phoenix_dataset={dataset_name}")
    print(f"phoenix_examples={len(examples)}")
    print(f"phoenix_experiment={_experiment_identifier(experiment)}")


def _run_phoenix_experiment(
    examples: list[dict[str, Any]],
    settings: Any,
    dataset_name: str,
    experiment_name: str,
    ragas_metric_names: list[str] | None = None,
) -> None:
    return run_phoenix_experiment(
        examples=examples,
        settings=settings,
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        ragas_metric_names=ragas_metric_names,
    )


def _configure_observability(settings: Any) -> None:
    from imperial_rag.observability import configure_observability

    configure_observability(settings)


def _log_eval_completion(started_at: float, **fields: Any) -> None:
    from imperial_rag.observability import log_event

    log_event(
        "imperial_rag.phoenix_eval",
        component="cli",
        duration_ms=_duration_ms(started_at),
        **fields,
    )


def _log_failure(operation: str, exc: BaseException, started_at: float, **fields: Any) -> None:
    from imperial_rag.observability import log_failure

    log_failure(operation, exc, component="cli", duration_ms=_duration_ms(started_at), **fields)


def _duration_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)


def parse_phoenix_ragas_metrics(raw_metrics: str | None) -> list[str]:
    from imperial_rag.ragas_eval import parse_ragas_metric_names

    return parse_ragas_metric_names(raw_metrics, default=("faithfulness",), allow_none=True)


def _validate_phoenix_ragas_metric_requirements(
    metric_names: list[str],
    examples: list[dict[str, Any]],
) -> None:
    from imperial_rag.ragas_eval import validate_ragas_metric_requirements

    validate_ragas_metric_requirements(
        metric_names,
        examples,
        reference_key="reference_answer",
        row_label_key="question",
    )


def _phoenix_evaluators(metric_names: list[str]) -> list[Any]:
    unsupported = sorted(set(metric_names) - {"faithfulness"})
    if unsupported:
        raise SystemExit(
            "Phoenix Ragas evaluators currently support only faithfulness. "
            "Run scripts/run_ragas_eval.py for reference-based Ragas metrics."
        )
    evaluators: list[Any] = [phoenix_citation_behavior, phoenix_source_hint_behavior, phoenix_retrieval_relevance]
    if "faithfulness" in metric_names:
        evaluators.append(phoenix_ragas_faithfulness)
    return evaluators


def _phoenix_experiment_description(metric_names: list[str]) -> str:
    if "faithfulness" in metric_names:
        return "Imperial RAG deterministic citation/refusal/source-hint checks plus Ragas Faithfulness."
    return "Imperial RAG deterministic citation/refusal/source-hint checks."


def _to_phoenix_dataset_rows(
    examples: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for row_index, example in enumerate(examples):
        expected = {
            "expected_behavior": example["expected_behavior"],
            "expected_source_hints": example.get("expected_source_hints", []),
        }
        if example.get("reference_answer"):
            expected["reference_answer"] = example["reference_answer"]
        stable_payload = json.dumps(
            {"question": example["question"], "expected": expected},
            ensure_ascii=False,
            sort_keys=True,
        )
        example_id = str(example.get("id") or hashlib.sha1(stable_payload.encode("utf-8")).hexdigest())
        inputs.append({"question": example["question"]})
        outputs.append(expected)
        metadata.append({"id": example_id, "row_index": row_index, "source": str(DEFAULT_QUESTIONS_PATH)})
    return inputs, outputs, metadata


def retrieval_relevance_metrics(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
    *,
    k: int = DEFAULT_RETRIEVAL_METRIC_K,
) -> dict[str, Any]:
    hints = _reference_hints(reference_outputs or inputs)
    if not hints:
        return {
            "score": None,
            "label": "skipped",
            "explanation": "Retrieval relevance requires expected_source_hints.",
            "metadata": {"k": k, "reason": "missing_expected_source_hints", "document_scores": []},
        }

    documents = list(outputs.get("documents") or outputs.get("evidence") or [])
    document_scores = [_document_relevance_score(document, hints) for document in documents]
    ranked_scores = document_scores[:k]
    hit = any(score > 0 for score in ranked_scores)
    precision = sum(ranked_scores) / k if k > 0 else 0.0
    ndcg = _ndcg(ranked_scores)
    relevant_count = int(sum(1 for score in document_scores if score > 0))
    return {
        "score": precision,
        "label": "hit" if hit else "miss",
        "explanation": f"{relevant_count} of {len(document_scores)} retrieved documents matched expected source hints.",
        "metadata": {
            "k": k,
            f"hit_at_{k}": hit,
            f"precision_at_{k}": precision,
            f"ndcg_at_{k}": ndcg,
            "document_scores": document_scores,
            "document_count": len(document_scores),
            "relevant_document_count": relevant_count,
        },
    }


def log_phoenix_eval_annotations(
    client: Any,
    *,
    span_id: str | None = None,
    retrieval_span_id: str | None = None,
    answer_metrics: list[dict[str, Any]] | None = None,
    retrieval_metrics: dict[str, Any] | None = None,
    sync: bool = False,
) -> None:
    span_annotations: list[dict[str, Any]] = []
    for metric in answer_metrics or []:
        if not span_id:
            continue
        result = _annotation_result(metric)
        if result:
            span_annotations.append(
                {
                    "name": str(metric["name"]),
                    "span_id": span_id,
                    "annotator_kind": "CODE",
                    "result": result,
                }
            )

    retrieval_metadata = dict((retrieval_metrics or {}).get("metadata") or {})
    if retrieval_span_id:
        for metric_name in _retrieval_span_metric_names(retrieval_metadata):
            span_annotations.append(
                {
                    "name": metric_name.replace("_at_", "@"),
                    "span_id": retrieval_span_id,
                    "annotator_kind": "CODE",
                    "result": {"score": float(retrieval_metadata[metric_name])},
                }
            )
    if span_annotations:
        client.spans.log_span_annotations(span_annotations=span_annotations, sync=sync)

    document_scores = retrieval_metadata.get("document_scores") or []
    if retrieval_span_id and document_scores:
        client.spans.log_document_annotations(
            document_annotations=[
                {
                    "name": "relevance",
                    "span_id": retrieval_span_id,
                    "document_position": position,
                    "annotator_kind": "CODE",
                    "result": {
                        "score": float(score),
                        "label": "relevant" if float(score) > 0 else "not_relevant",
                    },
                }
                for position, score in enumerate(document_scores)
            ],
            sync=sync,
        )


def _configure_tracing(settings: Any, enabled: bool) -> None:
    from imperial_rag.tracing import configure_phoenix_tracing

    configure_phoenix_tracing(settings, enabled=enabled)


def _get_ragas_faithfulness_scorer() -> Any:
    global _RAGAS_FAITHFULNESS_SCORER
    if _RAGAS_FAITHFULNESS_SCORER is None:
        from imperial_rag.ragas_eval import build_faithfulness_scorer

        _RAGAS_FAITHFULNESS_SCORER = build_faithfulness_scorer()
    return _RAGAS_FAITHFULNESS_SCORER


def _build_settings(workspace_root: Path | None) -> Any:
    from imperial_rag.config import Settings

    if workspace_root is None:
        return Settings()
    try:
        return Settings(workspace_root=workspace_root)
    except TypeError:
        os.environ["IMPERIAL_RAG_WORKSPACE_ROOT"] = str(workspace_root)
        return Settings()


def _load_project_env(workspace_root: Path | None) -> None:
    from imperial_rag.env import load_project_env

    load_project_env(workspace_root)


def _looks_like_refusal(answer: str) -> bool:
    normalized = answer.casefold()
    return any(text.casefold() in normalized for text in REFUSAL_FALLBACKS)


def _mentions_conflict(answer: str) -> bool:
    normalized = answer.casefold()
    return any(marker in normalized for marker in ("противореч", "конфликт", "disagree", "conflict"))


def _coerce_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {
        "answer": getattr(result, "answer", ""),
        "citations": getattr(result, "citations", []),
        "sources": getattr(result, "sources", []),
        "evidence": getattr(result, "evidence", []),
    }


def _document_payload(document: Any) -> dict[str, Any]:
    if isinstance(document, dict):
        return document
    return {
        "page_content": str(getattr(document, "page_content", "")),
        "metadata": dict(getattr(document, "metadata", {}) or {}),
    }


def _document_search_text(document: dict[str, Any]) -> str:
    metadata = document.get("metadata", {}) or {}
    return " ".join(
        [
            str(document.get("page_content", "")),
            *(
                str(metadata.get(field, ""))
                for field in ("relative_path", "file_name", "parent_folder", "section_heading")
            ),
        ]
    )


def _reference_hints(reference: Mapping[str, Any]) -> list[str]:
    return [str(hint).casefold() for hint in reference.get("expected_source_hints", []) if str(hint).strip()]


def _document_relevance_score(document: Any, hints: list[str]) -> float:
    if isinstance(document, Mapping):
        haystack = _document_search_text(dict(document))
    else:
        haystack = _document_search_text(_document_payload(document))
    normalized = haystack.casefold()
    return 1.0 if any(hint in normalized for hint in hints) else 0.0


def _ndcg(scores: list[float]) -> float:
    if not scores:
        return 0.0
    ideal = sorted(scores, reverse=True)
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(scores) / ideal_dcg


def _dcg(scores: list[float]) -> float:
    return sum(float(score) / math.log2(index + 2) for index, score in enumerate(scores))


def _annotation_result(metric: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if metric.get("score") is not None:
        result["score"] = float(metric["score"])
    if metric.get("label") is not None:
        result["label"] = str(metric["label"])
    if metric.get("explanation") is not None:
        result["explanation"] = str(metric["explanation"])
    return result


def _retrieval_span_metric_names(metadata: Mapping[str, Any]) -> list[str]:
    return [
        key
        for key in metadata
        if (key.startswith("precision_at_") or key.startswith("ndcg_at_")) and metadata.get(key) is not None
    ]


def _experiment_identifier(experiment: Any) -> str:
    fields = ("id", "experiment_id", "name")
    if isinstance(experiment, Mapping):
        for field in fields:
            value = experiment.get(field)
            if value:
                return str(value)
    for field in fields:
        value = getattr(experiment, field, None)
        if value:
            return str(value)
    return str(experiment)


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


if __name__ == "__main__":
    main()
