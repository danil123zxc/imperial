from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Any


DEFAULT_QUESTIONS_PATH = Path("evals/questions.jsonl")
DEFAULT_EXPERIMENT_NAME = "imperial-rag-citation-grounding"
DEFAULT_RETRIEVAL_METRIC_K = 5
VALID_EXPECTED_BEHAVIORS = {"cite_answer", "refuse_if_not_found", "surface_conflict"}
VALID_LANES = {
    "indexed_answerability",
    "conflict_version_behavior",
    "refusal_out_of_corpus_behavior",
    "known_missing_document_coverage",
}
LANES_BY_EXPECTED_BEHAVIOR = {
    "cite_answer": {"indexed_answerability", "known_missing_document_coverage"},
    "surface_conflict": {"conflict_version_behavior"},
    "refuse_if_not_found": {"refusal_out_of_corpus_behavior", "known_missing_document_coverage"},
}
CITE_ANSWER_BEHAVIOR = "cite_answer"
ANSWER_QUALITY_METRIC_KEYS = {
    "faithfulness": "ragas_faithfulness",
    "answer_relevancy": "ragas_answer_relevancy",
}
REFUSAL_FALLBACKS = (
    "I could not find this clearly in the indexed documents.",
    "не удалось найти",
    "не найдено",
    "нет в проиндексированных документах",
)
_RAGAS_FAITHFULNESS_SCORER: Any | None = None
_RAGAS_ANSWER_RELEVANCY_SCORER: Any | None = None


def load_questions(path: Path = DEFAULT_QUESTIONS_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        errors.extend(_validate_question_row(payload, line_number=line_number, seen_ids=seen_ids))
        rows.append(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return rows


def _validate_question_row(
    payload: Mapping[str, Any],
    *,
    line_number: int,
    seen_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    row_id = str(payload.get("id") or "").strip()
    if not row_id:
        errors.append(f"line {line_number}: missing id")
    elif row_id in seen_ids:
        errors.append(f"line {line_number}: duplicate id {row_id}")
    else:
        seen_ids.add(row_id)

    if not str(payload.get("suite") or "").strip():
        errors.append(f"line {line_number}: missing suite")
    if not payload.get("question"):
        errors.append(f"line {line_number}: missing question")
    expected_behavior = payload.get("expected_behavior")
    if expected_behavior not in VALID_EXPECTED_BEHAVIORS:
        errors.append(f"line {line_number}: invalid expected_behavior")
    lane = str(payload.get("lane") or "").strip()
    if not lane:
        errors.append(f"line {line_number}: missing lane")
    elif lane not in VALID_LANES:
        errors.append(f"line {line_number}: invalid lane")
    elif expected_behavior in LANES_BY_EXPECTED_BEHAVIOR and lane not in LANES_BY_EXPECTED_BEHAVIOR[expected_behavior]:
        errors.append(f"line {line_number}: lane is not valid for expected_behavior")
    if not str(payload.get("reference_answer") or "").strip():
        errors.append(f"line {line_number}: missing reference_answer")

    source_hints = payload.get("expected_source_hints", [])
    if not isinstance(source_hints, list) or any(not isinstance(hint, str) for hint in source_hints):
        errors.append(f"line {line_number}: expected_source_hints must be a list of strings")
    tags = payload.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in tags):
        errors.append(f"line {line_number}: tags must be a list of non-empty strings")
    if "reference_context_ids" in payload:
        context_ids = payload.get("reference_context_ids")
        if not isinstance(context_ids, list) or any(
            not isinstance(context_id, str) or not context_id.strip() for context_id in context_ids
        ):
            errors.append(f"line {line_number}: reference_context_ids must be a list of non-empty strings")
    return errors


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

    if not _answer_quality_metric_applies(expected or input or {}):
        return _metric_not_applicable_result("faithfulness", expected or input or {})
    return score_faithfulness_for_phoenix(
        input=input or {},
        output=output or {},
        scorer=_get_ragas_faithfulness_scorer(),
    )


def phoenix_ragas_answer_relevancy(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from imperial_rag.ragas_eval import score_answer_relevancy_for_phoenix

    if not _answer_quality_metric_applies(expected or input or {}):
        return _metric_not_applicable_result("answer_relevancy", expected or input or {})
    return score_answer_relevancy_for_phoenix(
        input=input or {},
        output=output or {},
        scorer=_get_ragas_answer_relevancy_scorer(),
    )


def phoenix_id_context_recall(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from imperial_rag.ragas_eval import score_id_context_recall_for_phoenix

    return score_id_context_recall_for_phoenix(
        input=input or {},
        output=output or {},
        expected=expected or {},
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
        default="faithfulness,answer_relevancy",
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
    if ragas_metric_names is None:
        from imperial_rag.ragas_eval import DEFAULT_RAGAS_METRICS

        resolved_ragas_metric_names = list(DEFAULT_RAGAS_METRICS)
    else:
        resolved_ragas_metric_names = list(ragas_metric_names)
    _validate_phoenix_ragas_metric_requirements(resolved_ragas_metric_names, examples)
    evaluators = _phoenix_evaluators(resolved_ragas_metric_names)

    try:
        from phoenix.client import Client
    except ImportError as exc:
        raise SystemExit("Phoenix client is not installed; install arize-phoenix-client.") from exc

    has_answer_quality_rows = any(
        example.get("expected_behavior") == CITE_ANSWER_BEHAVIOR for example in examples
    )
    if "faithfulness" in resolved_ragas_metric_names and has_answer_quality_rows:
        _get_ragas_faithfulness_scorer()
    if "answer_relevancy" in resolved_ragas_metric_names and has_answer_quality_rows:
        _get_ragas_answer_relevancy_scorer()
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
    from imperial_rag.cli import configure_observability

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
    from imperial_rag.cli import log_failure

    log_failure(operation, exc, started_at, **fields)


def _duration_ms(started_at: float) -> int:
    from imperial_rag.cli import duration_ms

    return duration_ms(started_at)


def parse_phoenix_ragas_metrics(raw_metrics: str | None) -> list[str]:
    from imperial_rag.ragas_eval import DEFAULT_RAGAS_METRICS, parse_ragas_metric_names

    return parse_ragas_metric_names(raw_metrics, default=DEFAULT_RAGAS_METRICS, allow_none=True)


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
    unsupported = sorted(set(metric_names) - {"faithfulness", "answer_relevancy", "id_context_recall"})
    if unsupported:
        raise SystemExit(
            "Phoenix Ragas evaluators currently support faithfulness, answer_relevancy, and id_context_recall. "
            "Run scripts/run_ragas_eval.py for reference-based Ragas metrics."
        )
    evaluators: list[Any] = [phoenix_citation_behavior, phoenix_source_hint_behavior, phoenix_retrieval_relevance]
    if "faithfulness" in metric_names:
        evaluators.append(phoenix_ragas_faithfulness)
    if "answer_relevancy" in metric_names:
        evaluators.append(phoenix_ragas_answer_relevancy)
    if "id_context_recall" in metric_names:
        evaluators.append(phoenix_id_context_recall)
    return evaluators


def _phoenix_experiment_description(metric_names: list[str]) -> str:
    if "faithfulness" in metric_names and "answer_relevancy" in metric_names and "id_context_recall" in metric_names:
        return (
            "Imperial RAG deterministic citation/refusal/source-hint checks plus Ragas Faithfulness, "
            "Answer Relevancy, and ID context recall."
        )
    if "faithfulness" in metric_names and "answer_relevancy" in metric_names:
        return (
            "Imperial RAG deterministic citation/refusal/source-hint checks plus Ragas Faithfulness "
            "and Answer Relevancy."
        )
    if "answer_relevancy" in metric_names and "id_context_recall" in metric_names:
        return (
            "Imperial RAG deterministic citation/refusal/source-hint checks plus Ragas Answer Relevancy "
            "and ID context recall."
        )
    if "faithfulness" in metric_names and "id_context_recall" in metric_names:
        return "Imperial RAG deterministic citation/refusal/source-hint checks plus Ragas Faithfulness and ID context recall."
    if "faithfulness" in metric_names:
        return "Imperial RAG deterministic citation/refusal/source-hint checks plus Ragas Faithfulness."
    if "answer_relevancy" in metric_names:
        return "Imperial RAG deterministic citation/refusal/source-hint checks plus Ragas Answer Relevancy."
    if "id_context_recall" in metric_names:
        return "Imperial RAG deterministic citation/refusal/source-hint checks plus Ragas ID context recall."
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
        if example.get("lane"):
            expected["lane"] = example["lane"]
        if example.get("reference_answer"):
            expected["reference_answer"] = example["reference_answer"]
        if "reference_context_ids" in example:
            expected["reference_context_ids"] = [
                str(context_id).strip() for context_id in example.get("reference_context_ids") or []
            ]
        if example.get("quarantine_reason"):
            expected["quarantine_reason"] = str(example["quarantine_reason"]).strip()
        stable_payload = json.dumps(
            {"question": example["question"], "expected": expected},
            ensure_ascii=False,
            sort_keys=True,
        )
        example_id = str(example.get("id") or hashlib.sha1(stable_payload.encode("utf-8")).hexdigest())
        inputs.append({"question": example["question"]})
        outputs.append(expected)
        metadata.append({"id": example_id, "row_index": row_index, "source": str(DEFAULT_QUESTIONS_PATH)})
        if example.get("suite"):
            metadata[-1]["suite"] = example["suite"]
        if "tags" in example:
            metadata[-1]["tags"] = list(example.get("tags") or [])
        if example.get("lane"):
            metadata[-1]["lane"] = str(example["lane"])
        if example.get("quarantine_reason"):
            metadata[-1]["quarantine_reason"] = str(example["quarantine_reason"]).strip()
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


def build_eval_artifact_row(
    *,
    example: Mapping[str, Any],
    output: Mapping[str, Any],
    ragas_results: Mapping[str, Mapping[str, Any]] | None = None,
    phoenix_experiment: str | None = None,
) -> dict[str, Any]:
    from imperial_rag.ragas_eval import retrieved_context_ids_from_output

    inputs = {"question": example["question"]}
    reference_outputs = {
        "expected_behavior": example["expected_behavior"],
        "expected_source_hints": example.get("expected_source_hints", []),
        "reference_context_ids": list(example.get("reference_context_ids") or []),
    }
    citation_verdict = citation_behavior(inputs, dict(output), reference_outputs)["score"]
    source_hint_verdict = source_hint_behavior(inputs, dict(output), reference_outputs)["score"]
    retrieval_metrics = retrieval_relevance_metrics(inputs, dict(output), reference_outputs)
    retrieval_metadata = retrieval_metrics.get("metadata", {})
    deterministic = {
        "citation_behavior": citation_verdict,
        "source_hint_behavior": source_hint_verdict,
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
    resolved_ragas_results = ragas_results or {}
    ragas_scores = {name: result.get("score") for name, result in resolved_ragas_results.items()}
    ragas_explanations = {name: result.get("explanation") for name, result in resolved_ragas_results.items()}
    return {
        "id": example.get("id"),
        "suite": example.get("suite"),
        "tags": list(example.get("tags") or []),
        "question": example["question"],
        "expected_behavior": example["expected_behavior"],
        "answer": str(output.get("answer") or ""),
        "citations": list(output.get("citations") or output.get("sources") or []),
        "retrieved_context_ids": retrieved_context_ids_from_output(dict(output)),
        "reference_context_ids": list(example.get("reference_context_ids") or []),
        "deterministic": deterministic,
        "ragas_scores": ragas_scores,
        "ragas_explanations": ragas_explanations,
        "phoenix_experiment": phoenix_experiment,
        "failure_class": classify_eval_failure(
            expected_behavior=str(example.get("expected_behavior") or ""),
            deterministic=deterministic,
            ragas_scores=ragas_scores,
        ),
    }


def classify_eval_failure(
    *,
    expected_behavior: str,
    deterministic: Mapping[str, Any],
    ragas_scores: Mapping[str, Any],
) -> str | None:
    if expected_behavior == "refuse_if_not_found" and deterministic.get("citation_behavior") is not True:
        return "bad_refusal"
    if expected_behavior == "surface_conflict" and deterministic.get("citation_behavior") is not True:
        return "bad_conflict_handling"
    if expected_behavior == "cite_answer" and deterministic.get("citation_behavior") is not True:
        return "missing_citation"
    if deterministic.get(f"retrieval_hit_at_{DEFAULT_RETRIEVAL_METRIC_K}") is False:
        return "retrieval_miss"
    if deterministic.get("source_hint_behavior") is False:
        return "noisy_retrieval"
    if _below_threshold(ragas_scores.get("faithfulness")):
        return "ungrounded_answer"
    if _below_threshold(ragas_scores.get("answer_relevancy")):
        return "irrelevant_answer"
    if _below_threshold(ragas_scores.get("factual_correctness")):
        return "incorrect_answer"
    return None


def _answer_quality_metric_applies(reference: Mapping[str, Any]) -> bool:
    behavior = reference.get("expected_behavior")
    return behavior in (None, "", CITE_ANSWER_BEHAVIOR)


def _metric_not_applicable_result(metric_name: str, reference: Mapping[str, Any]) -> dict[str, Any]:
    metric_key = ANSWER_QUALITY_METRIC_KEYS[metric_name]
    return {
        "score": None,
        "label": "skipped",
        "explanation": "Ragas answer-quality metrics are only applicable to cite_answer rows.",
        "metadata": {
            "metric": metric_key,
            "reason": "metric_not_applicable_for_behavior",
            "expected_behavior": reference.get("expected_behavior"),
        },
    }


def _below_threshold(value: Any, threshold: float = 0.5) -> bool:
    if value is None:
        return False
    try:
        return float(value) < threshold
    except (TypeError, ValueError):
        return False


def _configure_tracing(settings: Any, enabled: bool) -> None:
    from imperial_rag.cli import configure_tracing

    configure_tracing(settings, enabled=enabled)


def _get_ragas_faithfulness_scorer() -> Any:
    global _RAGAS_FAITHFULNESS_SCORER
    if _RAGAS_FAITHFULNESS_SCORER is None:
        from imperial_rag.ragas_eval import build_faithfulness_scorer

        _RAGAS_FAITHFULNESS_SCORER = build_faithfulness_scorer()
    return _RAGAS_FAITHFULNESS_SCORER


def _get_ragas_answer_relevancy_scorer() -> Any:
    global _RAGAS_ANSWER_RELEVANCY_SCORER
    if _RAGAS_ANSWER_RELEVANCY_SCORER is None:
        from imperial_rag.ragas_eval import build_answer_relevancy_scorer

        _RAGAS_ANSWER_RELEVANCY_SCORER = build_answer_relevancy_scorer()
    return _RAGAS_ANSWER_RELEVANCY_SCORER


def _build_settings(workspace_root: Path | None) -> Any:
    from imperial_rag.cli import build_settings

    return build_settings(workspace_root)


def _load_project_env(workspace_root: Path | None) -> None:
    from imperial_rag.cli import load_project_environment

    load_project_environment(workspace_root)


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
