from __future__ import annotations

import argparse
import asyncio
import hashlib
import math
import sys
from collections import Counter
from collections.abc import Mapping
from anyio.to_thread import run_sync as run_sync_in_worker_thread
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple, Sequence, cast

import anyio

from imperial_rag.cli import (  # noqa: E402
    build_settings as _build_settings,
    configure_observability as _configure_observability,
    configure_tracing as _configure_tracing,
    duration_ms as _duration_ms,
    load_project_environment as _load_project_env,
    log_failure as _log_failure,
    positive_int,
)
from imperial_rag.evals.dataset_input import (
    PhoenixInput, add_dataset_input_arguments, has_phoenix_input, load_phoenix_input,
    validate_dataset_input_arguments,
)
from imperial_rag.evals.corpus import clean_context_ids as _clean_context_ids  # noqa: E402
from imperial_rag.evals.corpus import unique_nonempty as _unique_nonempty  # noqa: E402
from imperial_rag.jsonl import iter_jsonl_with_line_numbers  # noqa: E402
from imperial_rag.serialization import stable_json_dumps  # noqa: E402


DEFAULT_QUESTIONS_PATH = Path("evals/questions.jsonl")
DEFAULT_EXPERIMENT_NAME = "imperial-rag-citation-grounding"
DEFAULT_PHOENIX_CONCURRENCY = 3
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


class _RankedIdOverlap(NamedTuple):
    reference_set: set[str]
    ranked_scores: list[float]
    matched_ids: list[str]
    recall: float

    @property
    def hit(self) -> bool:
        return bool(self.matched_ids)

    def precision_at(self, k: int) -> float:
        return sum(self.ranked_scores) / k if k > 0 else 0.0


def load_questions(path: Path = DEFAULT_QUESTIONS_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_ids: set[str] = set()
    for line_number, payload in iter_jsonl_with_line_numbers(path):
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
    output = {
        "answer": str(result.get("answer", "")),
        "citations": list(result.get("citations") or result.get("sources") or []),
        "sources": list(result.get("sources") or result.get("citations") or []),
        "documents": [_document_payload(document) for document in evidence],
        "retrieval": dict(result.get("retrieval") or {}),
    }
    if "ranked_documents" in result:
        output["ranked_documents"] = [_document_payload(doc) for doc in result["ranked_documents"]]
    if result.get("error"):
        output["error"] = result["error"]
    return output


def build_runtime(settings: Any | None = None) -> Any:
    try:
        from imperial_rag.answering.runtime import create_runtime
    except (ImportError, AttributeError):
        create_runtime = None
    if create_runtime is not None:
        return create_runtime(settings) if settings is not None else create_runtime()

    try:
        from imperial_rag.answering.runtime import Runtime
    except (ImportError, AttributeError):
        Runtime = None

    if Runtime is not None:
        RuntimeClass = cast(Any, Runtime)
        return RuntimeClass(settings=settings) if settings is not None else RuntimeClass()

    from imperial_rag.answering.runtime import build_live_query_workflow

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


def citation_grounding_behavior(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference = reference_outputs or inputs
    citations = _citation_values(outputs)
    resolved_ids: list[str] = []
    unresolved: list[str] = []
    documents = list(outputs.get("documents") or outputs.get("evidence") or [])
    for citation in citations:
        document_ids = _document_ids_matching_citation(citation, documents)
        if document_ids:
            resolved_ids.extend(document_ids)
        else:
            unresolved.append(citation)

    cited_ids = _unique_nonempty(resolved_ids)
    reference_ids = _clean_context_ids(reference.get("reference_context_ids") or [])
    gold_overlap = not reference_ids or bool(set(cited_ids) & set(reference_ids))
    expected_behavior = reference.get("expected_behavior")
    if not citations:
        score: bool | None = False if expected_behavior in {"cite_answer", "surface_conflict"} else None
    else:
        score = not unresolved and gold_overlap
    return {
        "key": "citation_grounding_behavior",
        "score": score,
        "label": _bool_label(score),
        "metadata": {
            "citation_count": len(citations),
            "resolved_citation_count": len(citations) - len(unresolved),
            "unresolved_citations": unresolved,
            "cited_context_ids": cited_ids,
            "reference_context_ids": reference_ids,
            "gold_overlap": gold_overlap,
        },
    }


def conflict_behavior(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reference = reference_outputs or inputs
    if reference.get("expected_behavior") != "surface_conflict":
        return {
            "key": "conflict_behavior",
            "score": None,
            "label": "skipped",
            "metadata": {"reason": "metric_not_applicable_for_behavior"},
        }

    answer = str(outputs.get("answer", ""))
    grounding = citation_grounding_behavior(inputs, outputs, reference)
    grounding_metadata = grounding.get("metadata", {})
    reference_ids = _clean_context_ids(reference.get("reference_context_ids") or [])
    cited_ids = _clean_context_ids(grounding_metadata.get("cited_context_ids") or [])
    cited_reference_count = len(set(cited_ids) & set(reference_ids))
    required_reference_count = min(2, len(reference_ids)) if reference_ids else 2
    mentions_conflict = _mentions_conflict(answer)
    decisive_claim = _has_decisive_version_claim(answer)
    score = (
        mentions_conflict
        and not decisive_claim
        and bool(grounding.get("score"))
        and cited_reference_count >= required_reference_count
    )
    return {
        "key": "conflict_behavior",
        "score": bool(score),
        "label": _bool_label(bool(score)),
        "metadata": {
            "mentions_conflict": mentions_conflict,
            "decisive_version_claim": decisive_claim,
            "cited_reference_context_count": cited_reference_count,
            "required_reference_context_count": required_reference_count,
            "cited_context_ids": cited_ids,
            "reference_context_ids": reference_ids,
            "citation_grounding_score": grounding.get("score"),
        },
    }


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


def phoenix_citation_grounding_behavior(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return citation_grounding_behavior(input or {}, output or {}, expected)


def phoenix_conflict_behavior(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return conflict_behavior(input or {}, output or {}, expected)


def phoenix_retrieval_relevance(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return retrieval_relevance_metrics(input or {}, output or {}, expected)


def phoenix_id_retrieval_relevance(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return id_retrieval_metrics(input or {}, output or {}, expected)


def phoenix_retrieval_relevance_for_k(k: int) -> Any:
    retrieval_k = positive_int(k)
    if retrieval_k == DEFAULT_RETRIEVAL_METRIC_K:
        return phoenix_retrieval_relevance

    def evaluator(
        output: dict[str, Any],
        expected: dict[str, Any] | None = None,
        input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return retrieval_relevance_metrics(input or {}, output or {}, expected, k=retrieval_k)

    evaluator.__name__ = f"phoenix_retrieval_relevance_at_{retrieval_k}"
    return evaluator


def phoenix_id_retrieval_relevance_for_k(k: int) -> Any:
    retrieval_k = positive_int(k)
    if retrieval_k == DEFAULT_RETRIEVAL_METRIC_K:
        return phoenix_id_retrieval_relevance

    def evaluator(
        output: dict[str, Any],
        expected: dict[str, Any] | None = None,
        input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return id_retrieval_metrics(input or {}, output or {}, expected, k=retrieval_k)

    evaluator.__name__ = f"phoenix_id_retrieval_relevance_at_{retrieval_k}"
    return evaluator


def phoenix_ragas_faithfulness(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from imperial_rag.evals.ragas import score_faithfulness_for_phoenix

    if not _answer_quality_metric_applies(expected or input or {}):
        return _metric_not_applicable_result("faithfulness", expected or input or {})
    return score_faithfulness_for_phoenix(
        input=input or {},
        output=output or {},
        scorer=_get_ragas_faithfulness_scorer(),
    )


async def phoenix_ragas_faithfulness_async(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from imperial_rag.evals.ragas import score_faithfulness_for_phoenix_async

    if not _answer_quality_metric_applies(expected or input or {}):
        return _metric_not_applicable_result("faithfulness", expected or input or {})
    return await score_faithfulness_for_phoenix_async(
        input=input or {},
        output=output or {},
        scorer=_get_ragas_faithfulness_scorer(),
    )


def phoenix_ragas_answer_relevancy(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from imperial_rag.evals.ragas import score_answer_relevancy_for_phoenix

    if not _answer_quality_metric_applies(expected or input or {}):
        return _metric_not_applicable_result("answer_relevancy", expected or input or {})
    return score_answer_relevancy_for_phoenix(
        input=input or {},
        output=output or {},
        scorer=_get_ragas_answer_relevancy_scorer(),
    )


async def phoenix_ragas_answer_relevancy_async(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from imperial_rag.evals.ragas import score_answer_relevancy_for_phoenix_async

    if not _answer_quality_metric_applies(expected or input or {}):
        return _metric_not_applicable_result("answer_relevancy", expected or input or {})
    return await score_answer_relevancy_for_phoenix_async(
        input=input or {},
        output=output or {},
        scorer=_get_ragas_answer_relevancy_scorer(),
    )


def phoenix_id_context_recall(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from imperial_rag.evals.ragas import score_id_context_recall_for_phoenix

    return score_id_context_recall_for_phoenix(
        input=input or {},
        output=output or {},
        expected=expected or {},
    )


async def phoenix_id_context_recall_async(
    output: dict[str, Any],
    expected: dict[str, Any] | None = None,
    input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from imperial_rag.evals.ragas import score_id_context_recall_for_phoenix_async

    return await score_id_context_recall_for_phoenix_async(
        input=input or {},
        output=output or {},
        expected=expected or {},
    )


def run_local_eval(
    examples: list[dict[str, Any]],
    runtime: Any | None = None,
    *,
    retrieval_k: int = DEFAULT_RETRIEVAL_METRIC_K,
) -> list[dict[str, Any]]:
    retrieval_k = positive_int(retrieval_k)
    resolved_runtime = runtime or build_runtime()
    rows: list[dict[str, Any]] = []
    for example in examples:
        inputs = {"question": example["question"]}
        reference_outputs = {
            "expected_behavior": example["expected_behavior"],
            "expected_source_hints": example.get("expected_source_hints", []),
            "reference_context_ids": example.get("reference_context_ids", []),
        }
        outputs = run_target(inputs, runtime=resolved_runtime)
        retrieval_metrics = retrieval_relevance_metrics(inputs, outputs, reference_outputs, k=retrieval_k)
        retrieval_metadata = retrieval_metrics.get("metadata", {})
        id_metrics = id_retrieval_metrics(inputs, outputs, reference_outputs, k=retrieval_k)
        id_metadata = id_metrics.get("metadata", {})
        citation_grounding = citation_grounding_behavior(inputs, outputs, reference_outputs)
        conflict = conflict_behavior(inputs, outputs, reference_outputs)
        rows.append(
            {
                "question": example["question"],
                "citation_behavior": citation_behavior(inputs, outputs, reference_outputs)["score"],
                "source_hint_behavior": source_hint_behavior(inputs, outputs, reference_outputs)["score"],
                "citation_grounding_behavior": citation_grounding["score"],
                "conflict_behavior": conflict["score"],
                **_deterministic_retrieval_values(
                    retrieval_metadata,
                    id_metadata,
                    retrieval_k=retrieval_k,
                ),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    _ensure_src_on_path()
    parser = argparse.ArgumentParser(description="Run Imperial RAG citation/refusal evaluations.")
    add_dataset_input_arguments(parser)
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
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        default=DEFAULT_PHOENIX_CONCURRENCY,
        help="Maximum concurrent Phoenix experiment tasks.",
    )
    parser.add_argument(
        "--retrieval-k",
        type=positive_int,
        default=DEFAULT_RETRIEVAL_METRIC_K,
        help="Top-k cutoff for deterministic retrieval relevance metrics.",
    )
    args = parser.parse_args(argv)
    validate_dataset_input_arguments(parser, args)

    _load_project_env(args.workspace_root)
    settings = _build_settings(args.workspace_root)
    _configure_observability(settings)
    started_at = perf_counter()
    try:
        if args.trace_phoenix or args.use_phoenix:
            _configure_tracing(settings, enabled=True)
        source = _run_async(load_phoenix_input(args, settings)) if has_phoenix_input(args) else None
        examples = source.examples if source else load_questions(args.questions_path or DEFAULT_QUESTIONS_PATH)
        if source:
            print(f"phoenix_input={stable_json_dumps(source.binding)}")
        metric_names = parse_phoenix_ragas_metrics(args.ragas_metrics)

        if args.use_phoenix:
            run_phoenix_experiment(
                examples=examples,
                settings=settings,
                dataset_name=args.dataset_name or f"{settings.phoenix_project_name}-gold-questions",
                experiment_name=args.experiment_name,
                ragas_metric_names=metric_names,
                concurrency=args.concurrency,
                retrieval_k=args.retrieval_k,
                **({"phoenix_input": source} if source else {}),
            )
            _log_eval_completion(
                started_at,
                operation="phoenix_eval",
                status="success",
                example_count=len(examples),
                phoenix_mode=True,
                ragas_metrics=",".join(metric_names),
                retrieval_k=args.retrieval_k,
            )
            return

        rows = run_local_eval(
            examples,
            runtime=build_runtime(settings=settings),
            retrieval_k=args.retrieval_k,
        )
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
            retrieval_k=args.retrieval_k,
        )
    except (Exception, SystemExit) as exc:
        _log_failure(
            "phoenix_eval",
            exc,
            started_at,
            phoenix_mode=args.use_phoenix,
            ragas_metrics=args.ragas_metrics,
            retrieval_k=args.retrieval_k,
        )
        raise


def run_phoenix_experiment(
    examples: list[dict[str, Any]],
    settings: Any,
    dataset_name: str,
    experiment_name: str,
    ragas_metric_names: list[str] | None = None,
    *,
    concurrency: int = DEFAULT_PHOENIX_CONCURRENCY,
    retrieval_k: int = DEFAULT_RETRIEVAL_METRIC_K,
    evidence_snapshot: dict[str, Any] | None = None,
    phoenix_input: PhoenixInput | None = None,
) -> None:
    return _run_async(
        run_phoenix_experiment_async(
            examples=examples,
            settings=settings,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            ragas_metric_names=ragas_metric_names,
            concurrency=concurrency,
            retrieval_k=retrieval_k,
            evidence_snapshot=evidence_snapshot,
            phoenix_input=phoenix_input,
        )
    )


async def run_phoenix_experiment_async(
    examples: list[dict[str, Any]],
    settings: Any,
    dataset_name: str,
    experiment_name: str,
    ragas_metric_names: list[str] | None = None,
    *,
    concurrency: int = DEFAULT_PHOENIX_CONCURRENCY,
    retrieval_k: int = DEFAULT_RETRIEVAL_METRIC_K,
    evidence_snapshot: dict[str, Any] | None = None,
    phoenix_input: PhoenixInput | None = None,
) -> None:
    concurrency = positive_int(concurrency)
    retrieval_k = positive_int(retrieval_k)
    if phoenix_input is not None:
        from imperial_rag.evals.dataset_input import dataset_binding, map_phoenix_examples
        from imperial_rag.evals.evidence import assemble_benchmark

        if dataset_binding(phoenix_input.dataset, settings.phoenix_client_endpoint) != phoenix_input.binding:
            raise ValueError("Phoenix input binding changed")
        questions, annotations = map_phoenix_examples(phoenix_input.dataset)
        validated = (assemble_benchmark(questions, annotations, evidence_snapshot)["examples"]
                     if evidence_snapshot is not None else questions)
        if examples != validated:
            raise ValueError("Phoenix input does not match the experiment examples")
    if ragas_metric_names is None:
        from imperial_rag.evals.ragas import DEFAULT_RAGAS_METRICS

        resolved_ragas_metric_names = list(DEFAULT_RAGAS_METRICS)
    else:
        resolved_ragas_metric_names = list(ragas_metric_names)
    _validate_phoenix_ragas_metric_requirements(resolved_ragas_metric_names, examples)
    evaluators = _phoenix_evaluators(resolved_ragas_metric_names, async_mode=True, retrieval_k=retrieval_k)
    evidence_metadata: dict[str, Any] = {}
    if evidence_snapshot is not None:
        from imperial_rag.ingestion.provenance import digest

        if not examples or any("evidence" not in row or "split" not in row for row in examples):
            raise ValueError("Evidence evaluation requires assembled, validated benchmark examples")
        evidence_metadata = {
            "snapshot_hash": evidence_snapshot["snapshot_hash"],
            "dataset_hash": digest({"schema_version": "imperial-evidence-benchmark-v1",
                                    "snapshot_hash": evidence_snapshot["snapshot_hash"], "examples": examples}),
        }
        legacy_names = {"id_retrieval_relevance", "citation_grounding_behavior",
                        "conflict_behavior", "ragas_id_context_recall"}
        evaluators = {f"legacy_{name}" if name in legacy_names else name: evaluator
                      for name, evaluator in evaluators.items()}
        evaluators.update(_phoenix_evidence_evaluators(documents_key="ranked_documents"))

    try:
        from phoenix.client import AsyncClient
    except ImportError as exc:
        raise SystemExit("Phoenix client is not installed; install arize-phoenix-client.") from exc

    has_answer_quality_rows = any(
        example.get("expected_behavior") == CITE_ANSWER_BEHAVIOR for example in examples
    )
    if "faithfulness" in resolved_ragas_metric_names and has_answer_quality_rows:
        _get_ragas_faithfulness_scorer()
    if "answer_relevancy" in resolved_ragas_metric_names and has_answer_quality_rows:
        _get_ragas_answer_relevancy_scorer()
    client = AsyncClient(base_url=settings.phoenix_client_endpoint)
    if phoenix_input is not None:
        dataset = phoenix_input.dataset
        evidence_metadata["phoenix_dataset"] = phoenix_input.binding
        if phoenix_input.benchmark is not None:
            evidence_metadata["dataset_hash"] = phoenix_input.benchmark["dataset_hash"]
    else:
        inputs, outputs, metadata = _to_phoenix_dataset_rows(examples)
        for row, example in zip(metadata, examples):
            if evidence_metadata:
                row.update(evidence_metadata | {"split": example["split"]})
        dataset = await client.datasets.create_dataset(
            name=dataset_name,
            dataset_description=("Imperial RAG reviewed source evidence benchmark." if evidence_metadata else
                                 "Imperial RAG gold questions loaded from evals/questions.jsonl."),
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
    runtime = build_runtime(settings=settings)
    failed_questions: list[str] = []

    async def bound_target(inputs: dict[str, Any]) -> dict[str, Any]:
        if evidence_snapshot is None:
            return await run_sync_in_worker_thread(lambda: run_target(inputs, runtime=runtime))
        output = await run_sync_in_worker_thread(lambda: _run_evidence_target(inputs, runtime, evidence_snapshot))
        if not output["eligible"]:
            failed_questions.append(output["evidence_error"])
        return output

    experiment = await client.experiments.run_experiment(
        dataset=dataset,
        task=bound_target,
        experiment_metadata=evidence_metadata,
        evaluators=evaluators,
        experiment_name=experiment_name,
        experiment_description=(
            "Source evidence recall and full-evidence success; legacy_* checks are ID-based diagnostics. "
            if evidence_snapshot is not None else ""
        ) + _phoenix_experiment_description(resolved_ragas_metric_names),
        concurrency=concurrency,
    )
    print(f"phoenix_dataset={phoenix_input.binding['dataset_id'] if phoenix_input else dataset_name}")
    if phoenix_input is not None:
        print(f"phoenix_input={stable_json_dumps(phoenix_input.binding)}")
    print(f"phoenix_examples={len(examples)}")
    print(f"phoenix_experiment={_experiment_identifier(experiment)}")
    if failed_questions:
        raise RuntimeError(f"Evidence evaluation failed for {len(failed_questions)} task(s): "
                           + ", ".join(sorted(set(failed_questions))))


def _run_phoenix_experiment(
    examples: list[dict[str, Any]],
    settings: Any,
    dataset_name: str,
    experiment_name: str,
    ragas_metric_names: list[str] | None = None,
    *,
    concurrency: int = DEFAULT_PHOENIX_CONCURRENCY,
    retrieval_k: int = DEFAULT_RETRIEVAL_METRIC_K,
) -> None:
    return run_phoenix_experiment(
        examples=examples,
        settings=settings,
        dataset_name=dataset_name,
        experiment_name=experiment_name,
        ragas_metric_names=ragas_metric_names,
        concurrency=concurrency,
        retrieval_k=retrieval_k,
    )


def _log_eval_completion(started_at: float, **fields: Any) -> None:
    from imperial_rag.observability import log_event

    log_event(
        "imperial_rag.phoenix_eval",
        component="cli",
        duration_ms=_duration_ms(started_at),
        **fields,
    )


def parse_phoenix_ragas_metrics(raw_metrics: str | None) -> list[str]:
    from imperial_rag.evals.ragas import DEFAULT_RAGAS_METRICS, parse_ragas_metric_names

    return parse_ragas_metric_names(raw_metrics, default=DEFAULT_RAGAS_METRICS, allow_none=True)


def _validate_phoenix_ragas_metric_requirements(
    metric_names: Sequence[str],
    examples: list[dict[str, Any]],
) -> None:
    from imperial_rag.evals.ragas import validate_ragas_metric_requirements

    validate_ragas_metric_requirements(
        metric_names,
        examples,
        reference_key="reference_answer",
        row_label_key="question",
    )


def _phoenix_evaluators(
    metric_names: Sequence[str],
    *,
    async_mode: bool = False,
    retrieval_k: int = DEFAULT_RETRIEVAL_METRIC_K,
) -> dict[str, Any]:
    retrieval_k = positive_int(retrieval_k)
    unsupported = sorted(set(metric_names) - {"faithfulness", "answer_relevancy", "id_context_recall"})
    if unsupported:
        raise SystemExit(
            "Phoenix Ragas evaluators currently support faithfulness, answer_relevancy, and id_context_recall. "
            "Run scripts/run_ragas_eval.py for reference-based Ragas metrics."
        )
    evaluators: dict[str, Any] = {
        "citation_behavior": phoenix_citation_behavior,
        "source_hint_behavior": phoenix_source_hint_behavior,
        "citation_grounding_behavior": phoenix_citation_grounding_behavior,
        "conflict_behavior": phoenix_conflict_behavior,
        "retrieval_relevance": phoenix_retrieval_relevance_for_k(retrieval_k),
        "id_retrieval_relevance": phoenix_id_retrieval_relevance_for_k(retrieval_k),
    }
    if "faithfulness" in metric_names:
        evaluators["ragas_faithfulness"] = (
            phoenix_ragas_faithfulness_async if async_mode else phoenix_ragas_faithfulness
        )
    if "answer_relevancy" in metric_names:
        evaluators["ragas_answer_relevancy"] = (
            phoenix_ragas_answer_relevancy_async if async_mode else phoenix_ragas_answer_relevancy
        )
    if "id_context_recall" in metric_names:
        evaluators["ragas_id_context_recall"] = (
            phoenix_id_context_recall_async if async_mode else phoenix_id_context_recall
        )
    return evaluators


def _phoenix_experiment_description(metric_names: Sequence[str]) -> str:
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
        expected = _phoenix_dataset_expected_payload(example)
        inputs.append({"question": example["question"]})
        outputs.append(expected)
        metadata.append(_phoenix_dataset_metadata(example, row_index=row_index, expected=expected))
    return inputs, outputs, metadata


def _phoenix_dataset_expected_payload(example: Mapping[str, Any]) -> dict[str, Any]:
    expected: dict[str, Any] = {
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
    if "evidence" in example:
        expected["evidence"] = example["evidence"]
    return expected


def _phoenix_dataset_metadata(
    example: Mapping[str, Any],
    *,
    row_index: int,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    stable_payload = stable_json_dumps({"question": example["question"], "expected": expected})
    example_id = str(example.get("id") or hashlib.sha1(stable_payload.encode("utf-8")).hexdigest())
    metadata: dict[str, Any] = {"id": example_id, "row_index": row_index, "source": str(DEFAULT_QUESTIONS_PATH)}
    if example.get("suite"):
        metadata["suite"] = example["suite"]
    if "tags" in example:
        metadata["tags"] = list(example.get("tags") or [])
    if example.get("lane"):
        metadata["lane"] = str(example["lane"])
    if example.get("quarantine_reason"):
        metadata["quarantine_reason"] = str(example["quarantine_reason"]).strip()
    for key in ("split", "review_status", "question_hash", "snapshot_hash"):
        if key in example:
            metadata[key] = example[key]
    return metadata


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


def id_retrieval_metrics(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any] | None = None,
    *,
    k: int = DEFAULT_RETRIEVAL_METRIC_K,
) -> dict[str, Any]:
    from imperial_rag.evals.ragas import retrieved_context_ids_from_output

    reference_ids = _reference_context_ids(inputs, reference_outputs)
    retrieved_ids = retrieved_context_ids_from_output(outputs)
    if not reference_ids:
        return _missing_reference_context_id_metric(
            "ID retrieval relevance requires reference_context_ids.",
            k=k,
            retrieved_key="retrieved_context_ids",
            retrieved_ids=retrieved_ids,
            empty_fields={"id_document_scores": []},
        )

    overlap = _ranked_id_overlap(reference_ids, retrieved_ids, k)
    mrr = _mrr(overlap.ranked_scores)
    ndcg = _ndcg_with_ideal(overlap.ranked_scores, relevant_count=len(overlap.reference_set), k=k)
    return {
        "score": overlap.recall,
        "label": "hit" if overlap.hit else "miss",
        "explanation": f"{len(overlap.matched_ids)} of {len(overlap.reference_set)} gold context IDs appeared in the top {k}.",
        "metadata": {
            "k": k,
            f"id_hit_at_{k}": overlap.hit,
            f"id_precision_at_{k}": overlap.precision_at(k),
            f"id_recall_at_{k}": overlap.recall,
            f"id_mrr_at_{k}": mrr,
            f"id_ndcg_at_{k}": ndcg,
            "id_document_scores": overlap.ranked_scores,
            "retrieved_context_ids": retrieved_ids,
            "reference_context_ids": reference_ids,
            "matched_context_ids": overlap.matched_ids,
        },
    }


def _reference_context_ids(
    inputs: Mapping[str, Any],
    reference_outputs: Mapping[str, Any] | None,
) -> list[str]:
    return _clean_context_ids((reference_outputs or inputs).get("reference_context_ids") or [])


def _missing_reference_context_id_metric(
    explanation: str,
    *,
    k: int,
    retrieved_key: str,
    retrieved_ids: list[str],
    empty_fields: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "score": None,
        "label": "skipped",
        "explanation": explanation,
        "metadata": {
            "k": k,
            "reason": "missing_reference_context_ids",
            retrieved_key: retrieved_ids,
            "reference_context_ids": [],
            **empty_fields,
        },
    }


def _ranked_id_overlap(
    reference_ids: list[str],
    retrieved_ids: list[str],
    k: int,
) -> _RankedIdOverlap:
    reference_set = set(reference_ids)
    ranked_ids = retrieved_ids[:k]
    ranked_scores = [1.0 if context_id in reference_set else 0.0 for context_id in ranked_ids]
    matched_ids = _unique_nonempty(context_id for context_id in ranked_ids if context_id in reference_set)
    recall = len(matched_ids) / len(reference_set) if reference_set else 0.0
    return _RankedIdOverlap(
        reference_set=reference_set,
        ranked_scores=ranked_scores,
        matched_ids=matched_ids,
        recall=recall,
    )


def _deterministic_retrieval_values(
    retrieval_metadata: Mapping[str, Any],
    id_metadata: Mapping[str, Any],
    *,
    retrieval_k: int,
) -> dict[str, Any]:
    return {
        f"retrieval_hit_at_{retrieval_k}": retrieval_metadata.get(f"hit_at_{retrieval_k}"),
        f"retrieval_precision_at_{retrieval_k}": retrieval_metadata.get(f"precision_at_{retrieval_k}"),
        f"retrieval_ndcg_at_{retrieval_k}": retrieval_metadata.get(f"ndcg_at_{retrieval_k}"),
        f"id_hit_at_{retrieval_k}": id_metadata.get(f"id_hit_at_{retrieval_k}"),
        f"id_precision_at_{retrieval_k}": id_metadata.get(f"id_precision_at_{retrieval_k}"),
        f"id_recall_at_{retrieval_k}": id_metadata.get(f"id_recall_at_{retrieval_k}"),
        f"id_mrr_at_{retrieval_k}": id_metadata.get(f"id_mrr_at_{retrieval_k}"),
        f"id_ndcg_at_{retrieval_k}": id_metadata.get(f"id_ndcg_at_{retrieval_k}"),
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
    retrieval_k: int = DEFAULT_RETRIEVAL_METRIC_K,
) -> dict[str, Any]:
    from imperial_rag.evals.ragas import retrieved_chunk_ids_from_output, retrieved_context_ids_from_output

    retrieval_k = positive_int(retrieval_k)
    inputs = {"question": example["question"]}
    reference_outputs = {
        "expected_behavior": example["expected_behavior"],
        "expected_source_hints": example.get("expected_source_hints", []),
        "reference_context_ids": list(example.get("reference_context_ids") or []),
    }
    citation_verdict = citation_behavior(inputs, dict(output), reference_outputs)["score"]
    source_hint_verdict = source_hint_behavior(inputs, dict(output), reference_outputs)["score"]
    citation_grounding = citation_grounding_behavior(inputs, dict(output), reference_outputs)
    conflict_verdict = conflict_behavior(inputs, dict(output), reference_outputs)
    retrieval_metrics = retrieval_relevance_metrics(inputs, dict(output), reference_outputs, k=retrieval_k)
    retrieval_metadata = retrieval_metrics.get("metadata", {})
    id_metrics = id_retrieval_metrics(inputs, dict(output), reference_outputs, k=retrieval_k)
    id_metadata = id_metrics.get("metadata", {})
    deterministic = {
        "citation_behavior": citation_verdict,
        "source_hint_behavior": source_hint_verdict,
        "citation_grounding_behavior": citation_grounding.get("score"),
        "conflict_behavior": conflict_verdict.get("score"),
        **_deterministic_retrieval_values(
            retrieval_metadata,
            id_metadata,
            retrieval_k=retrieval_k,
        ),
    }
    resolved_ragas_results = ragas_results or {}
    ragas_scores = {name: result.get("score") for name, result in resolved_ragas_results.items()}
    ragas_explanations = {name: result.get("explanation") for name, result in resolved_ragas_results.items()}
    return {
        "id": example.get("id"),
        "suite": example.get("suite"),
        "tags": list(example.get("tags") or []),
        "lane": example.get("lane"),
        "question": example["question"],
        "expected_behavior": example["expected_behavior"],
        "answer": str(output.get("answer") or ""),
        "citations": list(output.get("citations") or output.get("sources") or []),
        "retrieved_context_ids": retrieved_context_ids_from_output(dict(output)),
        "retrieved_chunk_ids": retrieved_chunk_ids_from_output(dict(output)),
        "reference_context_ids": list(example.get("reference_context_ids") or []),
        "source_families": source_families_from_output(dict(output)),
        "deterministic": deterministic,
        "ragas_scores": ragas_scores,
        "ragas_explanations": ragas_explanations,
        "phoenix_experiment": phoenix_experiment,
        "failure_class": classify_eval_failure(
            expected_behavior=str(example.get("expected_behavior") or ""),
            deterministic=deterministic,
            ragas_scores=ragas_scores,
            retrieval_k=retrieval_k,
        ),
    }


def source_families_from_output(output: Mapping[str, Any]) -> list[str]:
    families: list[str] = []
    documents = list(output.get("documents") or output.get("evidence") or [])
    for document in documents:
        payload = dict(document) if isinstance(document, Mapping) else _document_payload(document)
        metadata = payload.get("metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            continue
        family = _source_family_from_metadata(metadata)
        if family:
            families.append(family)
    return _unique_nonempty(families)


def summarize_eval_artifact_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    row_list = [dict(row) for row in rows]
    return {
        "overall": _pass_rate_summary(row_list),
        "by_lane": _grouped_pass_rate_summary(row_list, lambda row: [str(row.get("lane") or "unknown")]),
        "by_tag": _grouped_pass_rate_summary(row_list, lambda row: _clean_group_values(row.get("tags") or [])),
        "by_source_family": _grouped_pass_rate_summary(
            row_list,
            lambda row: _clean_group_values(row.get("source_families") or []) or ["unknown"],
        ),
    }


def classify_eval_failure(
    *,
    expected_behavior: str,
    deterministic: Mapping[str, Any],
    ragas_scores: Mapping[str, Any],
    retrieval_k: int = DEFAULT_RETRIEVAL_METRIC_K,
) -> str | None:
    retrieval_k = positive_int(retrieval_k)
    if expected_behavior == "refuse_if_not_found" and deterministic.get("citation_behavior") is not True:
        return "bad_refusal"
    if expected_behavior == "surface_conflict" and deterministic.get("citation_behavior") is not True:
        return "bad_conflict_handling"
    if expected_behavior == "cite_answer" and deterministic.get("citation_behavior") is not True:
        return "missing_citation"
    if expected_behavior == "surface_conflict" and deterministic.get("conflict_behavior") is False:
        return "bad_conflict_handling"
    if deterministic.get(f"id_hit_at_{retrieval_k}") is False:
        return "retrieval_miss"
    if deterministic.get(f"retrieval_hit_at_{retrieval_k}") is False:
        return "retrieval_miss"
    if deterministic.get("citation_grounding_behavior") is False:
        return "ungrounded_citation"
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


def _get_ragas_faithfulness_scorer() -> Any:
    global _RAGAS_FAITHFULNESS_SCORER
    if _RAGAS_FAITHFULNESS_SCORER is None:
        from imperial_rag.evals.ragas import build_faithfulness_scorer

        _RAGAS_FAITHFULNESS_SCORER = build_faithfulness_scorer()
    return _RAGAS_FAITHFULNESS_SCORER


def _get_ragas_answer_relevancy_scorer() -> Any:
    global _RAGAS_ANSWER_RELEVANCY_SCORER
    if _RAGAS_ANSWER_RELEVANCY_SCORER is None:
        from imperial_rag.evals.ragas import build_answer_relevancy_scorer

        _RAGAS_ANSWER_RELEVANCY_SCORER = build_answer_relevancy_scorer()
    return _RAGAS_ANSWER_RELEVANCY_SCORER


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


def _source_family_from_metadata(metadata: Mapping[str, Any]) -> str:
    for field in ("relative_path", "file_path"):
        value = str(metadata.get(field) or "").strip()
        if not value:
            continue
        normalized = value.replace("\\", "/").strip("/")
        if normalized:
            return normalized.split("/", 1)[0]
    parent_folder = str(metadata.get("parent_folder") or "").strip()
    if parent_folder:
        return parent_folder.replace("\\", "/").strip("/").split("/", 1)[0]
    return ""


def _clean_group_values(values: Any) -> list[str]:
    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values or [])
    return _unique_nonempty(str(value).strip() for value in raw_values)


def _pass_rate_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    failure_classes = Counter(
        str(row.get("failure_class")).strip()
        for row in rows
        if str(row.get("failure_class") or "").strip()
    )
    failed = sum(failure_classes.values())
    passed = total - failed
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total else None,
        "failure_classes": dict(sorted(failure_classes.items())),
    }


def _grouped_pass_rate_summary(rows: Sequence[Mapping[str, Any]], key_fn: Any) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        keys = key_fn(row) or ["unknown"]
        for key in keys:
            groups.setdefault(str(key), []).append(row)
    return {key: _pass_rate_summary(groups[key]) for key in sorted(groups)}


def _document_relevance_score(document: Any, hints: list[str]) -> float:
    if isinstance(document, Mapping):
        haystack = _document_search_text(dict(document))
    else:
        haystack = _document_search_text(_document_payload(document))
    normalized = haystack.casefold()
    return 1.0 if any(hint in normalized for hint in hints) else 0.0


def _citation_values(outputs: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("citations", "sources"):
        raw = outputs.get(key) or []
        if isinstance(raw, str):
            values.append(raw)
        else:
            values.extend(str(value) for value in raw)
    return _unique_nonempty(values)


def _document_ids_matching_citation(citation: str, documents: Sequence[Any]) -> list[str]:
    normalized_citation = citation.casefold()
    matched_ids: list[str] = []
    for document in documents:
        payload = dict(document) if isinstance(document, Mapping) else _document_payload(document)
        metadata = payload.get("metadata", {}) or {}
        identity_values = _document_identity_values(payload, metadata if isinstance(metadata, Mapping) else {})
        if any(_identity_matches_citation(value, normalized_citation) for value in identity_values):
            matched_ids.extend(_document_context_ids(metadata if isinstance(metadata, Mapping) else {}))
    return _unique_nonempty(matched_ids)


def _document_context_ids(metadata: Mapping[str, Any]) -> list[str]:
    return _unique_nonempty(str(metadata.get(field) or "").strip() for field in ("file_id", "chunk_id", "citation_id"))


def _document_identity_values(payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for field in (
        "file_id",
        "chunk_id",
        "citation_id",
        "document_id",
        "source_id",
        "source",
        "relative_path",
        "file_name",
        "file_path",
        "parent_folder",
        "section_heading",
    ):
        value = str(metadata.get(field) or "").strip()
        if value:
            values.append(value)
    source = str(payload.get("source") or payload.get("citation") or "").strip()
    if source:
        values.append(source)
    return _unique_nonempty(values)


def _identity_matches_citation(identity_value: str, normalized_citation: str) -> bool:
    normalized_identity = identity_value.casefold().strip()
    if len(normalized_identity) < 3 or len(normalized_citation.strip()) < 3:
        return False
    return normalized_identity in normalized_citation or normalized_citation in normalized_identity


def _bool_label(score: bool | None) -> str:
    if score is None:
        return "skipped"
    return "pass" if score else "fail"


def _has_decisive_version_claim(answer: str) -> bool:
    normalized = answer.casefold()
    return any(
        marker in normalized
        for marker in (
            "действует новая",
            "действует стар",
            "действует только",
            "актуальна новая",
            "актуален новый",
            "единственная версия",
            "new version applies",
            "old version applies",
            "only valid version",
        )
    )


def _ndcg(scores: list[float]) -> float:
    if not scores:
        return 0.0
    ideal = sorted(scores, reverse=True)
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(scores) / ideal_dcg


def _ndcg_with_ideal(scores: list[float], *, relevant_count: int, k: int) -> float:
    if k <= 0:
        return 0.0
    actual_scores = [*scores[:k], *([0.0] * max(0, k - len(scores)))]
    ideal_scores = [1.0] * min(relevant_count, k)
    ideal_scores.extend([0.0] * max(0, k - len(ideal_scores)))
    ideal_dcg = _dcg(ideal_scores)
    if ideal_dcg == 0:
        return 0.0
    return _dcg(actual_scores) / ideal_dcg


def _mrr(scores: list[float]) -> float:
    for index, score in enumerate(scores, start=1):
        if score > 0:
            return 1.0 / index
    return 0.0


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
    metric_prefixes = (
        "precision_at_",
        "ndcg_at_",
        "id_precision_at_",
        "id_recall_at_",
        "id_mrr_at_",
        "id_ndcg_at_",
    )
    return [
        key
        for key in metadata
        if any(key.startswith(prefix) for prefix in metric_prefixes) and metadata.get(key) is not None
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
    root = Path(__file__).resolve().parents[3]
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
    raise RuntimeError("Cannot synchronously run async Phoenix evaluation inside a running event loop.")


async def _await_result(awaitable: Any) -> Any:
    return await awaitable


def _run_evidence_target(inputs: dict[str, Any], runtime: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    from langchain_core.documents import Document
    from imperial_rag.ingestion.provenance import validate_chunk

    output: dict[str, Any] = {"eligible": False}
    try:
        output.update(run_target(inputs, runtime=runtime))
        output["eligible"] = False
        if "ranked_documents" not in output:
            output["evidence_error"] = "missing_ranked_documents"
            return output
        sources = {row["source_id"]: row for row in snapshot["sources"]}
        for row in output["ranked_documents"]:
            validate_chunk(Document(**row), sources)
        diagnostics = output["retrieval"]
        error = output.get("error", {}).get("type")
        if diagnostics.get("fallbacks") or diagnostics.get("degraded"):
            output["evidence_error"] = "degraded_retrieval"
        elif error and error != "no_relevant_documents":
            output["evidence_error"] = "query_error"
        else:
            output["eligible"] = True
    except Exception as exc:
        # Exceptions can contain private text or credentials; publish only the class.
        output["evidence_error"] = type(exc).__name__
    return output


def phoenix_evidence_evaluator(
    *, metric: str, k: int | None = None, budget: int | None = None, documents_key: str = "documents",
):
    from imperial_rag.answering.strict import pack_context
    from imperial_rag.evals.evidence import evidence_recall
    from langchain_core.documents import Document

    def evaluate(output: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
        key = f"{metric}_{'at_' + str(k) if k is not None else 'budget_' + str(budget)}"
        if not output.get("eligible"):
            return {"key": key, "score": None,
                    "explanation": "Invalid or degraded retrieval run: " + output.get("evidence_error", "ineligible")}
        documents = [Document(**row) for row in output[documents_key]]
        selected = documents[:k] if k is not None else pack_context(documents, budget)["documents"]
        score = evidence_recall(expected["evidence"], selected)[metric]
        return {"key": key, "score": score,
                "explanation": "Recall is undefined for refusal questions" if score is None else "Exact source union coverage"}

    return evaluate


def _phoenix_evidence_evaluators(*, documents_key: str = "documents") -> dict[str, Any]:
    from imperial_rag.evals.evidence import BUDGETS, KS

    evaluators = {}
    for metric in ("evidence_recall", "full_evidence_success"):
        for k in KS:
            evaluators[f"{metric}_at_{k}"] = phoenix_evidence_evaluator(metric=metric, k=k, documents_key=documents_key)
        for budget in BUDGETS:
            evaluators[f"{metric}_budget_{budget}"] = phoenix_evidence_evaluator(
                metric=metric, budget=budget, documents_key=documents_key,
            )
    return evaluators


async def publish_evidence_comparison_async(root: Path, settings: Any, *, concurrency: int = 3) -> None:
    """Replay saved retrieval once per configuration against one pinned dataset version."""
    import json
    from phoenix.client import AsyncClient
    from imperial_rag.evals.chunk_comparison import load_comparison, write_json
    from imperial_rag.ingestion.provenance import digest

    concurrency = positive_int(concurrency)
    manifest, benchmark, results = load_comparison(root)
    examples = [row for row in benchmark["examples"] if row["split"] == manifest["split"]]
    client = AsyncClient(base_url=settings.phoenix_client_endpoint)
    if "phoenix_dataset" in manifest:
        from phoenix.client.resources.datasets import Dataset
        from imperial_rag.evals.dataset_input import get_pinned_dataset

        binding = manifest["phoenix_dataset"]
        dataset = await get_pinned_dataset(client, binding, settings.phoenix_client_endpoint)
        # Filter in memory, retaining server example IDs and the exact dataset version.
        selected_ids = {row["id"] for row in examples}
        payload = dataset.to_dict()
        payload["examples"] = [row for row in dataset.examples if row["metadata"]["id"] in selected_ids]
        dataset = Dataset.from_dict(payload)
    else:
        binding_root = root.parent / "phoenix-datasets"
        binding_root.mkdir(exist_ok=True)
        binding_path = binding_root / f"{manifest['dataset_hash']}-{manifest['split']}.json"
        inputs = [{"id": row["id"], "question": row["question"]} for row in examples]
        outputs = [_phoenix_dataset_expected_payload(row) for row in examples]
        if binding_path.exists():
            binding = json.loads(binding_path.read_text())
            if binding["endpoint"] != settings.phoenix_client_endpoint:
                raise ValueError("Pinned dataset belongs to another Phoenix endpoint")
            dataset = await client.datasets.get_dataset(dataset=binding["dataset_id"], version_id=binding["version_id"])
        else:
            dataset = await client.datasets.create_dataset(
                name=f"imperial-evidence-{manifest['dataset_hash']}-{manifest['split']}",
                inputs=inputs, outputs=outputs,
                metadata=[{"id": row["id"], "dataset_hash": manifest["dataset_hash"],
                           "snapshot_hash": manifest["snapshot_hash"], "split": row["split"]} for row in examples],
            )
            binding = {"dataset_id": dataset.id, "version_id": dataset.version_id,
                       "endpoint": settings.phoenix_client_endpoint}
            write_json(binding_path, binding)
        expected_rows = {row["id"]: {"input": row, "output": expected} for row, expected in zip(inputs, outputs)}
        actual_rows = {row["input"]["id"]: {"input": row["input"], "output": row["output"]} for row in dataset.examples}
        if len(dataset.examples) != len(examples) or digest(actual_rows) != digest(expected_rows):
            raise ValueError("Pinned Phoenix dataset does not match the local benchmark")
    evaluators = _phoenix_evidence_evaluators()
    published_path = root / "phoenix.json"
    published = json.loads(published_path.read_text()) if published_path.exists() else binding | {"experiments": {}}
    if any(published.get(key) != value for key, value in binding.items()):
        raise ValueError("Published comparison has a different Phoenix dataset binding")
    for config in manifest["configs"]:
        config_id = config["config_id"]
        if config_id in published["experiments"]:
            continue
        by_id = {row["id"]: row for row in results if row["config_id"] == config_id}

        async def replay(input: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
            return by_id[metadata["id"] if "phoenix_dataset" in manifest and metadata else input["id"]]

        experiment = await client.experiments.run_experiment(
            dataset=dataset, task=replay, evaluators=evaluators,
            experiment_name=f"{manifest['run_id']}-{config_id}",
            experiment_metadata={**manifest, "config": config, "mode": "saved-retrieval-replay"},
            concurrency=concurrency,
        )
        published["experiments"][config_id] = _experiment_identifier(experiment)
        write_json(published_path, published)


if __name__ == "__main__":
    main()
