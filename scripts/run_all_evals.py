from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from urllib import request

from _bootstrap import ensure_src_on_path as _ensure_src_on_path

_ensure_src_on_path(__file__)

import run_phoenix_eval as phoenix_eval
from imperial_rag.evals.dataset_input import (
    add_dataset_input_arguments, has_phoenix_input, load_phoenix_input, validate_dataset_input_arguments,
)
from imperial_rag.evals.evidence import assemble_benchmark
from imperial_rag.ingestion.provenance import load_snapshot
from imperial_rag.jsonl import read_jsonl
from imperial_rag.cli import (  # noqa: E402
    configure_observability as _configure_observability,
    duration_ms as _duration_ms,
    log_failure as _log_failure,
)


DEFAULT_EXPERIMENT_NAME = "imperial-rag-all-evals"
PHOENIX_START_COMMAND = "docker compose up -d phoenix"


def main(argv: list[str] | None = None) -> None:
    _ensure_src_on_path(__file__)
    parser = argparse.ArgumentParser(
        description="Run all currently runnable Imperial RAG evals and store one Phoenix experiment."
    )
    add_dataset_input_arguments(parser)
    parser.add_argument("--snapshot", type=Path, required=True, help="Frozen source snapshot for evidence recall.")
    parser.add_argument("--annotations", type=Path, help="Reviewed evidence sidecar matching the questions and snapshot.")
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument(
        "--ragas-metrics",
        default="faithfulness,answer_relevancy",
        help="Comma-separated Ragas metrics to attach, or 'none' for deterministic-only.",
    )
    parser.add_argument(
        "--concurrency",
        type=phoenix_eval.positive_int,
        default=phoenix_eval.DEFAULT_PHOENIX_CONCURRENCY,
        help="Maximum concurrent Phoenix experiment tasks.",
    )
    args = parser.parse_args(argv)
    validate_dataset_input_arguments(parser, args, evidence=True)

    snapshot = load_snapshot(args.snapshot)
    source = None
    if has_phoenix_input(args):
        phoenix_eval._load_project_env(args.workspace_root)
        source = phoenix_eval._run_async(load_phoenix_input(args, phoenix_eval._build_settings(args.workspace_root), snapshot=snapshot))
        benchmark = source.benchmark
        assert benchmark is not None
    else:
        benchmark = assemble_benchmark(phoenix_eval.load_questions(args.questions_path or phoenix_eval.DEFAULT_QUESTIONS_PATH),
                                       read_jsonl(args.annotations), snapshot)
    examples = benchmark["examples"]
    if not examples:
        parser.error("Evidence evaluation requires at least one question.")
    phoenix_eval._load_project_env(args.workspace_root)
    settings = phoenix_eval._build_settings(args.workspace_root)
    _configure_observability(settings)
    started_at = perf_counter()
    try:
        _assert_phoenix_reachable(settings.phoenix_client_endpoint)
        phoenix_eval._configure_tracing(settings, enabled=True)

        metric_names = phoenix_eval.parse_phoenix_ragas_metrics(args.ragas_metrics)
        phoenix_eval.run_phoenix_experiment(
            examples=examples,
            settings=settings,
            dataset_name=args.dataset_name or f"{settings.phoenix_project_name}-gold-questions",
            experiment_name=args.experiment_name,
            ragas_metric_names=metric_names,
            concurrency=args.concurrency,
            evidence_snapshot=snapshot,
            **({"phoenix_input": source} if source else {}),
        )
        _log_completion(started_at, example_count=len(examples), ragas_metrics=",".join(metric_names))
    except (Exception, SystemExit) as exc:
        _log_failure("all_evals", exc, started_at, ragas_metrics=args.ragas_metrics)
        raise


def _assert_phoenix_reachable(endpoint: str, timeout: float = 2.0) -> None:
    try:
        with request.urlopen(endpoint, timeout=timeout):
            return
    except Exception as exc:
        raise SystemExit(
            f"Phoenix is not reachable at {endpoint}. "
            f"Start it with `{PHOENIX_START_COMMAND}` and rerun this command."
        ) from exc


def _log_completion(started_at: float, *, example_count: int, ragas_metrics: str) -> None:
    from imperial_rag.observability import log_event

    log_event(
        "imperial_rag.all_evals",
        operation="all_evals",
        status="success",
        component="cli",
        duration_ms=_duration_ms(started_at),
        example_count=example_count,
        phoenix_mode=True,
        ragas_metrics=ragas_metrics,
    )


if __name__ == "__main__":
    main()
