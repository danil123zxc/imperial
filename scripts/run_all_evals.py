from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib import request


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_phoenix_eval as phoenix_eval


DEFAULT_EXPERIMENT_NAME = "imperial-rag-all-evals"
PHOENIX_START_COMMAND = "docker compose up -d phoenix"


def main(argv: list[str] | None = None) -> None:
    phoenix_eval._ensure_src_on_path()
    parser = argparse.ArgumentParser(
        description="Run all currently runnable Imperial RAG evals and store one Phoenix experiment."
    )
    parser.add_argument("--questions-path", type=Path, default=phoenix_eval.DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--dataset-name")
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument(
        "--ragas-metrics",
        default="faithfulness",
        help="Comma-separated Ragas metrics to attach, or 'none' for deterministic-only.",
    )
    args = parser.parse_args(argv)

    phoenix_eval._load_project_env(args.workspace_root)
    settings = phoenix_eval._build_settings(args.workspace_root)
    _assert_phoenix_reachable(settings.phoenix_client_endpoint)
    phoenix_eval._configure_tracing(settings, enabled=True)

    examples = phoenix_eval.load_questions(args.questions_path)
    phoenix_eval.run_phoenix_experiment(
        examples=examples,
        settings=settings,
        dataset_name=args.dataset_name or f"{settings.phoenix_project_name}-gold-questions",
        experiment_name=args.experiment_name,
        ragas_metric_names=phoenix_eval.parse_phoenix_ragas_metrics(args.ragas_metrics),
    )


def _assert_phoenix_reachable(endpoint: str, timeout: float = 2.0) -> None:
    try:
        with request.urlopen(endpoint, timeout=timeout):
            return
    except Exception as exc:
        raise SystemExit(
            f"Phoenix is not reachable at {endpoint}. "
            f"Start it with `{PHOENIX_START_COMMAND}` and rerun this command."
        ) from exc


if __name__ == "__main__":
    main()
