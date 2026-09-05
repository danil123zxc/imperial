"""Read Phoenix examples without mutating datasets; reuse the local benchmark contract."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imperial_rag.evals.evidence import assemble_benchmark
from imperial_rag.ingestion.provenance import digest


def add_dataset_input_arguments(parser: argparse.ArgumentParser, *, questions_flag: str = "--questions-path") -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument(questions_flag, type=Path, dest="questions_path", help="Local questions JSONL (default: evals/questions.jsonl).")
    source.add_argument("--phoenix-dataset-name", help="Read an existing Phoenix dataset by name; never upload it.")
    source.add_argument("--phoenix-dataset-id", help="Read an existing Phoenix dataset by ID; never upload it.")
    parser.add_argument("--phoenix-dataset-version-id", help="Read this immutable version; default: resolve latest once.")


def has_phoenix_input(args: argparse.Namespace) -> bool:
    return args.phoenix_dataset_name is not None or args.phoenix_dataset_id is not None


def validate_dataset_input_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace, *, evidence: bool = False,
) -> None:
    phoenix = has_phoenix_input(args)
    for flag in ("phoenix_dataset_name", "phoenix_dataset_id", "phoenix_dataset_version_id"):
        value = getattr(args, flag)
        if value is not None and not value.strip():
            parser.error(f"--{flag.replace('_', '-')} must not be empty")
    if args.phoenix_dataset_version_id is not None and not phoenix:
        parser.error("--phoenix-dataset-version-id requires a Phoenix input dataset")
    if phoenix and (getattr(args, "annotations", None) is not None or getattr(args, "dataset_name", None) is not None):
        parser.error("Phoenix input conflicts with --annotations and upload-only --dataset-name")
    if evidence and not phoenix and args.annotations is None:
        parser.error("Local evidence evaluation requires --annotations")


def map_phoenix_examples(dataset: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from imperial_rag.evals.phoenix_experiment import _validate_question_row

    questions, annotations = [], []
    seen: set[str] = set()
    for index, example in enumerate(dataset.examples, 1):
        inputs, output, metadata = (example.get(key) for key in ("input", "output", "metadata"))
        if not all(isinstance(value, dict) for value in (inputs, output, metadata)):
            raise ValueError(f"Phoenix example {index}: input, output and metadata must be objects")
        question = {"question": inputs.get("question")}
        for key in ("expected_behavior", "reference_answer", "expected_source_hints", "reference_context_ids",
                    "lane", "quarantine_reason"):
            if key in output:
                question[key] = output[key]
        for key in ("id", "suite", "tags", "lane", "quarantine_reason"):
            if key in metadata:
                if key in question and question[key] != metadata[key]:
                    raise ValueError(f"Phoenix example {index}: conflicting {key}")
                question[key] = metadata[key]
        if any(not isinstance(question.get(key), str) or not question[key].strip()
               for key in ("id", "suite", "question", "reference_answer", "expected_behavior", "lane")):
            raise ValueError(f"Phoenix example {index}: required question fields must be nonempty strings")
        errors = _validate_question_row(question, line_number=index, seen_ids=seen)
        if errors:
            raise ValueError("; ".join(errors))
        questions.append(question)
        annotations.append({"id": question["id"], "evidence": output.get("evidence"), **{
            key: metadata.get(key) for key in ("split", "review_status", "question_hash", "snapshot_hash")
        }})
    if not questions:
        raise ValueError("Phoenix input dataset must contain at least one question")
    return questions, annotations


def dataset_binding(dataset: Any, endpoint: str) -> dict[str, Any]:
    if not dataset.id or not dataset.version_id:
        raise ValueError("Phoenix did not return a dataset ID and resolved version ID")
    rows = [{key: example[key] for key in ("id", "input", "output", "metadata")} for example in dataset.examples]
    return {"dataset_id": dataset.id, "version_id": dataset.version_id, "dataset_name": dataset.name,
            "endpoint": endpoint, "examples_hash": digest(sorted(rows, key=lambda row: row["id"]))}


@dataclass
class PhoenixInput:
    dataset: Any
    binding: dict[str, Any]
    examples: list[dict[str, Any]]
    benchmark: dict[str, Any] | None


async def load_phoenix_input(
    args: argparse.Namespace, settings: Any, *, snapshot: dict[str, Any] | None = None,
) -> PhoenixInput:
    from phoenix.client import AsyncClient

    client = AsyncClient(base_url=settings.phoenix_client_endpoint)
    selector = {"id": args.phoenix_dataset_id} if args.phoenix_dataset_id is not None else {"name": args.phoenix_dataset_name}
    dataset = await client.datasets.get_dataset(dataset=selector, version_id=args.phoenix_dataset_version_id)
    if args.phoenix_dataset_version_id is not None and dataset.version_id != args.phoenix_dataset_version_id:
        raise ValueError("Phoenix returned a different dataset version")
    if args.phoenix_dataset_id is not None and dataset.id != args.phoenix_dataset_id:
        raise ValueError("Phoenix returned a different dataset ID")
    questions, annotations = map_phoenix_examples(dataset)
    binding = dataset_binding(dataset, settings.phoenix_client_endpoint)
    benchmark = None
    if snapshot is not None:
        benchmark = assemble_benchmark(questions, annotations, snapshot)
        benchmark["phoenix_dataset"] = binding
        benchmark["dataset_hash"] = digest({key: value for key, value in benchmark.items() if key != "dataset_hash"})
    return PhoenixInput(dataset, binding, benchmark["examples"] if benchmark is not None else questions, benchmark)


async def get_pinned_dataset(client: Any, binding: dict[str, Any], endpoint: str) -> Any:
    if binding["endpoint"] != endpoint:
        raise ValueError("Pinned dataset belongs to another Phoenix endpoint")
    dataset = await client.datasets.get_dataset(dataset={"id": binding["dataset_id"]}, version_id=binding["version_id"])
    actual = dataset_binding(dataset, endpoint)
    # Dataset names can change; the identity, version and example content cannot.
    if any(actual[key] != binding[key] for key in ("dataset_id", "version_id", "examples_hash")):
        raise ValueError("Pinned Phoenix dataset does not match the saved input")
    return dataset
