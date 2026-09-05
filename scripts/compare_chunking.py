"""Freeze, validate, compare, and inspect source-grounded chunking experiments."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from _bootstrap import ensure_src_on_path

ensure_src_on_path(__file__)

from imperial_rag.config import Settings  # noqa: E402
from imperial_rag.env import load_project_env  # noqa: E402
from imperial_rag.evals.chunk_comparison import generate_comparison_answers, run_comparison  # noqa: E402
from imperial_rag.evals.evidence import assemble_benchmark  # noqa: E402
from imperial_rag.evals.questions import load_questions  # noqa: E402
from imperial_rag.ingestion.provenance import freeze_extracted_sources, load_snapshot  # noqa: E402
from imperial_rag.jsonl import read_jsonl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze", help="Freeze existing extraction artifacts without OCR or providers")
    freeze.add_argument("--documents-root", type=Path, default=Path(".imperial_rag/extracted/documents"))
    freeze.add_argument("--authority", type=Path, default=Path("docs/document-authority.json"))
    freeze.add_argument("--output", type=Path, required=True)
    for command in ("validate", "run"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--snapshot", type=Path, required=True)
        cmd.add_argument("--annotations", type=Path, required=True)
        cmd.add_argument("--questions", type=Path, default=Path("evals/questions.jsonl"))
        if command == "run":
            cmd.add_argument("--output", type=Path, required=True)
            cmd.add_argument("--configs", help="size:overlap pairs, comma-separated; default baseline plus nine variants")
            cmd.add_argument("--concurrency", type=int, default=3)
            cmd.add_argument("--split", choices=("dev", "test"), default="dev")
    answers = sub.add_parser("answers", help="Generate baseline and shortlisted answers from saved retrieval")
    answers.add_argument("--run", type=Path, required=True)
    answers.add_argument("--concurrency", type=int, default=3)
    phoenix = sub.add_parser("phoenix", help="Publish saved retrieval results; never rerun providers")
    phoenix.add_argument("--run", type=Path, required=True)
    phoenix.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args(argv)
    load_project_env()
    settings = Settings()
    if args.command == "freeze":
        snapshot = freeze_extracted_sources(args.documents_root, args.authority, args.output)
        print(f"sources={len(snapshot['sources'])}; snapshot_hash={snapshot['snapshot_hash']}")
    elif args.command in {"validate", "run"}:
        snapshot = load_snapshot(args.snapshot)
        benchmark = assemble_benchmark(load_questions(args.questions), read_jsonl(args.annotations), snapshot)
        if args.command == "validate":
            print(f"reviewed_questions={len(benchmark['examples'])}; dataset_hash={benchmark['dataset_hash']}")
        else:
            configs = None
            if args.configs:
                configs = []
                for entry in args.configs.split(","):
                    size, overlap = entry.split(":")
                    configs.append((int(size), int(overlap)))
            manifest = asyncio.run(run_comparison(snapshot, benchmark, settings, args.output,
                                                  configs=configs, concurrency=args.concurrency, split=args.split))
            print(f"comparison={args.output}; active_indexes_unchanged=true")
            if not manifest["eligible"]:
                return 1
    elif args.command == "answers":
        rows = asyncio.run(generate_comparison_answers(args.run, concurrency=args.concurrency))
        print(f"answer_review_packets={len(rows)}")
        if any(row.get("error") for row in rows):
            return 1
    else:
        from imperial_rag.evals.phoenix_experiment import publish_evidence_comparison_async

        asyncio.run(publish_evidence_comparison_async(args.run, settings, concurrency=args.concurrency))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
