from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from imperial_rag.jsonl import read_jsonl, write_jsonl


DEFAULT_CHUNKS_PATH = Path(".imperial_rag/extracted/chunks.jsonl")
DEFAULT_OUTPUT_PATH = Path(".imperial_rag/evals/question-drafts.jsonl")
GENERATOR_ID = "imperial_eval_question_drafts_v1"


def load_chunks(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def build_question_drafts(chunks: Iterable[Mapping[str, Any]], *, limit: int = 25) -> list[dict[str, Any]]:
    drafts: list[dict[str, Any]] = []
    seen_documents: set[str] = set()
    for chunk in chunks:
        metadata = dict(chunk.get("metadata") or {})
        file_name = str(metadata.get("file_name") or "").strip()
        relative_path = str(metadata.get("relative_path") or "").strip()
        citation_id = str(metadata.get("citation_id") or "").strip()
        if not file_name or not relative_path:
            continue
        document_key = relative_path.casefold()
        if document_key in seen_documents:
            continue
        seen_documents.add(document_key)
        title = Path(file_name).stem.strip()
        drafts.append(
            {
                "question": f"Какие правила описаны в документе {title}?",
                "expected_behavior": "draft_review_required",
                "expected_source_hints": _source_hints(file_name, relative_path),
                "reference_answer": "",
                "review_status": "draft",
                "draft_source_citation_id": citation_id,
                "generated_by": GENERATOR_ID,
            }
        )
        if len(drafts) >= limit:
            break
    return drafts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate review-required draft eval questions from extracted Imperial chunks."
    )
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args(argv)

    drafts = build_question_drafts(load_chunks(args.chunks_path), limit=args.limit)
    write_jsonl(args.output_path, drafts)
    print(f"draft_questions={len(drafts)}")
    print(f"draft_questions_path={args.output_path}")


def _source_hints(file_name: str, relative_path: str) -> list[str]:
    hints: list[str] = []
    for value in (file_name, relative_path):
        if value and value not in hints:
            hints.append(value)
    return hints


if __name__ == "__main__":
    main()
