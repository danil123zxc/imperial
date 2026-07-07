from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_on_path as _ensure_src_on_path


def main(argv: list[str] | None = None) -> None:
    _ensure_src_on_path(__file__)
    parser = argparse.ArgumentParser(description="Check whether a shadow ingestion run can be promoted.")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--shadow-root", type=Path, required=True)
    parser.add_argument("--questions-path", type=Path, default=Path("evals/questions.jsonl"))
    parser.add_argument("--min-locator-coverage", type=float, default=0.95)
    parser.add_argument("--expected-keyword-index")
    parser.add_argument("--expected-qdrant-collection")
    args = parser.parse_args(argv)

    from imperial_rag.ingestion.promotion import check_promotion_gates

    result = check_promotion_gates(
        args.baseline_root,
        args.shadow_root,
        questions_path=args.questions_path,
        min_locator_coverage=args.min_locator_coverage,
        expected_keyword_index=args.expected_keyword_index,
        expected_qdrant_collection=args.expected_qdrant_collection,
    )
    print(
        json.dumps(
            {"passed": result.passed, "errors": result.errors, "summary": result.summary},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if not result.passed:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
