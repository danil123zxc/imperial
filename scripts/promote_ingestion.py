from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_src_on_path


ensure_src_on_path(__file__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate and promote an isolated shadow ingestion run.")
    parser.add_argument("run_id")
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--questions", type=Path)
    args = parser.parse_args(argv)

    from imperial_rag.cli import build_settings, load_project_environment
    from imperial_rag.ingestion.shadow import promote_shadow_run

    load_project_environment(args.workspace_root)
    settings = build_settings(args.workspace_root, use_active_pointer=False)
    questions = args.questions or Path(settings.workspace_root) / "evals" / "questions.jsonl"
    result = promote_shadow_run(settings, args.run_id, questions_path=questions)
    for key, value in result.summary.items():
        print(f"{key}={value}")
    if not result.passed:
        for error in result.errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
