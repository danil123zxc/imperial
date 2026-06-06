from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> None:
    _ensure_src_on_path()
    parser = argparse.ArgumentParser(description="Ask a question against the Imperial RAG runtime.")
    parser.add_argument("question")
    parser.add_argument("--workspace-root", type=Path, help="Workspace root containing the processed RAG state.")
    parser.add_argument("--trace-phoenix", action="store_true", help="Send this run's traces to configured Phoenix.")
    args = parser.parse_args(argv)

    _load_project_env(args.workspace_root)
    settings = _build_settings(args.workspace_root)
    _configure_tracing(settings, args.trace_phoenix)
    result = _query(settings=settings, question=args.question)
    print(str(_result_value(result, "answer", "")))
    sources = _result_value(result, "sources", None) or _result_value(result, "citations", []) or []
    for source in sources:
        print(source)


def _query(settings: Any, question: str) -> dict[str, Any]:
    try:
        from imperial_rag.runtime import create_runtime
    except (ImportError, AttributeError):
        create_runtime = None
    if create_runtime is not None:
        return _coerce_result(create_runtime(settings).query(question))

    try:
        from imperial_rag.runtime import Runtime
    except (ImportError, AttributeError):
        Runtime = None

    if Runtime is not None:
        runtime = Runtime(settings=settings)
        return _coerce_result(runtime.query(question))

    from imperial_rag.runtime import build_live_query_workflow

    workflow = build_live_query_workflow(settings)
    return _coerce_result(workflow.invoke({"question": question}))


def _configure_tracing(settings: Any, trace_phoenix: bool) -> None:
    from imperial_rag.tracing import configure_phoenix_tracing

    configure_phoenix_tracing(settings, enabled=True if trace_phoenix else None)


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


def _coerce_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {
        "answer": getattr(result, "answer", ""),
        "sources": getattr(result, "sources", getattr(result, "citations", [])),
    }


def _result_value(result: Any, key: str, default: Any) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


if __name__ == "__main__":
    main()
