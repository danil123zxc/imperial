from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any


def main(argv: list[str] | None = None) -> None:
    _ensure_src_on_path()
    parser = argparse.ArgumentParser(description="Ask a question against the Imperial RAG runtime.")
    parser.add_argument("question")
    parser.add_argument("--workspace-root", type=Path, help="Workspace root containing the processed RAG state.")
    parser.add_argument("--trace-phoenix", action="store_true", help="Send this run's traces to configured Phoenix.")
    parser.add_argument("--trace-session-id", help="Phoenix session.id for grouping traces.")
    args = parser.parse_args(argv)

    _load_project_env(args.workspace_root)
    settings = _build_settings(args.workspace_root)
    _configure_observability(settings)
    _configure_tracing(settings, args.trace_phoenix)
    trace_session_id = _trace_session_id(args.trace_session_id)
    started_at = perf_counter()
    try:
        with _trace_context(trace_session_id):
            result = _query(settings=settings, question=args.question)
    except (Exception, SystemExit) as exc:
        _log_failure("query", exc, started_at)
        raise
    _log_query_completion(result, started_at)
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
    from imperial_rag.cli import configure_tracing

    configure_tracing(settings, trace_phoenix=trace_phoenix)


def _trace_context(session_id: str):
    from imperial_rag.cli import trace_context

    return trace_context(session_id)


def _trace_session_id(explicit: str | None) -> str:
    if explicit is not None and explicit.strip():
        return explicit.strip()
    env_value = os.environ.get("IMPERIAL_RAG_TRACE_SESSION_ID", "").strip()
    if env_value:
        return env_value
    return f"cli_{uuid.uuid4()}"


def _configure_observability(settings: Any) -> None:
    from imperial_rag.cli import configure_observability

    configure_observability(settings)


def _log_query_completion(result: Any, started_at: float) -> None:
    from imperial_rag.observability import log_event

    log_event(
        "imperial_rag.query",
        operation="query",
        status="success",
        component="cli",
        duration_ms=_duration_ms(started_at),
        **_query_log_fields(result),
    )


def _log_failure(operation: str, exc: BaseException, started_at: float) -> None:
    from imperial_rag.cli import log_failure

    log_failure(operation, exc, started_at)


def _build_settings(workspace_root: Path | None) -> Any:
    from imperial_rag.cli import build_settings

    return build_settings(workspace_root)


def _load_project_env(workspace_root: Path | None) -> None:
    from imperial_rag.cli import load_project_environment

    load_project_environment(workspace_root)


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


def _query_log_fields(result: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    retrieval = _result_value(result, "retrieval", {}) or {}
    if isinstance(retrieval, dict):
        for key in (
            "final_evidence",
            "vector_candidates",
            "keyword_candidates",
            "merged_candidates",
            "rerank_input_candidates",
            "reranked_candidates",
            "reranker",
        ):
            if key in retrieval:
                fields[key] = retrieval[key]
        fallbacks = retrieval.get("fallbacks")
        if isinstance(fallbacks, list):
            fields["fallback_count"] = len(fallbacks)
    evidence = _result_value(result, "evidence", None) or _result_value(result, "retrieved_documents", None)
    if evidence is not None and "final_evidence" not in fields:
        try:
            fields["final_evidence"] = len(evidence)
        except TypeError:
            pass
    return fields


def _duration_ms(started_at: float) -> int:
    from imperial_rag.cli import duration_ms

    return duration_ms(started_at)


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


if __name__ == "__main__":
    main()
