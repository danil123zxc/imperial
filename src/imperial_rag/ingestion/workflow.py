from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class IngestionState(TypedDict, total=False):
    settings: object
    ocr_client: object
    vector_store: object
    summary: object
    status: str
    counts: dict[str, int]


def build_ingestion_workflow(run_pipeline=None):
    def run_ingestion(state: IngestionState) -> IngestionState:
        if run_pipeline is not None:
            summary = _call_pipeline(run_pipeline, state)
        else:
            from imperial_rag.ingestion.pipeline import ingest_corpus

            summary = ingest_corpus(
                settings=state["settings"],
                ocr_client=state.get("ocr_client"),
                vector_store=state.get("vector_store"),
            )
        counts = _counts_from_summary(summary)
        status = (
            str(summary.get("status", "completed"))
            if isinstance(summary, Mapping)
            else str(getattr(summary, "status", "completed"))
        )
        return {"summary": summary, "status": status, "counts": counts}

    graph = StateGraph(IngestionState)
    graph.add_node("run_ingestion", run_ingestion)
    graph.add_edge(START, "run_ingestion")
    graph.add_edge("run_ingestion", END)
    return graph.compile()


def _call_pipeline(run_pipeline, state: Mapping[str, Any]):
    try:
        signature = inspect.signature(run_pipeline)
    except (TypeError, ValueError):
        return run_pipeline(state)
    positional_count = sum(
        1
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    return run_pipeline(state) if positional_count else run_pipeline()


def _counts_from_summary(summary: Any) -> dict[str, int]:
    if isinstance(summary, Mapping):
        summary_counts = summary.get("counts")
        if isinstance(summary_counts, Mapping):
            return {str(key): int(value) for key, value in summary_counts.items()}
        return {str(key): int(value) for key, value in summary.items() if isinstance(value, int)}
    result_counts: dict[str, int] = {}
    for source_name, target_name in (
        ("total_files", "files"),
        ("document_count", "documents"),
        ("documents", "documents"),
        ("chunk_count", "chunks"),
        ("chunks", "chunks"),
        ("indexed_count", "indexed"),
        ("indexed", "indexed"),
    ):
        value = getattr(summary, source_name, None)
        if isinstance(value, int):
            result_counts[target_name] = value
    return result_counts


__all__ = ["IngestionState", "build_ingestion_workflow"]
