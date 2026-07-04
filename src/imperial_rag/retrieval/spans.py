from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from imperial_rag.observability.phoenix import retrieval_documents_preview, trace_candidate_documents_enabled
from imperial_rag.retrieval.identity import _retrieval_id


def _set_documents_span_output(span: Any, documents: list[Document], **metadata: Any) -> None:
    output: dict[str, Any] = {"count": len(documents)}
    for key, value in metadata.items():
        if value is not None:
            output[key] = _trace_output_value(value)
    output.update(_document_summary_output(documents))
    previews = retrieval_documents_preview(documents)
    if previews:
        output["top_documents"] = previews
    span.set_output(output)


def _set_candidate_documents_omitted(span: Any) -> None:
    span.set_attribute("retrieval.documents.omitted", True)
    span.set_attribute("retrieval.documents.omitted_reason", "candidate_tracing_disabled")


def _record_candidate_documents(span: Any, documents: list[Document]) -> None:
    if trace_candidate_documents_enabled():
        span.set_retrieval_documents(documents)
    elif documents:
        _set_candidate_documents_omitted(span)


def _keyword_match_mode(documents: list[Document]) -> str | None:
    modes = sorted(
        {
            str(document.metadata.get("_keyword_match_mode"))
            for document in documents
            if document.metadata.get("_keyword_match_mode") is not None
        }
    )
    if not modes:
        return None
    if len(modes) == 1:
        return modes[0]
    return f"mixed:{','.join(modes)}"


def _compact_fusion_span_output(
    vector_docs: list[Document],
    keyword_docs: list[Document],
    merged: list[Document],
    fused: list[Document],
    rerank_input: list[Document],
    rrf_k: int,
) -> dict[str, Any]:
    return {
        "vector_candidates": len(vector_docs),
        "keyword_candidates": len(keyword_docs),
        "merged_candidates": len(merged),
        "deduped_candidates": _deduped_candidate_count(vector_docs, keyword_docs, merged),
        "fused_candidates": len(fused),
        "rerank_input_candidates": len(rerank_input),
        "fusion": "rrf",
        "fusion_rrf_k": rrf_k,
        "source_mix": _source_mix(merged),
        "merged_top_ids": _top_trace_ids(merged),
        "fused_top_ids": _top_trace_ids(fused),
        "rerank_input_top_ids": _top_trace_ids(rerank_input),
    }


def _merge_candidates_span_output(
    vector_docs: list[Document],
    keyword_docs: list[Document],
    merged: list[Document],
    duplicate_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "vector_candidates": len(vector_docs),
        "keyword_candidates": len(keyword_docs),
        "merged_candidates": len(merged),
        "deduped_candidates": _deduped_candidate_count(vector_docs, keyword_docs, merged),
        "source_mix": _source_mix(merged),
        "merged_top_ids": _top_trace_ids(merged),
        "duplicate_groups": duplicate_groups,
    }


def _rrf_fusion_span_output(merged: list[Document], fused: list[Document], rrf_k: int) -> dict[str, Any]:
    return {
        "fusion": "rrf",
        "fusion_rrf_k": rrf_k,
        "input_candidates": len(merged),
        "fused_candidates": len(fused),
        "input_top_ids": _top_trace_ids(merged),
        "fused_top_ids": _top_trace_ids(fused),
        "rank_movements": _rank_movements(fused),
    }


def _deduped_candidate_count(vector_docs: list[Document], keyword_docs: list[Document], merged: list[Document]) -> int:
    return max(len(vector_docs) + len(keyword_docs) - len(merged), 0)


def _source_mix(documents: list[Document]) -> dict[str, int]:
    counts = {"hybrid": 0, "keyword_only": 0, "vector_only": 0}
    for document in documents:
        metadata = document.metadata or {}
        has_vector = _rank_value(metadata.get("_vector_rank")) is not None
        has_keyword = _rank_value(metadata.get("_keyword_rank")) is not None
        if has_vector and has_keyword:
            counts["hybrid"] += 1
        elif has_keyword:
            counts["keyword_only"] += 1
        elif has_vector:
            counts["vector_only"] += 1
    return counts


def _top_trace_ids(documents: list[Document], *, limit: int = 10) -> list[str]:
    return [_retrieval_id(document) for document in documents[:limit]]


def _rank_movements(documents: list[Document], *, limit: int = 10) -> list[dict[str, Any]]:
    movements: list[dict[str, Any]] = []
    for fusion_rank, document in enumerate(documents[:limit]):
        metadata = document.metadata or {}
        vector_rank = _rank_value(metadata.get("_vector_rank"))
        keyword_rank = _rank_value(metadata.get("_keyword_rank"))
        original_ranks = [rank for rank in (vector_rank, keyword_rank) if rank is not None]
        best_original_rank = min(original_ranks) if original_ranks else None
        rrf_score = _numeric_value(metadata.get("_rrf_score"))
        movements.append(
            {
                "id": _retrieval_id(document),
                "vector_rank": vector_rank,
                "keyword_rank": keyword_rank,
                "fusion_rank": fusion_rank,
                "rank_delta": None if best_original_rank is None else fusion_rank - best_original_rank,
                "rrf_score": None if rrf_score is None else round(rrf_score, 10),
            }
        )
    return movements


def _rank_value(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return None
    return int(value)


def _numeric_value(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _document_summary_output(documents: list[Document], *, limit: int = 5) -> dict[str, Any]:
    top_ids: list[str] = []
    top_files: list[str] = []
    top_scores: list[float] = []
    seen_files: set[str] = set()
    for document in documents[:limit]:
        metadata = document.metadata or {}
        document_id = _first_metadata_value(metadata, ("citation_id", "chunk_id", "_retrieval_id"))
        if document_id is not None:
            top_ids.append(document_id)
        file_name = _first_metadata_value(metadata, ("file_name",))
        if file_name is not None and file_name not in seen_files:
            top_files.append(file_name)
            seen_files.add(file_name)
        score = _first_numeric_value(metadata, ("relevance_score", "_fallback_score", "_keyword_score", "_rrf_score"))
        if score is not None:
            top_scores.append(score)
    output: dict[str, Any] = {}
    if top_ids:
        output["top_document_ids"] = top_ids
    if top_files:
        output["top_document_files"] = top_files
    if top_scores:
        output["top_score_summary"] = {
            "max": max(top_scores),
            "min": min(top_scores),
            "count": len(top_scores),
        }
    return output


def _final_evidence_span_output(documents: list[Document]) -> dict[str, Any]:
    citation_ids: list[str] = []
    files: list[str] = []
    seen_files: set[str] = set()
    context_chars = 0
    for document in documents:
        metadata = document.metadata or {}
        context_chars += len(str(document.page_content))
        citation_id = metadata.get("citation_id")
        if citation_id is not None:
            citation_ids.append(str(citation_id))
        file_name = metadata.get("file_name")
        if file_name is not None and str(file_name) not in seen_files:
            files.append(str(file_name))
            seen_files.add(str(file_name))
    return {
        "count": len(documents),
        "citation_ids": citation_ids,
        "files": files,
        "context_chars": context_chars,
    }


def _first_metadata_value(metadata: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _first_numeric_value(metadata: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _retrieval_summary_output(diagnostics: dict[str, Any]) -> dict[str, Any]:
    fallbacks = list(diagnostics.get("fallbacks") or [])
    keys = (
        "vector_candidates",
        "keyword_candidates",
        "merged_candidates",
        "deduped_candidates",
        "fused_candidates",
        "rerank_input_candidates",
        "fusion",
        "fusion_rrf_k",
        "reranked_candidates",
        "final_evidence",
        "vector_search_status",
        "keyword_search_status",
        "reranker",
    )
    output = {key: diagnostics[key] for key in keys if key in diagnostics}
    output["fallbacks"] = fallbacks
    output["degraded"] = bool(fallbacks)
    return output


def _retrieval_degraded_tags(diagnostics: dict[str, Any]) -> list[str]:
    fallbacks = list(diagnostics.get("fallbacks") or [])
    if not fallbacks:
        return []
    tags = ["degraded"]
    seen = set(tags)
    for fallback in fallbacks:
        reason = _bounded_fallback_tag(str(fallback))
        if not reason:
            continue
        tag = f"fallback:{reason}"
        if tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return tags


def _bounded_fallback_tag(reason: str) -> str:
    sanitized = "".join(character if character.isalnum() or character in {"_", "-", ":"} else "_" for character in reason)
    return sanitized[:80].strip("_")


def _trace_output_value(value: Any) -> Any:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _vector_error_diagnostics(vector_search: Any) -> dict[str, Any]:
    if not getattr(vector_search, "vector_unavailable", False):
        return {}
    error_type = getattr(vector_search, "error_type", None)
    return {"vector_search_error_type": str(error_type)} if error_type else {}
