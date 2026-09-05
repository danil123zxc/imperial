"""Reviewed source evidence and deterministic union-coverage metrics."""
from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from imperial_rag.answering.strict import pack_context
from imperial_rag.ingestion.provenance import digest, validate_chunk, validate_span

KS = (1, 3, 5, 10)
BUDGETS = (1000, 2000, 4000)


def assemble_benchmark(
    questions: list[dict[str, Any]], annotations: list[dict[str, Any]], snapshot: dict[str, Any],
) -> dict[str, Any]:
    by_id = {row["id"]: row for row in annotations}
    if len(by_id) != len(annotations) or set(by_id) != {row["id"] for row in questions}:
        raise ValueError("Sidecar must contain exactly one annotation per question")
    sources = {row["source_id"]: row for row in snapshot["sources"]}
    examples = []
    for question in questions:
        annotation = by_id[question["id"]]
        if annotation.get("review_status") != "reviewed" or annotation.get("split") not in {"dev", "test"}:
            raise ValueError(f"{question['id']}: reviewed annotations and an explicit split are required")
        if annotation.get("question_hash") != digest(question):
            raise ValueError(f"{question['id']}: question changed since annotation")
        if annotation.get("snapshot_hash") != snapshot["snapshot_hash"]:
            raise ValueError(f"{question['id']}: annotation snapshot mismatch")
        units = annotation.get("evidence")
        if not isinstance(units, list):
            raise ValueError("Evidence must be a list")
        refusal = question["expected_behavior"] == "refuse_if_not_found"
        if refusal and units or not refusal and not units:
            raise ValueError("Answerable questions require evidence; refusal questions require none")
        if len({unit.get("evidence_id") for unit in units}) != len(units):
            raise ValueError("Evidence IDs must be unique within a question")
        if question["expected_behavior"] == "surface_conflict" and len(units) < 2:
            raise ValueError("Conflict questions require at least two reviewed claim units")
        for unit in units:
            if not isinstance(unit.get("evidence_id"), str) or not unit["evidence_id"].strip():
                raise ValueError("Each evidence unit requires a nonempty ID")
            sets = unit.get("support_sets")
            if not isinstance(sets, list) or not sets:
                raise ValueError("Evidence requires nonempty alternative support sets")
            for support in sets:
                if not isinstance(support, list) or not support:
                    raise ValueError("A support set must contain at least one source span")
                for span in support:
                    if validate_span(span, sources) != span.get("quote"):
                        raise ValueError("Evidence quote does not match frozen source text")
        examples.append(question | {"evidence": units, "split": annotation["split"]})
    payload = {"schema_version": "imperial-evidence-benchmark-v1",
               "snapshot_hash": snapshot["snapshot_hash"], "examples": examples}
    return payload | {"dataset_hash": digest(payload)}


def covered(gold: dict[str, Any], retrieved: list[dict[str, Any]]) -> bool:
    cursor = gold["start"]
    intervals = sorted((max(gold["start"], span["start"]), min(gold["end"], span["end"]))
                       for span in retrieved
                       if (span["source_id"], span["text_sha256"]) == (gold["source_id"], gold["text_sha256"]))
    for start, end in intervals:
        if start > cursor:
            break
        cursor = max(cursor, end)
    return cursor >= gold["end"]


def evidence_recall(units: list[dict[str, Any]], documents: list[Document]) -> dict[str, Any]:
    spans = [span for doc in documents for span in doc.metadata.get("source_spans", [])]
    recovered = [unit["evidence_id"] for unit in units
                 if any(all(covered(gold, spans) for gold in support) for support in unit["support_sets"])]
    return {"evidence_recall": len(recovered) / len(units) if units else None,
            "full_evidence_success": int(len(recovered) == len(units)) if units else None,
            "recovered_evidence_ids": recovered}


def score_retrieval(
    example: dict[str, Any], documents: list[Document], snapshot: dict[str, Any],
) -> dict[str, Any]:
    sources = {row["source_id"]: row for row in snapshot["sources"]}
    for document in documents:
        validate_chunk(document, sources)
    result: dict[str, Any] = {"budgets": {}}
    for k in KS:
        metrics = evidence_recall(example["evidence"], documents[:k])
        result.update({f"{key}_at_{k}": value for key, value in metrics.items()})
    for budget in BUDGETS:
        packed = pack_context(documents, budget)
        result["budgets"][str(budget)] = evidence_recall(example["evidence"], packed["documents"]) | {
            "proxy_tokens": packed["proxy_tokens"], "budget_utilization": packed["budget_utilization"],
            "delivered_chunk_ids": [doc.metadata.get("chunk_id") for doc in packed["documents"]],
            "delivered_source_spans": [span for doc in packed["documents"] for span in doc.metadata["source_spans"]],
        }
    return result


def draft_annotations(
    questions: list[dict[str, Any]], chunks: list[dict[str, Any]], snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    """Navigation candidates only: a reviewer must select sufficient claim spans."""
    by_id = {row["metadata"].get("chunk_id"): row for row in chunks}
    output = []
    for question in questions:
        candidates = []
        for context_id in question.get("reference_context_ids", []):
            chunk = by_id.get(context_id)
            if chunk is None:
                continue
            metadata = chunk["metadata"]
            for source in snapshot["sources"]:
                source_metadata = source["metadata"]
                if metadata.get("file_id") != source_metadata.get("file_id"):
                    continue
                text = chunk["page_content"]
                start = source["page_content"].find(text)
                # Repeated matches remain separate review candidates, never auto-selected gold.
                while start >= 0:
                    candidates.append({"legacy_chunk_id": context_id, "source_id": source["source_id"],
                                       "text_sha256": source["text_sha256"], "start": start,
                                       "end": start + len(text), "quote": text})
                    start = source["page_content"].find(text, start + 1)
        output.append({"id": question["id"], "question_hash": digest(question),
                       "snapshot_hash": snapshot["snapshot_hash"], "split": "dev",
                       "review_status": "draft", "evidence": [], "candidates": candidates})
    return output
