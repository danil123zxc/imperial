"""Reviewed source evidence and deterministic union-coverage metrics."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from imperial_rag.answering.strict import pack_context
from imperial_rag.ingestion.provenance import digest, validate_chunk, validate_span
from imperial_rag.jsonl import read_jsonl

KS = (1, 3, 5, 10)
BUDGETS = (1000, 2000, 4000)
RANKING_VERSION = "complete-evidence-ranking-v1"
RANKING_METRICS = ("evidence_rr", "evidence_ap", "evidence_ndcg")


def _dcg(scores: list[float]) -> float:
    return sum(float(score) / math.log2(index + 2) for index, score in enumerate(scores))


def _mrr(scores: list[float]) -> float:
    return next((1.0 / rank for rank, score in enumerate(scores, 1) if score > 0), 0.0)


def _ndcg_with_ideal(scores: list[float], *, relevant_count: int, k: int) -> float:
    ideal = _dcg([1.0] * min(relevant_count, max(0, k)))
    return _dcg(scores[:k]) / ideal if ideal else 0.0


def _chunk_fingerprint(document: Document) -> str:
    # Retrieval adds scores and ranks; identity covers only text and its exact provenance.
    return digest([document.page_content, document.metadata.get("source_spans")])


@dataclass
class EvidenceRankingCorpus:
    fingerprints: dict[str, str]
    relevant_ids: dict[str, set[str]]
    corpus_hash: str

    def score(self, example: dict[str, Any], documents: list[Document]) -> dict[str, Any]:
        relevant = self.relevant_ids[example["id"]]
        scores: list[float] = []
        seen: set[str] = set()
        for document in documents:
            chunk_id = document.metadata.get("chunk_id")
            if self.fingerprints.get(chunk_id) != _chunk_fingerprint(document):
                raise ValueError("Retrieved chunk does not match the evaluation corpus")
            scores.append(float(chunk_id in relevant and chunk_id not in seen))
            seen.add(chunk_id)
        result: dict[str, Any] = {
            "ranking_metric_version": RANKING_VERSION, "ranking_corpus_hash": self.corpus_hash,
            "relevant_chunk_count": len(relevant) if example["evidence"] else None,
        }
        for k in KS:
            hits = 0.0
            precision_sum = 0.0
            for rank, score in enumerate(scores[:k], 1):
                hits += score
                precision_sum += score * hits / rank
            values = (_mrr(scores[:k]), precision_sum / len(relevant) if relevant else 0.0,
                      _ndcg_with_ideal(scores, relevant_count=len(relevant), k=k))
            result.update({f"{metric}_at_{k}": value if example["evidence"] else None
                           for metric, value in zip(RANKING_METRICS, values)})
        return result


def load_ranking_corpus(
    path: Path, snapshot: dict[str, Any], examples: list[dict[str, Any]],
) -> EvidenceRankingCorpus:
    documents = [Document(**row) for row in read_jsonl(path)]
    if not documents:
        raise ValueError("Ranking evaluation requires a nonempty chunk corpus")
    sources = {row["source_id"]: row for row in snapshot["sources"]}
    fingerprints: dict[str, str] = {}
    for document in documents:
        validate_chunk(document, sources)
        chunk_id = document.metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip() or chunk_id in fingerprints:
            raise ValueError("Ranking corpus requires unique nonempty chunk IDs")
        fingerprints[chunk_id] = _chunk_fingerprint(document)
    # ponytail: scan each question's corpus once; index spans if benchmark size makes this costly.
    relevant = {example["id"]: {doc.metadata["chunk_id"] for doc in documents
                                if evidence_recall(example["evidence"], [doc])["recovered_evidence_ids"]}
                for example in examples}
    return EvidenceRankingCorpus(fingerprints, relevant, digest(fingerprints))


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
    *, ranking_corpus: EvidenceRankingCorpus | None = None,
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
    if ranking_corpus is not None:
        result.update(ranking_corpus.score(example, documents))
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
