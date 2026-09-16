from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

from langchain_core.documents import Document
import pytest

from imperial_rag.answering.strict import build_context, pack_context
from imperial_rag.evals.chunk_comparison import evaluate_configuration, run_comparison, summarize, load_comparison
from imperial_rag.evals.evidence import (
    assemble_benchmark, draft_annotations, evidence_recall, load_evidence_corpus, score_retrieval,
)
from imperial_rag.ingestion.chunking import build_chunks, _body_start_index
from imperial_rag.ingestion.provenance import digest, document_payload, freeze_sources, load_snapshot, validate_chunk
from imperial_rag.jsonl import write_jsonl


def benchmark_case(tmp_path):
    doc = Document(page_content="Revenue 2023: 10.\nRevenue 2024: 12.\n" + "Other text. " * 40,
                   metadata={"file_id": "f", "source_locator": "body:1", "source_type": "body"})
    snapshot = freeze_sources([doc], tmp_path / "snapshot.json")
    source = snapshot["sources"][0]
    question = {"id": "q", "question": "How did revenue change?", "reference_answer": "20%",
                "expected_behavior": "cite_answer", "reference_context_ids": ["old"]}
    span = {key: source[key] for key in ("source_id", "text_sha256")}
    span.update(start=0, end=17, quote=doc.page_content[:17])
    annotation = {"id": "q", "question_hash": digest(question), "snapshot_hash": snapshot["snapshot_hash"],
                  "split": "dev", "review_status": "reviewed",
                  "evidence": [{"evidence_id": "revenue", "support_sets": [[span]]}]}
    benchmark = assemble_benchmark([question], [annotation], snapshot)
    return doc, snapshot, question, annotation, benchmark


def test_snapshot_and_gold_survive_rechunking(tmp_path):
    doc, snapshot, _, _, benchmark = benchmark_case(tmp_path)
    chunks = [build_chunks([doc], size, 0) for size in (8, 20, 40)]
    assert len({len(rows) for rows in chunks}) == 3
    for rows in chunks:
        assert score_retrieval(benchmark["examples"][0], rows, snapshot)["budgets"]["2000"]["evidence_recall"] == 1
    assert load_snapshot(tmp_path / "snapshot.json") == snapshot
    with pytest.raises(FileExistsError):
        freeze_sources([doc], tmp_path / "snapshot.json")
    changed = deepcopy(snapshot)
    changed["sources"][0]["page_content"] += "!"
    (tmp_path / "changed.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="modified"):
        load_snapshot(tmp_path / "changed.json")


def test_same_locator_different_extracted_elements_have_distinct_identities(tmp_path):
    docs = [Document(page_content=text, metadata={"file_id": "f", "source_locator": "section:13"})
            for text in ("First paragraph", "Second paragraph")]
    snapshot = freeze_sources(docs, tmp_path / "snapshot.json")
    assert len({row["source_id"] for row in snapshot["sources"]}) == 2


def test_vector_embedding_roundtrip_restores_source_text(tmp_path):
    from imperial_rag.indexing.vector import _embedding_document
    from imperial_rag.retrieval.identity import _annotate_retrieval_documents
    from imperial_rag.evals.phoenix_experiment import _document_payload

    doc, snapshot, _, _, _ = benchmark_case(tmp_path)
    doc.metadata["file_name"] = "Context header"
    chunk = build_chunks([doc])[0]
    stored = _embedding_document(chunk)
    assert stored.page_content != chunk.page_content
    returned = _annotate_retrieval_documents([stored], rank_key="_vector_rank")[0]
    validate_chunk(returned, {row["source_id"]: row for row in snapshot["sources"]})
    assert _document_payload(returned)["metadata"]["source_spans"] == chunk.metadata["source_spans"]


def test_table_headers_map_to_original_rows_and_unicode(tmp_path):
    doc = Document(page_content=" Заголовок | год\n" + "\n".join(f" Доход🙂{n} | {n} " for n in range(12)),
                   metadata={"file_id": "table", "source_type": "sheet", "source_locator": "sheet:1"})
    snapshot = freeze_sources([doc], tmp_path / "snapshot.json")
    sources = {row["source_id"]: row for row in snapshot["sources"]}
    chunks = build_chunks([doc], 15, 0)
    assert len(chunks) > 2
    for chunk in chunks:
        validate_chunk(chunk, sources)
        assert chunk.metadata["source_spans"][0]["start"] == 1
    assert chunks[-1].metadata["source_spans"][-1]["start"] > chunks[0].metadata["source_spans"][-1]["start"]
    bad = deepcopy(chunks[0])
    bad.page_content = "WRONG" + bad.page_content
    with pytest.raises(ValueError, match="does not match"):
        validate_chunk(bad, sources)
    with pytest.raises(ValueError, match="cannot be mapped"):
        _body_start_index("abc", "missing", {"start_index": 0}, 0)


def test_union_coverage_alternatives_and_duplicate_chunks():
    def span(start, end, source="s"):
        return {"source_id": source, "text_sha256": "v", "start": start, "end": end}

    gold = [{"evidence_id": "a", "support_sets": [[span(10, 30)], [span(50, 60), span(70, 80)]]},
            {"evidence_id": "b", "support_sets": [[span(0, 5, "other")]]}]
    def docs(*spans):
        return [Document(page_content="unused", metadata={"source_spans": [s]}) for s in spans]

    assert evidence_recall(gold, docs(span(0, 20), span(20, 35), span(0, 20)))["evidence_recall"] == .5
    assert evidence_recall(gold, docs(span(0, 19), span(20, 35)))["evidence_recall"] == 0
    assert evidence_recall(gold, docs(span(50, 60)))["evidence_recall"] == 0
    assert evidence_recall(gold, docs(span(50, 60), span(70, 80), span(0, 5, "other")))["full_evidence_success"] == 1
    assert evidence_recall(gold, docs(span(10, 30) | {"text_sha256": "old"}))["evidence_recall"] == 0
    assert evidence_recall(gold, [])["evidence_recall"] == 0
    assert evidence_recall([], [])["evidence_recall"] is None


def test_evidence_corpus_validation_survives_ranking_removal(tmp_path):
    doc, snapshot, _, _, benchmark = benchmark_case(tmp_path)
    documents = build_chunks([doc])
    path = tmp_path / "chunks.jsonl"
    write_jsonl(path, [document_payload(doc) for doc in documents])
    corpus = load_evidence_corpus(path, snapshot)
    result = score_retrieval(benchmark["examples"][0], documents, snapshot, evidence_corpus=corpus)
    assert result["evidence_recall_at_10"] == 1
    assert not any(key.startswith(("evidence_rr_", "evidence_ap_", "evidence_ndcg_")) for key in result)
    stale = deepcopy(documents[0])
    stale.page_content += "changed"
    with pytest.raises(ValueError, match="does not match"):
        corpus.validate_documents([stale])
    stale.metadata["source_spans"][0]["text_sha256"] = "stale"
    for invalid in ([], [documents[0], documents[0]], [stale]):
        write_jsonl(path, [document_payload(doc) for doc in invalid])
        with pytest.raises(ValueError):
            load_evidence_corpus(path, snapshot)
    with pytest.raises(FileNotFoundError):
        load_evidence_corpus(tmp_path / "missing.jsonl", snapshot)


@pytest.mark.parametrize("mutation", ["draft", "quote", "offset", "version", "question", "empty", "conflict", "missing"])
def test_annotation_validation_rejects_invalid_benchmark(tmp_path, mutation):
    _, snapshot, question, annotation, _ = benchmark_case(tmp_path)
    if mutation == "draft":
        annotation["review_status"] = "draft"
    elif mutation in {"quote", "offset", "version"}:
        key, value = {"quote": ("quote", "wrong"), "offset": ("start", -1), "version": ("text_sha256", "old")}[mutation]
        annotation["evidence"][0]["support_sets"][0][0][key] = value
    elif mutation == "question":
        question["question"] = "changed"
    elif mutation == "empty":
        annotation["evidence"] = []
    elif mutation == "conflict":
        question["expected_behavior"] = "surface_conflict"
        annotation["question_hash"] = digest(question)
    with pytest.raises(ValueError):
        assemble_benchmark([question], [] if mutation == "missing" else [annotation], snapshot)


def test_drafts_never_promote_legacy_chunks(tmp_path):
    doc, snapshot, question, _, _ = benchmark_case(tmp_path)
    drafts = draft_annotations([question], [{"page_content": doc.page_content[:17],
                                            "metadata": {"chunk_id": "old", "file_id": "f"}}], snapshot)
    assert drafts[0]["candidates"]
    assert drafts[0]["evidence"] == []
    assert drafts[0]["review_status"] == "draft"


def test_packing_matches_generation_and_excludes_unselected_citations():
    from imperial_rag.answering.workflow import build_query_workflow

    docs = [Document(page_content="large " * 1000, metadata={"source_type": "body"}),
            Document(page_content="Доход 🙂 12", metadata={"source_type": "body"})]
    packed = pack_context(docs, 40)
    assert packed["documents"] == docs[1:]
    assert packed["proxy_tokens"] <= 40
    assert packed["context"] == build_context(docs[1:])
    captured = []
    def generate(question, evidence):
        captured.append(build_context(evidence))
        return "Доход 12 [S1]"

    graph = build_query_workflow(retrieve=lambda _: docs, generate=generate, context_token_budget=40)
    result = graph.invoke({"question": "Revenue?"})
    assert captured == [packed["context"]]
    assert result["evidence"] == docs[1:]
    assert len(result["citations"]) == 1
    assert result["citations_valid"]
    assert pack_context(docs, 1)["documents"] == []
    assert pack_context(docs)["documents"] == docs


def test_async_retrieval_errors_and_empty_results(tmp_path):
    _, snapshot, _, _, benchmark = benchmark_case(tmp_path)
    examples = [benchmark["examples"][0] | {"id": str(i)} for i in range(3)]
    class Empty:
        def retrieve(self, question):
            return SimpleNamespace(evidence=[], diagnostics={})
    rows = asyncio.run(evaluate_configuration(examples, snapshot, Empty(), concurrency=2))
    assert summarize(rows)["eligible"]
    assert summarize(rows)["evidence_recall_at_1"] == 0
    class Degraded:
        def retrieve(self, question):
            return SimpleNamespace(evidence=[], diagnostics={"fallbacks": ["vector_search_failed"]})
    rows = asyncio.run(evaluate_configuration(examples, snapshot, Degraded()))
    assert not summarize(rows)["eligible"]
    assert summarize(rows)["errors"] == 3
    class Unmapped:
        def retrieve(self, question):
            return SimpleNamespace(evidence=[Document(page_content="unmapped")], diagnostics={})
    rows = asyncio.run(evaluate_configuration(examples, snapshot, Unmapped()))
    assert all(row["error"] == "ValueError" and not row["eligible"] for row in rows)


def test_refusal_gold_is_reviewed_but_recall_undefined(tmp_path):
    _, snapshot, question, annotation, _ = benchmark_case(tmp_path)
    question["expected_behavior"] = "refuse_if_not_found"
    annotation.update(question_hash=digest(question), evidence=[])
    benchmark = assemble_benchmark([question], [annotation], snapshot)
    assert score_retrieval(benchmark["examples"][0], [], snapshot)["evidence_recall_at_10"] is None


@pytest.mark.parametrize("mutation", ["none", "score", "corpus", "version", "legacy"])
def test_saved_ranking_integrity_and_legacy_artifacts(tmp_path, mutation):
    RANKING_VERSION = "complete-evidence-ranking-v1"
    from imperial_rag.evals.phoenix_experiment import _phoenix_evidence_evaluators

    doc, snapshot, _, _, benchmark = benchmark_case(tmp_path)
    chunks = build_chunks([doc])
    write_jsonl(tmp_path / "chunks.jsonl", [document_payload(doc) for doc in chunks])
    corpus = load_evidence_corpus(tmp_path / "chunks.jsonl", snapshot)
    row = {"eligible": True, "latency_ms": 1,
           "metrics": score_retrieval(benchmark["examples"][0], chunks, snapshot, evidence_corpus=corpus)}
    row["metrics"].update(ranking_metric_version=RANKING_VERSION, ranking_corpus_hash=corpus.corpus_hash,
                          evidence_ap_at_1=1.0)
    manifest = {"dataset_hash": benchmark["dataset_hash"], "ranking_metric_version": RANKING_VERSION,
                "configs": [{"config_id": "s400-o50", "ranking_corpus_hash": corpus.corpus_hash}],
                "results_hash": digest([[row]])}
    if mutation == "score":
        row["metrics"]["evidence_ap_at_1"] = .123
    elif mutation == "corpus":
        manifest["configs"][0]["ranking_corpus_hash"] = "changed"
    elif mutation == "version":
        manifest["ranking_metric_version"] = "future"
    elif mutation == "legacy":
        manifest.pop("ranking_metric_version")
        manifest["configs"][0].pop("ranking_corpus_hash")
        row["metrics"] = score_retrieval(benchmark["examples"][0], chunks, snapshot)
        manifest["results_hash"] = digest([[row]])
    write_jsonl(tmp_path / "s400-o50" / "results.jsonl", [row])
    (tmp_path / "benchmark.json").write_text(json.dumps(benchmark))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    if mutation in {"score", "corpus", "version"}:
        with pytest.raises(ValueError):
            load_comparison(tmp_path)
    else:
        loaded, _, rows = load_comparison(tmp_path)
        evaluators = _phoenix_evidence_evaluators()
        assert "evidence_ap_at_1" not in evaluators
        assert "evidence_map_at_1" not in summarize(rows)


def test_comparison_isolated_and_phoenix_dataset_pinned(tmp_path, monkeypatch):
    from imperial_rag.config import Settings
    from imperial_rag.evals import chunk_comparison as module
    from imperial_rag.evals.phoenix_experiment import publish_evidence_comparison_async
    from imperial_rag.integrations.dashscope import QwenProviderSettings
    import phoenix.client

    doc, snapshot, _, _, benchmark = benchmark_case(tmp_path)
    settings = Settings(workspace_root=tmp_path)
    targets = []
    monkeypatch.setattr(QwenProviderSettings, "require_api_key", lambda self: "test")
    import imperial_rag.integrations.dashscope as provider
    def forbidden_chat(*args, **kwargs):
        raise AssertionError("Retrieval-only comparison must never construct an answer model")
    monkeypatch.setattr(provider, "create_chat_model", forbidden_chat)
    def build(snapshot, shadow, retrieval):
        targets.append(shadow)
        assert retrieval.rerank_top_n == 100
        chunks = build_chunks([doc], retrieval.chunk_size, retrieval.chunk_overlap)
        write_jsonl(shadow.extraction_root / "chunks.jsonl", [document_payload(doc) for doc in chunks])
        class Service:
            def retrieve(self, question):
                return SimpleNamespace(evidence=chunks, diagnostics={})
        return Service(), ()
    monkeypatch.setattr(module, "build_shadow_retriever", build)
    root = tmp_path / "run"
    asyncio.run(run_comparison(snapshot, benchmark, settings, root, configs=[(400, 50), (20, 0)]))
    assert len({target.qdrant_collection for target in targets}) == 2
    assert all(target.qdrant_collection != settings.qdrant_collection for target in targets)
    assert not (tmp_path / ".imperial_rag" / "active-ingestion.json").exists()
    manifest, _, rows = load_comparison(root)
    assert {row["dataset_hash"] for row in rows} == {manifest["dataset_hash"]}
    assert all(row["eligible"] for row in rows)
    created, versions, experiments = [], [], []
    class FakeClient:
        def __init__(self, **kwargs):
            self.datasets = self.experiments = self
        async def create_dataset(self, **kwargs):
            dataset = SimpleNamespace(id="dataset", version_id="version-1", examples=[
                {"input": inp, "output": out} for inp, out in zip(kwargs["inputs"], kwargs["outputs"])
            ])
            created.append(dataset)
            return dataset
        async def get_dataset(self, **kwargs):
            versions.append(kwargs["version_id"])
            return created[0]
        async def run_experiment(self, **kwargs):
            experiments.append(kwargs["dataset"].version_id)
            for example in kwargs["dataset"].examples:
                output = await kwargs["task"](example["input"])
                assert kwargs["evaluators"]["evidence_recall_budget_2000"](output, example["output"])["score"] == 1
                for metric in ("rr", "ap", "ndcg"):
                    key = f"evidence_{metric}_at_10"
                    assert key not in kwargs["evaluators"]
                    assert key not in output["metrics"]
            return {"id": "experiment"}
    monkeypatch.setattr(phoenix.client, "AsyncClient", FakeClient)
    asyncio.run(publish_evidence_comparison_async(root, settings))
    asyncio.run(publish_evidence_comparison_async(root, settings))
    assert len(created) == 1
    assert versions == ["version-1"]
    assert experiments == ["version-1", "version-1"]

    from langchain_core.runnables import RunnableLambda
    captured = []
    def answer(prompt):
        captured.append(prompt.to_string())
        return "Revenue increased [S1]"
    monkeypatch.setattr(provider, "create_chat_model", lambda: RunnableLambda(answer))
    answers = asyncio.run(module.generate_comparison_answers(root))
    assert len(answers) == 2
    assert len(captured) == 2
    assert all(any(row["context"] in prompt for prompt in captured) for row in answers)
    assert all(row["checks"]["citations_valid"] for row in answers)
    with pytest.raises(FileExistsError):
        asyncio.run(module.generate_comparison_answers(root))
