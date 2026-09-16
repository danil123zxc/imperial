from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document
from phoenix.client.resources.experiments import _bind_task_signature, _validate_task_signature
import pytest

pytestmark = pytest.mark.usefixtures("fake_phoenix_retrieval_judge")

from imperial_rag.answering.workflow import build_query_workflow
from imperial_rag.evals import phoenix_experiment
from imperial_rag.evals.evidence import assemble_benchmark
from imperial_rag.ingestion.chunking import build_chunks
from imperial_rag.ingestion.provenance import digest, document_payload, freeze_sources
from imperial_rag.jsonl import write_jsonl


@pytest.fixture
def evidence_case(tmp_path):
    document = Document(page_content="Revenue increased by 20%.", metadata={"file_id": "file", "source_locator": "body:1"})
    snapshot_path = tmp_path / "snapshot.json"
    snapshot = freeze_sources([document], snapshot_path)
    source = snapshot["sources"][0]
    question = {"id": "q", "suite": "answer", "lane": "indexed_answerability",
                "question": "How much did revenue increase?", "reference_answer": "20%",
                "expected_behavior": "cite_answer", "reference_context_ids": ["obsolete-chunk"]}
    span = {key: source[key] for key in ("source_id", "text_sha256")}
    span.update(start=0, end=len(document.page_content), quote=document.page_content)
    annotation = {"id": "q", "question_hash": digest(question), "snapshot_hash": snapshot["snapshot_hash"],
                  "split": "dev", "review_status": "reviewed",
                  "evidence": [{"evidence_id": "revenue", "support_sets": [[span]]}]}
    questions_path, annotations_path = tmp_path / "questions.jsonl", tmp_path / "annotations.jsonl"
    write_jsonl(questions_path, [question])
    write_jsonl(annotations_path, [annotation])
    return SimpleNamespace(
        snapshot=snapshot, question=question, annotation=annotation, documents=build_chunks([document]),
        snapshot_path=snapshot_path, questions_path=questions_path, annotations_path=annotations_path,
        argv=["--snapshot", str(snapshot_path), "--annotations", str(annotations_path),
              "--questions-path", str(questions_path)],
    )


def prepare_runner(monkeypatch):
    module = _load_all_evals_runner()
    settings = SimpleNamespace(phoenix_client_endpoint="http://localhost:6006", phoenix_project_name="imperial-rag")
    monkeypatch.setattr(module.phoenix_eval, "_load_project_env", lambda root: None)
    monkeypatch.setattr(module.phoenix_eval, "_build_settings", lambda root: settings)
    monkeypatch.setattr(module.phoenix_eval, "_configure_tracing", lambda settings, enabled: None)
    monkeypatch.setattr(module, "_configure_observability", lambda settings: None)
    monkeypatch.setattr(module, "_assert_phoenix_reachable", lambda endpoint: None)
    return module


@pytest.mark.parametrize("args,metrics", [([], ["faithfulness", "answer_relevancy"]),
                                         (["--ragas-metrics", "none"], []),
                                         (["--ragas-metrics", "id_context_recall"], ["id_context_recall"])])
def test_all_evals_forwards_validated_evidence_and_ragas(monkeypatch, evidence_case, args, metrics):
    module = prepare_runner(monkeypatch)
    captured = {}
    monkeypatch.setattr(module.phoenix_eval, "run_phoenix_experiment", lambda **kwargs: captured.update(kwargs))
    module.main(evidence_case.argv + args + ["--concurrency", "5"])
    benchmark = assemble_benchmark([evidence_case.question], [evidence_case.annotation], evidence_case.snapshot)
    assert captured["examples"] == benchmark["examples"]
    assert captured["evidence_snapshot"] == evidence_case.snapshot
    assert captured["ragas_metric_names"] == metrics
    assert captured["concurrency"] == 5
    assert captured["retrieval_k"] == 5
    assert captured["experiment_name"] == "imperial-rag-all-evals"


@pytest.mark.parametrize("args", [[], ["--snapshot", "snapshot.json"], ["--annotations", "annotations.jsonl"],
                                  ["--snapshot", "snapshot.json", "--annotations", "annotations.jsonl", "--concurrency", "0"]])
def test_all_evals_rejects_missing_evidence_or_invalid_concurrency(monkeypatch, args):
    module = _load_all_evals_runner()
    monkeypatch.setattr(module, "load_snapshot", lambda path: pytest.fail("read snapshot after invalid arguments"))
    with pytest.raises(SystemExit) as exc_info:
        module.main(args)
    assert exc_info.value.code == 2


@pytest.mark.parametrize("mutation", ["draft", "question", "snapshot", "snapshot_hash", "missing", "empty"])
def test_all_evals_validates_before_any_external_calls(monkeypatch, evidence_case, mutation):
    module = _load_all_evals_runner()
    monkeypatch.setattr(module.phoenix_eval, "_load_project_env", lambda root: pytest.fail("external setup before validation"))
    case = evidence_case
    if mutation == "draft":
        case.annotation["review_status"] = "draft"
    elif mutation == "question":
        case.question["question"] += " Changed?"
    elif mutation == "snapshot":
        case.snapshot["sources"][0]["page_content"] += " changed"
    elif mutation == "snapshot_hash":
        case.annotation["snapshot_hash"] = "old"
    case.snapshot_path.write_text(json.dumps(case.snapshot))
    write_jsonl(case.questions_path, [] if mutation == "empty" else [case.question])
    write_jsonl(case.annotations_path, [] if mutation in {"empty", "missing"} else [case.annotation])
    with pytest.raises((ValueError, SystemExit)):
        module.main(case.argv)


@pytest.mark.parametrize("scenario", ["packed", "empty", "refusal", "mapping", "degraded", "fallback", "provider", "missing_ranked", "judge_error"])
def test_all_evals_evidence_experiment_end_to_end(monkeypatch, evidence_case, scenario):
    module = prepare_runner(monkeypatch)
    case = evidence_case
    extraction_root = case.snapshot_path.parent / "extracted"
    write_jsonl(extraction_root / "chunks.jsonl", [document_payload(doc) for doc in case.documents])
    settings = module.phoenix_eval._build_settings(None)
    settings.extraction_root = extraction_root
    if scenario == "refusal":
        case.question.update(expected_behavior="refuse_if_not_found", lane="refusal_out_of_corpus_behavior")
        case.annotation.update(evidence=[], question_hash=digest(case.question))
        write_jsonl(case.questions_path, [case.question])
        write_jsonl(case.annotations_path, [case.annotation])
    documents = [] if scenario == "empty" else case.documents
    if scenario == "mapping":
        documents[0].metadata.pop("source_spans")
    diagnostics = {"degraded": scenario == "degraded", "fallbacks": ["keyword_only"] if scenario == "fallback" else []}
    workflow = build_query_workflow(
        retrieve=lambda query: {"retrieved_documents": documents, "retrieval": diagnostics},
        generate=lambda *args: pytest.fail("answer model called despite empty packed context"),
        context_token_budget=1,
    )
    captured = {}
    if scenario == "judge_error":
        async def fail_judging(example, output):
            return {"status": "error", "error_type": "HTTPStatusError", "error_stage": "log_annotations"}
        monkeypatch.setattr(phoenix_experiment, "PhoenixRetrievalJudge",
                            lambda *a, **kw: SimpleNamespace(evaluate=fail_judging, metadata={}))

    def query(question):
        if scenario == "provider":
            raise OSError("private provider failure")
        result = workflow.invoke({"question": question})
        if scenario == "missing_ranked":
            result.pop("ranked_documents")
        assert result["evidence"] == []
        assert result["retrieved_documents"] == []
        return result

    monkeypatch.setattr(phoenix_experiment, "build_runtime", lambda settings: SimpleNamespace(query=query))

    async def create_dataset(**kwargs):
        captured["dataset"] = kwargs
        return kwargs

    async def run_experiment(**kwargs):
        captured["experiment"] = kwargs
        signature = inspect.signature(kwargs["task"])
        _validate_task_signature(signature)
        bound = _bind_task_signature(signature, {
            "id": "example-1", "input": kwargs["dataset"]["inputs"][0],
            "output": kwargs["dataset"]["outputs"][0], "metadata": kwargs["dataset"]["metadata"][0],
        })
        output = await kwargs["task"](*bound.args, **bound.kwargs)
        expected = kwargs["dataset"]["outputs"][0]
        captured["output"] = output
        captured["scores"] = {name: evaluator(output=output, expected=expected)
                              for name, evaluator in kwargs["evaluators"].items()}
        return SimpleNamespace(id="experiment-1")

    monkeypatch.setitem(sys.modules, "phoenix.client", SimpleNamespace(AsyncClient=lambda **kwargs: SimpleNamespace(
        datasets=SimpleNamespace(create_dataset=create_dataset), experiments=SimpleNamespace(run_experiment=run_experiment),
    )))
    invalid = scenario in {"mapping", "degraded", "fallback", "provider", "missing_ranked"}
    if invalid or scenario == "judge_error":
        with pytest.raises(RuntimeError, match="Evaluation failed") as exc:
            module.main(case.argv + ["--ragas-metrics", "none"])
        assert "private provider failure" not in str(exc.value)
    else:
        module.main(case.argv + ["--ragas-metrics", "none"])
    if scenario == "judge_error":
        assert captured["output"]["retrieval_evaluation"]["status"] == "error"
    scores = captured["scores"]
    evidence_scores = {key: row["score"] for key, row in scores.items()
                       if key.startswith(("evidence_recall_", "full_evidence_success_"))}
    assert len(evidence_scores) == 14
    ranking_scores = {key: row["score"] for key, row in scores.items()
                      if key.startswith(("evidence_rr_", "evidence_ap_", "evidence_ndcg_"))}
    assert ranking_scores == {}
    expected_score = None if invalid or scenario == "refusal" else 0 if scenario == "empty" else 1
    assert set(evidence_scores.values()) == {expected_score}
    assert "legacy_chunk_recall" not in scores and "chunk_recall" not in scores
    assert "legacy_citation_grounding_behavior" in scores
    assert "legacy_conflict_behavior" in scores
    if scenario == "packed":
        assert captured["output"]["documents"] == []
        assert captured["output"]["ranked_documents"]
        assert captured["output"]["retrieval_evaluation"]["document_count"] == len(case.documents)
        assert case.documents[0].metadata["chunk_id"] != "obsolete-chunk"
    benchmark = assemble_benchmark([case.question], [case.annotation], case.snapshot)
    metadata = captured["dataset"]["metadata"][0]
    assert metadata["dataset_hash"] == benchmark["dataset_hash"]
    assert metadata["snapshot_hash"] == case.snapshot["snapshot_hash"]
    assert metadata["split"] == "dev"
    assert captured["dataset"]["outputs"][0]["evidence"] == case.annotation["evidence"]


def test_all_evals_preflight_fails_with_phoenix_start_hint(monkeypatch):
    module = _load_all_evals_runner()

    def broken_urlopen(url: str, timeout: float):
        raise OSError("connection refused")

    monkeypatch.setattr(module.request, "urlopen", broken_urlopen)
    with pytest.raises(SystemExit) as exc_info:
        module._assert_phoenix_reachable("http://localhost:6006")
    assert "Phoenix is not reachable at http://localhost:6006" in str(exc_info.value)
    assert "docker compose up -d phoenix" in str(exc_info.value)


def test_missing_evidence_corpus_fails_before_query_or_experiment(monkeypatch, evidence_case):
    module = prepare_runner(monkeypatch)
    settings = module.phoenix_eval._build_settings(None)
    settings.extraction_root = evidence_case.snapshot_path.parent / "missing"
    monkeypatch.setattr(phoenix_experiment, "build_runtime", lambda **kwargs: pytest.fail("runtime before corpus validation"))
    monkeypatch.setitem(sys.modules, "phoenix.client", SimpleNamespace(
        AsyncClient=lambda **kwargs: pytest.fail("Phoenix client before corpus validation"),
    ))
    with pytest.raises(FileNotFoundError):
        module.main(evidence_case.argv + ["--ragas-metrics", "none"])


def _load_all_evals_runner():
    spec = importlib.util.spec_from_file_location("run_all_evals_for_test", Path("scripts/run_all_evals.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
