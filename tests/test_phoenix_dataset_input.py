from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from langchain_core.documents import Document
import phoenix.client
from phoenix.client.resources.datasets import Dataset
import pytest

from imperial_rag.config import Settings
from imperial_rag.evals import chunk_comparison, dataset_input, phoenix_experiment, ragas_runner
from imperial_rag.ingestion.chunking import build_chunks
from imperial_rag.ingestion.provenance import digest, document_payload, freeze_sources
from imperial_rag.jsonl import read_jsonl, write_jsonl


@pytest.fixture
def case(tmp_path, monkeypatch):
    doc = Document(page_content="Revenue increased by 20%.", metadata={"file_id": "f", "source_locator": "body:1"})
    snapshot_path = tmp_path / "snapshot.json"
    snapshot = freeze_sources([doc], snapshot_path)
    question = {"id": "q", "suite": "answer", "lane": "indexed_answerability", "tags": ["revenue"],
                "question": "How much did revenue increase?", "reference_answer": "20%",
                "expected_behavior": "cite_answer", "expected_source_hints": ["revenue"],
                "reference_context_ids": ["legacy"]}
    source = snapshot["sources"][0]
    span = {key: source[key] for key in ("source_id", "text_sha256")}
    span.update(start=0, end=len(doc.page_content), quote=doc.page_content)
    annotation = {"id": "q", "split": "dev", "review_status": "reviewed", "question_hash": digest(question),
                  "snapshot_hash": snapshot["snapshot_hash"],
                  "evidence": [{"evidence_id": "revenue", "support_sets": [[span], [deepcopy(span)]]}]}
    inputs, outputs, metadata = phoenix_experiment._to_phoenix_dataset_rows([question | annotation])
    dataset = Dataset.from_dict({"id": "dataset-1", "name": "reviewed-questions", "version_id": "version-1",
                                 "examples": [{"id": "example-1", "input": inputs[0], "output": outputs[0],
                                               "metadata": metadata[0]}]})
    state = SimpleNamespace(dataset=dataset, gets=[], experiments=[], pinned={})

    class Client:
        def __init__(self, **kwargs):
            self.datasets = self.experiments = self

        async def get_dataset(self, **kwargs):
            state.gets.append(kwargs)
            return state.pinned.get(kwargs["version_id"], state.dataset)

        async def create_dataset(self, **kwargs):
            pytest.fail("Phoenix-input runs must never create or modify a dataset")

        async def run_experiment(self, **kwargs):
            state.experiments.append(kwargs)
            for example in kwargs["dataset"].examples:
                if kwargs["experiment_name"].startswith("test"):
                    await kwargs["task"](example["input"], metadata=example["metadata"])
                else:
                    output = await kwargs["task"](example["input"], example["metadata"])
                    assert output["id"] == example["metadata"]["id"]
            return {"id": "experiment-1"}

    monkeypatch.setattr(phoenix.client, "AsyncClient", Client)
    monkeypatch.setattr(phoenix_experiment, "build_runtime", lambda **kw: pytest.fail("Unexpected runtime construction"))
    monkeypatch.setattr(ragas_runner, "build_runtime", lambda **kw: pytest.fail("Unexpected runtime construction"))
    settings = Settings(workspace_root=tmp_path)
    write_jsonl(settings.extraction_root / "chunks.jsonl", [document_payload(chunk) for chunk in build_chunks([doc])])
    return SimpleNamespace(doc=doc, snapshot=snapshot, snapshot_path=snapshot_path, question=question,
                           annotation=annotation, state=state, settings=settings, client=Client())


def args(**overrides):
    return argparse.Namespace(phoenix_dataset_name="reviewed-questions", phoenix_dataset_id=None,
                              phoenix_dataset_version_id=None, **overrides)


def load(case, *, evidence=True, selection=None):
    return asyncio.run(dataset_input.load_phoenix_input(selection or args(), case.settings,
                                                       snapshot=case.snapshot if evidence else None))


def script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_mapping_preserves_contract_and_never_invents_review(case):
    questions, annotations = dataset_input.map_phoenix_examples(case.state.dataset)
    assert questions == [case.question]
    assert annotations == [case.annotation]
    source = load(case)
    assert source.benchmark["snapshot_hash"] == case.snapshot["snapshot_hash"]
    assert source.examples[0]["evidence"] == case.annotation["evidence"]
    assert source.binding["version_id"] == "version-1"
    row = case.state.dataset.examples[0]
    row["metadata"].pop("review_status")
    assert dataset_input.map_phoenix_examples(case.state.dataset)[1][0]["review_status"] is None
    with pytest.raises(ValueError, match="reviewed"):
        load(case)


@pytest.mark.parametrize("selector", ["name", "id"])
@pytest.mark.parametrize("version", [None, "version-1"])
def test_resolves_version_once(case, selector, version):
    selection = args()
    selection.phoenix_dataset_version_id = version
    if selector == "id":
        selection.phoenix_dataset_name = None
        selection.phoenix_dataset_id = "dataset-1"
    source = load(case, selection=selection)
    assert case.state.gets == [{"dataset": {selector: "dataset-1" if selector == "id" else "reviewed-questions"},
                                "version_id": version}]
    assert source.binding["dataset_id"] == "dataset-1"
    assert source.binding["version_id"] == "version-1"
    assert source.benchmark["phoenix_dataset"] == source.binding


@pytest.mark.parametrize("mutation", ["draft", "question_hash", "snapshot_hash", "question", "missing_evidence",
                                      "empty_evidence", "quote", "offset", "source", "text_hash", "split", "support"])
@pytest.mark.parametrize("command", ["all", "validate", "run"])
def test_evidence_input_rejects_invalid_annotations_before_execution(case, monkeypatch, mutation, command, tmp_path):
    row = case.state.dataset.examples[0]
    if mutation == "draft":
        row["metadata"]["review_status"] = "draft"
    elif mutation in {"question_hash", "snapshot_hash"}:
        row["metadata"][mutation] = "stale"
    elif mutation == "question":
        row["input"]["question"] += " Changed?"
    elif mutation == "missing_evidence":
        row["output"].pop("evidence")
    elif mutation == "empty_evidence":
        row["output"]["evidence"] = []
    elif mutation == "split":
        row["metadata"]["split"] = "train"
    elif mutation == "support":
        row["output"]["evidence"][0]["support_sets"] = []
    else:
        span = row["output"]["evidence"][0]["support_sets"][0][0]
        key, value = {"quote": ("quote", "wrong"), "offset": ("end", 9999),
                      "source": ("source_id", "unknown"), "text_hash": ("text_sha256", "stale")}[mutation]
        span[key] = value
    argv = ["--snapshot", str(case.snapshot_path), "--phoenix-dataset-name", "reviewed-questions"]
    if command == "all":
        module = script("run_all_evals")
        monkeypatch.setattr(module.phoenix_eval, "_load_project_env", lambda *a: None)
        monkeypatch.setattr(module.phoenix_eval, "_build_settings", lambda *a: case.settings)
        monkeypatch.setattr(module, "_assert_phoenix_reachable", lambda *a: pytest.fail("Execution before validation"))
    else:
        module = script("compare_chunking")
        monkeypatch.setattr(module, "load_project_env", lambda: None)
        monkeypatch.setattr(module, "run_comparison", lambda *a, **kw: pytest.fail("Indexing before validation"))
        argv = [command, *argv]
        if command == "run":
            argv += ["--output", str(tmp_path / "run")]
    with pytest.raises(ValueError):
        module.main(argv)
    assert len(case.state.gets) == 1
    assert not case.state.experiments
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("command", ["phoenix", "ragas", "all", "validate", "run"])
@pytest.mark.parametrize("conflict", ["local", "selectors", "version", "empty"])
def test_cli_rejects_conflicting_input_before_loading(command, conflict, monkeypatch):
    module = {"phoenix": phoenix_experiment, "ragas": ragas_runner}.get(command)
    if module is None:
        module = script("run_all_evals" if command == "all" else "compare_chunking")
    questions_flag = "--questions" if command in {"validate", "run"} else "--questions-path"
    argv = {"local": ["--phoenix-dataset-name", "gold", questions_flag, "evals/questions.jsonl"],
            "selectors": ["--phoenix-dataset-name", "gold", "--phoenix-dataset-id", "id"],
            "version": ["--phoenix-dataset-version-id", "v1"],
            "empty": ["--phoenix-dataset-name", ""]}[conflict]
    if command in {"all", "validate", "run"}:
        argv += ["--snapshot", "missing.json"]
    if command in {"validate", "run"}:
        argv = [command, *argv]
    if command == "run":
        argv += ["--output", "missing-run"]
    with pytest.raises(SystemExit) as exc:
        module.main(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("flag", ["--annotations", "--dataset-name"])
def test_existing_dataset_conflicts_with_sidecar_or_upload_name(flag):
    with pytest.raises(SystemExit) as exc:
        script("run_all_evals").main(["--snapshot", "missing", "--phoenix-dataset-id", "id", flag, "conflict"])
    assert exc.value.code == 2


def test_basic_dataset_without_evidence_runs_on_original_dataset(case, monkeypatch):
    row = case.state.dataset.examples[0]
    row["output"].pop("evidence")
    for key in ("review_status", "question_hash", "snapshot_hash", "split"):
        row["metadata"].pop(key)
    source = load(case, evidence=False)
    monkeypatch.setattr(phoenix_experiment, "build_runtime", lambda **kw: SimpleNamespace(query=lambda q: {}))
    asyncio.run(phoenix_experiment.run_phoenix_experiment_async(
        examples=source.examples, settings=case.settings, dataset_name="unused-upload-name", experiment_name="test-basic",
        ragas_metric_names=[], phoenix_input=source,
    ))
    assert len(case.state.gets) == 1
    experiment = case.state.experiments[0]
    assert experiment["dataset"] is case.state.dataset
    assert experiment["experiment_metadata"]["phoenix_dataset"] == source.binding


@pytest.mark.parametrize("runner", [phoenix_experiment, ragas_runner])
def test_basic_cli_reads_phoenix_input_and_records_binding(case, monkeypatch, runner, tmp_path, capsys):
    for name in ("_load_project_env", "_configure_observability", "_configure_tracing"):
        if hasattr(runner, name):
            monkeypatch.setattr(runner, name, lambda *a, **kw: None)
    monkeypatch.setattr(runner, "_build_settings", lambda *a: case.settings)
    monkeypatch.setattr(runner, "build_runtime", lambda **kw: SimpleNamespace(query=lambda q: {}))
    argv = ["--phoenix-dataset-name", "reviewed-questions"]
    if runner is phoenix_experiment:
        argv += ["--use-phoenix", "--ragas-metrics", "none", "--experiment-name", "test-cli"]
    else:
        def prepare(examples, **kwargs):
            assert examples == [case.question]
            return ragas_runner.PreparedRagasRows([{"id": "q"}], 0)
        monkeypatch.setattr(runner, "build_ragas_rows", prepare)
        monkeypatch.setattr(runner, "evaluate_ragas_rows", lambda *a, **kw: [{"id": "q", "score": 1}])
        argv += ["--output-path", str(tmp_path / "ragas.jsonl")]
    runner.main(argv)
    assert len(case.state.gets) == 1
    assert "version-1" in capsys.readouterr().out
    if runner is ragas_runner:
        assert read_jsonl(tmp_path / "ragas.jsonl")[0]["phoenix_dataset"]["version_id"] == "version-1"


def test_local_validate_remains_offline(case, tmp_path):
    write_jsonl(tmp_path / "questions.jsonl", [case.question])
    write_jsonl(tmp_path / "annotations.jsonl", [case.annotation])
    assert script("compare_chunking").main([
        "validate", "--snapshot", str(case.snapshot_path), "--questions", str(tmp_path / "questions.jsonl"),
        "--annotations", str(tmp_path / "annotations.jsonl"),
    ]) == 0
    assert not case.state.gets


@pytest.mark.parametrize("mutation", ["duplicate", "missing_id", "lane", "tags", "inputs", "empty"])
def test_basic_mapping_validates_question_contract(case, mutation):
    payload = case.state.dataset.to_dict()
    row = payload["examples"][0]
    if mutation == "duplicate":
        payload["examples"].append(deepcopy(row))
    elif mutation == "missing_id":
        row["metadata"].pop("id")
    elif mutation == "lane":
        row["output"]["lane"] = "conflict_version_behavior"
        row["metadata"].pop("lane")
    elif mutation == "tags":
        row["metadata"]["tags"] = "not-a-list"
    elif mutation == "inputs":
        row["input"] = "not-an-object"
    else:
        payload["examples"] = []
    with pytest.raises(ValueError):
        dataset_input.map_phoenix_examples(Dataset.from_dict(payload))


def test_comparison_answers_and_publication_preserve_original_version(case, tmp_path, monkeypatch):
    from imperial_rag.integrations import dashscope
    from langchain_core.runnables import RunnableLambda

    payload = case.state.dataset.to_dict()
    other = deepcopy(payload["examples"][0])
    other["id"] = "example-test"
    other["metadata"].update(id="q-test", split="test", question_hash=digest(case.question | {"id": "q-test"}))
    payload["examples"].append(other)
    case.state.dataset = Dataset.from_dict(payload)
    source = load(case)
    monkeypatch.setattr(dashscope.QwenProviderSettings, "require_api_key", lambda self: "offline")
    def build(snapshot, settings, retrieval):
        chunks = build_chunks([case.doc], retrieval.chunk_size, retrieval.chunk_overlap)
        write_jsonl(settings.extraction_root / "chunks.jsonl", [document_payload(doc) for doc in chunks])
        return SimpleNamespace(retrieve=lambda q: SimpleNamespace(evidence=chunks, diagnostics={})), ()
    monkeypatch.setattr(chunk_comparison, "build_shadow_retriever", build)
    root = tmp_path / "comparison"
    manifest = asyncio.run(chunk_comparison.run_comparison(case.snapshot, source.benchmark, case.settings, root,
                                                           configs=[(400, 50), (256, 0)]))
    assert len(case.state.gets) == 1
    assert manifest["phoenix_dataset"] == source.binding
    assert len({config["qdrant_collection"] for config in manifest["configs"]}) == 2
    assert all(config["qdrant_collection"] != case.settings.qdrant_collection for config in manifest["configs"])
    for config in manifest["configs"]:
        rows = read_jsonl(root / config["config_id"] / "results.jsonl")
        assert [row["id"] for row in rows] == ["q"]
        assert all(row["phoenix_dataset"] == source.binding for row in rows)
    # A later edit advances latest; answers and replay must still use version-1.
    case.state.pinned["version-1"] = case.state.dataset
    latest = case.state.dataset.to_dict()
    latest["version_id"] = "version-2"
    latest["examples"][0]["input"]["question"] = "A later question"
    case.state.dataset = Dataset.from_dict(latest)
    monkeypatch.setattr(dashscope, "create_chat_model", lambda: RunnableLambda(lambda prompt: "Revenue increased [S1]"))
    answers = asyncio.run(chunk_comparison.generate_comparison_answers(root))
    assert all(row["phoenix_dataset"] == source.binding for row in answers)
    assert all(row["question"] == case.question["question"] for row in answers)
    assert len(case.state.gets) == 1
    asyncio.run(phoenix_experiment.publish_evidence_comparison_async(root, case.settings))
    asyncio.run(phoenix_experiment.publish_evidence_comparison_async(root, case.settings))
    assert case.state.gets[1:] == [{"dataset": {"id": "dataset-1"}, "version_id": "version-1"}] * 2
    assert len(case.state.experiments) == 2
    for experiment in case.state.experiments:
        assert experiment["dataset"].version_id == "version-1"
        assert [row["id"] for row in experiment["dataset"].examples] == ["example-1"]
        assert experiment["experiment_metadata"]["phoenix_dataset"] == source.binding
    assert len(case.state.dataset.examples) == 2
    assert not (root.parent / "phoenix-datasets").exists()
    assert json.loads((root / "phoenix.json").read_text())["version_id"] == "version-1"
    manifest["phoenix_dataset"] = manifest["phoenix_dataset"] | {"version_id": "version-2"}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="binding changed"):
        chunk_comparison.load_comparison(root)


@pytest.mark.parametrize("mutation", ["endpoint", "version", "content"])
def test_pinned_read_fails_closed(case, mutation):
    source = load(case)
    endpoint = case.settings.phoenix_client_endpoint
    if mutation == "endpoint":
        endpoint += "/another"
    else:
        payload = case.state.dataset.to_dict()
        if mutation == "version":
            payload["version_id"] = "version-2"
        else:
            payload["examples"][0]["output"]["reference_answer"] = "Changed"
        case.state.dataset = Dataset.from_dict(payload)
    with pytest.raises(ValueError, match="Pinned"):
        asyncio.run(dataset_input.get_pinned_dataset(case.client, source.binding, endpoint))
    assert not case.state.experiments


def test_full_runner_executes_on_existing_evidence_dataset(case, monkeypatch):
    module = script("run_all_evals")
    monkeypatch.setattr(module.phoenix_eval, "_load_project_env", lambda *a: None)
    monkeypatch.setattr(module.phoenix_eval, "_build_settings", lambda *a: case.settings)
    monkeypatch.setattr(module.phoenix_eval, "_configure_tracing", lambda *a, **kw: None)
    monkeypatch.setattr(module, "_configure_observability", lambda *a: None)
    monkeypatch.setattr(module, "_assert_phoenix_reachable", lambda *a: None)
    monkeypatch.setattr(phoenix_experiment, "build_runtime", lambda **kw: object())
    monkeypatch.setattr(phoenix_experiment, "_run_evidence_target", lambda *a: {"eligible": True})
    module.main(["--snapshot", str(case.snapshot_path), "--phoenix-dataset-id", "dataset-1",
                 "--phoenix-dataset-version-id", "version-1", "--ragas-metrics", "none",
                 "--experiment-name", "test-evidence"])
    assert len(case.state.gets) == len(case.state.experiments) == 1
    experiment = case.state.experiments[0]
    assert experiment["dataset"] is case.state.dataset
    assert experiment["experiment_metadata"]["snapshot_hash"] == case.snapshot["snapshot_hash"]
    assert experiment["experiment_metadata"]["phoenix_dataset"]["version_id"] == "version-1"
    assert "evidence_recall_at_10" in experiment["evaluators"]


def test_requested_version_mismatch_is_rejected(case):
    selection = args()
    selection.phoenix_dataset_version_id = "missing-version"
    with pytest.raises(ValueError, match="different dataset version"):
        load(case, selection=selection)
    assert not case.state.experiments


def test_phoenix_validate_is_read_only(case, capsys):
    assert script("compare_chunking").main(["validate", "--snapshot", str(case.snapshot_path),
                                           "--phoenix-dataset-id", "dataset-1"]) == 0
    assert len(case.state.gets) == 1
    assert "phoenix_version_id=version-1" in capsys.readouterr().out
    assert not case.state.experiments
