from __future__ import annotations

from collections import Counter
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_eval_questions_are_russian_jsonl_with_expected_behavior():
    lines = Path("evals/questions.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    ids = [row["id"] for row in rows]
    behavior_counts = Counter(row["expected_behavior"] for row in rows)

    assert len(rows) >= 37
    assert len(ids) == len(set(ids))
    assert behavior_counts["surface_conflict"] >= 5
    assert behavior_counts["refuse_if_not_found"] >= 5
    assert 10 <= sum(bool(row.get("reference_context_ids")) for row in rows) <= 15
    for payload in rows:
        assert payload["id"].startswith("imperial-")
        assert payload["suite"]
        assert isinstance(payload.get("tags", []), list)
        assert all(isinstance(tag, str) and tag for tag in payload.get("tags", []))
        assert payload["question"]
        assert _contains_cyrillic(payload["question"])
        assert payload["expected_behavior"] in {"cite_answer", "refuse_if_not_found", "surface_conflict"}
        assert isinstance(payload.get("expected_source_hints", []), list)
        assert payload["reference_answer"]
        assert _contains_cyrillic(payload["reference_answer"])
        if "reference_context_ids" in payload:
            assert all(isinstance(context_id, str) and context_id.strip() for context_id in payload["reference_context_ids"])


def test_russian_judge_calibration_fixture_has_locked_human_labels():
    rows = [
        json.loads(line)
        for line in Path("evals/russian_judge_calibration.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = Counter(row["human_label"] for row in rows)
    ids = [row["id"] for row in rows]

    assert len(rows) == 20
    assert labels == {"correct": 10, "incorrect": 10}
    assert len(ids) == len(set(ids))
    for row in rows:
        assert row["id"].startswith("judge-calibration-")
        assert row["suite"] == "russian_judge_calibration"
        assert row["locked"] is True
        assert row["expected_behavior"] == "cite_answer"
        assert row["question"]
        assert row["reference_answer"]
        assert row["candidate_answer"]
        assert _contains_cyrillic(row["question"])
        assert _contains_cyrillic(row["reference_answer"])
        assert row["human_label"] in {"correct", "incorrect"}


def test_load_questions_validates_required_metadata_and_uniqueness(tmp_path):
    module = _load_eval_runner()
    path = tmp_path / "questions.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "duplicate",
                        "suite": "core",
                        "tags": ["retrieval"],
                        "question": "Что делать с браком?",
                        "expected_behavior": "cite_answer",
                        "expected_source_hints": ["брак"],
                        "reference_answer": "Нужно оформить возврат брака.",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "duplicate",
                        "suite": "core",
                        "tags": "retrieval",
                        "question": "Какова столица Австралии?",
                        "expected_behavior": "unsupported",
                        "expected_source_hints": [],
                        "reference_answer": "Ответа нет в корпусе.",
                        "reference_context_ids": ["file-a", ""],
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        module.load_questions(path)

    message = str(exc_info.value)
    assert "duplicate id" in message
    assert "expected_behavior" in message
    assert "tags must be a list" in message
    assert "reference_context_ids" in message


def test_eval_runner_deterministic_citation_behavior():
    module = _load_eval_runner()

    assert module.citation_behavior(
        {"question": "x"},
        {"answer": "Ответ. [/docs/a.docx#chunk]", "citations": ["[/docs/a.docx#chunk] body"]},
        {"expected_behavior": "cite_answer"},
    )["score"] is True
    assert module.citation_behavior(
        {"question": "x"},
        {"answer": "I could not find this clearly in the indexed documents.", "citations": []},
        {"expected_behavior": "refuse_if_not_found"},
    )["score"] is True
    assert module.citation_behavior(
        {"question": "x"},
        {"answer": "Документы противоречат друг другу. [a] [b]", "citations": ["[a] body", "[b] body"]},
        {"expected_behavior": "surface_conflict"},
    )["score"] is True


def test_phoenix_evaluator_wrappers_accept_phoenix_bound_keywords(monkeypatch):
    module = _load_eval_runner()

    assert module.phoenix_citation_behavior(
        output={"answer": "Ответ. [/docs/a.docx#chunk]", "citations": ["[/docs/a.docx#chunk] body"]},
        expected={"expected_behavior": "cite_answer"},
    ) is True
    assert module.phoenix_source_hint_behavior(
        output={"sources": ["source contains брак"]},
        expected={"expected_source_hints": ["брак"]},
    ) is True

    from imperial_rag import ragas_eval

    captured: dict[str, object] = {}

    def fake_score_faithfulness_for_phoenix(**kwargs):
        captured.update(kwargs)
        return {"score": 0.75, "label": "faithfulness", "metadata": {"metric": "ragas_faithfulness"}}

    monkeypatch.setattr(module, "_get_ragas_faithfulness_scorer", lambda: "fake-scorer")
    monkeypatch.setattr(ragas_eval, "score_faithfulness_for_phoenix", fake_score_faithfulness_for_phoenix)

    assert module.phoenix_ragas_faithfulness(
        input={"question": "Что делать с браком?"},
        output={"answer": "Ответ", "documents": [{"page_content": "Контекст"}]},
    ) == {"score": 0.75, "label": "faithfulness", "metadata": {"metric": "ragas_faithfulness"}}
    assert captured == {
        "input": {"question": "Что делать с браком?"},
        "output": {"answer": "Ответ", "documents": [{"page_content": "Контекст"}]},
        "scorer": "fake-scorer",
    }

    def fake_score_answer_relevancy_for_phoenix(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return {"score": 0.8, "label": "answer_relevancy", "metadata": {"metric": "ragas_answer_relevancy"}}

    monkeypatch.setattr(module, "_get_ragas_answer_relevancy_scorer", lambda: "fake-answer-scorer")
    monkeypatch.setattr(ragas_eval, "score_answer_relevancy_for_phoenix", fake_score_answer_relevancy_for_phoenix)

    assert module.phoenix_ragas_answer_relevancy(
        input={"question": "Что делать с браком?"},
        output={"answer": "Ответ", "documents": []},
    ) == {"score": 0.8, "label": "answer_relevancy", "metadata": {"metric": "ragas_answer_relevancy"}}
    assert captured == {
        "input": {"question": "Что делать с браком?"},
        "output": {"answer": "Ответ", "documents": []},
        "scorer": "fake-answer-scorer",
    }

    def fake_score_id_context_recall_for_phoenix(**kwargs):
        captured.clear()
        captured.update(kwargs)
        return {"score": 1.0, "label": "id_context_recall", "metadata": {"metric": "ragas_id_context_recall"}}

    monkeypatch.setattr(ragas_eval, "score_id_context_recall_for_phoenix", fake_score_id_context_recall_for_phoenix)

    assert module.phoenix_id_context_recall(
        input={"question": "Что делать с браком?"},
        output={"documents": [{"metadata": {"file_id": "file-a"}}]},
        expected={"reference_context_ids": ["file-a"]},
    ) == {"score": 1.0, "label": "id_context_recall", "metadata": {"metric": "ragas_id_context_recall"}}
    assert captured == {
        "input": {"question": "Что делать с браком?"},
        "output": {"documents": [{"metadata": {"file_id": "file-a"}}]},
        "expected": {"reference_context_ids": ["file-a"]},
    }


def test_phoenix_ragas_wrappers_skip_answer_quality_for_non_citation_rows(monkeypatch):
    module = _load_eval_runner()

    monkeypatch.setattr(module, "_get_ragas_faithfulness_scorer", lambda: pytest.fail("faithfulness scorer built"))
    monkeypatch.setattr(module, "_get_ragas_answer_relevancy_scorer", lambda: pytest.fail("answer relevancy scorer built"))

    refusal = module.phoenix_ragas_faithfulness(
        input={"question": "Какова столица Австралии?"},
        output={"answer": "I could not find this clearly in the indexed documents.", "documents": []},
        expected={"expected_behavior": "refuse_if_not_found"},
    )
    conflict = module.phoenix_ragas_answer_relevancy(
        input={"question": "Какая версия регламента действует?"},
        output={"answer": "Документы противоречат друг другу.", "documents": []},
        expected={"expected_behavior": "surface_conflict"},
    )

    assert refusal["score"] is None
    assert refusal["label"] == "skipped"
    assert refusal["metadata"] == {
        "metric": "ragas_faithfulness",
        "reason": "metric_not_applicable_for_behavior",
        "expected_behavior": "refuse_if_not_found",
    }
    assert conflict["score"] is None
    assert conflict["label"] == "skipped"
    assert conflict["metadata"] == {
        "metric": "ragas_answer_relevancy",
        "reason": "metric_not_applicable_for_behavior",
        "expected_behavior": "surface_conflict",
    }


def test_phoenix_id_context_recall_reports_not_applicable_without_gold_ids():
    module = _load_eval_runner()

    result = module.phoenix_id_context_recall(
        input={"question": "Что делать с браком?"},
        output={"documents": [{"metadata": {"file_id": "file-a"}}]},
        expected={"expected_behavior": "cite_answer"},
    )

    assert result["score"] is None
    assert result["label"] == "not_applicable"
    assert result["metadata"]["reason"] == "missing_reference_context_ids"


def test_source_hint_behavior_scans_citations_when_sources_are_non_empty():
    module = _load_eval_runner()

    assert module.source_hint_behavior(
        {"question": "x"},
        {"sources": ["unrelated source"], "citations": ["citation contains брак"]},
        {"expected_source_hints": ["брак"]},
    )["score"] is True


def test_eval_runner_includes_retrieval_diagnostics_in_outputs():
    module = _load_eval_runner()
    diagnostics = {"final_evidence": 0, "reranker": "fallback:deterministic"}

    class FakeRuntime:
        def query(self, question: str) -> dict[str, object]:
            return {
                "answer": "I could not find this clearly in the indexed documents.",
                "citations": [],
                "sources": [],
                "evidence": [],
                "retrieval": diagnostics,
            }

    output = module.run_target({"question": "Что делать?"}, runtime=FakeRuntime())

    assert output["retrieval"] == diagnostics


def test_retrieval_relevance_metrics_use_source_hints_and_rank_order():
    module = _load_eval_runner()

    metrics = module.retrieval_relevance_metrics(
        {"question": "Как оформить возврат брака?"},
        {
            "documents": [
                {"page_content": "Нерелевантный текст.", "metadata": {"relative_path": "documents/other.docx"}},
                {"page_content": "Регламент описывает возврат брака из магазина.", "metadata": {}},
            ]
        },
        {"expected_source_hints": ["возврат брака"]},
        k=2,
    )

    assert metrics["score"] == 0.5
    assert metrics["label"] == "hit"
    assert metrics["metadata"]["document_scores"] == [0.0, 1.0]
    assert metrics["metadata"]["hit_at_2"] is True
    assert metrics["metadata"]["precision_at_2"] == 0.5
    assert metrics["metadata"]["ndcg_at_2"] == pytest.approx(0.6309297536)


def test_run_local_eval_includes_retrieval_quality_metrics():
    module = _load_eval_runner()

    class FakeRuntime:
        def query(self, question: str) -> dict[str, object]:
            return {
                "answer": f"Ответ на {question}",
                "citations": ["[/docs/reglament.docx#chunk] body"],
                "sources": ["[/docs/reglament.docx#chunk] documents/reglament.docx"],
                "evidence": [{"page_content": "Регламент описывает возврат брака.", "metadata": {}}],
            }

    rows = module.run_local_eval(
        [
            {
                "question": "Как оформить возврат брака?",
                "expected_behavior": "cite_answer",
                "expected_source_hints": ["возврат брака"],
            }
        ],
        runtime=FakeRuntime(),
    )

    assert rows == [
        {
            "question": "Как оформить возврат брака?",
            "citation_behavior": True,
            "source_hint_behavior": True,
            "retrieval_hit_at_5": True,
            "retrieval_precision_at_5": 0.2,
            "retrieval_ndcg_at_5": 1.0,
        }
    ]


def test_build_eval_artifact_row_includes_verdicts_ragas_scores_and_failure_class():
    module = _load_eval_runner()
    example = {
        "id": "imperial-cite-001",
        "suite": "core",
        "tags": ["returns"],
        "question": "Как оформить возврат брака?",
        "expected_behavior": "cite_answer",
        "expected_source_hints": ["возврат брака"],
        "reference_answer": "Возврат брака оформляется по регламенту.",
        "reference_context_ids": ["file-a"],
    }
    output = {
        "answer": "Возврат брака оформляется по регламенту.",
        "citations": [],
        "sources": [],
        "documents": [
            {"page_content": "Возврат брака оформляется по регламенту.", "metadata": {"file_id": "file-a"}}
        ],
    }

    artifact = module.build_eval_artifact_row(
        example=example,
        output=output,
        ragas_results={
            "faithfulness": {"score": 0.9, "explanation": "grounded"},
            "id_context_recall": {"score": 1.0, "explanation": None},
        },
        phoenix_experiment="experiment-1",
    )

    assert artifact == {
        "id": "imperial-cite-001",
        "suite": "core",
        "tags": ["returns"],
        "question": "Как оформить возврат брака?",
        "expected_behavior": "cite_answer",
        "answer": "Возврат брака оформляется по регламенту.",
        "citations": [],
        "retrieved_context_ids": ["file-a"],
        "reference_context_ids": ["file-a"],
        "deterministic": {
            "citation_behavior": False,
            "source_hint_behavior": True,
            "retrieval_hit_at_5": True,
            "retrieval_precision_at_5": 0.2,
            "retrieval_ndcg_at_5": 1.0,
        },
        "ragas_scores": {"faithfulness": 0.9, "id_context_recall": 1.0},
        "ragas_explanations": {"faithfulness": "grounded", "id_context_recall": None},
        "phoenix_experiment": "experiment-1",
        "failure_class": "missing_citation",
    }


def test_eval_failure_class_prioritizes_behavior_failures():
    module = _load_eval_runner()

    assert module.classify_eval_failure(
        expected_behavior="refuse_if_not_found",
        deterministic={"citation_behavior": False, "retrieval_hit_at_5": False},
        ragas_scores={},
    ) == "bad_refusal"
    assert module.classify_eval_failure(
        expected_behavior="surface_conflict",
        deterministic={"citation_behavior": False, "retrieval_hit_at_5": True},
        ragas_scores={},
    ) == "bad_conflict_handling"
    assert module.classify_eval_failure(
        expected_behavior="cite_answer",
        deterministic={"citation_behavior": True, "retrieval_hit_at_5": False},
        ragas_scores={},
    ) == "retrieval_miss"


def test_eval_runner_uses_phoenix_dataset_and_experiment_api_shape():
    source = Path("scripts/run_phoenix_eval.py").read_text(encoding="utf-8")
    legacy_name = "".join(("lang", "smith"))
    legacy_runner = Path("scripts") / f"run_{legacy_name}_eval.py"

    assert not legacy_runner.exists()
    assert "from phoenix.client import Client" in source
    assert "client.datasets.create_dataset" in source
    assert "inputs=inputs" in source
    assert "outputs=outputs" in source
    assert "metadata=metadata" in source
    assert "client.experiments.run_experiment" in source
    assert legacy_name not in source.casefold()


def test_phoenix_dataset_rows_have_stable_metadata_ids():
    module = _load_eval_runner()
    example = {
        "id": "imperial-cite-001",
        "suite": "core",
        "tags": ["returns"],
        "question": "Что делать с браком?",
        "expected_behavior": "cite_answer",
        "expected_source_hints": ["брак"],
        "reference_answer": "Ответ про брак.",
    }

    inputs, outputs, metadata = module._to_phoenix_dataset_rows([example])
    _, _, repeated_metadata = module._to_phoenix_dataset_rows([example])

    assert metadata[0]["id"] == "imperial-cite-001"
    assert metadata[0]["id"] == repeated_metadata[0]["id"]
    assert metadata[0]["suite"] == "core"
    assert metadata[0]["tags"] == ["returns"]
    assert inputs == [{"question": "Что делать с браком?"}]
    assert outputs == [
        {
            "expected_behavior": "cite_answer",
            "expected_source_hints": ["брак"],
            "reference_answer": "Ответ про брак.",
        }
    ]
    assert metadata[0]["row_index"] == 0
    assert metadata[0]["source"] == "evals/questions.jsonl"


def test_phoenix_dataset_rows_preserve_reference_context_ids():
    module = _load_eval_runner()
    example = {
        "question": "Что делать с браком?",
        "expected_behavior": "cite_answer",
        "expected_source_hints": ["брак"],
        "reference_context_ids": ["file-a", "file-b"],
    }

    _, outputs, metadata = module._to_phoenix_dataset_rows([example])
    _, _, repeated_metadata = module._to_phoenix_dataset_rows([example])

    assert outputs == [
        {
            "expected_behavior": "cite_answer",
            "expected_source_hints": ["брак"],
            "reference_context_ids": ["file-a", "file-b"],
        }
    ]
    assert metadata[0]["id"] == repeated_metadata[0]["id"]


def test_parse_phoenix_ragas_metrics_supports_none_and_rejects_unknown():
    module = _load_eval_runner()

    assert module.parse_phoenix_ragas_metrics("none") == []
    assert module.parse_phoenix_ragas_metrics("faithfulness") == ["faithfulness"]
    assert module.parse_phoenix_ragas_metrics("answer_relevance") == ["answer_relevancy"]
    assert module.parse_phoenix_ragas_metrics("id-based-context-recall") == ["id_context_recall"]
    assert module.parse_phoenix_ragas_metrics("") == ["faithfulness", "answer_relevancy"]
    assert module.parse_phoenix_ragas_metrics(" faithfulness , NONE ") == []

    with pytest.raises(SystemExit, match="Unsupported Ragas metrics"):
        module.parse_phoenix_ragas_metrics("answer_correctness")


def test_phoenix_experiment_uses_documented_python_dataset_arguments(monkeypatch):
    module = _load_eval_runner()
    captured: dict[str, object] = {}

    class FakeDatasets:
        def create_dataset(self, **kwargs):
            captured["dataset"] = kwargs
            return {"dataset_id": "dataset-1"}

    class FakeExperiments:
        def run_experiment(self, **kwargs):
            captured["experiment"] = kwargs
            return SimpleNamespace(id="experiment-1")

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.datasets = FakeDatasets()
            self.experiments = FakeExperiments()

    class FakeRuntime:
        def query(self, question: str) -> dict[str, object]:
            return {"answer": f"Ответ на {question}", "citations": ["[/docs/a.docx#chunk] body"]}

    fake_phoenix = types.ModuleType("phoenix")
    fake_client_module = types.ModuleType("phoenix.client")
    fake_client_module.Client = FakeClient
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.client", fake_client_module)
    monkeypatch.setattr(module, "build_runtime", lambda settings=None: FakeRuntime())
    monkeypatch.setattr(module, "_get_ragas_faithfulness_scorer", lambda: object())
    monkeypatch.setattr(module, "_get_ragas_answer_relevancy_scorer", lambda: object())

    module._run_phoenix_experiment(
        examples=[
            {
                "question": "Что делать с браком?",
                "expected_behavior": "cite_answer",
                "expected_source_hints": ["брак"],
            }
        ],
        settings=SimpleNamespace(phoenix_client_endpoint="http://localhost:6006"),
        dataset_name="imperial-rag-gold-questions",
        experiment_name="imperial-rag-citation-grounding",
    )

    assert captured["client"] == {"base_url": "http://localhost:6006"}
    dataset_args = captured["dataset"]
    assert dataset_args["name"] == "imperial-rag-gold-questions"
    assert dataset_args["dataset_description"] == "Imperial RAG gold questions loaded from evals/questions.jsonl."
    assert dataset_args["inputs"] == [{"question": "Что делать с браком?"}]
    assert dataset_args["outputs"] == [{"expected_behavior": "cite_answer", "expected_source_hints": ["брак"]}]
    assert dataset_args["metadata"][0]["id"]
    assert dataset_args["metadata"][0]["row_index"] == 0
    assert "examples" not in dataset_args
    experiment_args = captured["experiment"]
    assert experiment_args["dataset"] == {"dataset_id": "dataset-1"}
    assert callable(experiment_args["task"])
    assert experiment_args["evaluators"] == [
        module.phoenix_citation_behavior,
        module.phoenix_source_hint_behavior,
        module.phoenix_retrieval_relevance,
        module.phoenix_ragas_faithfulness,
        module.phoenix_ragas_answer_relevancy,
    ]
    assert experiment_args["experiment_name"] == "imperial-rag-citation-grounding"


def test_phoenix_experiment_can_run_id_context_recall_without_reference_answer(monkeypatch):
    module = _load_eval_runner()
    captured: dict[str, object] = {}

    class FakeDatasets:
        def create_dataset(self, **kwargs):
            captured["dataset"] = kwargs
            return {"dataset_id": "dataset-1"}

    class FakeExperiments:
        def run_experiment(self, **kwargs):
            captured["experiment"] = kwargs
            return SimpleNamespace(id="experiment-1")

    class FakeClient:
        def __init__(self, **kwargs):
            self.datasets = FakeDatasets()
            self.experiments = FakeExperiments()

    class FakeRuntime:
        def query(self, question: str) -> dict[str, object]:
            return {"answer": f"Ответ на {question}", "evidence": [{"metadata": {"file_id": "file-a"}}]}

    fake_phoenix = types.ModuleType("phoenix")
    fake_client_module = types.ModuleType("phoenix.client")
    fake_client_module.Client = FakeClient
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.client", fake_client_module)
    monkeypatch.setattr(module, "build_runtime", lambda settings=None: FakeRuntime())

    module._run_phoenix_experiment(
        examples=[
            {
                "question": "Что делать с браком?",
                "expected_behavior": "cite_answer",
                "expected_source_hints": ["брак"],
                "reference_context_ids": ["file-a"],
            }
        ],
        settings=SimpleNamespace(phoenix_client_endpoint="http://localhost:6006"),
        dataset_name="imperial-rag-gold-questions",
        experiment_name="imperial-rag-citation-grounding",
        ragas_metric_names=["id_context_recall"],
    )

    assert captured["experiment"]["evaluators"] == [
        module.phoenix_citation_behavior,
        module.phoenix_source_hint_behavior,
        module.phoenix_retrieval_relevance,
        module.phoenix_id_context_recall,
    ]


def test_phoenix_annotation_hook_logs_span_and_document_metrics():
    module = _load_eval_runner()
    captured: dict[str, object] = {}

    class FakeSpans:
        def log_span_annotations(self, **kwargs):
            captured["span_annotations"] = kwargs

        def log_document_annotations(self, **kwargs):
            captured["document_annotations"] = kwargs

    module.log_phoenix_eval_annotations(
        SimpleNamespace(spans=FakeSpans()),
        span_id="query-span",
        retrieval_span_id="retrieval-span",
        answer_metrics=[
            {"name": "citation_behavior", "score": 1.0, "label": "pass", "explanation": "citation present"}
        ],
        retrieval_metrics={
            "metadata": {
                "document_scores": [1.0, 0.0],
                "precision_at_2": 0.5,
                "ndcg_at_2": 1.0,
            }
        },
        sync=True,
    )

    span_annotations = captured["span_annotations"]["span_annotations"]
    document_annotations = captured["document_annotations"]["document_annotations"]

    assert captured["span_annotations"]["sync"] is True
    assert span_annotations == [
        {
            "name": "citation_behavior",
            "span_id": "query-span",
            "annotator_kind": "CODE",
            "result": {"score": 1.0, "label": "pass", "explanation": "citation present"},
        },
        {
            "name": "precision@2",
            "span_id": "retrieval-span",
            "annotator_kind": "CODE",
            "result": {"score": 0.5},
        },
        {
            "name": "ndcg@2",
            "span_id": "retrieval-span",
            "annotator_kind": "CODE",
            "result": {"score": 1.0},
        },
    ]
    assert document_annotations == [
        {
            "name": "relevance",
            "span_id": "retrieval-span",
            "document_position": 0,
            "annotator_kind": "CODE",
            "result": {"score": 1.0, "label": "relevant"},
        },
        {
            "name": "relevance",
            "span_id": "retrieval-span",
            "document_position": 1,
            "annotator_kind": "CODE",
            "result": {"score": 0.0, "label": "not_relevant"},
        },
    ]


def test_phoenix_experiment_can_disable_ragas_evaluators(monkeypatch):
    module = _load_eval_runner()
    captured: dict[str, object] = {}

    class FakeDatasets:
        def create_dataset(self, **kwargs):
            captured["dataset"] = kwargs
            return {"dataset_id": "dataset-1"}

    class FakeExperiments:
        def run_experiment(self, **kwargs):
            captured["experiment"] = kwargs
            return SimpleNamespace(id="experiment-1")

    class FakeClient:
        def __init__(self, **kwargs):
            self.datasets = FakeDatasets()
            self.experiments = FakeExperiments()

    class FakeRuntime:
        def query(self, question: str) -> dict[str, object]:
            return {"answer": f"Ответ на {question}", "citations": ["[/docs/a.docx#chunk] body"]}

    fake_phoenix = types.ModuleType("phoenix")
    fake_client_module = types.ModuleType("phoenix.client")
    fake_client_module.Client = FakeClient
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.client", fake_client_module)
    monkeypatch.setattr(module, "build_runtime", lambda settings=None: FakeRuntime())
    monkeypatch.setattr(module, "_get_ragas_faithfulness_scorer", lambda: pytest.fail("Ragas scorer was built"))
    monkeypatch.setattr(module, "_get_ragas_answer_relevancy_scorer", lambda: pytest.fail("Ragas scorer was built"))

    module._run_phoenix_experiment(
        examples=[
            {
                "question": "Что делать с браком?",
                "expected_behavior": "cite_answer",
                "expected_source_hints": ["брак"],
            }
        ],
        settings=SimpleNamespace(phoenix_client_endpoint="http://localhost:6006"),
        dataset_name="imperial-rag-gold-questions",
        experiment_name="imperial-rag-citation-grounding",
        ragas_metric_names=[],
    )

    assert captured["experiment"]["evaluators"] == [
        module.phoenix_citation_behavior,
        module.phoenix_source_hint_behavior,
        module.phoenix_retrieval_relevance,
    ]


def test_phoenix_experiment_rejects_reference_ragas_metrics_without_references():
    module = _load_eval_runner()

    with pytest.raises(SystemExit, match="reference_answer"):
        module._run_phoenix_experiment(
            examples=[
                {
                    "question": "Что делать с браком?",
                    "expected_behavior": "cite_answer",
                    "expected_source_hints": ["брак"],
                }
            ],
            settings=SimpleNamespace(phoenix_client_endpoint="http://localhost:6006"),
            dataset_name="imperial-rag-gold-questions",
            experiment_name="imperial-rag-citation-grounding",
            ragas_metric_names=["context_recall"],
        )


def test_experiment_identifier_reads_mapping_result():
    module = _load_eval_runner()

    assert module._experiment_identifier({"experiment_id": "experiment-1"}) == "experiment-1"


def _contains_cyrillic(value: str) -> bool:
    return any("а" <= character.casefold() <= "я" for character in value)


def _load_eval_runner():
    spec = importlib.util.spec_from_file_location("run_phoenix_eval_for_test", Path("scripts/run_phoenix_eval.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
