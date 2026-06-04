from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_eval_questions_are_russian_jsonl_with_expected_behavior():
    lines = Path("evals/questions.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(lines) == 30
    for line in lines:
        payload = json.loads(line)
        assert payload["question"]
        assert _contains_cyrillic(payload["question"])
        assert payload["expected_behavior"] in {"cite_answer", "refuse_if_not_found", "surface_conflict"}
        assert isinstance(payload.get("expected_source_hints", []), list)


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
        "question": "Что делать с браком?",
        "expected_behavior": "cite_answer",
        "expected_source_hints": ["брак"],
    }

    inputs, outputs, metadata = module._to_phoenix_dataset_rows([example])
    _, _, repeated_metadata = module._to_phoenix_dataset_rows([example])

    assert metadata[0]["id"] == repeated_metadata[0]["id"]
    assert inputs == [{"question": "Что делать с браком?"}]
    assert outputs == [
        {
            "expected_behavior": "cite_answer",
            "expected_source_hints": ["брак"],
        }
    ]
    assert metadata[0]["row_index"] == 0
    assert metadata[0]["source"] == "evals/questions.jsonl"


def test_parse_phoenix_ragas_metrics_supports_none_and_rejects_unknown():
    module = _load_eval_runner()

    assert module.parse_phoenix_ragas_metrics("none") == []
    assert module.parse_phoenix_ragas_metrics("faithfulness") == ["faithfulness"]
    assert module.parse_phoenix_ragas_metrics("") == ["faithfulness"]
    assert module.parse_phoenix_ragas_metrics(" faithfulness , NONE ") == []

    with pytest.raises(SystemExit, match="Unsupported Ragas metrics"):
        module.parse_phoenix_ragas_metrics("answer_relevancy")


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
        module.phoenix_ragas_faithfulness,
    ]
    assert experiment_args["experiment_name"] == "imperial-rag-citation-grounding"


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
