from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_ragas_eval_script_imports_without_importing_ragas_at_module_load():
    module = _load_ragas_runner()

    assert hasattr(module, "build_ragas_rows")
    assert hasattr(module, "evaluate_ragas_rows")


def test_build_ragas_rows_uses_runtime_outputs_and_skips_refusals():
    module = _load_ragas_runner()
    examples = [
        {
            "question": "Как оформить возврат брака?",
            "expected_behavior": "cite_answer",
            "expected_source_hints": ["брак"],
            "reference_answer": "Возврат брака оформляется по регламенту.",
        },
        {
            "question": "Какова столица Австралии?",
            "expected_behavior": "refuse_if_not_found",
            "expected_source_hints": [],
        },
    ]

    class FakeRuntime:
        def query(self, question: str) -> dict[str, object]:
            if "Австралии" in question:
                return {
                    "answer": "I could not find this clearly in the indexed documents.",
                    "citations": [],
                    "sources": [],
                    "evidence": [],
                }
            return {
                "answer": "Возврат брака оформляется по регламенту. [doc#1]",
                "citations": ["[doc#1] internal_document"],
                "sources": ["[doc#1] documents/reglament.docx"],
                "evidence": [
                    {
                        "page_content": "Возврат брака оформляется по регламенту.",
                        "metadata": {"relative_path": "documents/reglament.docx"},
                    },
                    {"page_content": "   ", "metadata": {}},
                ],
            }

    prepared = module.build_ragas_rows(examples, runtime=FakeRuntime())

    assert prepared.skipped == 1
    assert prepared.rows == [
        {
            "user_input": "Как оформить возврат брака?",
            "response": "Возврат брака оформляется по регламенту. [doc#1]",
            "retrieved_contexts": ["Возврат брака оформляется по регламенту."],
            "expected_behavior": "cite_answer",
            "expected_source_hints": ["брак"],
            "reference": "Возврат брака оформляется по регламенту.",
        }
    ]


def test_validate_metric_requirements_rejects_reference_metrics_without_reference():
    module = _load_ragas_runner()

    with pytest.raises(SystemExit, match="reference_answer"):
        module.validate_metric_requirements(
            ["faithfulness", "context_recall"],
            [{"user_input": "q", "response": "a", "retrieved_contexts": ["ctx"]}],
        )


def test_evaluate_ragas_rows_uses_dataset_metrics_and_evaluator_llm(monkeypatch):
    module = _load_ragas_runner()
    captured: dict[str, object] = {}

    monkeypatch.setattr(module, "build_ragas_dataset", lambda rows: {"dataset": rows})
    monkeypatch.setattr(module, "build_ragas_metrics", lambda names: [f"metric:{name}" for name in names])
    monkeypatch.setattr(module, "build_evaluator_llm", lambda: "wrapped-llm")

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"scores": [{"faithfulness": 1.0}]}

    rows = [{"user_input": "q", "response": "a", "retrieved_contexts": ["ctx"]}]
    result = module.evaluate_ragas_rows(rows, ["faithfulness"], evaluate_fn=fake_evaluate)

    assert result == {"scores": [{"faithfulness": 1.0}]}
    assert captured == {
        "dataset": {"dataset": rows},
        "metrics": ["metric:faithfulness"],
        "llm": "wrapped-llm",
    }


def test_build_ragas_metrics_imports_installed_ragas_metrics():
    module = _load_ragas_runner()

    metrics = module.build_ragas_metrics(["faithfulness"])

    assert [type(metric).__name__ for metric in metrics] == ["Faithfulness"]


def test_result_records_support_scores_and_pandas_like_results():
    module = _load_ragas_runner()

    assert module.result_records(type("ScoresResult", (), {"scores": [{"faithfulness": 1.0}]})()) == [
        {"faithfulness": 1.0}
    ]

    class FakeFrame:
        def to_dict(self, orient: str):
            assert orient == "records"
            return [{"faithfulness": 0.5}]

    class PandasResult:
        def to_pandas(self):
            return FakeFrame()

    assert module.result_records(PandasResult()) == [{"faithfulness": 0.5}]


def _load_ragas_runner():
    spec = importlib.util.spec_from_file_location("run_ragas_eval_for_test", Path("scripts/run_ragas_eval.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
