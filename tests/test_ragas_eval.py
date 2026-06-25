from __future__ import annotations

import asyncio
import importlib.util
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest


def test_faithfulness_row_from_run_output_extracts_docs_and_skips_empty_cases():
    from imperial_rag import ragas_eval

    row = ragas_eval.faithfulness_row_from_run_output(
        {"question": " Как оформить возврат брака? "},
        {
            "answer": " Возврат оформляется по регламенту. ",
            "documents": [
                {"page_content": " Регламент описывает возврат брака. "},
                {"page_content": "   "},
            ],
        },
    )

    assert row == {
        "user_input": "Как оформить возврат брака?",
        "response": "Возврат оформляется по регламенту.",
        "retrieved_contexts": ["Регламент описывает возврат брака."],
    }
    assert (
        ragas_eval.faithfulness_row_from_run_output(
            {"question": "Как оформить возврат брака?"},
            {"answer": "", "documents": [{"page_content": "Контекст"}]},
        )
        is None
    )
    assert (
        ragas_eval.faithfulness_row_from_run_output(
            {"question": "Как оформить возврат брака?"},
            {"answer": "Ответ", "documents": []},
        )
        is None
    )


def test_build_faithfulness_scorer_uses_openai_compatible_dashscope_client(monkeypatch):
    from imperial_rag import ragas_eval

    captured: dict[str, Any] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured["client"] = self

    class FakeFaithfulness:
        def __init__(self, **kwargs):
            captured["metric"] = kwargs

    def fake_llm_factory(model, **kwargs):
        captured["llm_factory_model"] = model
        captured["llm_factory_kwargs"] = kwargs
        return "ragas-llm"

    monkeypatch.setattr(ragas_eval, "_import_async_openai", lambda: FakeAsyncOpenAI)
    monkeypatch.setattr(ragas_eval, "_import_llm_factory", lambda: fake_llm_factory)
    monkeypatch.setattr(ragas_eval, "_import_faithfulness_metric", lambda: FakeFaithfulness)

    scorer = ragas_eval.build_faithfulness_scorer(
        cast(Any, SimpleNamespace(
            compat_base_url="https://dashscope.example/compatible-mode/v1",
            chat_model="qwen-test",
            require_api_key=lambda: "dashscope-key",
        ))
    )

    assert isinstance(scorer, FakeFaithfulness)
    assert captured["client"].kwargs == {
        "api_key": "dashscope-key",
        "base_url": "https://dashscope.example/compatible-mode/v1",
    }
    assert captured["llm_factory_model"] == "qwen-test"
    assert captured["llm_factory_kwargs"] == {"client": captured["client"], "provider": "openai"}
    assert captured["metric"] == {"llm": "ragas-llm"}


def test_build_answer_relevancy_scorer_uses_qwen_llm_and_embeddings(monkeypatch):
    from imperial_rag import ragas_eval

    captured: dict[str, Any] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured["client"] = self

    class FakeAnswerRelevancy:
        def __init__(self, **kwargs):
            captured["metric"] = kwargs

    def fake_llm_factory(model, **kwargs):
        captured["llm_factory_model"] = model
        captured["llm_factory_kwargs"] = kwargs
        return "ragas-llm"

    def fake_embedding_factory(provider, **kwargs):
        captured["embedding_factory_provider"] = provider
        captured["embedding_factory_kwargs"] = kwargs
        return "ragas-embeddings"

    monkeypatch.setattr(ragas_eval, "_import_async_openai", lambda: FakeAsyncOpenAI)
    monkeypatch.setattr(ragas_eval, "_import_llm_factory", lambda: fake_llm_factory)
    monkeypatch.setattr(ragas_eval, "_import_embedding_factory", lambda: fake_embedding_factory)
    monkeypatch.setattr(ragas_eval, "_import_answer_relevancy_metric", lambda: FakeAnswerRelevancy)

    scorer = ragas_eval.build_answer_relevancy_scorer(
        cast(Any, SimpleNamespace(
            compat_base_url="https://dashscope.example/compatible-mode/v1",
            chat_model="qwen-test",
            embedding_model="text-embedding-v4",
            require_api_key=lambda: "dashscope-key",
        ))
    )

    assert isinstance(scorer, FakeAnswerRelevancy)
    assert captured["client"].kwargs == {
        "api_key": "dashscope-key",
        "base_url": "https://dashscope.example/compatible-mode/v1",
    }
    assert captured["llm_factory_model"] == "qwen-test"
    assert captured["llm_factory_kwargs"] == {"client": captured["client"], "provider": "openai"}
    assert captured["embedding_factory_provider"] == "openai"
    assert captured["embedding_factory_kwargs"] == {
        "model": "text-embedding-v4",
        "client": captured["client"],
    }
    assert captured["metric"] == {"llm": "ragas-llm", "embeddings": "ragas-embeddings"}


def test_import_faithfulness_metric_prefers_ragas_collections_api():
    from imperial_rag import ragas_eval

    metric_cls = ragas_eval._import_faithfulness_metric()

    assert metric_cls.__name__ == "Faithfulness"
    assert ".collections." in metric_cls.__module__


def test_score_faithfulness_for_phoenix_returns_score_dictionary():
    from imperial_rag import ragas_eval

    captured: dict[str, Any] = {}

    class FakeResult:
        value = 0.8
        reason = "all generated claims are supported"

    class FakeScorer:
        def score(self, **kwargs):
            captured.update(kwargs)
            return FakeResult()

    result = ragas_eval.score_faithfulness_for_phoenix(
        input={"question": "Что делать с браком?"},
        output={"answer": "Оформить по регламенту.", "documents": [{"page_content": "Регламент описывает возврат."}]},
        scorer=FakeScorer(),
    )

    assert captured == {
        "user_input": "Что делать с браком?",
        "response": "Оформить по регламенту.",
        "retrieved_contexts": ["Регламент описывает возврат."],
    }
    assert result == {
        "score": 0.8,
        "label": "faithfulness",
        "explanation": "all generated claims are supported",
        "metadata": {"metric": "ragas_faithfulness", "retrieved_context_count": 1},
    }


def test_score_faithfulness_for_phoenix_skips_empty_response_or_contexts():
    from imperial_rag import ragas_eval

    result = ragas_eval.score_faithfulness_for_phoenix(
        input={"question": "Что делать с браком?"},
        output={"answer": "", "documents": [{"page_content": "Контекст"}]},
        scorer=object(),
    )

    assert result["score"] is None
    assert result["label"] == "skipped"
    assert result["metadata"]["reason"] == "missing_response_or_contexts"


def test_score_answer_relevancy_for_phoenix_uses_question_and_response_without_contexts():
    from imperial_rag import ragas_eval

    captured: dict[str, Any] = {}

    class FakeResult:
        value = 0.7
        reason = "response addresses the question"

    class FakeScorer:
        def score(self, **kwargs):
            captured.update(kwargs)
            return FakeResult()

    result = ragas_eval.score_answer_relevancy_for_phoenix(
        input={"question": "Что делать с браком?"},
        output={"answer": "Оформить возврат по регламенту.", "documents": []},
        scorer=FakeScorer(),
    )

    assert captured == {
        "user_input": "Что делать с браком?",
        "response": "Оформить возврат по регламенту.",
    }
    assert result == {
        "score": 0.7,
        "label": "answer_relevancy",
        "explanation": "response addresses the question",
        "metadata": {"metric": "ragas_answer_relevancy"},
    }


def test_score_answer_relevancy_for_phoenix_skips_empty_question_or_response():
    from imperial_rag import ragas_eval

    result = ragas_eval.score_answer_relevancy_for_phoenix(
        input={"question": "Что делать с браком?"},
        output={"answer": ""},
        scorer=object(),
    )

    assert result["score"] is None
    assert result["label"] == "skipped"
    assert result["metadata"]["reason"] == "missing_user_input_or_response"


def test_retrieved_context_ids_from_output_extracts_unique_file_ids():
    from imperial_rag import ragas_eval

    output = {
        "documents": [
            {"page_content": "one", "metadata": {"file_id": "file-a", "chunk_id": "chunk-1"}},
            {"page_content": "two", "metadata": {"file_id": "file-b"}},
            {"page_content": "duplicate", "metadata": {"file_id": "file-a"}},
            {"page_content": "missing", "metadata": {"chunk_id": "chunk-4"}},
        ]
    }

    assert ragas_eval.retrieved_context_ids_from_output(output) == ["file-a", "file-b"]


def test_score_id_context_recall_row_uses_ragas_single_turn_sample():
    from imperial_rag import ragas_eval

    captured: dict[str, Any] = {}

    class FakeScorer:
        def single_turn_ascore(self, sample):
            captured["retrieved_context_ids"] = sample.retrieved_context_ids
            captured["reference_context_ids"] = sample.reference_context_ids
            return 0.5

    result = ragas_eval.score_id_context_recall_row(
        {
            "user_input": "q",
            "retrieved_context_ids": ["file-a", "file-b"],
            "reference_context_ids": ["file-b", "file-c"],
        },
        scorer=FakeScorer(),
    )

    assert captured == {
        "retrieved_context_ids": ["file-a", "file-b"],
        "reference_context_ids": ["file-b", "file-c"],
    }
    assert result == {
        "score": 0.5,
        "label": "id_context_recall",
        "explanation": None,
        "metadata": {
            "metric": "ragas_id_context_recall",
            "retrieved_context_id_count": 2,
            "reference_context_id_count": 2,
        },
    }


def test_score_id_context_recall_row_skips_missing_ids():
    from imperial_rag import ragas_eval

    missing_reference = ragas_eval.score_id_context_recall_row(
        {"user_input": "q", "retrieved_context_ids": ["file-a"], "reference_context_ids": []},
        scorer=object(),
    )
    missing_retrieved = ragas_eval.score_id_context_recall_row(
        {"user_input": "q", "retrieved_context_ids": [], "reference_context_ids": ["file-a"]},
        scorer=object(),
    )

    assert missing_reference["score"] is None
    assert missing_reference["label"] == "skipped"
    assert missing_reference["metadata"]["reason"] == "missing_reference_context_ids"
    assert missing_retrieved["score"] is None
    assert missing_retrieved["label"] == "skipped"
    assert missing_retrieved["metadata"]["reason"] == "missing_retrieved_context_ids"


def test_single_turn_sample_from_row_populates_text_and_context_id_fields(monkeypatch):
    from imperial_rag import ragas_eval

    captured: dict[str, Any] = {}

    class FakeSingleTurnSample:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ragas_eval, "_import_single_turn_sample", lambda: FakeSingleTurnSample)

    sample = ragas_eval.single_turn_sample_from_row(
        {
            "user_input": " Что делать с браком? ",
            "response": " Оформить возврат. ",
            "retrieved_contexts": [" Регламент возврата. ", ""],
            "reference": "Возврат оформляется по регламенту.",
            "retrieved_context_ids": ["file-a", "file-a", ""],
            "reference_context_ids": ["file-b", " "],
        }
    )

    assert isinstance(sample, FakeSingleTurnSample)
    assert captured == {
        "user_input": "Что делать с браком?",
        "response": "Оформить возврат.",
        "retrieved_contexts": ["Регламент возврата."],
        "reference": "Возврат оформляется по регламенту.",
        "retrieved_context_ids": ["file-a"],
        "reference_context_ids": ["file-b"],
    }


def test_evaluation_dataset_from_rows_uses_single_turn_samples(monkeypatch):
    from imperial_rag import ragas_eval

    captured: dict[str, Any] = {}

    class FakeDataset:
        def __init__(self, *, samples):
            self.samples = samples
            captured["samples"] = samples

    monkeypatch.setattr(ragas_eval, "_import_evaluation_dataset", lambda: FakeDataset)
    monkeypatch.setattr(ragas_eval, "single_turn_sample_from_row", lambda row: {"sample": row["user_input"]})

    dataset = ragas_eval.evaluation_dataset_from_rows(
        [
            {"user_input": "q1", "response": "a1"},
            {"user_input": "q2", "response": "a2"},
        ]
    )

    assert isinstance(dataset, FakeDataset)
    assert captured["samples"] == [{"sample": "q1"}, {"sample": "q2"}]


def test_parse_ragas_metric_names_accepts_id_context_recall_aliases():
    from imperial_rag import ragas_eval

    assert ragas_eval.parse_ragas_metric_names("id-context-recall") == ["id_context_recall"]
    assert ragas_eval.parse_ragas_metric_names("id_based_context_recall") == ["id_context_recall"]
    assert ragas_eval.parse_ragas_metric_names("answer-relevancy") == ["answer_relevancy"]
    assert ragas_eval.parse_ragas_metric_names("answer_relevance") == ["answer_relevancy"]
    assert ragas_eval.parse_ragas_metric_names("response_relevancy") == ["answer_relevancy"]


def test_evaluate_id_context_recall_rows_returns_sidecar_records():
    from imperial_rag import ragas_eval

    class FakeScorer:
        def single_turn_ascore(self, sample):
            return 1.0 if sample.reference_context_ids == ["file-a"] else 0.0

    rows = [
        {"user_input": "q1", "retrieved_context_ids": ["file-a"], "reference_context_ids": ["file-a"]},
        {"user_input": "q2", "retrieved_context_ids": ["file-b"], "reference_context_ids": []},
    ]

    assert ragas_eval.evaluate_id_context_recall_rows(rows, scorer=FakeScorer()) == [
        {
            "user_input": "q1",
            "id_context_recall": 1.0,
            "label": "id_context_recall",
            "explanation": None,
            "retrieved_context_id_count": 1,
            "reference_context_id_count": 1,
        },
        {
            "user_input": "q2",
            "id_context_recall": None,
            "label": "skipped",
            "explanation": "Ragas ID context recall requires reference_context_ids.",
            "retrieved_context_id_count": 1,
            "reference_context_id_count": 0,
        },
    ]


def test_evaluate_answer_relevancy_rows_returns_sidecar_records():
    from imperial_rag import ragas_eval

    class FakeScorer:
        def score(self, **kwargs):
            return SimpleNamespace(value=0.9, reason="relevant")

    rows = [{"user_input": "q", "response": "a"}]

    assert ragas_eval.evaluate_answer_relevancy_rows(rows, scorer=FakeScorer()) == [
        {
            "user_input": "q",
            "answer_relevancy": 0.9,
            "label": "answer_relevancy",
            "explanation": "relevant",
        }
    ]


def test_evaluate_faithfulness_rows_returns_sidecar_records():
    from imperial_rag import ragas_eval

    class FakeScorer:
        def score(self, **kwargs):
            return SimpleNamespace(value=1.0, reason="supported")

    rows = [{"user_input": "q", "response": "a", "retrieved_contexts": ["ctx"]}]

    assert ragas_eval.evaluate_faithfulness_rows(rows, scorer=FakeScorer()) == [
        {
            "user_input": "q",
            "faithfulness": 1.0,
            "label": "faithfulness",
            "explanation": "supported",
            "retrieved_context_count": 1,
        }
    ]


def test_ragas_eval_script_imports_without_importing_ragas_at_module_load():
    module = _load_ragas_runner()

    assert hasattr(module, "build_ragas_rows")
    assert hasattr(module, "evaluate_ragas_rows")


def test_run_coroutine_rejects_sync_bridge_inside_running_loop():
    from imperial_rag import ragas_eval

    async def runner():
        with pytest.raises(RuntimeError, match="running event loop"):
            ragas_eval._run_coroutine(asyncio.sleep(0))

    asyncio.run(runner())


def test_ragas_runner_sync_wrapper_rejects_running_loop():
    module = _load_ragas_runner()

    async def runner():
        with pytest.raises(RuntimeError, match="running event loop"):
            module.evaluate_ragas_rows([], ["faithfulness"])

    asyncio.run(runner())


def test_ragas_eval_rejects_non_positive_cli_concurrency_before_running():
    module = _load_ragas_runner()

    module._load_project_env = lambda workspace_root: pytest.fail("loaded env after invalid concurrency")
    module._build_settings = lambda workspace_root: pytest.fail("built settings after invalid concurrency")

    with pytest.raises(SystemExit) as exc_info:
        module.main(["--concurrency", "0"])

    assert exc_info.value.code == 2


def test_async_faithfulness_row_awaits_scorer_inside_running_loop(monkeypatch):
    from imperial_rag import ragas_eval

    monkeypatch.setattr(ragas_eval, "_run_coroutine", lambda awaitable: pytest.fail("sync bridge used"))

    class FakeResult:
        value = 0.6
        reason = "supported asynchronously"

    class FakeScorer:
        async def ascore(self, **kwargs):
            return FakeResult()

    async def runner():
        return await ragas_eval.score_faithfulness_row_async(
            {"user_input": "q", "response": "a", "retrieved_contexts": ["ctx"]},
            scorer=FakeScorer(),
        )

    assert asyncio.run(runner()) == {
        "score": 0.6,
        "label": "faithfulness",
        "explanation": "supported asynchronously",
        "metadata": {"metric": "ragas_faithfulness", "retrieved_context_count": 1},
    }


def test_async_answer_relevancy_rows_preserve_order_under_concurrency():
    from imperial_rag import ragas_eval

    class FakeScorer:
        async def ascore(self, **kwargs):
            if kwargs["user_input"] == "slow":
                await asyncio.sleep(0.01)
            return SimpleNamespace(value=1.0 if kwargs["user_input"] == "slow" else 0.5)

    async def runner():
        return await ragas_eval.evaluate_answer_relevancy_rows_async(
            [
                {"user_input": "slow", "response": "a"},
                {"user_input": "fast", "response": "b"},
            ],
            scorer=FakeScorer(),
            concurrency=2,
        )

    records = asyncio.run(runner())

    assert [record["user_input"] for record in records] == ["slow", "fast"]
    assert [record["answer_relevancy"] for record in records] == [1.0, 0.5]


def test_build_ragas_rows_uses_runtime_outputs_and_skips_refusals():
    module = _load_ragas_runner()
    examples = [
        {
            "question": "Как оформить возврат брака?",
            "expected_behavior": "cite_answer",
            "expected_source_hints": ["брак"],
            "reference_answer": "Возврат брака оформляется по регламенту.",
            "reference_context_ids": ["file-a"],
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
                        "metadata": {"relative_path": "documents/reglament.docx", "file_id": "file-a"},
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
            "retrieved_context_ids": ["file-a"],
            "expected_behavior": "cite_answer",
            "expected_source_hints": ["брак"],
            "reference": "Возврат брака оформляется по регламенту.",
            "reference_context_ids": ["file-a"],
        }
    ]


def test_build_ragas_rows_skips_conflict_rows_for_answer_quality_metrics():
    module = _load_ragas_runner()
    examples = [
        {
            "id": "imperial-conflict-001",
            "question": "Какая версия регламента действует?",
            "expected_behavior": "surface_conflict",
            "expected_source_hints": ["регламент"],
            "reference_answer": "Ответ должен показать конфликт.",
        }
    ]

    class FakeRuntime:
        def query(self, question: str) -> dict[str, object]:
            return {
                "answer": "Документы противоречат друг другу. [a] [b]",
                "citations": ["[a]", "[b]"],
                "evidence": [{"page_content": "Версия А", "metadata": {"file_id": "file-a"}}],
            }

    prepared = module.build_ragas_rows(examples, runtime=FakeRuntime())

    assert prepared.skipped == 1
    assert prepared.rows == []


def test_validate_metric_requirements_rejects_reference_metrics_without_reference():
    module = _load_ragas_runner()

    with pytest.raises(SystemExit, match="reference_answer"):
        module.validate_metric_requirements(
            ["faithfulness", "context_recall"],
            [{"user_input": "q", "response": "a", "retrieved_contexts": ["ctx"]}],
        )


def test_evaluate_ragas_rows_delegates_default_faithfulness_to_shared_helper(monkeypatch):
    module = _load_ragas_runner()
    captured: dict[str, Any] = {}

    async def fake_evaluate_faithfulness_rows_async(rows, *, concurrency):
        captured["rows"] = rows
        captured["concurrency"] = concurrency
        return [{"faithfulness": 1.0}]

    rows = [{"user_input": "q", "response": "a", "retrieved_contexts": ["ctx"]}]
    monkeypatch.setattr(module, "evaluate_faithfulness_rows_async", fake_evaluate_faithfulness_rows_async)

    result = module.evaluate_ragas_rows(rows, ["faithfulness"], concurrency=2)

    assert result == [{"faithfulness": 1.0}]
    assert captured == {"rows": rows, "concurrency": 2}


def test_evaluate_ragas_rows_merges_faithfulness_and_answer_relevancy(monkeypatch):
    module = _load_ragas_runner()
    captured: dict[str, Any] = {}

    async def fake_evaluate_faithfulness_rows_async(rows, *, concurrency):
        captured["faithfulness_rows"] = rows
        return [{"user_input": "q", "faithfulness": 1.0}]

    async def fake_evaluate_answer_relevancy_rows_async(rows, *, concurrency):
        captured["answer_rows"] = rows
        return [{"user_input": "q", "answer_relevancy": 0.8}]

    rows = [{"user_input": "q", "response": "a", "retrieved_contexts": ["ctx"]}]
    monkeypatch.setattr(module, "evaluate_faithfulness_rows_async", fake_evaluate_faithfulness_rows_async)
    monkeypatch.setattr(module, "evaluate_answer_relevancy_rows_async", fake_evaluate_answer_relevancy_rows_async)

    result = module.evaluate_ragas_rows(rows, ["faithfulness", "answer_relevancy"])

    assert result == [{"user_input": "q", "faithfulness": 1.0, "answer_relevancy": 0.8}]
    assert captured == {"faithfulness_rows": rows, "answer_rows": rows}


def test_evaluate_ragas_rows_merges_id_context_recall_without_llm(monkeypatch):
    module = _load_ragas_runner()
    captured: dict[str, Any] = {}

    async def fake_evaluate_id_context_recall_rows_async(rows, *, concurrency):
        captured["rows"] = rows
        return [{"user_input": "q", "id_context_recall": 0.5}]

    monkeypatch.setattr(module, "evaluate_id_context_recall_rows_async", fake_evaluate_id_context_recall_rows_async)
    monkeypatch.setattr(module, "build_evaluator_llm", lambda: pytest.fail("ID recall should not build an evaluator LLM"))

    rows = [{"user_input": "q", "retrieved_context_ids": ["file-a"], "reference_context_ids": ["file-b"]}]
    result = module.evaluate_ragas_rows(rows, ["id_context_recall"])

    assert result == [{"user_input": "q", "id_context_recall": 0.5}]
    assert captured == {"rows": rows}


def test_evaluate_ragas_rows_runs_sidecars_concurrently_and_merges_metric_order(monkeypatch):
    module = _load_ragas_runner()
    running = 0
    max_running = 0
    started: list[str] = []

    def sidecar(name: str, metric_key: str, score: float):
        async def evaluate(rows, *, concurrency):
            nonlocal running, max_running
            started.append(name)
            running += 1
            max_running = max(max_running, running)
            await asyncio.sleep(0.02)
            running -= 1
            return [{"user_input": row["user_input"], metric_key: score} for row in rows]

        return evaluate

    monkeypatch.setattr(module, "evaluate_faithfulness_rows_async", sidecar("faithfulness", "faithfulness", 1.0))
    monkeypatch.setattr(
        module,
        "evaluate_answer_relevancy_rows_async",
        sidecar("answer_relevancy", "answer_relevancy", 0.8),
    )
    monkeypatch.setattr(
        module,
        "evaluate_id_context_recall_rows_async",
        sidecar("id_context_recall", "id_context_recall", 0.5),
    )

    rows = [
        {
            "user_input": "q",
            "response": "a",
            "retrieved_contexts": ["ctx"],
            "retrieved_context_ids": ["file-a"],
            "reference_context_ids": ["file-a"],
        }
    ]

    result = asyncio.run(
        module.evaluate_ragas_rows_async(
            rows,
            ["faithfulness", "answer_relevancy", "id_context_recall"],
            concurrency=2,
        )
    )

    assert result == [
        {
            "user_input": "q",
            "faithfulness": 1.0,
            "answer_relevancy": 0.8,
            "id_context_recall": 0.5,
        }
    ]
    assert started == ["faithfulness", "answer_relevancy", "id_context_recall"]
    assert max_running > 1


def test_evaluate_ragas_rows_keeps_reference_metric_path(monkeypatch):
    module = _load_ragas_runner()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(module, "build_ragas_dataset", lambda rows: {"dataset": rows})
    monkeypatch.setattr(module, "build_evaluator_llm", lambda: "ragas-llm")

    def fake_build_ragas_metrics(names, evaluator_llm):
        captured["metric_names"] = names
        captured["metric_llm"] = evaluator_llm
        return [f"metric:{name}" for name in names]

    monkeypatch.setattr(module, "build_ragas_metrics", fake_build_ragas_metrics)

    def fake_evaluate(**kwargs):
        captured["evaluate_kwargs"] = kwargs
        return {"scores": [{"context_recall": 1.0}]}

    rows = [
        {
            "id": "imperial-cite-001",
            "lane": "indexed_answerability",
            "user_input": "q",
            "response": "a",
            "retrieved_contexts": ["ctx"],
            "reference": "ref",
        }
    ]
    result = module.evaluate_ragas_rows(rows, ["context_recall"], evaluate_fn=fake_evaluate)

    assert result == [{"id": "imperial-cite-001", "lane": "indexed_answerability", "context_recall": 1.0}]
    assert captured == {
        "metric_names": ["context_recall"],
        "metric_llm": "ragas-llm",
        "evaluate_kwargs": {
            "dataset": {"dataset": rows},
            "metrics": ["metric:context_recall"],
        },
    }


def test_evaluate_ragas_rows_uses_async_ragas_reference_path_by_default(monkeypatch):
    module = _load_ragas_runner()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(module, "build_ragas_dataset", lambda rows: {"dataset": rows})
    monkeypatch.setattr(module, "build_evaluator_llm", lambda: "ragas-llm")
    monkeypatch.setattr(module, "build_ragas_metrics", lambda names, evaluator_llm: [f"metric:{name}" for name in names])

    async def fake_aevaluate(**kwargs):
        captured["aevaluate_kwargs"] = kwargs
        return {"scores": [{"context_recall": 0.75}]}

    monkeypatch.setattr(module, "_import_ragas_aevaluate", lambda: fake_aevaluate)

    rows = [{"id": "imperial-cite-001", "user_input": "q", "response": "a", "reference": "ref"}]
    result = module.evaluate_ragas_rows(rows, ["context_recall"], batch_size=3)

    assert result == [{"id": "imperial-cite-001", "context_recall": 0.75}]
    assert captured == {
        "aevaluate_kwargs": {
            "dataset": {"dataset": rows},
            "metrics": ["metric:context_recall"],
            "batch_size": 3,
        }
    }


def test_evaluate_ragas_rows_suppresses_ragas_aevaluate_deprecation_warning(monkeypatch):
    module = _load_ragas_runner()

    monkeypatch.setattr(module, "build_ragas_dataset", lambda rows: {"dataset": rows})
    monkeypatch.setattr(module, "build_evaluator_llm", lambda: "ragas-llm")
    monkeypatch.setattr(module, "build_ragas_metrics", lambda names, evaluator_llm: [f"metric:{name}" for name in names])

    async def fake_aevaluate(**kwargs):
        warnings.warn("ragas.aevaluate() is deprecated; use @experiment", DeprecationWarning, stacklevel=2)
        return {"scores": [{"context_recall": 0.75}]}

    rows = [{"id": "imperial-cite-001", "user_input": "q", "response": "a", "reference": "ref"}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = module.evaluate_ragas_rows(rows, ["context_recall"], evaluate_fn=fake_aevaluate)

    assert result == [{"id": "imperial-cite-001", "context_recall": 0.75}]
    assert [warning for warning in caught if issubclass(warning.category, DeprecationWarning)] == []


def test_build_ragas_dataset_uses_structured_evaluation_dataset(monkeypatch):
    module = _load_ragas_runner()
    rows = [{"user_input": "q", "response": "a", "retrieved_contexts": ["ctx"], "reference": "ref"}]
    captured: dict[str, Any] = {}

    def fake_evaluation_dataset_from_rows(received_rows):
        captured["rows"] = received_rows
        return {"structured_dataset": received_rows}

    monkeypatch.setattr(module, "evaluation_dataset_from_rows", fake_evaluation_dataset_from_rows)

    assert module.build_ragas_dataset(rows) == {"structured_dataset": rows}
    assert captured == {"rows": rows}


def test_build_evaluator_llm_uses_modern_ragas_dashscope_client(monkeypatch):
    module = _load_ragas_runner()
    captured: dict[str, Any] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured["client"] = self

    def fake_llm_factory(model, **kwargs):
        captured["llm_factory_model"] = model
        captured["llm_factory_kwargs"] = kwargs
        return "modern-ragas-llm"

    monkeypatch.setattr(module, "_import_async_openai", lambda: FakeAsyncOpenAI)
    monkeypatch.setattr(module, "_import_llm_factory", lambda: fake_llm_factory)

    evaluator_llm = module.build_evaluator_llm(
        cast(Any, SimpleNamespace(
            compat_base_url="https://dashscope.example/compatible-mode/v1",
            chat_model="qwen-test",
            require_api_key=lambda: "dashscope-key",
        ))
    )

    assert evaluator_llm == "modern-ragas-llm"
    assert captured["client"].kwargs == {
        "api_key": "dashscope-key",
        "base_url": "https://dashscope.example/compatible-mode/v1",
    }
    assert captured["llm_factory_model"] == "qwen-test"
    assert captured["llm_factory_kwargs"] == {"client": captured["client"], "provider": "openai"}


def test_build_ragas_metrics_imports_installed_ragas_metrics():
    module = _load_ragas_runner()
    module._install_ragas_langchain_community_compat()

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.base import Metric

    client = AsyncOpenAI(api_key="test-key", base_url="https://dashscope.example/compatible-mode/v1")
    evaluator_llm = llm_factory("qwen-test", client=client, provider="openai")
    metrics = module.build_ragas_metrics(["context_recall", "factual_correctness"], evaluator_llm)

    assert [metric.name for metric in metrics] == ["context_recall", "factual_correctness"]
    assert all(isinstance(metric, Metric) for metric in metrics)


def test_import_answer_relevancy_metric_prefers_ragas_collections_api():
    from imperial_rag import ragas_eval

    metric_cls = ragas_eval._import_answer_relevancy_metric()

    assert metric_cls.__name__ == "AnswerRelevancy"
    assert ".collections." in metric_cls.__module__


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


def test_map_rows_bounded_asserts_all_result_slots_are_filled():
    from imperial_rag import ragas_eval

    async def evaluator(row):
        return None

    async def runner():
        with pytest.raises(AssertionError, match="did not fill all result slots"):
            await ragas_eval._map_rows_bounded([{"user_input": "q"}], evaluator, concurrency=1)

    asyncio.run(runner())


def _load_ragas_runner():
    spec = importlib.util.spec_from_file_location("run_ragas_eval_for_test", Path("scripts/run_ragas_eval.py"))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
