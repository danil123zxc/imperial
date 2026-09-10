from __future__ import annotations

import asyncio
from types import SimpleNamespace

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from phoenix.evals.metrics import DocumentRelevanceEvaluator
import pytest

from imperial_rag.evals.phoenix_retrieval import PhoenixRetrievalJudge
from imperial_rag.observability import phoenix as tracing


@pytest.fixture
def judge_case(monkeypatch):
    for name in ("OPENINFERENCE_HIDE_INPUTS", "OPENINFERENCE_HIDE_INPUT_TEXT", "OPENINFERENCE_HIDE_OUTPUTS",
                 "IMPERIAL_RAG_TRACE_DOCUMENT_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("IMPERIAL_RAG_DASHSCOPE_COMPAT_BASE_URL", "https://example.invalid/v1")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", lambda *a, **kw: provider.get_tracer("test"))
    monkeypatch.setattr(tracing, "_CONFIGURED_PROVIDER", provider)
    state = SimpleNamespace(calls=[], annotations=[], active=0, peak=0, failure=None, logging_failure=None,
                            provider=provider, exporter=exporter)

    async def evaluate(self, eval_input):
        state.active += 1
        state.peak = max(state.peak, state.active)
        state.calls.append(eval_input)
        await asyncio.sleep(0.001)
        state.active -= 1
        if state.failure == "provider":
            raise RuntimeError("private provider response with a secret")
        if state.failure == "malformed":
            return []
        relevant = "revenue" in eval_input["document_text"]
        return [SimpleNamespace(score=float(relevant), label="relevant" if relevant else "unrelated",
                                explanation="Matches the requested topic" if relevant else "Different topic")]

    async def log_document_annotations(*, document_annotations, sync):
        assert sync is True
        assert state.exporter.get_finished_spans(), "Annotations must follow exported retrieval spans"
        if state.logging_failure == "exception":
            raise RuntimeError("private annotation response")
        state.annotations.extend(document_annotations)
        return [] if state.logging_failure == "partial" else [{"id": str(i)} for i in range(len(document_annotations))]

    monkeypatch.setattr(DocumentRelevanceEvaluator, "async_evaluate", evaluate)
    state.client = SimpleNamespace(spans=SimpleNamespace(log_document_annotations=log_document_annotations))
    yield state
    provider.shutdown()


def documents(*texts):
    return [{"page_content": text, "metadata": {"chunk_id": str(index)}} for index, text in enumerate(texts)]


def test_async_judging_preserves_ranked_positions_and_shared_concurrency(judge_case):
    state = judge_case
    judge = PhoenixRetrievalJudge(state.client, k=3, concurrency=2)
    example = {"question": "How did revenue change?", "expected_behavior": "cite_answer"}
    output = {"ranked_documents": documents("unrelated", "revenue", "revenue", "outside cutoff"),
              "documents": documents("packed text must not be judged")}

    async def run():
        return await asyncio.gather(judge.evaluate(example, output), judge.evaluate(example, output))

    results = asyncio.run(run())
    assert all(result["status"] == "completed" for result in results)
    assert all(result["question_kind"] == "answerable" for result in results)
    assert state.peak == 2
    assert len(state.calls) == 6
    assert {row["document_text"] for row in state.calls} == {"unrelated", "revenue"}
    assert all(row["input"] == example["question"] for row in state.calls)
    spans = state.exporter.get_finished_spans()
    for result in results:
        annotations = [row for row in state.annotations if row["span_id"] == result["span_id"]]
        assert [row["document_position"] for row in annotations] == [0, 1, 2]
        assert [row["result"]["score"] for row in annotations] == [0, 1, 1]
        assert all(row["annotator_kind"] == "LLM" and row["result"]["explanation"] for row in annotations)
        span = next(span for span in spans if f"{span.context.span_id:016x}" == result["span_id"])
        assert span.attributes["retrieval.documents.0.document.content"] == "unrelated"
        assert span.attributes["retrieval.documents.2.document.id"] == "2"
        assert "retrieval.documents.3.document.id" not in span.attributes
        assert result["ranking_evaluation_version"] == "phoenix-document-relevance-v1"
        assert not any(key in result for key in ("ndcg", "mrr", "precision"))


@pytest.mark.parametrize("case", ["empty", "missing", "provider", "malformed", "annotation", "partial", "flush"])
def test_empty_and_failed_judging_never_become_successful_scores(judge_case, monkeypatch, case):
    state = judge_case
    judge = PhoenixRetrievalJudge(state.client, k=3, concurrency=2)
    output = {"ranked_documents": documents("revenue", "unrelated")}
    if case == "empty":
        output["ranked_documents"] = []
    elif case == "missing":
        output = {"documents": documents("must not fall back")}
    elif case in {"provider", "malformed"}:
        state.failure = case
    elif case in {"annotation", "partial"}:
        state.logging_failure = "exception" if case == "annotation" else "partial"
    else:
        monkeypatch.setattr(state.provider, "force_flush", lambda: False)
    result = asyncio.run(judge.evaluate({"question": "revenue?", "expected_behavior": "refuse_if_not_found"}, output))
    assert result["status"] == ("empty" if case == "empty" else "error")
    assert result["question_kind"] == "refusal"
    assert "private" not in str(result)
    assert state.active == 0
    if case in {"empty", "missing"}:
        assert state.calls == []
    if case != "partial":
        assert state.annotations == []


@pytest.mark.parametrize("flag", ["OPENINFERENCE_HIDE_INPUTS", "OPENINFERENCE_HIDE_INPUT_TEXT",
                                 "OPENINFERENCE_HIDE_OUTPUTS", "IMPERIAL_RAG_TRACE_DOCUMENT_LIMIT", "tracing"])
def test_incompatible_tracing_fails_before_judge_calls(judge_case, monkeypatch, flag):
    if flag == "tracing":
        monkeypatch.setattr(tracing, "_CONFIGURED_PROVIDER", None)
    else:
        monkeypatch.setenv(flag, "1")
    with pytest.raises(ValueError):
        PhoenixRetrievalJudge(judge_case.client, k=3, concurrency=2)
    assert judge_case.calls == []


@pytest.mark.parametrize("failure", ["eventual", "permanent", "server"])
def test_annotation_retry_waits_for_ingestion_without_repeating_judging(judge_case, monkeypatch, failure):
    import httpx

    state = judge_case
    original = state.client.spans.log_document_annotations
    attempts = []

    async def log(**kwargs):
        attempts.append(kwargs)
        if failure == "eventual" and len(attempts) == 2:
            return await original(**kwargs)
        response = httpx.Response(500 if failure == "server" else 404,
                                  request=httpx.Request("POST", "http://localhost/v1/document_annotations"))
        raise httpx.HTTPStatusError("Not ingested", request=response.request, response=response)

    async def sleep(delay):
        pass

    monkeypatch.setattr(state.client.spans, "log_document_annotations", log)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    judge = PhoenixRetrievalJudge(state.client, k=2, concurrency=1)
    result = asyncio.run(judge.evaluate({"question": "revenue?"}, {"ranked_documents": documents("revenue")}))
    assert len(state.calls) == 1
    assert len(attempts) == {"eventual": 2, "permanent": 6, "server": 1}[failure]
    assert result["status"] == ("completed" if failure == "eventual" else "error")
