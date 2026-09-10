"""Phoenix LLM relevance annotations; ranking arithmetic belongs to Phoenix."""
from __future__ import annotations

import asyncio
from importlib.metadata import version
from typing import Any, cast

import httpx
from langchain_core.documents import Document
from opentelemetry import trace

from imperial_rag.config import env_bool, env_int
from imperial_rag.integrations.dashscope import QwenProviderSettings
from imperial_rag.observability import phoenix as tracing

EVALUATOR_VERSION = "phoenix-document-relevance-v1"


class PhoenixRetrievalJudge:
    def __init__(self, client: Any, *, k: int, concurrency: int) -> None:
        if k < 1 or concurrency < 1:
            raise ValueError("Retrieval cutoff and concurrency must be positive")
        hidden = [key for key in ("OPENINFERENCE_HIDE_INPUTS", "OPENINFERENCE_HIDE_INPUT_TEXT",
                                 "OPENINFERENCE_HIDE_OUTPUTS") if env_bool(key, False)]
        if hidden:
            raise ValueError("Phoenix retrieval judging needs visible evaluation inputs/outputs; incompatible: "
                             + ", ".join(hidden))
        if env_int("IMPERIAL_RAG_TRACE_DOCUMENT_LIMIT", k) < k:
            raise ValueError("IMPERIAL_RAG_TRACE_DOCUMENT_LIMIT must cover --retrieval-k for Phoenix judging")
        self.provider = tracing._CONFIGURED_PROVIDER
        if self.provider is None:
            raise ValueError("Phoenix retrieval judging requires configured Phoenix tracing; enable it before evaluation")
        from phoenix.evals import LLM
        from phoenix.evals.metrics import DocumentRelevanceEvaluator

        settings = QwenProviderSettings.from_env()
        self.evaluator = DocumentRelevanceEvaluator(
            llm=LLM(provider="openai", client="openai", model=settings.chat_model,
                    api_key=settings.require_api_key(), base_url=settings.compat_base_url,
                    timeout=60),
            temperature=0, extra_body={"enable_thinking": False},
        )
        self.client = client
        self.k = k
        self.semaphore = asyncio.Semaphore(concurrency)
        self.metadata = {"ranking_evaluation_version": EVALUATOR_VERSION,
                         "phoenix_evals_version": version("arize-phoenix-evals"),
                         "judge_model": settings.chat_model, "retrieval_k": k}

    async def evaluate(self, example: dict[str, Any], output: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = self.metadata | {
            "status": "error", "document_count": 0,
            "question_kind": "refusal" if example.get("expected_behavior") == "refuse_if_not_found" else "answerable",
        }
        stage = "validate_ranked_documents"
        try:
            if "ranked_documents" not in output:
                raise ValueError("Missing ranked_documents before answer packing")
            documents = [Document(**row) for row in output["ranked_documents"][:self.k]]
            result["document_count"] = len(documents)
            if not documents:
                return result | {"status": "empty"}
            if any(not doc.page_content.strip() for doc in documents):
                raise ValueError("Cannot judge a chunk without text")
            stage = "judge"
            attrs = {f"imperial.eval.{key}": value for key, value in result.items() if key != "status"}
            with tracing.trace_retrieval_step("evaluation.retrieval_relevance", example["question"], attributes=attrs) as span:
                current = trace.get_current_span()
                if not current.is_recording():
                    raise ValueError("Phoenix evaluation span is not recording; check tracing and sampling")
                result["span_id"] = f"{current.get_span_context().span_id:016x}"
                result["trace_id"] = f"{current.get_span_context().trace_id:032x}"
                span.set_retrieval_documents(documents)

                async def judge(document: Document) -> Any:
                    async with self.semaphore:
                        scores = await self.evaluator.async_evaluate(
                            {"input": example["question"], "document_text": document.page_content}
                        )
                        if len(scores) != 1 or scores[0].score not in (0, 1) or not scores[0].explanation:
                            raise ValueError("Incomplete Phoenix document relevance judgment")
                        return scores[0]

                # Wait for every request before ending the span, including on a partial failure.
                scores = await asyncio.gather(*(judge(doc) for doc in documents), return_exceptions=True)
                error = next((score for score in scores if isinstance(score, BaseException)), None)
                if error is not None:
                    # Do not let provider exception text enter trace payloads.
                    raise RuntimeError(f"Document relevance judge failed: {type(error).__name__}") from None
                span.set_output({"status": "judged", "document_count": len(documents)})
            stage = "export_trace"
            if not await asyncio.to_thread(self.provider.force_flush):
                raise RuntimeError("Phoenix trace export did not flush")
            annotations = [{"name": "relevance", "span_id": result["span_id"], "document_position": position,
                            "annotator_kind": "LLM", "metadata": self.metadata,
                            "result": {"score": score.score, "label": score.label, "explanation": score.explanation}}
                           for position, score in enumerate(cast(list[Any], scores))]
            stage = "log_annotations"
            # Phoenix ingests exported spans asynchronously; a 404 has not inserted annotations.
            for attempt in range(6):
                try:
                    inserted = await self.client.spans.log_document_annotations(document_annotations=annotations, sync=True)
                    break
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 404 or attempt == 5:
                        raise
                    await asyncio.sleep(min(2 ** attempt, 4))
            if inserted is None or len(inserted) != len(documents):
                raise RuntimeError("Phoenix did not acknowledge every document annotation")
            return result | {"status": "completed"}
        except Exception as exc:
            # Error text may include provider credentials or private document contents.
            return result | {"error_type": type(exc).__name__, "error_stage": stage}
