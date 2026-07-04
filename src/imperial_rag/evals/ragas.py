from __future__ import annotations

import asyncio
import importlib
import inspect
import math
import sys
import types
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import anyio
from anyio.to_thread import run_sync as run_sync_in_worker_thread

from imperial_rag.evals.corpus import CHUNK_ID_METADATA_FIELDS, unique_nonempty
from imperial_rag.integrations.dashscope import MissingDashScopeKeyError, QwenProviderSettings


FAITHFULNESS_LABEL = "faithfulness"
FAITHFULNESS_METADATA_KEY = "ragas_faithfulness"
ANSWER_RELEVANCY_LABEL = "answer_relevancy"
ANSWER_RELEVANCY_METADATA_KEY = "ragas_answer_relevancy"
ID_CONTEXT_RECALL_LABEL = "id_context_recall"
ID_CONTEXT_RECALL_METADATA_KEY = "ragas_id_context_recall"
DEFAULT_RAGAS_METRICS = (FAITHFULNESS_LABEL, ANSWER_RELEVANCY_LABEL)
DEFAULT_RAGAS_CONCURRENCY = 4
REFERENCE_REQUIRED_RAGAS_METRICS = {"context_recall", "factual_correctness"}
SUPPORTED_RAGAS_METRICS = DEFAULT_RAGAS_METRICS + (
    "context_recall",
    "factual_correctness",
    ID_CONTEXT_RECALL_LABEL,
)
RAGAS_METRIC_ALIASES = {
    "answer_relevance": ANSWER_RELEVANCY_LABEL,
    "response_relevance": ANSWER_RELEVANCY_LABEL,
    "response_relevancy": ANSWER_RELEVANCY_LABEL,
    "id_based_context_recall": ID_CONTEXT_RECALL_LABEL,
}
NO_RAGAS_METRIC_ALIASES = {"none", "no", "off", "false", "0"}


def parse_ragas_metric_names(
    raw_metrics: str | None,
    *,
    default: Sequence[str] = DEFAULT_RAGAS_METRICS,
    allow_none: bool = False,
) -> list[str]:
    if raw_metrics is None or not raw_metrics.strip():
        return list(default)
    normalized_names = [
        name.strip().casefold().replace("-", "_")
        for name in raw_metrics.split(",")
        if name.strip()
    ]
    names: list[str] = [RAGAS_METRIC_ALIASES.get(name, name) for name in normalized_names]
    if allow_none and any(name in NO_RAGAS_METRIC_ALIASES for name in names):
        return []
    unsupported = sorted(set(names) - set(SUPPORTED_RAGAS_METRICS))
    if unsupported:
        supported_names: list[str] = [*SUPPORTED_RAGAS_METRICS]
        if allow_none:
            supported_names.append("none")
        supported = ", ".join(supported_names)
        raise SystemExit(f"Unsupported Ragas metrics: {', '.join(unsupported)}. Supported metrics: {supported}.")
    return names


def validate_ragas_metric_requirements(
    metric_names: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_key: str,
    row_label_key: str,
) -> None:
    reference_metrics = sorted(set(metric_names) & REFERENCE_REQUIRED_RAGAS_METRICS)
    if not reference_metrics:
        return
    missing_reference = [str(row.get(row_label_key) or "") for row in rows if not row.get(reference_key)]
    if missing_reference:
        joined_metrics = ", ".join(reference_metrics)
        raise SystemExit(
            f"Ragas metrics {joined_metrics} require reference_answer in evals/questions.jsonl. "
            f"Missing reference_answer for {len(missing_reference)} prepared rows."
        )


def faithfulness_row_from_run_output(
    input: Mapping[str, Any] | None,
    output: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    row = answer_relevancy_row_from_run_output(input, output)
    if row is None:
        return None
    retrieved_contexts = retrieved_contexts_from_output(output or {})
    if not retrieved_contexts:
        return None
    return {
        **row,
        "retrieved_contexts": retrieved_contexts,
    }


def answer_relevancy_row_from_run_output(
    input: Mapping[str, Any] | None,
    output: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    resolved_input = input or {}
    resolved_output = output or {}
    user_input = str(
        resolved_input.get("question")
        or resolved_input.get("user_input")
        or resolved_output.get("question")
        or resolved_output.get("user_input")
        or ""
    ).strip()
    response = str(resolved_output.get("answer") or resolved_output.get("response") or "").strip()
    if not user_input or not response:
        return None
    return {
        "user_input": user_input,
        "response": response,
    }


def retrieved_contexts_from_output(output: Mapping[str, Any]) -> list[str]:
    direct_contexts = output.get("retrieved_contexts")
    if direct_contexts:
        return _clean_texts(direct_contexts)

    documents = output.get("documents") or output.get("evidence") or []
    contexts: list[str] = []
    for document in documents:
        if isinstance(document, Mapping):
            text = str(document.get("page_content") or document.get("text") or "").strip()
        else:
            text = str(getattr(document, "page_content", "") or getattr(document, "text", "")).strip()
        if text:
            contexts.append(text)
    return contexts


def retrieved_context_ids_from_output(output: Mapping[str, Any]) -> list[str]:
    return _ids_from_output(output, direct_key="retrieved_context_ids", metadata_fields=("file_id",))


def retrieved_chunk_ids_from_output(output: Mapping[str, Any]) -> list[str]:
    return _ids_from_output(output, direct_key="retrieved_chunk_ids", metadata_fields=CHUNK_ID_METADATA_FIELDS)


def preferred_retrieved_context_ids(output: Mapping[str, Any]) -> list[str]:
    """Chunk-level retrieved IDs when available, else legacy file-level IDs."""
    return retrieved_chunk_ids_from_output(output) or retrieved_context_ids_from_output(output)


def _ids_from_output(
    output: Mapping[str, Any],
    *,
    direct_key: str,
    metadata_fields: tuple[str, ...],
) -> list[str]:
    direct_ids = output.get(direct_key)
    if direct_ids:
        return _clean_context_ids(direct_ids)

    documents = output.get("documents") or output.get("evidence") or []
    ids: list[str] = []
    for document in documents:
        metadata = _document_metadata(document)
        for field in metadata_fields:
            value = str(metadata.get(field) or "").strip()
            if value:
                ids.append(value)
                break
    return _clean_context_ids(ids)


def build_faithfulness_scorer(provider_settings: QwenProviderSettings | None = None) -> Any:
    settings = provider_settings or QwenProviderSettings.from_env()
    try:
        api_key = settings.require_api_key()
    except MissingDashScopeKeyError as exc:
        raise SystemExit(
            "DASHSCOPE_API_KEY is required to run Ragas Faithfulness. "
            "Set DASHSCOPE_API_KEY and rerun the eval command."
        ) from exc

    AsyncOpenAI = _import_async_openai()
    llm_factory = _import_llm_factory()
    Faithfulness = _import_faithfulness_metric()

    client = AsyncOpenAI(api_key=api_key, base_url=settings.compat_base_url)
    llm = llm_factory(settings.chat_model, client=client, provider="openai")
    return Faithfulness(llm=llm)


def build_answer_relevancy_scorer(provider_settings: QwenProviderSettings | None = None) -> Any:
    settings = provider_settings or QwenProviderSettings.from_env()
    try:
        api_key = settings.require_api_key()
    except MissingDashScopeKeyError as exc:
        raise SystemExit(
            "DASHSCOPE_API_KEY is required to run Ragas Answer Relevancy. "
            "Set DASHSCOPE_API_KEY and rerun the eval command."
        ) from exc

    AsyncOpenAI = _import_async_openai()
    llm_factory = _import_llm_factory()
    embedding_factory = _import_embedding_factory()
    AnswerRelevancy = _import_answer_relevancy_metric()

    client = AsyncOpenAI(api_key=api_key, base_url=settings.compat_base_url)
    llm = llm_factory(settings.chat_model, client=client, provider="openai")
    embeddings = embedding_factory("openai", model=settings.embedding_model, client=client)
    return AnswerRelevancy(llm=llm, embeddings=embeddings)


def build_id_context_recall_scorer() -> Any:
    return _import_id_based_context_recall_metric()()


def score_faithfulness_for_phoenix(
    input: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
    scorer: Any | None = None,
) -> dict[str, Any]:
    return _run_coroutine(score_faithfulness_for_phoenix_async(input=input, output=output, scorer=scorer))


async def score_faithfulness_for_phoenix_async(
    input: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
    scorer: Any | None = None,
) -> dict[str, Any]:
    row = faithfulness_row_from_run_output(input, output)
    if row is None:
        return _skipped_result("missing_response_or_contexts")
    return await score_faithfulness_row_async(row, scorer=scorer)


def score_answer_relevancy_for_phoenix(
    input: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
    scorer: Any | None = None,
) -> dict[str, Any]:
    return _run_coroutine(score_answer_relevancy_for_phoenix_async(input=input, output=output, scorer=scorer))


async def score_answer_relevancy_for_phoenix_async(
    input: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
    scorer: Any | None = None,
) -> dict[str, Any]:
    row = answer_relevancy_row_from_run_output(input, output)
    if row is None:
        return _answer_relevancy_skipped_result("missing_user_input_or_response")
    return await score_answer_relevancy_row_async(row, scorer=scorer)


def score_id_context_recall_for_phoenix(
    input: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
    expected: Mapping[str, Any] | None = None,
    scorer: Any | None = None,
) -> dict[str, Any]:
    return _run_coroutine(
        score_id_context_recall_for_phoenix_async(input=input, output=output, expected=expected, scorer=scorer)
    )


async def score_id_context_recall_for_phoenix_async(
    input: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
    expected: Mapping[str, Any] | None = None,
    scorer: Any | None = None,
) -> dict[str, Any]:
    resolved_input = input or {}
    resolved_output = output or {}
    resolved_expected = expected or resolved_input
    reference_context_ids = _clean_context_ids(resolved_expected.get("reference_context_ids") or [])
    retrieved_context_ids = preferred_retrieved_context_ids(resolved_output)
    if not reference_context_ids:
        return _id_context_recall_not_applicable_result(
            "missing_reference_context_ids",
            retrieved_context_count=len(retrieved_context_ids),
            reference_context_count=0,
        )
    row = {
        "user_input": str(resolved_input.get("question") or resolved_input.get("user_input") or ""),
        "retrieved_context_ids": retrieved_context_ids,
        "reference_context_ids": reference_context_ids,
    }
    return await score_id_context_recall_row_async(row, scorer=scorer)


def score_faithfulness_row(row: Mapping[str, Any], scorer: Any | None = None) -> dict[str, Any]:
    return _run_coroutine(score_faithfulness_row_async(row, scorer=scorer))


async def score_faithfulness_row_async(row: Mapping[str, Any], scorer: Any | None = None) -> dict[str, Any]:
    retrieved_contexts = _clean_texts(row.get("retrieved_contexts") or [])
    user_input = str(row.get("user_input") or "").strip()
    response = str(row.get("response") or "").strip()
    if not user_input or not response or not retrieved_contexts:
        return _skipped_result("missing_response_or_contexts")

    resolved_scorer = scorer or build_faithfulness_scorer()
    raw_result = await _score_with_ragas_async(
        resolved_scorer,
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
    )
    score = _coerce_score_value(raw_result)
    return {
        "score": score,
        "label": FAITHFULNESS_LABEL,
        "explanation": _result_explanation(raw_result),
        "metadata": {
            "metric": FAITHFULNESS_METADATA_KEY,
            "retrieved_context_count": len(retrieved_contexts),
        },
    }


def score_answer_relevancy_row(row: Mapping[str, Any], scorer: Any | None = None) -> dict[str, Any]:
    return _run_coroutine(score_answer_relevancy_row_async(row, scorer=scorer))


async def score_answer_relevancy_row_async(row: Mapping[str, Any], scorer: Any | None = None) -> dict[str, Any]:
    user_input = str(row.get("user_input") or "").strip()
    response = str(row.get("response") or "").strip()
    if not user_input or not response:
        return _answer_relevancy_skipped_result("missing_user_input_or_response")

    resolved_scorer = scorer or build_answer_relevancy_scorer()
    raw_result = await _score_answer_relevancy_with_ragas_async(
        resolved_scorer,
        user_input=user_input,
        response=response,
    )
    score = _coerce_score_value(raw_result)
    return {
        "score": score,
        "label": ANSWER_RELEVANCY_LABEL,
        "explanation": _result_explanation(raw_result),
        "metadata": {"metric": ANSWER_RELEVANCY_METADATA_KEY},
    }


def score_id_context_recall_row(row: Mapping[str, Any], scorer: Any | None = None) -> dict[str, Any]:
    return _run_coroutine(score_id_context_recall_row_async(row, scorer=scorer))


async def score_id_context_recall_row_async(row: Mapping[str, Any], scorer: Any | None = None) -> dict[str, Any]:
    retrieved_context_ids = _clean_context_ids(row.get("retrieved_context_ids") or [])
    reference_context_ids = _clean_context_ids(row.get("reference_context_ids") or [])
    if not reference_context_ids:
        return _id_context_recall_skipped_result(
            "missing_reference_context_ids",
            retrieved_context_count=len(retrieved_context_ids),
            reference_context_count=0,
        )
    if not retrieved_context_ids:
        return _id_context_recall_skipped_result(
            "missing_retrieved_context_ids",
            retrieved_context_count=0,
            reference_context_count=len(reference_context_ids),
        )

    resolved_scorer = scorer or build_id_context_recall_scorer()
    raw_result = await _score_id_context_recall_with_ragas_async(
        resolved_scorer,
        retrieved_context_ids=retrieved_context_ids,
        reference_context_ids=reference_context_ids,
    )
    score = _coerce_score_value(raw_result)
    return {
        "score": score,
        "label": ID_CONTEXT_RECALL_LABEL,
        "explanation": _result_explanation(raw_result),
        "metadata": {
            "metric": ID_CONTEXT_RECALL_METADATA_KEY,
            "retrieved_context_id_count": len(retrieved_context_ids),
            "reference_context_id_count": len(reference_context_ids),
        },
    }


def evaluate_faithfulness_rows(
    rows: Sequence[Mapping[str, Any]],
    scorer: Any | None = None,
    *,
    concurrency: int = DEFAULT_RAGAS_CONCURRENCY,
) -> list[dict[str, Any]]:
    return _run_coroutine(evaluate_faithfulness_rows_async(rows, scorer=scorer, concurrency=concurrency))


async def evaluate_faithfulness_rows_async(
    rows: Sequence[Mapping[str, Any]],
    scorer: Any | None = None,
    *,
    concurrency: int = DEFAULT_RAGAS_CONCURRENCY,
) -> list[dict[str, Any]]:
    resolved_scorer = scorer
    if resolved_scorer is None and any(_has_scoreable_fields(row) for row in rows):
        resolved_scorer = build_faithfulness_scorer()

    async def evaluate(row: Mapping[str, Any]) -> dict[str, Any]:
        result = await score_faithfulness_row_async(row, scorer=resolved_scorer)
        return {
            "user_input": str(row.get("user_input") or ""),
            "faithfulness": result["score"],
            "label": result["label"],
            "explanation": result.get("explanation"),
            "retrieved_context_count": result["metadata"].get("retrieved_context_count", 0),
        }

    return await _map_rows_bounded(rows, evaluate, concurrency=concurrency)


def evaluate_answer_relevancy_rows(
    rows: Sequence[Mapping[str, Any]],
    scorer: Any | None = None,
    *,
    concurrency: int = DEFAULT_RAGAS_CONCURRENCY,
) -> list[dict[str, Any]]:
    return _run_coroutine(evaluate_answer_relevancy_rows_async(rows, scorer=scorer, concurrency=concurrency))


async def evaluate_answer_relevancy_rows_async(
    rows: Sequence[Mapping[str, Any]],
    scorer: Any | None = None,
    *,
    concurrency: int = DEFAULT_RAGAS_CONCURRENCY,
) -> list[dict[str, Any]]:
    resolved_scorer = scorer
    if resolved_scorer is None and any(_has_scoreable_answer_relevancy_fields(row) for row in rows):
        resolved_scorer = build_answer_relevancy_scorer()

    async def evaluate(row: Mapping[str, Any]) -> dict[str, Any]:
        result = await score_answer_relevancy_row_async(row, scorer=resolved_scorer)
        return {
            "user_input": str(row.get("user_input") or ""),
            ANSWER_RELEVANCY_LABEL: result["score"],
            "label": result["label"],
            "explanation": result.get("explanation"),
        }

    return await _map_rows_bounded(rows, evaluate, concurrency=concurrency)


def evaluate_id_context_recall_rows(
    rows: Sequence[Mapping[str, Any]],
    scorer: Any | None = None,
    *,
    concurrency: int = DEFAULT_RAGAS_CONCURRENCY,
) -> list[dict[str, Any]]:
    return _run_coroutine(evaluate_id_context_recall_rows_async(rows, scorer=scorer, concurrency=concurrency))


async def evaluate_id_context_recall_rows_async(
    rows: Sequence[Mapping[str, Any]],
    scorer: Any | None = None,
    *,
    concurrency: int = DEFAULT_RAGAS_CONCURRENCY,
) -> list[dict[str, Any]]:
    resolved_scorer = scorer
    if resolved_scorer is None and any(_has_scoreable_id_context_fields(row) for row in rows):
        resolved_scorer = build_id_context_recall_scorer()

    async def evaluate(row: Mapping[str, Any]) -> dict[str, Any]:
        result = await score_id_context_recall_row_async(row, scorer=resolved_scorer)
        metadata = result["metadata"]
        return {
            "user_input": str(row.get("user_input") or ""),
            ID_CONTEXT_RECALL_LABEL: result["score"],
            "label": result["label"],
            "explanation": result.get("explanation"),
            "retrieved_context_id_count": metadata.get("retrieved_context_id_count", 0),
            "reference_context_id_count": metadata.get("reference_context_id_count", 0),
        }

    return await _map_rows_bounded(rows, evaluate, concurrency=concurrency)


def _score_with_ragas(
    scorer: Any,
    *,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
) -> Any:
    return _run_coroutine(
        _score_with_ragas_async(
            scorer,
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
    )


async def _score_with_ragas_async(
    scorer: Any,
    *,
    user_input: str,
    response: str,
    retrieved_contexts: list[str],
) -> Any:
    kwargs = {
        "user_input": user_input,
        "response": response,
        "retrieved_contexts": retrieved_contexts,
    }
    if hasattr(scorer, "ascore"):
        return await _resolve_awaitable_async(scorer.ascore(**kwargs))
    if hasattr(scorer, "single_turn_ascore"):
        sample = _import_single_turn_sample()(**kwargs)
        return await _resolve_awaitable_async(scorer.single_turn_ascore(sample))
    if hasattr(scorer, "score"):
        result = await run_sync_in_worker_thread(lambda: scorer.score(**kwargs))
        return await _resolve_awaitable_async(result)
    raise TypeError("Ragas Faithfulness scorer does not expose score/ascore methods.")


def _score_answer_relevancy_with_ragas(
    scorer: Any,
    *,
    user_input: str,
    response: str,
) -> Any:
    return _run_coroutine(
        _score_answer_relevancy_with_ragas_async(
            scorer,
            user_input=user_input,
            response=response,
        )
    )


async def _score_answer_relevancy_with_ragas_async(
    scorer: Any,
    *,
    user_input: str,
    response: str,
) -> Any:
    kwargs = {
        "user_input": user_input,
        "response": response,
    }
    if hasattr(scorer, "ascore"):
        return await _resolve_awaitable_async(scorer.ascore(**kwargs))
    if hasattr(scorer, "single_turn_ascore"):
        sample = _import_single_turn_sample()(**kwargs)
        return await _resolve_awaitable_async(scorer.single_turn_ascore(sample))
    if hasattr(scorer, "score"):
        result = await run_sync_in_worker_thread(lambda: scorer.score(**kwargs))
        return await _resolve_awaitable_async(result)
    raise TypeError("Ragas AnswerRelevancy scorer does not expose score/ascore methods.")


def _score_id_context_recall_with_ragas(
    scorer: Any,
    *,
    retrieved_context_ids: list[str],
    reference_context_ids: list[str],
) -> Any:
    return _run_coroutine(
        _score_id_context_recall_with_ragas_async(
            scorer,
            retrieved_context_ids=retrieved_context_ids,
            reference_context_ids=reference_context_ids,
        )
    )


async def _score_id_context_recall_with_ragas_async(
    scorer: Any,
    *,
    retrieved_context_ids: list[str],
    reference_context_ids: list[str],
) -> Any:
    sample = single_turn_sample_from_row(
        {
            "retrieved_context_ids": retrieved_context_ids,
            "reference_context_ids": reference_context_ids,
        }
    )
    if hasattr(scorer, "single_turn_ascore"):
        return await _resolve_awaitable_async(scorer.single_turn_ascore(sample))
    if hasattr(scorer, "score"):
        result = await run_sync_in_worker_thread(lambda: scorer.score(sample))
        return await _resolve_awaitable_async(result)
    raise TypeError("Ragas IDBasedContextRecall scorer does not expose single_turn_ascore/score methods.")


def single_turn_sample_from_row(row: Mapping[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for source_key, target_key in (
        ("user_input", "user_input"),
        ("response", "response"),
        ("reference", "reference"),
    ):
        value = str(row.get(source_key) or "").strip()
        if value:
            kwargs[target_key] = value

    retrieved_contexts = _clean_texts(row.get("retrieved_contexts") or [])
    if retrieved_contexts:
        kwargs["retrieved_contexts"] = retrieved_contexts
    retrieved_context_ids = _clean_context_ids(row.get("retrieved_context_ids") or [])
    if retrieved_context_ids:
        kwargs["retrieved_context_ids"] = retrieved_context_ids
    reference_context_ids = _clean_context_ids(row.get("reference_context_ids") or [])
    if reference_context_ids:
        kwargs["reference_context_ids"] = reference_context_ids

    return _import_single_turn_sample()(**kwargs)


def evaluation_dataset_from_rows(rows: Sequence[Mapping[str, Any]]) -> Any:
    return _import_evaluation_dataset()(
        samples=[single_turn_sample_from_row(row) for row in rows],
    )


def _import_async_openai() -> Any:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit("OpenAI client is not installed; run `uv sync --extra dev`.") from exc
    return AsyncOpenAI


def _import_llm_factory() -> Any:
    _install_ragas_langchain_community_compat()
    try:
        from ragas.llms.base import llm_factory
    except ImportError as exc:
        try:
            from ragas.llms import llm_factory
        except ImportError:
            raise SystemExit("Ragas LLM factory is not installed; run `uv sync --extra dev`.") from exc
    return llm_factory


def _import_embedding_factory() -> Any:
    _install_ragas_langchain_community_compat()
    try:
        from ragas.embeddings.base import embedding_factory
    except ImportError as exc:
        raise SystemExit("Ragas embedding factory is not installed; run `uv sync --extra dev`.") from exc
    return embedding_factory


def _import_faithfulness_metric() -> Any:
    _install_ragas_langchain_community_compat()
    try:
        from ragas.metrics.collections.faithfulness import Faithfulness as CollectionsFaithfulness

        return CollectionsFaithfulness
    except ImportError as collections_exc:
        try:
            from ragas.metrics.collections import Faithfulness as CollectionFaithfulness

            return CollectionFaithfulness
        except ImportError:
            try:
                from ragas.metrics import Faithfulness as MetricsFaithfulness

                return MetricsFaithfulness
            except ImportError:
                raise SystemExit("Ragas Faithfulness metric is not installed; run `uv sync --extra dev`.") from (
                    collections_exc
                )


def _import_answer_relevancy_metric() -> Any:
    _install_ragas_langchain_community_compat()
    try:
        from ragas.metrics.collections import AnswerRelevancy as CollectionsAnswerRelevancy

        return CollectionsAnswerRelevancy
    except ImportError as collections_exc:
        try:
            from ragas.metrics import AnswerRelevancy as MetricsAnswerRelevancy

            return MetricsAnswerRelevancy
        except ImportError:
            raise SystemExit("Ragas AnswerRelevancy metric is not installed; run `uv sync --extra dev`.") from (
                collections_exc
            )


def _import_id_based_context_recall_metric() -> Any:
    _install_ragas_langchain_community_compat()
    try:
        return getattr(importlib.import_module("ragas.metrics.collections"), "IDBasedContextRecall")
    except (AttributeError, ImportError) as collections_exc:
        try:
            return getattr(importlib.import_module("ragas.metrics._context_recall"), "IDBasedContextRecall")
        except (AttributeError, ImportError):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="Importing IDBasedContextRecall from 'ragas.metrics' is deprecated.*",
                        category=DeprecationWarning,
                    )
                    return getattr(importlib.import_module("ragas.metrics"), "IDBasedContextRecall")
            except (AttributeError, ImportError):
                raise SystemExit(
                    "Ragas IDBasedContextRecall metric is not installed; run `uv sync --extra dev`."
                ) from collections_exc


def _import_single_turn_sample() -> Any:
    _install_ragas_langchain_community_compat()
    try:
        from ragas.dataset_schema import SingleTurnSample
    except ImportError as exc:
        raise SystemExit("Ragas SingleTurnSample is not installed; run `uv sync --extra dev`.") from exc
    return SingleTurnSample


def _import_evaluation_dataset() -> Any:
    _install_ragas_langchain_community_compat()
    try:
        from ragas import EvaluationDataset
    except ImportError as exc:
        raise SystemExit("Ragas EvaluationDataset is not installed; run `uv sync --extra dev`.") from exc
    return EvaluationDataset


def _install_ragas_langchain_community_compat() -> None:
    module_name = "langchain_community.chat_models.vertexai"
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="`langchain-community` is being sunset.*",
                category=DeprecationWarning,
            )
            importlib.import_module(module_name)
        return
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise

    module = types.ModuleType(module_name)

    class ChatVertexAI:
        pass

    setattr(module, "ChatVertexAI", ChatVertexAI)
    sys.modules[module_name] = module


def _clean_texts(values: Any) -> list[str]:
    if isinstance(values, str):
        raw_values = [values]
    else:
        raw_values = list(values or [])
    texts: list[str] = []
    for value in raw_values:
        text = str(value).strip()
        if text:
            texts.append(text)
    return texts


def _clean_context_ids(values: Any) -> list[str]:
    if values is None:
        raw_values: list[Any] = []
    elif isinstance(values, str) or not isinstance(values, Sequence):
        raw_values = [values]
    else:
        raw_values = list(values)
    return unique_nonempty(raw_values)


def _has_scoreable_fields(row: Mapping[str, Any]) -> bool:
    return bool(str(row.get("user_input") or "").strip()) and bool(str(row.get("response") or "").strip()) and bool(
        _clean_texts(row.get("retrieved_contexts") or [])
    )


def _has_scoreable_answer_relevancy_fields(row: Mapping[str, Any]) -> bool:
    return bool(str(row.get("user_input") or "").strip()) and bool(str(row.get("response") or "").strip())


def _has_scoreable_id_context_fields(row: Mapping[str, Any]) -> bool:
    return bool(_clean_context_ids(row.get("retrieved_context_ids") or [])) and bool(
        _clean_context_ids(row.get("reference_context_ids") or [])
    )


async def _map_rows_bounded(
    rows: Sequence[Mapping[str, Any]],
    evaluator: Any,
    *,
    concurrency: int,
) -> list[dict[str, Any]]:
    if concurrency < 1:
        raise ValueError("Ragas concurrency must be at least 1.")
    if not rows:
        return []

    limiter = anyio.Semaphore(concurrency)
    results: list[dict[str, Any] | None] = [None] * len(rows)

    async def run_one(index: int, row: Mapping[str, Any]) -> None:
        async with limiter:
            result = await evaluator(row)
            if result is not None:
                results[index] = dict(result)

    async with anyio.create_task_group() as task_group:
        for index, row in enumerate(rows):
            task_group.start_soon(run_one, index, row)

    filled_results: list[dict[str, Any]] = []
    for result in results:
        if result is None:
            raise AssertionError("Ragas row evaluator did not fill all result slots.")
        filled_results.append(result)
    return filled_results


def _document_metadata(document: Any) -> Mapping[str, Any]:
    if isinstance(document, Mapping):
        metadata = document.get("metadata") or {}
    else:
        metadata = getattr(document, "metadata", {}) or {}
    return metadata if isinstance(metadata, Mapping) else {}


def _coerce_score_value(result: Any) -> float | None:
    value = getattr(result, "value", result)
    if value is None:
        return None
    score = float(value)
    if math.isnan(score):
        return None
    return score


def _result_explanation(result: Any) -> str | None:
    reason = getattr(result, "reason", None)
    if reason:
        return str(reason)
    explanation = getattr(result, "explanation", None)
    if explanation:
        return str(explanation)
    return None


def _skipped_result(reason: str) -> dict[str, Any]:
    return {
        "score": None,
        "label": "skipped",
        "explanation": "Ragas Faithfulness requires a non-empty response and retrieved contexts.",
        "metadata": {"metric": FAITHFULNESS_METADATA_KEY, "reason": reason, "retrieved_context_count": 0},
    }


def _answer_relevancy_skipped_result(reason: str) -> dict[str, Any]:
    return {
        "score": None,
        "label": "skipped",
        "explanation": "Ragas Answer Relevancy requires a non-empty user input and response.",
        "metadata": {"metric": ANSWER_RELEVANCY_METADATA_KEY, "reason": reason},
    }


def _id_context_recall_skipped_result(
    reason: str,
    *,
    retrieved_context_count: int,
    reference_context_count: int,
) -> dict[str, Any]:
    if reason == "missing_reference_context_ids":
        explanation = "Ragas ID context recall requires reference_context_ids."
    else:
        explanation = "Ragas ID context recall requires retrieved_context_ids."
    return {
        "score": None,
        "label": "skipped",
        "explanation": explanation,
        "metadata": {
            "metric": ID_CONTEXT_RECALL_METADATA_KEY,
            "reason": reason,
            "retrieved_context_id_count": retrieved_context_count,
            "reference_context_id_count": reference_context_count,
        },
    }


def _id_context_recall_not_applicable_result(
    reason: str,
    *,
    retrieved_context_count: int,
    reference_context_count: int,
) -> dict[str, Any]:
    return {
        "score": None,
        "label": "not_applicable",
        "explanation": "Ragas ID context recall is only applicable to rows with reference_context_ids.",
        "metadata": {
            "metric": ID_CONTEXT_RECALL_METADATA_KEY,
            "reason": reason,
            "retrieved_context_id_count": retrieved_context_count,
            "reference_context_id_count": reference_context_count,
        },
    }


def _resolve_awaitable(result: Any) -> Any:
    if inspect.isawaitable(result):
        return _run_coroutine(result)
    return result


async def _resolve_awaitable_async(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result


def _run_coroutine(awaitable: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return anyio.run(_await_result, awaitable)
    if not loop.is_running():
        return loop.run_until_complete(awaitable)
    if hasattr(awaitable, "close"):
        awaitable.close()
    raise RuntimeError("Cannot synchronously resolve a Ragas awaitable inside a running event loop; use the async variant.")


async def _await_result(awaitable: Any) -> Any:
    return await awaitable
