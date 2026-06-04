from __future__ import annotations

import asyncio
import importlib
import inspect
import math
import sys
import threading
import types
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

from imperial_rag.providers import MissingDashScopeKeyError, QwenProviderSettings


FAITHFULNESS_LABEL = "faithfulness"
FAITHFULNESS_METADATA_KEY = "ragas_faithfulness"
DEFAULT_RAGAS_METRICS = ("faithfulness",)
REFERENCE_REQUIRED_RAGAS_METRICS = {"context_recall", "factual_correctness"}
SUPPORTED_RAGAS_METRICS = DEFAULT_RAGAS_METRICS + ("context_recall", "factual_correctness")
NO_RAGAS_METRIC_ALIASES = {"none", "no", "off", "false", "0"}


def parse_ragas_metric_names(
    raw_metrics: str | None,
    *,
    default: Sequence[str] = DEFAULT_RAGAS_METRICS,
    allow_none: bool = False,
) -> list[str]:
    if raw_metrics is None or not raw_metrics.strip():
        return list(default)
    names = [name.strip().casefold().replace("-", "_") for name in raw_metrics.split(",") if name.strip()]
    if allow_none and any(name in NO_RAGAS_METRIC_ALIASES for name in names):
        return []
    unsupported = sorted(set(names) - set(SUPPORTED_RAGAS_METRICS))
    if unsupported:
        supported_names = list(SUPPORTED_RAGAS_METRICS)
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
    retrieved_contexts = retrieved_contexts_from_output(resolved_output)
    if not user_input or not response or not retrieved_contexts:
        return None
    return {
        "user_input": user_input,
        "response": response,
        "retrieved_contexts": retrieved_contexts,
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


def score_faithfulness_for_phoenix(
    input: Mapping[str, Any] | None = None,
    output: Mapping[str, Any] | None = None,
    scorer: Any | None = None,
) -> dict[str, Any]:
    row = faithfulness_row_from_run_output(input, output)
    if row is None:
        return _skipped_result("missing_response_or_contexts")
    return score_faithfulness_row(row, scorer=scorer)


def score_faithfulness_row(row: Mapping[str, Any], scorer: Any | None = None) -> dict[str, Any]:
    retrieved_contexts = _clean_texts(row.get("retrieved_contexts") or [])
    user_input = str(row.get("user_input") or "").strip()
    response = str(row.get("response") or "").strip()
    if not user_input or not response or not retrieved_contexts:
        return _skipped_result("missing_response_or_contexts")

    resolved_scorer = scorer or build_faithfulness_scorer()
    raw_result = _score_with_ragas(
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


def evaluate_faithfulness_rows(rows: Sequence[Mapping[str, Any]], scorer: Any | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    resolved_scorer = scorer
    for row in rows:
        if _has_scoreable_fields(row) and resolved_scorer is None:
            resolved_scorer = build_faithfulness_scorer()
        result = score_faithfulness_row(row, scorer=resolved_scorer)
        records.append(
            {
                "user_input": str(row.get("user_input") or ""),
                "faithfulness": result["score"],
                "label": result["label"],
                "explanation": result.get("explanation"),
                "retrieved_context_count": result["metadata"].get("retrieved_context_count", 0),
            }
        )
    return records


def _score_with_ragas(
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
    if hasattr(scorer, "score"):
        result = scorer.score(**kwargs)
        return _resolve_awaitable(result)
    if hasattr(scorer, "ascore"):
        return _run_coroutine(scorer.ascore(**kwargs))
    if hasattr(scorer, "single_turn_ascore"):
        sample = _import_single_turn_sample()(**kwargs)
        return _run_coroutine(scorer.single_turn_ascore(sample))
    raise TypeError("Ragas Faithfulness scorer does not expose score/ascore methods.")


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


def _import_faithfulness_metric() -> Any:
    _install_ragas_langchain_community_compat()
    try:
        from ragas.metrics.collections.faithfulness import Faithfulness

        return Faithfulness
    except ImportError as collections_exc:
        try:
            from ragas.metrics.collections import Faithfulness

            return Faithfulness
        except ImportError:
            try:
                from ragas.metrics import Faithfulness

                return Faithfulness
            except ImportError:
                raise SystemExit("Ragas Faithfulness metric is not installed; run `uv sync --extra dev`.") from (
                    collections_exc
                )


def _import_single_turn_sample() -> Any:
    _install_ragas_langchain_community_compat()
    try:
        from ragas.dataset_schema import SingleTurnSample
    except ImportError as exc:
        raise SystemExit("Ragas SingleTurnSample is not installed; run `uv sync --extra dev`.") from exc
    return SingleTurnSample


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

    module.ChatVertexAI = ChatVertexAI
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


def _has_scoreable_fields(row: Mapping[str, Any]) -> bool:
    return bool(str(row.get("user_input") or "").strip()) and bool(str(row.get("response") or "").strip()) and bool(
        _clean_texts(row.get("retrieved_contexts") or [])
    )


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


def _resolve_awaitable(result: Any) -> Any:
    if inspect.isawaitable(result):
        return _run_coroutine(result)
    return result


def _run_coroutine(awaitable: Any) -> Any:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    if not loop.is_running():
        return loop.run_until_complete(awaitable)

    result_box: dict[str, Any] = {}

    def run_in_thread() -> None:
        try:
            result_box["result"] = asyncio.run(awaitable)
        except BaseException as exc:  # pragma: no cover - defensive cross-thread propagation
            result_box["error"] = exc

    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()
    if "error" in result_box:
        raise result_box["error"]
    return result_box.get("result")
