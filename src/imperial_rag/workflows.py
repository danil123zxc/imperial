from __future__ import annotations

import inspect
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from imperial_rag.answering import (
    REFUSAL_TEXT,
    STRICT_SYSTEM_PROMPT,
    build_evidence_prompt,
    build_strict_answer_chain,
    format_citations,
    format_sources,
    validate_citations,
)
from imperial_rag.document_ids import metadata_or_content_id
from imperial_rag.retrieval import CandidateMerger
from imperial_rag.tracing import imperial_trace_attributes, trace_answer_step, trace_llm_step

PROMPT_VERSION = "strict-rag-v1"
_EVIDENCE_PROMPT_SKELETON = """You are answering questions about internal company documents.
Use only the evidence below.
Do not use general model knowledge.
Every meaningful factual claim must cite a source from the evidence.
Use the short source labels exactly as shown, for example [S1].
Do not include uncited introductions or summaries.
If the evidence is insufficient, answer exactly: {refusal_text}

Question:
{question}

Evidence:
{evidence}
"""
_EVIDENCE_PROMPT_SKELETON_HASH = hashlib.sha256(_EVIDENCE_PROMPT_SKELETON.encode("utf-8")).hexdigest()


class VectorSearch(Protocol):
    def similarity_search(self, query: str, k: int) -> list[Document]:
        ...


class KeywordSearch(Protocol):
    def search(self, query: str, limit: int = 5) -> list[Document]:
        ...


class ChatModel(Protocol):
    def invoke(self, messages):
        ...


class QueryState(TypedDict, total=False):
    question: str
    normalized_query: str
    vector_candidates: list[Document]
    keyword_candidates: list[Document]
    evidence: list[Document]
    retrieved_documents: list[Document]
    answer: str
    citations: list[str]
    sources: list[str]
    citations_valid: bool
    invalid_citations: list[str]
    retrieval: dict[str, Any]


def _legacy_openai_chat_model():
    from imperial_rag.providers import QwenProviderSettings

    if not QwenProviderSettings.from_env().allow_legacy_openai:
        raise RuntimeError(
            "Legacy OpenAI chat is disabled. Use Qwen provider defaults or set "
            "IMPERIAL_RAG_ALLOW_LEGACY_OPENAI=true."
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-4.1-mini", temperature=0)


def _document_key(document: Document) -> str:
    metadata = document.metadata or {}
    return metadata_or_content_id(metadata.get("citation_id"), metadata.get("chunk_id"), content=document.page_content)


def _content_key(document: Document) -> str:
    return " ".join(document.page_content.split()).casefold()


def _contains_query_terms(query: str, text: str) -> bool:
    normalized_text = text.casefold()
    return all(term in normalized_text for term in query.casefold().split() if term)


def rank_hybrid_candidates(
    query: str,
    vector_docs: list[Document],
    keyword_docs: list[Document],
    limit: int = 12,
    k: int | None = None,
) -> list[Document]:
    if k is not None:
        limit = k
    candidates = CandidateMerger().merge(vector_docs, keyword_docs)
    keyword_keys = {_document_key(document) for document in keyword_docs}
    keyword_contents = {_content_key(document) for document in keyword_docs}

    def score(document: Document) -> tuple[int, int]:
        searchable = " ".join(
            [
                document.page_content,
                str(document.metadata.get("file_name", "")),
                str(document.metadata.get("relative_path", "")),
                str(document.metadata.get("section_heading", "")),
                str(document.metadata.get("source_type", "")),
            ]
        )
        exact_boost = 1 if _contains_query_terms(query, searchable) else 0
        keyword_boost = 1 if _document_key(document) in keyword_keys or _content_key(document) in keyword_contents else 0
        return exact_boost, keyword_boost

    return sorted(candidates, key=score, reverse=True)[:limit]


def _documents_from_first_key(
    retrieved: Mapping[str, Any],
    keys: Sequence[str],
    default: Sequence[Document],
) -> list[Document]:
    for key in keys:
        if key in retrieved:
            value = retrieved[key]
            return [] if value is None else list(value)
    return list(default)


def _coerce_retrieved_documents(retrieved: Any, query: str) -> list[Document]:
    if retrieved is None:
        return []
    if isinstance(retrieved, Mapping):
        for key in ("retrieved_documents", "documents", "docs", "evidence"):
            if key in retrieved:
                direct_docs = retrieved[key]
                return [] if direct_docs is None else list(direct_docs)
        vector_docs = _documents_from_first_key(
            retrieved,
            ("vector_docs", "vector_documents", "vector_candidates"),
            [],
        )
        keyword_docs = _documents_from_first_key(
            retrieved,
            ("keyword_docs", "keyword_documents", "keyword_candidates"),
            [],
        )
        return rank_hybrid_candidates(query, vector_docs, keyword_docs)
    if isinstance(retrieved, tuple) and len(retrieved) == 2:
        return rank_hybrid_candidates(query, list(retrieved[0]), list(retrieved[1]))
    return list(retrieved)


def _coerce_answer(answer: Any) -> str:
    if isinstance(answer, Mapping) and "answer" in answer:
        return str(answer["answer"])
    content = getattr(answer, "content", None)
    if content is not None:
        return str(content)
    return str(answer)


def _coerce_trace_attributes(answer: Any) -> dict[str, Any]:
    if not isinstance(answer, Mapping):
        return {}
    trace_attributes = answer.get("trace_attributes")
    if isinstance(trace_attributes, Mapping):
        return dict(trace_attributes)
    return {}


def _call_pipeline(run_pipeline, state: Mapping[str, Any]):
    try:
        signature = inspect.signature(run_pipeline)
    except (TypeError, ValueError):
        return run_pipeline(state)
    positional_count = sum(
        1
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    return run_pipeline(state) if positional_count else run_pipeline()


def build_query_workflow(
    vector_search: VectorSearch | None = None,
    keyword_search: KeywordSearch | None = None,
    chat_model: ChatModel | None = None,
    retrieve=None,
    generate=None,
):
    model = chat_model

    def normalize_query(state: QueryState) -> QueryState:
        return {"normalized_query": state["question"].strip()}

    def retrieve_node(state: QueryState) -> QueryState:
        query = state["normalized_query"]
        if retrieve is not None:
            retrieved = retrieve(query)
            evidence = _coerce_retrieved_documents(retrieved, query)
            update: QueryState = {
                "vector_candidates": (
                    _documents_from_first_key(retrieved, ("vector_docs", "vector_candidates"), evidence)
                    if isinstance(retrieved, Mapping)
                    else evidence
                ),
                "keyword_candidates": (
                    _documents_from_first_key(retrieved, ("keyword_docs", "keyword_candidates"), [])
                    if isinstance(retrieved, Mapping)
                    else []
                ),
                "evidence": evidence,
                "retrieved_documents": evidence,
            }
            if isinstance(retrieved, Mapping) and isinstance(retrieved.get("retrieval"), Mapping):
                update["retrieval"] = dict(retrieved["retrieval"])
            return update
        vector_docs = vector_search.similarity_search(query, k=8) if vector_search is not None else []
        keyword_docs = keyword_search.search(query, limit=8) if keyword_search is not None else []
        evidence = rank_hybrid_candidates(query, vector_docs, keyword_docs)
        return {
            "vector_candidates": vector_docs,
            "keyword_candidates": keyword_docs,
            "evidence": evidence,
            "retrieved_documents": evidence,
            "retrieval": {
                "vector_candidates": len(vector_docs),
                "keyword_candidates": len(keyword_docs),
                "merged_candidates": len(evidence),
                "reranked_candidates": len(evidence),
                "final_evidence": len(evidence),
                "reranker": "legacy:rank_hybrid_candidates",
                "fallbacks": [],
            },
        }

    def call_model(state: QueryState) -> QueryState:
        evidence = state.get("evidence", [])
        citations = format_citations(evidence)
        sources = format_sources(evidence)
        with trace_answer_step(
            "answer.generate",
            state["question"],
            attributes=imperial_trace_attributes(
                "answer",
                "generate",
                _answer_trace_attributes(evidence, citations),
            ),
        ) as span:
            with trace_answer_step(
                "answer.prepare_context",
                state["question"],
                attributes=imperial_trace_attributes(
                    "answer",
                    "prepare_context",
                    _answer_trace_attributes(evidence, citations),
                ),
            ) as context_span:
                context_span.set_output(_answer_context_trace_output(evidence, citations, sources))
            if not evidence:
                update: QueryState = {
                    "answer": REFUSAL_TEXT,
                    "citations": [],
                    "sources": [],
                    "citations_valid": True,
                    "invalid_citations": [],
                }
                _set_answer_trace_output(span, update, evidence_count=0, citation_count=0)
                return update
            with trace_llm_step(
                "answer.call_model",
                state["question"],
                attributes=imperial_trace_attributes(
                    "answer",
                    "call_model",
                    {"answer.evidence_count": len(evidence), "answer.citation_count": len(citations)},
                ),
            ) as model_span:
                _set_model_prompt_trace_attributes(model_span, state["question"], evidence, citations)
                if generate is not None:
                    generated = generate(state["question"], evidence)
                    answer = _coerce_answer(generated)
                    _set_model_generation_trace_attributes(model_span, generated)
                else:
                    resolved_model = model or _legacy_openai_chat_model()
                    _set_model_trace_attributes(model_span, resolved_model)
                    answer = build_strict_answer_chain(resolved_model).invoke(
                        {"evidence_prompt": build_evidence_prompt(state["question"], evidence)}
                    )
                    model_span.set_attribute("answer.model_status", "ok")
                _set_model_output_trace_attributes(model_span, answer)
                model_span.set_output(
                    {
                        "answer_chars": len(str(answer)),
                        "evidence_count": len(evidence),
                        "citation_count": len(citations),
                    }
                )
            with trace_answer_step(
                "answer.validate_citations",
                state["question"],
                attributes=imperial_trace_attributes(
                    "answer",
                    "validate_citations",
                    {"answer.evidence_count": len(evidence), "answer.citation_count": len(citations)},
                ),
            ) as validation_span:
                valid, invalid = validate_citations(answer, evidence)
                validation_span.set_output(
                    {
                        "citations_valid": valid,
                        "invalid_citations": invalid,
                        "evidence_count": len(evidence),
                        "citation_count": len(citations),
                    }
                )
            update = {
                "answer": answer,
                "citations": citations,
                "sources": sources,
                "citations_valid": valid,
                "invalid_citations": invalid,
            }
            _set_answer_trace_output(span, update, evidence_count=len(evidence), citation_count=len(citations))
            return update

    graph = StateGraph(QueryState)
    graph.add_node("normalize_query", normalize_query)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("call_model", call_model)
    graph.add_edge(START, "normalize_query")
    graph.add_edge("normalize_query", "retrieve")
    graph.add_edge("retrieve", "call_model")
    graph.add_edge("call_model", END)
    return graph.compile()


def _answer_trace_attributes(evidence: Sequence[Document], citations: Sequence[str]) -> dict[str, Any]:
    return {
        "answer.evidence_count": len(evidence),
        "answer.citation_count": len(citations),
        "answer.citation_ids": [
            str(document.metadata.get("citation_id"))
            for document in evidence
            if document.metadata.get("citation_id") is not None
        ],
        "answer.context_chars": sum(len(str(document.page_content)) for document in evidence),
    }


def _set_model_prompt_trace_attributes(
    span: Any,
    question: str,
    evidence: Sequence[Document],
    citations: Sequence[str],
) -> None:
    attrs = _answer_trace_attributes(evidence, citations)
    span.set_attribute("answer.prompt_version", PROMPT_VERSION)
    span.set_attribute("answer.prompt_skeleton_hash", _EVIDENCE_PROMPT_SKELETON_HASH)
    span.set_attribute("llm.invocation_parameters", {"temperature": 0})
    for key, value in _llm_input_message_trace_attributes(question, evidence).items():
        span.set_attribute(key, value)
    for key in ("answer.evidence_count", "answer.citation_count", "answer.citation_ids", "answer.context_chars"):
        span.set_attribute(key, attrs[key])


def _set_model_trace_attributes(span: Any, model: Any) -> None:
    model_name = (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or getattr(model, "_model_name", None)
        or getattr(model, "model_id", None)
    )
    if model_name:
        span.set_attribute("llm.model_name", str(model_name))
    provider = getattr(model, "lc_namespace", None)
    if callable(provider):
        namespace = ".".join(str(part) for part in provider())
        if namespace:
            span.set_attribute("llm.provider", namespace)


def _set_model_generation_trace_attributes(span: Any, generated: Any) -> None:
    for key, value in _coerce_trace_attributes(generated).items():
        span.set_attribute(str(key), value)


def _set_model_output_trace_attributes(span: Any, answer: Any) -> None:
    span.set_attribute("llm.output_messages.0.message.role", "assistant")
    span.set_attribute("llm.output_messages.0.message.content", str(answer))


def _llm_input_message_trace_attributes(question: str, evidence: Sequence[Document]) -> dict[str, Any]:
    return {
        "llm.input_messages.0.message.role": "system",
        "llm.input_messages.0.message.content": STRICT_SYSTEM_PROMPT,
        "llm.input_messages.1.message.role": "user",
        "llm.input_messages.1.message.content": _safe_user_prompt_trace_content(question, evidence),
    }


def _safe_user_prompt_trace_content(question: str, evidence: Sequence[Document]) -> str:
    return (
        f"Question:\n{question}\n\n"
        "Evidence:\n"
        f"<{len(evidence)} retrieved chunk(s) elided; inspect retrieval.select_evidence for source metadata.>"
    )


def _answer_context_trace_output(
    evidence: Sequence[Document],
    citations: Sequence[str],
    sources: Sequence[str],
) -> dict[str, Any]:
    return {
        "evidence_count": len(evidence),
        "citation_count": len(citations),
        "citation_ids": [
            str(document.metadata.get("citation_id"))
            for document in evidence
            if document.metadata.get("citation_id") is not None
        ],
        "source_count": len(sources),
        "context_chars": sum(len(str(document.page_content)) for document in evidence),
    }


def _set_answer_trace_output(span: Any, update: Mapping[str, Any], *, evidence_count: int, citation_count: int) -> None:
    span.set_output(
        {
            "answer": update.get("answer", ""),
            "citations_valid": update.get("citations_valid"),
            "invalid_citations": update.get("invalid_citations", []),
            "refused": update.get("answer") == REFUSAL_TEXT,
            "evidence_count": evidence_count,
            "citation_count": citation_count,
        }
    )


class IngestionState(TypedDict, total=False):
    settings: object
    ocr_client: object
    vector_store: object
    summary: object
    status: str
    counts: dict[str, int]


def build_ingestion_workflow(run_pipeline=None):
    def run_ingestion(state: IngestionState) -> IngestionState:
        if run_pipeline is not None:
            summary = _call_pipeline(run_pipeline, state)
        else:
            from imperial_rag.pipeline import ingest_corpus

            summary = ingest_corpus(
                settings=state["settings"],
                ocr_client=state.get("ocr_client"),
                vector_store=state.get("vector_store"),
            )
        counts = _counts_from_summary(summary)
        status = str(summary.get("status", "completed")) if isinstance(summary, Mapping) else str(getattr(summary, "status", "completed"))
        return {"summary": summary, "status": status, "counts": counts}

    graph = StateGraph(IngestionState)
    graph.add_node("run_ingestion", run_ingestion)
    graph.add_edge(START, "run_ingestion")
    graph.add_edge("run_ingestion", END)
    return graph.compile()


def _counts_from_summary(summary: Any) -> dict[str, int]:
    if isinstance(summary, Mapping):
        counts = summary.get("counts")
        if isinstance(counts, Mapping):
            return {str(key): int(value) for key, value in counts.items()}
        return {str(key): int(value) for key, value in summary.items() if isinstance(value, int)}
    counts: dict[str, int] = {}
    for source_name, target_name in (
        ("total_files", "files"),
        ("document_count", "documents"),
        ("documents", "documents"),
        ("chunk_count", "chunks"),
        ("chunks", "chunks"),
        ("indexed_count", "indexed"),
        ("indexed", "indexed"),
    ):
        value = getattr(summary, source_name, None)
        if isinstance(value, int):
            counts[target_name] = value
    return counts
