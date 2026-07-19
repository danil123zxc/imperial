from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from imperial_rag.answering.strict import (
    REFUSAL_TEXT,
    STRICT_SYSTEM_PROMPT,
    build_evidence_prompt,
    build_strict_answer_chain,
    format_citations,
    format_sources,
    validate_citations,
)
from imperial_rag.document_ids import content_key, document_key
from imperial_rag.retrieval.lexical import searchable_document_text
from imperial_rag.retrieval.service import CandidateMerger
from imperial_rag.observability.phoenix import imperial_trace_attributes, trace_answer_step, trace_llm_step

PROMPT_VERSION = "strict-rag-v2"
_EVIDENCE_PROMPT_SKELETON = """You are answering questions about internal company documents.
Use only the evidence below.
Do not use general model knowledge.
Every meaningful factual claim must cite a source from the evidence.
Use the short source labels exactly as shown, for example [S1].
Write each source label in its own brackets. For multiple sources, write [S1] [S4], never [S1, S4].
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
    error: dict[str, Any]


_RETRIEVED_DOCUMENT_KEYS = ("retrieved_documents", "documents", "docs", "evidence")
_VECTOR_CANDIDATE_KEYS = ("vector_docs", "vector_documents", "vector_candidates")
_KEYWORD_CANDIDATE_KEYS = ("keyword_docs", "keyword_documents", "keyword_candidates")


@dataclass(frozen=True)
class _CoercedRetrieval:
    evidence: list[Document]
    vector_candidates: list[Document]
    keyword_candidates: list[Document]
    retrieval: dict[str, Any] | None = None


def _legacy_openai_chat_model():
    from imperial_rag.integrations.dashscope import QwenProviderSettings

    if not QwenProviderSettings.from_env().allow_legacy_openai:
        raise RuntimeError(
            "Legacy OpenAI chat is disabled. Use Qwen provider defaults or set "
            "IMPERIAL_RAG_ALLOW_LEGACY_OPENAI=true."
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-4.1-mini", temperature=0)


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
    keyword_keys = {document_key(document) for document in keyword_docs}
    keyword_contents = {content_key(document) for document in keyword_docs}

    def score(document: Document) -> tuple[int, int]:
        searchable = searchable_document_text(document)
        exact_boost = 1 if _contains_query_terms(query, searchable) else 0
        keyword_boost = 1 if document_key(document) in keyword_keys or content_key(document) in keyword_contents else 0
        return exact_boost, keyword_boost

    return sorted(candidates, key=score, reverse=True)[:limit]


def _documents_from_first_key(
    retrieved: Mapping[str, Any],
    keys: Sequence[str],
    default: Sequence[Document],
) -> list[Document]:
    found, value = _first_mapping_value(retrieved, keys)
    return _documents_from_value(value) if found else list(default)


def _first_mapping_value(retrieved: Mapping[str, Any], keys: Sequence[str]) -> tuple[bool, Any]:
    for key in keys:
        if key in retrieved:
            return True, retrieved[key]
    return False, None


def _documents_from_value(value: Any) -> list[Document]:
    return [] if value is None else list(value)


def _coerce_retrieval(retrieved: Any, query: str) -> _CoercedRetrieval:
    if retrieved is None:
        return _CoercedRetrieval(evidence=[], vector_candidates=[], keyword_candidates=[])
    if isinstance(retrieved, Mapping):
        return _coerce_mapping_retrieval(retrieved, query)
    if isinstance(retrieved, tuple) and len(retrieved) == 2:
        evidence = rank_hybrid_candidates(query, list(retrieved[0]), list(retrieved[1]))
        return _CoercedRetrieval(evidence=evidence, vector_candidates=evidence, keyword_candidates=[])
    evidence = list(retrieved)
    return _CoercedRetrieval(evidence=evidence, vector_candidates=evidence, keyword_candidates=[])


def _coerce_mapping_retrieval(retrieved: Mapping[str, Any], query: str) -> _CoercedRetrieval:
    has_direct_evidence, direct_evidence = _first_mapping_value(retrieved, _RETRIEVED_DOCUMENT_KEYS)
    if has_direct_evidence:
        evidence = _documents_from_value(direct_evidence)
        vector_candidates = _documents_from_first_key(retrieved, _VECTOR_CANDIDATE_KEYS, evidence)
        keyword_candidates = _documents_from_first_key(retrieved, _KEYWORD_CANDIDATE_KEYS, [])
    else:
        vector_candidates = _documents_from_first_key(retrieved, _VECTOR_CANDIDATE_KEYS, [])
        keyword_candidates = _documents_from_first_key(retrieved, _KEYWORD_CANDIDATE_KEYS, [])
        evidence = rank_hybrid_candidates(query, vector_candidates, keyword_candidates)
    retrieval = dict(retrieved["retrieval"]) if isinstance(retrieved.get("retrieval"), Mapping) else None
    return _CoercedRetrieval(
        evidence=evidence,
        vector_candidates=vector_candidates,
        keyword_candidates=keyword_candidates,
        retrieval=retrieval,
    )

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


def _coerce_error(answer: Any) -> dict[str, Any] | None:
    if not isinstance(answer, Mapping):
        return None
    error = answer.get("error")
    return dict(error) if isinstance(error, Mapping) else None


def build_query_workflow(
    vector_search: VectorSearch | None = None,
    keyword_search: KeywordSearch | None = None,
    chat_model: ChatModel | None = None,
    retrieve=None,
    generate=None,
):
    model = chat_model

    def normalize_query(state: QueryState) -> QueryState:
        return {"normalized_query": str(state.get("question", "")).strip()}

    def retrieve_node(state: QueryState) -> QueryState:
        query = str(state.get("normalized_query") or state.get("question") or "")
        if retrieve is not None:
            retrieved = retrieve(query)
            coerced = _coerce_retrieval(retrieved, query)
            update: QueryState = {
                "vector_candidates": coerced.vector_candidates,
                "keyword_candidates": coerced.keyword_candidates,
                "evidence": coerced.evidence,
                "retrieved_documents": coerced.evidence,
            }
            if coerced.retrieval is not None:
                update["retrieval"] = coerced.retrieval
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
        question = str(state.get("question", ""))
        evidence = state.get("evidence", [])
        citations = format_citations(evidence)
        sources = format_sources(evidence)
        with trace_answer_step(
            "answer.generate",
            question,
            attributes=imperial_trace_attributes(
                "answer",
                "generate",
                _answer_trace_attributes(evidence, citations, source_count=len(sources)),
            ),
        ) as span:
            if not evidence:
                update: QueryState = {
                    "answer": REFUSAL_TEXT,
                    "citations": [],
                    "sources": [],
                    "citations_valid": True,
                    "invalid_citations": [],
                }
                _set_answer_trace_output(span, update, evidence=evidence, citations=citations, sources=sources)
                return update
            with trace_llm_step(
                "answer.call_model",
                question,
                attributes=imperial_trace_attributes(
                    "answer",
                    "call_model",
                    {"answer.evidence_count": len(evidence), "answer.citation_count": len(citations)},
                ),
            ) as model_span:
                _set_model_prompt_trace_attributes(model_span, question, evidence, citations)
                if generate is not None:
                    generated = generate(question, evidence)
                    answer = _coerce_answer(generated)
                    model_error = _coerce_error(generated)
                    _set_model_generation_trace_attributes(model_span, generated)
                else:
                    resolved_model = model or _legacy_openai_chat_model()
                    _set_model_trace_attributes(model_span, resolved_model)
                    answer = build_strict_answer_chain(resolved_model).invoke(
                        {"evidence_prompt": build_evidence_prompt(question, evidence)}
                    )
                    model_error = None
                    model_span.set_attribute("answer.model_status", "ok")
                _set_model_output_trace_attributes(model_span, answer)
                model_span.set_output(
                    {
                        "answer_chars": len(str(answer)),
                        "evidence_count": len(evidence),
                        "citation_count": len(citations),
                    }
                )
            if model_error is not None:
                update = {
                    "answer": answer,
                    "citations": citations,
                    "sources": sources,
                    "citations_valid": True,
                    "invalid_citations": [],
                    "error": model_error,
                }
                _set_answer_trace_output(span, update, evidence=evidence, citations=citations, sources=sources)
                return update
            with trace_answer_step(
                "answer.citation_check",
                question,
                attributes=imperial_trace_attributes(
                    "answer",
                    "citation_check",
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
            _set_answer_trace_output(span, update, evidence=evidence, citations=citations, sources=sources)
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


def _answer_trace_attributes(
    evidence: Sequence[Document],
    citations: Sequence[str],
    *,
    source_count: int | None = None,
) -> dict[str, Any]:
    attributes = {
        "answer.evidence_count": len(evidence),
        "answer.citation_count": len(citations),
        "answer.citation_ids": [
            str(document.metadata.get("citation_id"))
            for document in evidence
            if document.metadata.get("citation_id") is not None
        ],
        "answer.context_chars": sum(len(str(document.page_content)) for document in evidence),
    }
    if source_count is not None:
        attributes["answer.source_count"] = source_count
    return attributes


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
        namespace_parts = provider()
        namespace = (
            ".".join(str(part) for part in namespace_parts)
            if isinstance(namespace_parts, Sequence) and not isinstance(namespace_parts, (str, bytes))
            else str(namespace_parts)
        )
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
        f"<{len(evidence)} retrieved chunk(s) elided; inspect retrieval.final_evidence for source metadata.>"
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


def _set_answer_trace_output(
    span: Any,
    update: Mapping[str, Any],
    *,
    evidence: Sequence[Document],
    citations: Sequence[str],
    sources: Sequence[str],
) -> None:
    span.set_output(
        {
            "answer": update.get("answer", ""),
            "citations_valid": update.get("citations_valid"),
            "invalid_citations": update.get("invalid_citations", []),
            "refused": update.get("answer") == REFUSAL_TEXT,
            **_answer_context_trace_output(evidence, citations, sources),
        }
    )


from imperial_rag.ingestion.workflow import IngestionState, build_ingestion_workflow  # noqa: F401
