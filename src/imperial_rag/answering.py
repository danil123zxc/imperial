from __future__ import annotations

import re

from langchain_core.documents import Document


REFUSAL_TEXT = "I could not find this clearly in the indexed documents."


def _doc_citation_marker(document: Document) -> str:
    if document.metadata.get("citation_id"):
        return str(document.metadata["citation_id"])
    file_path = document.metadata.get("file_path", "unknown")
    chunk_id = document.metadata.get("chunk_id", "unknown")
    return f"{file_path}#{chunk_id}"


def format_citations(documents: list[Document]) -> list[str]:
    citations: list[str] = []
    for document in documents:
        marker = _doc_citation_marker(document)
        source_type = document.metadata.get("source_type", "unknown")
        citations.append(f"[{marker}] {source_type}")
    return citations


def format_sources(documents: list[Document]) -> list[str]:
    sources: list[str] = []
    for document in documents:
        marker = _doc_citation_marker(document)
        source = (
            document.metadata.get("file_path")
            or document.metadata.get("relative_path")
            or document.metadata.get("file_name")
            or "unknown"
        )
        details = []
        if document.metadata.get("source_type"):
            details.append(str(document.metadata["source_type"]))
        details.extend(
            f"{field}={document.metadata[field]}"
            for field in (
                "section_heading",
                "page_number",
                "sheet_name",
                "table_index",
                "row_range",
                "image_index",
                "embedded_media_name",
            )
            if document.metadata.get(field)
        )
        suffix = f" {' '.join(details)}" if details else ""
        sources.append(f"[{marker}] {source}{suffix}")
    return sources


def build_context(documents: list[Document]) -> str:
    return "\n\n".join(
        f"Source: {citation}\nText:\n{document.page_content}"
        for citation, document in zip(format_citations(documents), documents, strict=False)
    )


def build_evidence_prompt(question: str, documents: list[Document]) -> str:
    return f"""You are answering questions about internal company documents.
Use only the evidence below.
Do not use general model knowledge.
Every meaningful factual claim must cite a source from the evidence.
If the evidence is insufficient, answer exactly: {REFUSAL_TEXT}

Question:
{question}

Evidence:
{build_context(documents)}
"""


def build_strict_messages(question: str, documents: list[Document]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a strict-citation RAG assistant. Use only the provided context. "
                "Do not use general model knowledge. Answer only from context and "
                "cite every factual claim. Refuse when the documents do not support the answer."
            ),
        },
        {"role": "user", "content": build_evidence_prompt(question, documents)},
    ]


def citation_marker(citation: str) -> str:
    return citation.split("]", maxsplit=1)[0] + "]" if "]" in citation else citation


def _markers_in_text(text: str) -> list[str]:
    return re.findall(r"\[[^\]]+\]", text)


def answer_has_required_citations(answer: str, citations: list[str]) -> bool:
    stripped = answer.strip()
    if not citations:
        return stripped == REFUSAL_TEXT
    if stripped == REFUSAL_TEXT:
        return True
    known_markers = {citation_marker(citation) for citation in citations}
    paragraphs = [paragraph.strip() for paragraph in answer.splitlines() if paragraph.strip()]
    if not paragraphs:
        return False
    for paragraph in paragraphs:
        markers = set(_markers_in_text(paragraph))
        if not markers:
            return False
        if not markers.issubset(known_markers):
            return False
    return True


def validate_citations(answer: str, documents: list[Document]) -> tuple[bool, list[str]]:
    known = {citation_marker(citation) for citation in format_citations(documents)}
    stripped = answer.strip()
    if stripped in {REFUSAL_TEXT, "No indexed evidence was enough to answer."}:
        return True, []
    invalid = [marker.strip("[]") for marker in _markers_in_text(answer) if marker not in known]
    if invalid:
        return False, invalid
    return answer_has_required_citations(answer, format_citations(documents)), invalid


def refuse_message(question: str | None = None) -> str:
    return REFUSAL_TEXT
