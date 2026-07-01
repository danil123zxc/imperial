from __future__ import annotations

import re
import unicodedata

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


REFUSAL_TEXT = "I could not find this clearly in the indexed documents."

STRICT_SYSTEM_PROMPT = (
    "You are a strict-citation RAG assistant. Use only the provided context. "
    "Do not use general model knowledge. Answer only from context and "
    "cite every factual claim. Use concise bullets or short paragraphs. "
    "Do not include uncited introductions or summaries. "
    "Refuse when the documents do not support the answer."
)


def _short_citation_marker(index: int) -> str:
    return f"[S{index + 1}]"


def _doc_citation_marker(document: Document) -> str:
    if document.metadata.get("citation_id"):
        return str(document.metadata["citation_id"])
    file_path = document.metadata.get("file_path", "unknown")
    chunk_id = document.metadata.get("chunk_id", "unknown")
    return f"{file_path}#{chunk_id}"


def _legacy_citations(documents: list[Document]) -> list[str]:
    citations: list[str] = []
    for document in documents:
        marker = _doc_citation_marker(document)
        source_type = document.metadata.get("source_type", "unknown")
        citations.append(f"[{marker}] {source_type}")
    return citations


def _validation_citations(documents: list[Document]) -> list[str]:
    return [*format_citations(documents), *_legacy_citations(documents)]


def format_citations(documents: list[Document]) -> list[str]:
    citations: list[str] = []
    for index, document in enumerate(documents):
        marker = _short_citation_marker(index)
        source_type = document.metadata.get("source_type", "unknown")
        citations.append(f"{marker} {source_type}")
    return citations


def format_sources(documents: list[Document]) -> list[str]:
    sources: list[str] = []
    for index, document in enumerate(documents):
        marker = _short_citation_marker(index)
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
        sources.append(f"{marker} {source}{suffix}")
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
Use the short source labels exactly as shown, for example [S1].
Do not include uncited introductions or summaries.
If the evidence is insufficient, answer exactly: {REFUSAL_TEXT}

Question:
{question}

Evidence:
{build_context(documents)}
"""


def build_strict_messages(question: str, documents: list[Document]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": STRICT_SYSTEM_PROMPT},
        {"role": "user", "content": build_evidence_prompt(question, documents)},
    ]


STRICT_ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", STRICT_SYSTEM_PROMPT),
        ("human", "{evidence_prompt}"),
    ]
)


def build_strict_answer_chain(chat_model):
    """LCEL chain that renders the strict-citation prompt, calls the model, and returns text."""
    return STRICT_ANSWER_PROMPT | chat_model | StrOutputParser()


def citation_marker(citation: str) -> str:
    return citation.split("]", maxsplit=1)[0] + "]" if "]" in citation else citation


def _normalize_marker(marker: str) -> str:
    return unicodedata.normalize("NFC", marker)


def _markers_in_text(text: str) -> list[str]:
    return re.findall(r"\[[^\]]+\]", text)


def _citation_markers_in_text(text: str, known_markers: set[str]) -> list[str]:
    return [marker for marker in _markers_in_text(text) if _normalize_marker(marker) in known_markers]


def _is_unknown_citation_marker(marker: str, known_markers: set[str]) -> bool:
    normalized = _normalize_marker(marker)
    if normalized in known_markers:
        return False
    label = marker.strip()[1:-1].strip()
    if re.fullmatch(r"S\d+", label):
        return True
    return bool(re.fullmatch(r"[A-Za-z0-9_.:/#-]+", label))


def _is_structural_heading(text: str) -> bool:
    stripped = text.strip()
    if re.fullmatch(r"#{1,6}\s+\S.*", stripped):
        return True
    bold_label = re.fullmatch(r"(\*\*|__)(.+)\1", stripped)
    return bool(bold_label and bold_label.group(2).strip().endswith(":"))


def answer_has_required_citations(answer: str, citations: list[str]) -> bool:
    stripped = answer.strip()
    if not citations:
        return stripped == REFUSAL_TEXT
    if stripped == REFUSAL_TEXT:
        return True
    known_markers = {_normalize_marker(citation_marker(citation)) for citation in citations}
    paragraphs = [paragraph.strip() for paragraph in answer.splitlines() if paragraph.strip()]
    if not paragraphs:
        return False
    checked_factual_line = False
    for paragraph in paragraphs:
        if _is_structural_heading(paragraph):
            continue
        checked_factual_line = True
        markers = {_normalize_marker(marker) for marker in _citation_markers_in_text(paragraph, known_markers)}
        if not markers:
            return False
        if not markers.issubset(known_markers):
            return False
    return checked_factual_line


def validate_citations(answer: str, documents: list[Document]) -> tuple[bool, list[str]]:
    citations = _validation_citations(documents)
    known = {_normalize_marker(citation_marker(citation)) for citation in citations}
    stripped = answer.strip()
    if stripped in {REFUSAL_TEXT, "No indexed evidence was enough to answer."}:
        return True, []
    invalid = [marker.strip("[]") for marker in _markers_in_text(answer) if _is_unknown_citation_marker(marker, known)]
    if invalid:
        return False, invalid
    return answer_has_required_citations(answer, citations), invalid


def refuse_message(question: str | None = None) -> str:
    return REFUSAL_TEXT
