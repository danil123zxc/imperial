from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document


_ENDING_RE = re.compile(r"(иями|ями|ами|ого|его|ому|ему|ыми|ими|ов|ев|ей|ый|ий|ой|ая|яя|ое|ее|ам|ям|ах|ях|ом|ем|а|я|ы|и|у|ю|е|о|ь)$")
_QUERY_STOPWORDS = frozenset(
    {
        "а",
        "без",
        "в",
        "во",
        "где",
        "для",
        "до",
        "есть",
        "если",
        "имеет",
        "из",
        "или",
        "и",
        "как",
        "каки",
        "каку",
        "каков",
        "когда",
        "кто",
        "к",
        "ко",
        "ли",
        "на",
        "не",
        "но",
        "об",
        "о",
        "от",
        "по",
        "почему",
        "при",
        "про",
        "найт",
        "с",
        "со",
        "что",
    }
)
_MAX_RELAXED_QUERY_ATTEMPTS = 24
_MAX_ONE_DROP_RELAXATION_TOKENS = 8


@dataclass(frozen=True)
class KeywordHit:
    document: Document
    score: float


class KeywordSearch(Protocol):
    def replace_all(self, documents: list[Document]) -> None: ...
    def index_documents(self, documents: list[Document]) -> None: ...
    def search(self, query: str, limit: int = 5, k: int | None = None) -> list[Document]: ...
    def search_with_scores(self, query: str, limit: int = 5, k: int | None = None) -> list[KeywordHit]: ...


def stem_token(token: str) -> str:
    token = token.casefold().replace("ё", "е")
    while len(token) > 4:
        shortened = _ENDING_RE.sub("", token)
        if shortened == token:
            break
        token = shortened
    return token


def normalize_search_text(text: str) -> str:
    return " ".join(
        stem_token(token)
        for token in re.findall(r"\w+", text.casefold().replace("-", " "), flags=re.UNICODE)
    )


def keyword_query_tokens(query: str) -> list[str]:
    tokens = [token for token in normalize_search_text(query).split() if token]
    content_tokens = _content_query_tokens(tokens)
    return content_tokens or tokens


def content_keyword_query_tokens(query: str) -> list[str]:
    return _content_query_tokens([token for token in normalize_search_text(query).split() if token])


def _content_query_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in _QUERY_STOPWORDS and len(token) > 2]


def relaxed_query_token_sets(tokens: list[str]) -> list[list[str]]:
    if len(tokens) < 3:
        return []
    relaxed: list[list[str]] = []

    if len(tokens) <= _MAX_ONE_DROP_RELAXATION_TOKENS:
        for drop_index in range(len(tokens)):
            relaxed.append([token for index, token in enumerate(tokens) if index != drop_index])
            if len(relaxed) >= _MAX_RELAXED_QUERY_ATTEMPTS:
                return relaxed

    for pair in _bounded_adjacent_pairs(tokens, _MAX_RELAXED_QUERY_ATTEMPTS - len(relaxed)):
        if pair not in relaxed:
            relaxed.append(pair)
        if len(relaxed) >= _MAX_RELAXED_QUERY_ATTEMPTS:
            return relaxed
    return relaxed


def build_elasticsearch_token_query(tokens: list[str]) -> dict:
    return {"bool": {"must": [{"match": {"normalized_text": token}} for token in tokens]}}


def searchable_document_text(document: Document) -> str:
    metadata = document.metadata or {}
    return " ".join(
        [
            document.page_content,
            str(metadata.get("file_name", "")),
            str(metadata.get("relative_path", "")),
            str(metadata.get("section_heading", "")),
            str(metadata.get("source_type", "")),
        ]
    )


def relaxed_candidate_sort_key(candidate: tuple[int, int, int, object]) -> tuple[int, int, int]:
    matched_token_count, query_order, row_order, _row = candidate
    return (-matched_token_count, query_order, row_order)


def _bounded_adjacent_pairs(tokens: list[str], budget: int) -> list[list[str]]:
    if budget <= 0:
        return []
    pairs = [tokens[index : index + 2] for index in range(len(tokens) - 1)]
    if len(pairs) <= budget:
        return pairs
    head_count = budget // 2
    tail_count = budget - head_count
    return pairs[:head_count] + pairs[-tail_count:]
