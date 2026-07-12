from __future__ import annotations

import hashlib
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


RUSSIAN_STRUCTURE_SEPARATORS = ["\n\n", "\n", ". ", "; ", ": ", " - ", " ", ""]


def estimated_token_count(text: str) -> int:
    if not text.strip():
        return 0
    tokens = re.findall(r"[\w]+|[^\w\s]", text, flags=re.UNICODE)
    return max(1, len(tokens))


def _source_locator(metadata: dict) -> str:
    locator = metadata.get("source_locator")
    if locator is not None:
        return str(locator)
    page = metadata.get("page_number")
    if page is not None:
        return f"page:{page}"
    sheet = metadata.get("sheet_name")
    if sheet is not None:
        return f"sheet:{sheet}"
    image = metadata.get("image_index")
    if image is not None:
        return f"image:{image}"
    heading = metadata.get("section_heading")
    if heading is not None:
        return f"section:{heading}"
    return "body:1"


def _locator_for_id(locator: str) -> str:
    return locator.replace(":", "-").replace("/", "-").replace(" ", "-")


def _citation_id(metadata: dict, chunk_index: int) -> str:
    relative_path = metadata.get("relative_path", "unknown")
    source_type = metadata.get("source_type", "unknown")
    locator = _locator_for_id(_source_locator(metadata))
    start = metadata.get("body_start_index", metadata.get("start_index", 0))
    return f"{relative_path}#{source_type}:{locator}:start-{start}:chunk-{chunk_index}"


def _embedding_text(page_content: str, metadata: dict) -> str:
    context = [
        metadata.get("department") or metadata.get("inferred_category"),
        metadata.get("file_name"),
        metadata.get("section_heading"),
        metadata.get("sheet_name"),
    ]
    prefix = " | ".join(str(value).strip() for value in context if value is not None and str(value).strip())
    return page_content if not prefix else f"{prefix}\n\n{page_content}"


def _tabular_windows(document: Document, chunk_size: int) -> list[Document]:
    metadata = dict(document.metadata or {})
    if metadata.get("source_type") not in {"table", "sheet"} and metadata.get("layout_route") != "table":
        return [document]
    lines = [line.strip() for line in document.page_content.splitlines() if line.strip()]
    if len(lines) <= 2 or estimated_token_count(document.page_content) <= chunk_size:
        return [document]
    header, data_rows = lines[0], lines[1:]
    windows: list[Document] = []
    current: list[str] = []
    first_row = 2

    def flush() -> None:
        nonlocal current, first_row
        if not current:
            return
        last_row = first_row + len(current) - 1
        window_metadata = dict(metadata)
        base_locator = str(metadata.get("source_locator") or _source_locator(metadata))
        base_locator = re.sub(r":rows:\d+-\d+$", "", base_locator)
        window_metadata.update(
            {
                "source_locator": f"{base_locator}:header:1:rows:{first_row}-{last_row}",
                "header_row": 1,
                "row_start": first_row,
                "row_end": last_row,
                "table_header": header,
            }
        )
        windows.append(Document(page_content="\n".join([header, *current]), metadata=window_metadata))
        first_row = last_row + 1
        current = []

    for row in data_rows:
        candidate = "\n".join([header, *current, row])
        if current and estimated_token_count(candidate) > chunk_size:
            flush()
        current.append(row)
    flush()
    return windows


def _body_start_index(source_text: str, chunk_text: str, metadata: dict, search_from: int) -> tuple[int, int]:
    raw_start = metadata.get("start_index")
    if raw_start is not None:
        start = int(raw_start)
        if start >= 0:
            return start, max(search_from, start + 1)

    found = source_text.find(chunk_text, max(0, search_from))
    if found < 0:
        stripped = chunk_text.strip()
        found = source_text.find(stripped, max(0, search_from)) if stripped else -1
    start = max(0, found)
    return start, max(search_from, start + 1)


def build_chunks(documents: list[Document], chunk_size: int = 400, chunk_overlap: int = 50) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=estimated_token_count,
        separators=RUSSIAN_STRUCTURE_SEPARATORS,
        add_start_index=True,
    )
    chunks: list[Document] = []
    for document in documents:
        source_documents = _tabular_windows(document, chunk_size)
        chunk_sequence = 0
        for source_document in source_documents:
            split_docs = splitter.split_documents([source_document])
            search_from = 0
            for chunk in split_docs:
                index = chunk_sequence
                chunk_sequence += 1
                metadata = dict(chunk.metadata)
                base_source_locator = _source_locator(metadata)
                source_locator = f"{base_source_locator}:chunk:{index}"
                metadata["base_source_locator"] = base_source_locator
                metadata["source_locator"] = source_locator
                metadata["chunk_index"] = index
                body_start_index, search_from = _body_start_index(
                    source_document.page_content,
                    chunk.page_content,
                    metadata,
                    search_from,
                )
                metadata["body_start_index"] = body_start_index
                metadata["body_token_count"] = estimated_token_count(chunk.page_content)
                metadata["citation_id"] = _citation_id(metadata, index)
                metadata["embedding_text"] = _embedding_text(chunk.page_content, metadata)
                if "chunk_id" not in metadata:
                    base = (
                        f"{metadata.get('source_doc_id') or metadata.get('file_id')}:"
                        f"{metadata.get('source_type')}:{source_locator}:{index}:"
                        f"{metadata.get('body_start_index')}"
                    )
                    digest = hashlib.sha1(f"{base}:{chunk.page_content}".encode("utf-8")).hexdigest()[:10]
                    metadata["chunk_id"] = f"{base}:{digest}"
                chunks.append(Document(page_content=chunk.page_content, metadata=metadata))
    return chunks
