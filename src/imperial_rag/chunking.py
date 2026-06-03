from __future__ import annotations

import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _source_locator(metadata: dict) -> str:
    page = metadata.get("page_number")
    if page is not None:
        return f":page-{page}"
    sheet = metadata.get("sheet_name")
    if sheet is not None:
        return f":sheet-{sheet}"
    image = metadata.get("image_index")
    if image is not None:
        return f":image-{image}"
    heading = metadata.get("section_heading")
    if heading is not None:
        return f":section-{heading}"
    return ""


def _citation_id(metadata: dict, chunk_index: int) -> str:
    relative_path = metadata.get("relative_path", "unknown")
    source_type = metadata.get("source_type", "unknown")
    locator = _source_locator(metadata)
    return f"{relative_path}#{source_type}{locator}:chunk-{chunk_index}"


def build_chunks(documents: list[Document], chunk_size: int = 400, chunk_overlap: int = 50) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for document in documents:
        split_docs = splitter.split_documents([document])
        for index, chunk in enumerate(split_docs):
            metadata = dict(chunk.metadata)
            metadata["chunk_index"] = index
            metadata["citation_id"] = _citation_id(metadata, index)
            if "chunk_id" not in metadata:
                base = f"{metadata.get('file_id')}:{metadata.get('source_type')}:{_source_locator(metadata)}:{index}"
                digest = hashlib.sha1(f"{base}:{chunk.page_content}".encode("utf-8")).hexdigest()[:10]
                metadata["chunk_id"] = f"{base}:{digest}"
            chunks.append(Document(page_content=chunk.page_content, metadata=metadata))
    return chunks
