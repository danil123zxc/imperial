from langchain_core.documents import Document

from imperial_rag.ingestion.chunking import build_chunks


def test_build_chunks_defaults_to_structure_token_budget_and_overlap():
    source = Document(
        page_content=" ".join(f"токен{i}" for i in range(900)),
        metadata={"file_id": "file123", "relative_path": "policy.docx", "source_type": "body", "source_locator": "body:1"},
    )

    chunks = build_chunks([source])

    assert len(chunks) == 3
    assert all(chunk.metadata["body_token_count"] <= 400 for chunk in chunks)
    assert all(chunk.metadata["body_start_index"] >= 0 for chunk in chunks)
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["chunk_index"] == 1


def test_build_chunks_preserves_citation_metadata_and_adds_citation_id():
    source = Document(
        page_content="Возврат брака оформляется актом. " * 80,
        metadata={
            "file_path": "/docs/reglament.pdf",
            "relative_path": "reglament.pdf",
            "source_type": "pdf_page",
            "page_number": 3,
            "file_id": "file123",
            "file_hash": "abc",
            "file_name": "reglament.pdf",
        },
    )

    chunks = build_chunks([source], chunk_size=180, chunk_overlap=30)

    assert len(chunks) > 1
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["file_path"] == "/docs/reglament.pdf"
        assert chunk.metadata["relative_path"] == "reglament.pdf"
        assert chunk.metadata["source_type"] == "pdf_page"
        assert chunk.metadata["page_number"] == 3
        assert chunk.metadata["file_hash"] == "abc"
        assert chunk.metadata["chunk_index"] == index
        assert isinstance(chunk.metadata["body_start_index"], int)
        assert chunk.metadata["body_token_count"] > 0
        assert chunk.metadata["citation_id"] == (
            f"reglament.pdf#pdf_page:page-3-chunk-{index}:start-{chunk.metadata['body_start_index']}:chunk-{index}"
        )


def test_build_chunks_uses_sheet_name_in_citation_identity():
    documents = [
        Document(
            page_content="Строки первого листа",
            metadata={"relative_path": "book.xlsx", "source_type": "sheet", "sheet_name": "Склад"},
        ),
        Document(
            page_content="Строки второго листа",
            metadata={"relative_path": "book.xlsx", "source_type": "sheet", "sheet_name": "Продажи"},
        ),
    ]

    chunks = build_chunks(documents)

    assert [chunk.metadata["citation_id"] for chunk in chunks] == [
        "book.xlsx#sheet:sheet-Склад-chunk-0:start-0:chunk-0",
        "book.xlsx#sheet:sheet-Продажи-chunk-0:start-0:chunk-0",
    ]
    assert chunks[0].metadata["chunk_id"] != chunks[1].metadata["chunk_id"]


def test_build_chunks_repeats_table_header_and_tracks_exact_row_ranges():
    source = Document(
        page_content="Header A | Header B\n" + "\n".join(f"row-{index} | value-{index}" for index in range(1, 15)),
        metadata={
            "file_id": "table-file",
            "relative_path": "table.docx",
            "source_type": "table",
            "source_locator": "table:1:rows:1-15",
        },
    )

    chunks = build_chunks([source], chunk_size=24, chunk_overlap=0)

    assert len(chunks) > 1
    assert all(chunk.page_content.startswith("Header A | Header B") for chunk in chunks)
    assert all(chunk.metadata["header_row"] == 1 for chunk in chunks)
    assert all(":header:1:rows:" in chunk.metadata["base_source_locator"] for chunk in chunks)
    assert len({chunk.metadata["source_locator"] for chunk in chunks}) == len(chunks)
    assert all(chunk.metadata["embedding_text"].endswith(chunk.page_content) for chunk in chunks)
