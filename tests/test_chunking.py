from langchain_core.documents import Document

from imperial_rag.chunking import build_chunks


def test_build_chunks_defaults_to_accuracy_spec_size_and_overlap():
    expected_overlap = "0123456789" * 5
    source = Document(
        page_content=("А" * 350) + expected_overlap + ("Б" * 100),
        metadata={"file_id": "file123", "relative_path": "policy.docx", "source_type": "body"},
    )

    chunks = build_chunks([source])

    assert len(chunks) == 2
    assert len(chunks[0].page_content) <= 400
    assert chunks[0].page_content[-50:] == expected_overlap
    assert chunks[1].page_content[:51] == expected_overlap + "Б"
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
        assert chunk.metadata["citation_id"] == f"reglament.pdf#pdf_page:page-3:chunk-{index}"


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
        "book.xlsx#sheet:sheet-Склад:chunk-0",
        "book.xlsx#sheet:sheet-Продажи:chunk-0",
    ]
    assert chunks[0].metadata["chunk_id"] != chunks[1].metadata["chunk_id"]
