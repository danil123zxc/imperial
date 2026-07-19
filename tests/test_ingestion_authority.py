from __future__ import annotations

import json

from langchain_core.documents import Document

from imperial_rag.ingestion.authority import apply_authority_and_exact_deduplication, load_authority_catalog


def test_authority_catalog_selects_one_exact_copy_and_preserves_all_paths(tmp_path):
    catalog_path = tmp_path / "authority.json"
    catalog_path.write_text(
        json.dumps(
            {
                "documents": [
                    {"relative_path": "draft/policy.docx", "status": "draft", "authoritative_rank": 10},
                    {
                        "relative_path": "active/policy.docx",
                        "status": "active",
                        "authoritative_rank": 5,
                        "department": "Operations",
                        "version_group": "return-policy",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    documents = [
        Document(page_content="same", metadata={"file_hash": "abc", "relative_path": path, "file_id": path})
        for path in ("draft/policy.docx", "active/policy.docx")
    ]

    retained = apply_authority_and_exact_deduplication(documents, load_authority_catalog(catalog_path))

    assert len(retained) == 1
    assert retained[0].metadata["canonical_source_path"] == "active/policy.docx"
    assert retained[0].metadata["provenance_paths"] == ["active/policy.docx", "draft/policy.docx"]
    assert retained[0].metadata["department"] == "Operations"
    assert retained[0].metadata["version_group"] == "return-policy"
