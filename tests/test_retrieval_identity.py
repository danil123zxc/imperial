from langchain_core.documents import Document

from imperial_rag.retrieval.identity import _annotate_retrieval_documents


def test_vector_retrieval_restores_citation_text_after_contextual_embedding():
    indexed = Document(
        page_content="Operations | Policy\n\nCitation body",
        metadata={"citation_id": "c1", "citation_text": "Citation body"},
    )

    restored = _annotate_retrieval_documents([indexed], rank_key="_vector_rank")

    assert restored[0].page_content == "Citation body"
    assert "citation_text" not in restored[0].metadata
