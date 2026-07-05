from __future__ import annotations


def test_clean_context_ids_normalizes_scalars_sequences_and_iterables():
    from imperial_rag.evals.corpus import clean_context_ids

    assert clean_context_ids(None) == []
    assert clean_context_ids(" chunk-a ") == ["chunk-a"]
    assert clean_context_ids(["chunk-a", "", " chunk-b ", "chunk-a"]) == ["chunk-a", "chunk-b"]
    assert clean_context_ids(value for value in [" chunk-a ", None, "chunk-b"]) == ["chunk-a", "chunk-b"]
