from __future__ import annotations


def test_clean_context_ids_normalizes_scalars_sequences_and_iterables():
    from imperial_rag.evals.corpus import clean_context_ids

    assert clean_context_ids(None) == []
    assert clean_context_ids(" chunk-a ") == ["chunk-a"]
    assert clean_context_ids(["chunk-a", "", " chunk-b ", "chunk-a"]) == ["chunk-a", "chunk-b"]
    assert clean_context_ids(value for value in [" chunk-a ", None, "chunk-b"]) == ["chunk-a", "chunk-b"]


def test_clean_string_list_can_keep_scalars_opt_in():
    from imperial_rag.evals.corpus import clean_string_list

    assert clean_string_list(None) == []
    assert clean_string_list(" tag ") == []
    assert clean_string_list([" tag ", "", None, "other"]) == ["tag", "None", "other"]
    assert clean_string_list(" chunk-a ", allow_scalar=True) == ["chunk-a"]
