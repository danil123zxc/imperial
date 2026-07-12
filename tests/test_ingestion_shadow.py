from __future__ import annotations

import json
from types import SimpleNamespace

from imperial_rag.ingestion.shadow import (
    _active_baseline_root,
    _swap_elasticsearch_alias,
    _swap_qdrant_alias,
)


def test_active_baseline_root_uses_promoted_pointer(tmp_path):
    processed = tmp_path / ".imperial_rag"
    processed.mkdir()
    promoted = processed / "shadow-runs" / "v1" / "extracted"
    (processed / "active-ingestion.json").write_text(
        json.dumps({"artifact_root": str(promoted)}),
        encoding="utf-8",
    )
    settings = SimpleNamespace(processed_root=processed, extraction_root=processed / "extracted")

    assert _active_baseline_root(settings) == promoted


def test_elasticsearch_alias_swap_is_one_update_and_returns_previous_target():
    calls = []

    class Indices:
        def exists(self, *, index):
            return True

        def exists_alias(self, *, name):
            return True

        def get_alias(self, *, name):
            return {"old-index": {"aliases": {name: {}}}}

        def update_aliases(self, *, actions):
            calls.append(actions)

    previous = _swap_elasticsearch_alias(SimpleNamespace(indices=Indices()), "active", "new-index")

    assert previous == "old-index"
    assert calls == [[
        {"remove": {"index": "*", "alias": "active"}},
        {"add": {"index": "new-index", "alias": "active"}},
    ]]


def test_qdrant_alias_swap_returns_previous_collection():
    calls = []

    class Client:
        def get_aliases(self):
            return SimpleNamespace(aliases=[SimpleNamespace(alias_name="active", collection_name="old")])

        def update_collection_aliases(self, *, change_aliases_operations):
            calls.append(change_aliases_operations)

    previous = _swap_qdrant_alias(Client(), "active", "new")

    assert previous == "old"
    assert len(calls) == 1
    assert len(calls[0]) == 2
