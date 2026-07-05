from __future__ import annotations

from datetime import date

from imperial_rag.serialization import stable_json_dumps


def test_stable_json_dumps_sorts_keys_preserves_unicode_and_stringifies_unknown_values() -> None:
    payload = {"when": date(2026, 7, 5), "name": "Империал", "count": 2}

    assert stable_json_dumps(payload) == '{"count": 2, "name": "Империал", "when": "2026-07-05"}'
