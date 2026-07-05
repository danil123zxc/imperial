from __future__ import annotations

from imperial_rag.jsonl import write_jsonl


def test_write_jsonl_creates_parent_and_writes_sorted_rows(tmp_path):
    path = tmp_path / "nested" / "rows.jsonl"
    rows = (row for row in [{"b": 2, "a": "alpha"}, {"a": "beta"}])

    write_jsonl(path, rows)

    assert path.read_text(encoding="utf-8") == '{"a": "alpha", "b": 2}\n{"a": "beta"}\n'


def test_write_jsonl_empty_rows_leave_empty_file(tmp_path):
    path = tmp_path / "rows.jsonl"

    write_jsonl(path, [])

    assert path.read_text(encoding="utf-8") == ""
