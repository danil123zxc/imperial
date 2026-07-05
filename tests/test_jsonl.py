from __future__ import annotations

from imperial_rag.jsonl import read_jsonl, write_jsonl


def test_read_jsonl_skips_blank_lines_and_returns_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('\n{"a": 1}\n   \n{"b": "beta"}\n', encoding="utf-8")

    assert read_jsonl(path) == [{"a": 1}, {"b": "beta"}]


def test_write_jsonl_creates_parent_and_writes_sorted_rows(tmp_path):
    path = tmp_path / "nested" / "rows.jsonl"
    rows = (row for row in [{"b": 2, "a": "alpha"}, {"a": "beta"}])

    write_jsonl(path, rows)

    assert path.read_text(encoding="utf-8") == '{"a": "alpha", "b": 2}\n{"a": "beta"}\n'


def test_write_jsonl_empty_rows_leave_empty_file(tmp_path):
    path = tmp_path / "rows.jsonl"

    write_jsonl(path, [])

    assert path.read_text(encoding="utf-8") == ""
