from pathlib import Path

from imperial_rag.manifest import FileStatus, assign_duplicate_groups, scan_files, stable_file_id


def test_scan_files_records_every_file_including_temp_and_archives(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "~$lock.docx").write_bytes(b"lock")
    (docs / ".~lock.file.docx#").write_bytes(b"lock2")
    (docs / "Thumbs.db").write_bytes(b"thumbs")
    (docs / "archive.rar").write_bytes(b"rar")
    (docs / "policy.docx").write_bytes(b"docx")

    records = scan_files(docs)

    assert {record.relative_path for record in records} == {
        Path("~$lock.docx"),
        Path(".~lock.file.docx#"),
        Path("Thumbs.db"),
        Path("archive.rar"),
        Path("policy.docx"),
    }


def test_scan_files_hashes_duplicates_without_removing_them(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "a.docx").write_bytes(b"same")
    (docs / "b.docx").write_bytes(b"same")

    records = scan_files(docs)

    assert len(records) == 2
    assert records[0].sha256 == records[1].sha256
    assert records[0].duplicate_group_id is not None
    assert records[0].duplicate_group_id == records[1].duplicate_group_id
    assert all(record.status == FileStatus.PENDING for record in records)


def test_scan_files_records_audit_fields_and_stable_path_id(tmp_path):
    docs = tmp_path / "documents"
    nested = docs / "policies"
    nested.mkdir(parents=True)
    path = nested / "Policy.PDF"
    path.write_bytes(b"pdf")

    record = scan_files(docs)[0]

    assert record.file_id == stable_file_id(Path("policies/Policy.PDF"))
    assert record.absolute_path == path.resolve()
    assert record.relative_path == Path("policies/Policy.PDF")
    assert record.filename == "Policy.PDF"
    assert record.extension == ".pdf"
    assert record.size_bytes == 3
    assert len(record.sha256) == 64
    assert record.modified_ns > 0
    assert record.parent_folder == Path("policies")
    assert record.inferred_category == "policies"
    assert record.extraction_method is None
    assert record.error_message is None
    assert record.chunk_count == 0
    assert record.duplicate_group_id is None
    assert record.keyword_index_status.value == "pending"
    assert record.vector_index_status.value == "pending"
    assert record.embedding_model is None
    assert record.index_error_message is None
    assert record.last_indexed_ns == 0


def test_assign_duplicate_groups_marks_same_hash_records(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "a.pdf").write_bytes(b"same")
    (docs / "b.pdf").write_bytes(b"same")
    (docs / "c.pdf").write_bytes(b"different")

    records = assign_duplicate_groups(scan_files(docs))
    duplicate_groups = {record.relative_path: record.duplicate_group_id for record in records}

    assert duplicate_groups[Path("a.pdf")] is not None
    assert duplicate_groups[Path("a.pdf")] == duplicate_groups[Path("b.pdf")]
    assert duplicate_groups[Path("c.pdf")] is None
