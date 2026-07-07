import sqlite3
from pathlib import Path

import pytest

from imperial_rag.ingestion.manifest import (
    FileStatus,
    IndexStatus,
    ManifestStore,
    assign_duplicate_groups,
    scan_files,
)
from imperial_rag.ingestion import manifest as manifest_module


class TrackingConnection:
    def __init__(self, connection: sqlite3.Connection) -> None:
        object.__setattr__(self, "_connection", connection)
        object.__setattr__(self, "closed", False)

    def __getattr__(self, name: str):
        return getattr(self._connection, name)

    def __setattr__(self, name: str, value) -> None:
        if name in {"_connection", "closed"}:
            object.__setattr__(self, name, value)
            return
        setattr(self._connection, name, value)

    def close(self) -> None:
        object.__setattr__(self, "closed", True)
        self._connection.close()


def test_manifest_store_persists_every_scanned_file(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.docx").write_bytes(b"docx")
    (docs / "archive.rar").write_bytes(b"rar")
    records = assign_duplicate_groups(scan_files(docs))

    store = ManifestStore(tmp_path / "manifest.sqlite3")
    store.replace_records(records)
    loaded = store.list_records()

    assert {record.relative_path for record in loaded} == {Path("policy.docx"), Path("archive.rar")}
    assert all(record.status == FileStatus.PENDING for record in loaded)


def test_manifest_store_replaces_records(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "old.docx").write_bytes(b"old")
    first = scan_files(docs)
    (docs / "new.docx").write_bytes(b"new")
    second = [record for record in scan_files(docs) if record.filename == "new.docx"]

    store = ManifestStore(tmp_path / "manifest.sqlite3")
    store.replace_records(first)
    store.replace_records(second)

    assert [record.relative_path for record in store.list_records()] == [Path("new.docx")]


def test_manifest_store_roundtrips_path_and_enum_fields(tmp_path):
    docs = tmp_path / "documents"
    nested = docs / "folder"
    nested.mkdir(parents=True)
    (nested / "policy.docx").write_bytes(b"docx")
    record = assign_duplicate_groups(scan_files(docs))[0]

    store = ManifestStore(tmp_path / "manifest.sqlite3")
    store.replace_records([record])
    loaded = store.get_record(record.file_id)

    assert loaded.absolute_path == record.absolute_path
    assert isinstance(loaded.absolute_path, Path)
    assert loaded.relative_path == Path("folder/policy.docx")
    assert isinstance(loaded.relative_path, Path)
    assert loaded.parent_folder == Path("folder")
    assert isinstance(loaded.parent_folder, Path)
    assert loaded.status is FileStatus.PENDING
    assert loaded.keyword_index_status is IndexStatus.PENDING
    assert loaded.vector_index_status is IndexStatus.PENDING


def test_manifest_store_records_status_errors_and_chunk_counts(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "scan.pdf").write_bytes(b"%PDF")
    record = scan_files(docs)[0]
    store = ManifestStore(tmp_path / "manifest.sqlite3")
    store.replace_records([record])

    store.update_status(
        file_id=record.file_id,
        status=FileStatus.FAILED,
        extraction_method="pdf_ocr",
        error_message="render failed",
        chunk_count=0,
    )
    loaded = store.get_record(record.file_id)

    assert loaded.status == FileStatus.FAILED
    assert loaded.extraction_method == "pdf_ocr"
    assert loaded.error_message == "render failed"
    assert loaded.chunk_count == 0


def test_manifest_store_records_keyword_and_vector_index_status(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.docx").write_bytes(b"docx")
    record = scan_files(docs)[0]
    store = ManifestStore(tmp_path / "manifest.sqlite3")
    store.replace_records([record])

    store.update_index_status(
        file_id=record.file_id,
        keyword_index_status=IndexStatus.INDEXED,
        vector_index_status=IndexStatus.FAILED,
        embedding_model="text-embedding-3-large",
        index_error_message="qdrant unavailable",
    )
    loaded = store.get_record(record.file_id)

    assert loaded.keyword_index_status == IndexStatus.INDEXED
    assert loaded.vector_index_status == IndexStatus.FAILED
    assert loaded.embedding_model == "text-embedding-3-large"
    assert loaded.index_error_message == "qdrant unavailable"
    assert loaded.last_indexed_ns > 0


def test_manifest_store_context_manager_closes_connection(tmp_path):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.docx").write_bytes(b"docx")
    record = scan_files(docs)[0]

    with ManifestStore(tmp_path / "manifest.sqlite3") as store:
        store.replace_records([record])
        assert store.list_records()[0].file_id == record.file_id

    with pytest.raises(sqlite3.ProgrammingError):
        store.list_records()


def test_manifest_store_finalizer_closes_unclosed_connection(monkeypatch, tmp_path):
    real_connect = sqlite3.connect
    opened: list[TrackingConnection] = []

    def tracking_connect(*args, **kwargs):
        connection = TrackingConnection(real_connect(*args, **kwargs))
        opened.append(connection)
        return connection

    monkeypatch.setattr(manifest_module.sqlite3, "connect", tracking_connect)
    store = ManifestStore(tmp_path / "manifest.sqlite3")

    del store

    assert opened
    assert all(connection.closed for connection in opened)
