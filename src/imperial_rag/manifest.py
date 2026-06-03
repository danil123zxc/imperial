from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path


class FileStatus(StrEnum):
    PENDING = "pending"
    INDEXED = "indexed"
    MANIFEST_ONLY = "manifest_only"
    NO_TEXT = "no_text"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class IndexStatus(StrEnum):
    PENDING = "pending"
    INDEXED = "indexed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class FileRecord:
    file_id: str
    absolute_path: Path
    relative_path: Path
    filename: str
    extension: str
    size_bytes: int
    sha256: str
    modified_ns: int
    parent_folder: Path
    inferred_category: str
    status: FileStatus = FileStatus.PENDING
    extraction_method: str | None = None
    error_message: str | None = None
    chunk_count: int = 0
    duplicate_group_id: str | None = None
    keyword_index_status: IndexStatus = IndexStatus.PENDING
    vector_index_status: IndexStatus = IndexStatus.PENDING
    embedding_model: str | None = None
    index_error_message: str | None = None
    last_indexed_ns: int = 0


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_file_id(relative_path: Path) -> str:
    return hashlib.sha256(relative_path.as_posix().encode("utf-8")).hexdigest()[:16]


def scan_files(documents_root: Path) -> list[FileRecord]:
    records: list[FileRecord] = []
    for path in sorted(documents_root.rglob("*")):
        if not path.is_file():
            continue
        stat = path.stat()
        relative_path = path.relative_to(documents_root)
        records.append(
            FileRecord(
                file_id=stable_file_id(relative_path),
                absolute_path=path.resolve(),
                relative_path=relative_path,
                filename=path.name,
                extension=path.suffix.lower(),
                size_bytes=stat.st_size,
                sha256=hash_file(path),
                modified_ns=stat.st_mtime_ns,
                parent_folder=relative_path.parent,
                inferred_category=relative_path.parts[0] if relative_path.parts else "",
            )
        )
    return assign_duplicate_groups(records)


def assign_duplicate_groups(records: list[FileRecord]) -> list[FileRecord]:
    records_by_hash: dict[str, list[FileRecord]] = {}
    for record in records:
        records_by_hash.setdefault(record.sha256, []).append(record)

    grouped: list[FileRecord] = []
    for record in records:
        matches = records_by_hash[record.sha256]
        group_id = f"sha256:{record.sha256}" if len(matches) > 1 else None
        grouped.append(replace(record, duplicate_group_id=group_id))
    return grouped


class ManifestStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def replace_records(self, records: list[FileRecord]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM files")
            self._conn.executemany(
                """
                INSERT INTO files (
                    file_id,
                    absolute_path,
                    relative_path,
                    filename,
                    extension,
                    size_bytes,
                    sha256,
                    modified_ns,
                    parent_folder,
                    inferred_category,
                    status,
                    extraction_method,
                    error_message,
                    chunk_count,
                    duplicate_group_id,
                    keyword_index_status,
                    vector_index_status,
                    embedding_model,
                    index_error_message,
                    last_indexed_ns
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [self._to_row(record) for record in records],
            )

    def list_records(self) -> list[FileRecord]:
        rows = self._conn.execute("SELECT * FROM files ORDER BY relative_path").fetchall()
        return [self._from_row(row) for row in rows]

    def get_record(self, file_id: str) -> FileRecord:
        row = self._conn.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
        if row is None:
            raise KeyError(file_id)
        return self._from_row(row)

    def update_status(
        self,
        file_id: str,
        status: FileStatus,
        extraction_method: str | None = None,
        error_message: str | None = None,
        chunk_count: int = 0,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE files
                SET status = ?,
                    extraction_method = ?,
                    error_message = ?,
                    chunk_count = ?,
                    last_updated_ns = ?
                WHERE file_id = ?
                """,
                (status.value, extraction_method, error_message, chunk_count, time.time_ns(), file_id),
            )

    def update_index_status(
        self,
        file_id: str,
        keyword_index_status: IndexStatus,
        vector_index_status: IndexStatus,
        embedding_model: str | None = None,
        index_error_message: str | None = None,
    ) -> None:
        with self._conn:
            self._conn.execute(
                """
                UPDATE files
                SET keyword_index_status = ?,
                    vector_index_status = ?,
                    embedding_model = ?,
                    index_error_message = ?,
                    last_indexed_ns = ?
                WHERE file_id = ?
                """,
                (
                    keyword_index_status.value,
                    vector_index_status.value,
                    embedding_model,
                    index_error_message,
                    time.time_ns(),
                    file_id,
                ),
            )

    def close(self) -> None:
        self._conn.close()

    def _create_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                file_id TEXT PRIMARY KEY,
                absolute_path TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                filename TEXT NOT NULL,
                extension TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                modified_ns INTEGER NOT NULL,
                parent_folder TEXT NOT NULL,
                inferred_category TEXT NOT NULL,
                status TEXT NOT NULL,
                extraction_method TEXT,
                error_message TEXT,
                chunk_count INTEGER NOT NULL,
                duplicate_group_id TEXT,
                keyword_index_status TEXT NOT NULL,
                vector_index_status TEXT NOT NULL,
                embedding_model TEXT,
                index_error_message TEXT,
                last_updated_ns INTEGER NOT NULL DEFAULT 0,
                last_indexed_ns INTEGER NOT NULL DEFAULT 0
            )
            """
        )

    @staticmethod
    def _to_row(record: FileRecord) -> tuple[object, ...]:
        return (
            record.file_id,
            str(record.absolute_path),
            record.relative_path.as_posix(),
            record.filename,
            record.extension,
            record.size_bytes,
            record.sha256,
            record.modified_ns,
            record.parent_folder.as_posix(),
            record.inferred_category,
            record.status.value,
            record.extraction_method,
            record.error_message,
            record.chunk_count,
            record.duplicate_group_id,
            record.keyword_index_status.value,
            record.vector_index_status.value,
            record.embedding_model,
            record.index_error_message,
            record.last_indexed_ns,
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> FileRecord:
        return FileRecord(
            file_id=row["file_id"],
            absolute_path=Path(row["absolute_path"]),
            relative_path=Path(row["relative_path"]),
            filename=row["filename"],
            extension=row["extension"],
            size_bytes=row["size_bytes"],
            sha256=row["sha256"],
            modified_ns=row["modified_ns"],
            parent_folder=Path(row["parent_folder"]),
            inferred_category=row["inferred_category"],
            status=FileStatus(row["status"]),
            extraction_method=row["extraction_method"],
            error_message=row["error_message"],
            chunk_count=row["chunk_count"],
            duplicate_group_id=row["duplicate_group_id"],
            keyword_index_status=IndexStatus(row["keyword_index_status"]),
            vector_index_status=IndexStatus(row["vector_index_status"]),
            embedding_model=row["embedding_model"],
            index_error_message=row["index_error_message"],
            last_indexed_ns=row["last_indexed_ns"],
        )
