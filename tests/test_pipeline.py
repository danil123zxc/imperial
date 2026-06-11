from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

from imperial_rag.pipeline import run_ingestion


class FileStatus(str, Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    MANIFEST_ONLY = "manifest_only"
    NO_TEXT = "no_text"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class IndexStatus(str, Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class FakeSettings:
    workspace_root: Path
    elasticsearch_url: str = "http://127.0.0.1:9200"
    elasticsearch_index: str = "test_keyword_chunks"

    @property
    def documents_root(self) -> Path:
        return self.workspace_root / "documents"

    @property
    def manifest_db_path(self) -> Path:
        return self.workspace_root / ".imperial_rag" / "manifest.sqlite3"

    @property
    def extraction_root(self) -> Path:
        return self.workspace_root / ".imperial_rag" / "extracted"


class FakeManifestStore:
    last: "FakeManifestStore | None" = None

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.records = []
        self.status_updates = []
        self.index_updates = []
        FakeManifestStore.last = self

    def replace_records(self, records):
        self.records = list(records)

    def update_status(self, **kwargs):
        self.status_updates.append(kwargs)

    def update_index_status(self, **kwargs):
        self.index_updates.append(kwargs)


class FakeKeywordIndex:
    last_docs = None
    last_settings = None

    def __init__(self, settings) -> None:
        self.settings = settings
        FakeKeywordIndex.last_settings = settings

    def replace_all(self, documents):
        FakeKeywordIndex.last_docs = list(documents)


def test_run_ingestion_persists_chunks_and_updates_manifest(tmp_path, monkeypatch):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.txt").write_text("Регламент возврата брака.", encoding="utf-8")
    _install_fake_dependencies(monkeypatch)

    summary = run_ingestion(settings=FakeSettings(tmp_path), enable_ocr=False, index_vectors=False)

    chunks_path = tmp_path / ".imperial_rag" / "extracted" / "chunks.jsonl"
    rows = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    assert summary.total_files == 1
    assert summary.indexed_files == 1
    assert summary.chunk_count == 1
    assert rows[0]["metadata"]["relative_path"] == "policy.txt"
    assert rows[0]["metadata"]["chunk_id"] == "file1:body:0"
    assert FakeKeywordIndex.last_docs is not None
    assert FakeKeywordIndex.last_settings == FakeSettings(tmp_path)
    assert FakeManifestStore.last is not None
    assert FakeManifestStore.last.status_updates[0]["chunk_count"] == 1
    assert FakeManifestStore.last.index_updates[0]["keyword_index_status"] == IndexStatus.INDEXED
    assert FakeManifestStore.last.index_updates[0]["vector_index_status"] == IndexStatus.SKIPPED


def test_run_ingestion_uses_retrieval_chunk_settings(tmp_path, monkeypatch):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.txt").write_text("Регламент возврата брака.", encoding="utf-8")
    monkeypatch.setenv("IMPERIAL_RAG_CHUNK_SIZE", "321")
    monkeypatch.setenv("IMPERIAL_RAG_CHUNK_OVERLAP", "45")
    _install_fake_dependencies(monkeypatch)

    run_ingestion(settings=FakeSettings(tmp_path), enable_ocr=False, index_vectors=False)

    build_chunks = sys.modules["imperial_rag.chunking"].build_chunks
    assert build_chunks.calls == [{"chunk_size": 321, "chunk_overlap": 45}]


def test_run_ingestion_records_embedding_model_when_vector_indexed(tmp_path, monkeypatch):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.txt").write_text("Регламент возврата брака.", encoding="utf-8")
    _install_fake_dependencies(monkeypatch)

    summary = run_ingestion(settings=FakeSettings(tmp_path), enable_ocr=False, index_vectors=True)

    assert summary.vector_indexed is True
    assert FakeManifestStore.last is not None
    assert FakeManifestStore.last.index_updates[0]["vector_index_status"] == IndexStatus.INDEXED
    assert FakeManifestStore.last.index_updates[0]["embedding_model"] == "text-embedding-v4:2048"


def test_run_ingestion_records_embedding_model_only_for_indexed_vectors(tmp_path, monkeypatch):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.txt").write_text("Регламент возврата брака.", encoding="utf-8")
    (docs / "empty.txt").write_text("", encoding="utf-8")
    _install_fake_dependencies(monkeypatch, include_no_text_record=True)

    summary = run_ingestion(settings=FakeSettings(tmp_path), enable_ocr=False, index_vectors=True)

    assert summary.vector_indexed is True
    assert FakeManifestStore.last is not None
    updates = {update["file_id"]: update for update in FakeManifestStore.last.index_updates}
    assert updates["file1"]["vector_index_status"] == IndexStatus.INDEXED
    assert updates["file1"]["embedding_model"] == "text-embedding-v4:2048"
    assert updates["file2"]["vector_index_status"] == IndexStatus.SKIPPED
    assert updates["file2"]["embedding_model"] is None


def test_run_ingestion_traces_aggregate_lifecycle_without_vector_stage(tmp_path, monkeypatch):
    from imperial_rag import pipeline as pipeline_module

    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.txt").write_text("Регламент возврата брака.", encoding="utf-8")
    _install_fake_dependencies(monkeypatch)
    records = _capture_pipeline_spans(monkeypatch, pipeline_module)

    summary = run_ingestion(settings=FakeSettings(tmp_path), enable_ocr=False, index_vectors=False)

    assert summary.chunk_count == 1
    assert [record["name"] for record in records] == [
        "ingest.corpus",
        "ingest.scan_files",
        "ingest.extract_files",
        "ingest.build_chunks",
        "ingest.keyword_index",
    ]
    assert records[0]["output"] == {
        "total_files": 1,
        "indexed_files": 1,
        "manifest_only_files": 0,
        "no_text_files": 0,
        "unsupported_files": 0,
        "failed_files": 0,
        "chunk_count": 1,
        "keyword_indexed": True,
        "vector_indexed": False,
    }
    assert records[1]["output"] == {"total_files": 1}
    assert records[2]["output"] == {
        "document_count": 1,
        "status_counts": {"indexed": 1},
        "failed_file_count": 0,
    }
    assert records[3]["output"] == {"document_count": 1, "chunk_count": 1, "chunk_size": 400, "chunk_overlap": 50}
    assert records[4]["output"] == {"chunk_count": 1, "indexed": True}


def test_run_ingestion_traces_vector_stage_only_when_enabled(tmp_path, monkeypatch):
    from imperial_rag import pipeline as pipeline_module

    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.txt").write_text("Регламент возврата брака.", encoding="utf-8")
    _install_fake_dependencies(monkeypatch)
    records = _capture_pipeline_spans(monkeypatch, pipeline_module)

    summary = run_ingestion(settings=FakeSettings(tmp_path), enable_ocr=False, index_vectors=True)

    assert summary.vector_indexed is True
    assert [record["name"] for record in records] == [
        "ingest.corpus",
        "ingest.scan_files",
        "ingest.extract_files",
        "ingest.build_chunks",
        "ingest.keyword_index",
        "ingest.vector_index",
    ]
    assert records[5]["output"] == {"chunk_count": 1, "indexed": True}


def test_run_ingestion_traces_extraction_failure_summary(tmp_path, monkeypatch):
    from imperial_rag import pipeline as pipeline_module

    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.txt").write_text("Регламент возврата брака.", encoding="utf-8")
    (docs / "broken.docx").write_text("broken", encoding="utf-8")
    _install_fake_dependencies(monkeypatch, include_failed_record=True)
    records = _capture_pipeline_spans(monkeypatch, pipeline_module)

    summary = run_ingestion(settings=FakeSettings(tmp_path), enable_ocr=False, index_vectors=False)

    assert summary.failed_files == 1
    assert records[2]["name"] == "ingest.extract_files"
    assert records[2]["output"] == {
        "document_count": 1,
        "status_counts": {"failed": 1, "indexed": 1},
        "failed_file_count": 1,
        "failed_files": [{"file_id": "file3", "message": "extract failed"}],
    }


def _capture_pipeline_spans(monkeypatch, pipeline_module):
    records = []

    @contextmanager
    def fake_trace_pipeline_step(name, input_value, *, attributes=None):
        record = {
            "name": name,
            "input": input_value,
            "attributes": dict(attributes or {}),
            "output": None,
        }

        class FakeSpan:
            def set_output(self, output):
                record["output"] = output

            def set_attribute(self, key, value):
                record.setdefault("attributes_set", {})[key] = value

        records.append(record)
        yield FakeSpan()

    monkeypatch.setattr(pipeline_module, "trace_pipeline_step", fake_trace_pipeline_step, raising=False)
    return records


def _install_fake_dependencies(
    monkeypatch,
    *,
    include_no_text_record: bool = False,
    include_failed_record: bool = False,
) -> None:
    config = ModuleType("imperial_rag.config")
    config.Settings = FakeSettings

    retrieval = ModuleType("imperial_rag.retrieval")

    class RetrievalSettings:
        def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        @classmethod
        def from_env(cls):
            return cls(
                chunk_size=_safe_env_int("IMPERIAL_RAG_CHUNK_SIZE", 400),
                chunk_overlap=_safe_env_int("IMPERIAL_RAG_CHUNK_OVERLAP", 50),
            )

    retrieval.RetrievalSettings = RetrievalSettings

    record = SimpleNamespace(
        file_id="file1",
        absolute_path=Path("/fake/policy.txt"),
        relative_path=Path("policy.txt"),
        filename="policy.txt",
    )
    no_text_record = SimpleNamespace(
        file_id="file2",
        absolute_path=Path("/fake/empty.txt"),
        relative_path=Path("empty.txt"),
        filename="empty.txt",
    )
    failed_record = SimpleNamespace(
        file_id="file3",
        absolute_path=Path("/fake/broken.docx"),
        relative_path=Path("broken.docx"),
        filename="broken.docx",
        extension=".docx",
    )
    records = [record]
    if include_no_text_record:
        records.append(no_text_record)
    if include_failed_record:
        records.append(failed_record)
    manifest = ModuleType("imperial_rag.manifest")
    manifest.FileStatus = FileStatus
    manifest.IndexStatus = IndexStatus
    manifest.ManifestStore = FakeManifestStore
    manifest.scan_files = lambda documents_root: records
    manifest.assign_duplicate_groups = lambda records: records

    document = SimpleNamespace(
        page_content="Регламент возврата брака.",
        metadata={"file_id": "file1", "relative_path": "policy.txt", "source_type": "body"},
    )
    extraction = ModuleType("imperial_rag.extraction")

    def extract_file(record, **kwargs):
        if record.file_id == "file3":
            raise RuntimeError("extract failed")
        if record.file_id == "file2":
            return SimpleNamespace(
                status=FileStatus.NO_TEXT,
                documents=[],
                extraction_method="fake",
                message="",
            )
        return SimpleNamespace(
            status=FileStatus.INDEXED,
            documents=[document],
            extraction_method="fake",
            message="",
        )

    extraction.extract_file = extract_file

    chunk = SimpleNamespace(
        page_content=document.page_content,
        metadata={**document.metadata, "chunk_id": "file1:body:0"},
    )
    chunking = ModuleType("imperial_rag.chunking")

    def build_chunks(documents, chunk_size=None, chunk_overlap=None):
        build_chunks.calls.append({"chunk_size": chunk_size, "chunk_overlap": chunk_overlap})
        return [chunk]

    build_chunks.calls = []
    chunking.build_chunks = build_chunks

    elasticsearch_keyword = ModuleType("imperial_rag.elasticsearch_keyword")
    elasticsearch_keyword.ElasticsearchKeywordIndex = FakeKeywordIndex

    indexing = ModuleType("imperial_rag.indexing")
    indexing.ElasticsearchKeywordIndex = FakeKeywordIndex
    indexing.KeywordIndex = FakeKeywordIndex
    indexing.create_qdrant_vector_store = lambda settings: SimpleNamespace(add_documents=lambda documents, ids: ids)
    indexing.index_vector_documents = lambda documents, settings=None, vector_store=None: [
        doc.metadata["chunk_id"] for doc in documents
    ]
    indexing.index_documents = lambda vector_store, documents: [doc.metadata["chunk_id"] for doc in documents]
    indexing.embedding_model_identifier = lambda: "text-embedding-v4:2048"

    for module in (config, retrieval, manifest, extraction, chunking, elasticsearch_keyword, indexing):
        monkeypatch.setitem(sys.modules, module.__name__, module)


def _safe_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default
