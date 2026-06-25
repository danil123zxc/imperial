from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

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
    qdrant_collection: str = "test_qdrant_chunks"
    extraction_root_override: Path | None = None
    baseline_extraction_root: Path | None = None

    @property
    def documents_root(self) -> Path:
        return self.workspace_root / "documents"

    @property
    def manifest_db_path(self) -> Path:
        return self.workspace_root / ".imperial_rag" / "manifest.sqlite3"

    @property
    def extraction_root(self) -> Path:
        if self.extraction_root_override is not None:
            return self.extraction_root_override
        return self.workspace_root / ".imperial_rag" / "extracted"


def _fake_module(name: str) -> Any:
    return ModuleType(name)


class FakeManifestStore:
    last: "FakeManifestStore | None" = None

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.records = []
        self.status_updates = []
        self.index_updates = []
        self.closed = False
        FakeManifestStore.last = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True
        return False

    def replace_records(self, records):
        self.records = list(records)

    def update_status(self, **kwargs):
        self.status_updates.append(kwargs)

    def update_index_status(self, **kwargs):
        self.index_updates.append(kwargs)


class FakeKeywordSearchIndex:
    last_docs = None
    last_settings = None

    def __init__(self, settings) -> None:
        self.settings = settings
        FakeKeywordSearchIndex.last_settings = settings

    def replace_all(self, documents):
        FakeKeywordSearchIndex.last_docs = list(documents)


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
    assert FakeKeywordSearchIndex.last_docs is not None
    assert FakeKeywordSearchIndex.last_settings == FakeSettings(tmp_path)
    assert FakeManifestStore.last is not None
    assert FakeManifestStore.last.status_updates[0]["chunk_count"] == 1
    assert FakeManifestStore.last.index_updates[0]["keyword_index_status"] == IndexStatus.INDEXED
    assert FakeManifestStore.last.index_updates[0]["vector_index_status"] == IndexStatus.SKIPPED
    assert FakeManifestStore.last.closed is True


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
    monkeypatch.setenv("IMPERIAL_RAG_CHUNK_SIZE", "650")
    monkeypatch.setenv("IMPERIAL_RAG_CHUNK_OVERLAP", "80")
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
    assert {
        key: records[0]["output"][key]
        for key in (
            "total_files",
            "indexed_files",
            "manifest_only_files",
            "no_text_files",
            "unsupported_files",
            "failed_files",
            "chunk_count",
            "keyword_indexed",
            "vector_indexed",
        )
    } == {
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
    assert records[0]["output"]["corpus_version"].startswith("corpus_sha256:")
    assert records[0]["output"]["index_version"].startswith("index_sha256:")
    assert records[0]["output"]["keyword_index"] == "test_keyword_chunks"
    assert records[1]["output"] == {
        "total_files": 1,
        "supported_files": 1,
        "unsupported_files": 0,
        "extension_counts": {".txt": 1},
        "duplicate_group_count": 0,
    }
    assert records[2]["output"] == {
        "document_count": 1,
        "status_counts": {"indexed": 1},
        "extraction_methods": {"fake": 1},
        "failed_file_count": 0,
    }
    assert {
        key: records[3]["output"][key]
        for key in ("document_count", "chunk_count", "chunk_size", "chunk_overlap")
    } == {"document_count": 1, "chunk_count": 1, "chunk_size": 650, "chunk_overlap": 80}
    assert records[3]["output"]["corpus_version"].startswith("corpus_sha256:")
    assert records[3]["output"]["chunk_hashes"]["count"] == 1
    assert records[4]["output"] == {
        "chunk_count": 1,
        "indexed": True,
        "elasticsearch_index": "test_keyword_chunks",
        "indexed_count": 1,
        "replace_all_success": True,
    }


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
    assert records[5]["output"] == {
        "chunk_count": 1,
        "indexed": True,
        "qdrant_collection": "test_qdrant_chunks",
        "added_id_count": 1,
        "embedding_model": "text-embedding-v4:2048",
        "embedding_dimensions": 2048,
    }


def test_run_ingestion_traces_lineage_and_index_metadata(tmp_path, monkeypatch):
    from imperial_rag import pipeline as pipeline_module

    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.txt").write_text("Регламент возврата брака.", encoding="utf-8")
    _install_fake_dependencies(monkeypatch)
    records = _capture_pipeline_spans(monkeypatch, pipeline_module)

    run_ingestion(settings=FakeSettings(tmp_path), enable_ocr=False, index_vectors=True)

    ingest_run_id = records[0]["attributes"]["imperial.ingest_run_id"]
    assert ingest_run_id
    assert records[0]["attributes"]["imperial.phase"] == "ingest"
    assert records[0]["attributes"]["imperial.step"] == "corpus"
    assert records[0]["attributes"]["imperial.trace_schema_version"] == "rag-v2"
    for record in records[1:]:
        assert record["attributes"]["imperial.ingest_run_id"] == ingest_run_id
        assert record["attributes"]["imperial.trace_schema_version"] == "rag-v2"

    assert records[1]["output"] == {
        "total_files": 1,
        "supported_files": 1,
        "unsupported_files": 0,
        "extension_counts": {".txt": 1},
        "duplicate_group_count": 0,
    }
    assert records[3]["output"]["corpus_version"].startswith("corpus_sha256:")
    assert records[3]["output"]["chunk_hashes"]["count"] == 1
    assert records[3]["output"]["chunk_hashes"]["top"][0].startswith("content_sha256:")
    assert records[4]["attributes"]["imperial.keyword_index"] == "test_keyword_chunks"
    assert records[4]["output"] == {
        "chunk_count": 1,
        "indexed": True,
        "elasticsearch_index": "test_keyword_chunks",
        "indexed_count": 1,
        "replace_all_success": True,
    }
    assert records[5]["attributes"]["imperial.qdrant_collection"] == "test_qdrant_chunks"
    assert records[5]["attributes"]["imperial.embedding_model"] == "text-embedding-v4:2048"
    assert records[5]["output"] == {
        "chunk_count": 1,
        "indexed": True,
        "qdrant_collection": "test_qdrant_chunks",
        "added_id_count": 1,
        "embedding_model": "text-embedding-v4:2048",
        "embedding_dimensions": 2048,
    }
    assert records[0]["output"]["corpus_version"] == records[3]["output"]["corpus_version"]
    assert records[0]["output"]["index_version"].startswith("index_sha256:")
    assert records[0]["output"]["keyword_index"] == "test_keyword_chunks"
    assert records[0]["output"]["qdrant_collection"] == "test_qdrant_chunks"
    assert records[0]["output"]["embedding_model"] == "text-embedding-v4:2048"
    lineage = json.loads((tmp_path / ".imperial_rag" / "extracted" / "index-lineage.json").read_text(encoding="utf-8"))
    assert lineage == {
        "ingest_run_id": ingest_run_id,
        "corpus_version": records[0]["output"]["corpus_version"],
        "index_version": records[0]["output"]["index_version"],
        "keyword_index": "test_keyword_chunks",
        "qdrant_collection": "test_qdrant_chunks",
        "embedding_model": "text-embedding-v4:2048",
        "keyword_indexed": True,
        "vector_indexed": True,
    }


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
        "extraction_methods": {"fake": 1, "python_docx": 1},
        "failed_file_count": 1,
        "failed_files": [{"file_id": "file3", "message": "extract failed"}],
    }


def test_run_ingestion_writes_old_to_new_id_map(tmp_path, monkeypatch):
    docs = tmp_path / "documents"
    docs.mkdir()
    (docs / "policy.txt").write_text("Регламент возврата брака.", encoding="utf-8")
    extracted = tmp_path / ".imperial_rag" / "extracted"
    extracted.mkdir(parents=True)
    (extracted / "chunks.jsonl").write_text(
        json.dumps(
            {
                "page_content": "old",
                "metadata": {"file_id": "file1", "chunk_id": "old-chunk", "citation_id": "old-citation"},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _install_fake_dependencies(monkeypatch)

    run_ingestion(settings=FakeSettings(tmp_path), enable_ocr=False, index_vectors=False)

    id_map_path = extracted / "old-to-new-id-map.json"
    id_map = json.loads(id_map_path.read_text(encoding="utf-8"))
    assert id_map["schema_version"] == "old-to-new-id-map-v1"
    assert id_map["rows"][0]["old_chunk_id"] == "old-chunk"
    assert id_map["rows"][0]["old_citation_id"] == "old-citation"
    assert id_map["rows"][0]["new_chunk_id"] == "file1:body:0"
    assert id_map["rows"][0]["status"] == "mapped"


def test_run_ingestion_can_write_shadow_artifacts_without_mutating_canonical_root(tmp_path, monkeypatch):
    canonical = tmp_path / ".imperial_rag" / "extracted"
    canonical.mkdir(parents=True)
    (canonical / "chunks.jsonl").write_text(
        json.dumps({"page_content": "canonical", "metadata": {"file_id": "file1", "chunk_id": "old"}}) + "\n",
        encoding="utf-8",
    )
    shadow = tmp_path / ".imperial_rag" / "extracted-shadow-v2"
    _install_fake_dependencies(monkeypatch)

    settings = FakeSettings(
        tmp_path,
        extraction_root_override=shadow,
        baseline_extraction_root=canonical,
    )
    run_ingestion(settings=settings, enable_ocr=False, index_vectors=False)

    assert "canonical" in (canonical / "chunks.jsonl").read_text(encoding="utf-8")
    assert (shadow / "chunks.jsonl").exists()
    assert (shadow / "old-to-new-id-map.json").exists()


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
    config = _fake_module("imperial_rag.config")
    config.Settings = FakeSettings

    retrieval = _fake_module("imperial_rag.retrieval")

    class RetrievalSettings:
        def __init__(self, chunk_size: int = 650, chunk_overlap: int = 80) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        @classmethod
        def from_env(cls):
            return cls(
                chunk_size=_safe_env_int("IMPERIAL_RAG_CHUNK_SIZE", 650),
                chunk_overlap=_safe_env_int("IMPERIAL_RAG_CHUNK_OVERLAP", 80),
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
    manifest = _fake_module("imperial_rag.manifest")
    manifest.FileStatus = FileStatus
    manifest.IndexStatus = IndexStatus
    manifest.ManifestStore = FakeManifestStore
    manifest.scan_files = lambda documents_root: records
    manifest.assign_duplicate_groups = lambda records: records

    document = SimpleNamespace(
        page_content="Регламент возврата брака.",
        metadata={"file_id": "file1", "relative_path": "policy.txt", "source_type": "body"},
    )
    extraction = _fake_module("imperial_rag.extraction")

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
    chunking = _fake_module("imperial_rag.chunking")

    class FakeBuildChunks:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def __call__(self, documents, chunk_size=None, chunk_overlap=None):
            self.calls.append({"chunk_size": chunk_size, "chunk_overlap": chunk_overlap})
            return [chunk]

    build_chunks = FakeBuildChunks()
    chunking.build_chunks = build_chunks

    elasticsearch_keyword = _fake_module("imperial_rag.elasticsearch_keyword")
    elasticsearch_keyword.ElasticsearchKeywordIndex = FakeKeywordSearchIndex

    indexing = _fake_module("imperial_rag.indexing")
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
