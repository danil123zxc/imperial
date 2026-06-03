# Imperial RAG Database Schema Diagram

Generated from the live workspace on 2026-06-03.

## Live Local Databases

| Database | Path | Tables | Current rows |
| --- | --- | --- | --- |
| Manifest SQLite | `.imperial_rag/manifest.sqlite3` | `files` | 162 files |
| Keyword SQLite FTS | `.imperial_rag/keyword.sqlite3` | `chunks_fts` plus FTS5 internal tables | 970 chunks |

The `documents/**/Thumbs.db` files are Windows thumbnail artifacts, not project databases.

## App-Owned Schema

```mermaid
erDiagram
    FILES {
        TEXT file_id PK
        TEXT absolute_path
        TEXT relative_path
        TEXT filename
        TEXT extension
        INTEGER size_bytes
        TEXT sha256
        INTEGER modified_ns
        TEXT parent_folder
        TEXT inferred_category
        TEXT status
        TEXT extraction_method
        TEXT error_message
        INTEGER chunk_count
        TEXT duplicate_group_id
        TEXT keyword_index_status
        TEXT vector_index_status
        TEXT embedding_model
        TEXT index_error_message
        INTEGER last_updated_ns
        INTEGER last_indexed_ns
    }

    CHUNKS_FTS {
        TEXT chunk_id
        TEXT text
        TEXT normalized_text
        TEXT metadata
    }

    QDRANT_IMPERIAL_CHUNKS {
        UUID point_id PK
        FLOAT_VECTOR embedding
        TEXT page_content
        JSON metadata
    }

    OCR_CACHE {
        TEXT file_hash PK
        TEXT image_id PK
        TEXT text
        TEXT method
        INTEGER updated_ns
    }

    FILES ||--o{ CHUNKS_FTS : "file_id inside metadata JSON"
    FILES ||--o{ QDRANT_IMPERIAL_CHUNKS : "file_id inside metadata payload"
    FILES ||--o{ OCR_CACHE : "sha256 to file_hash"
    CHUNKS_FTS ||--o| QDRANT_IMPERIAL_CHUNKS : "same chunk content/metadata"
```

Notes:

- `FILES` is the manifest table for scanned corpus files.
- `CHUNKS_FTS` is the searchable SQLite FTS5 table. Its `metadata` column is JSON containing chunk/file citation metadata.
- `QDRANT_IMPERIAL_CHUNKS` is the configured Qdrant collection name from `QDRANT_COLLECTION`, defaulting to `imperial_chunks`. Qdrant was not running during this check, so this is the configured application shape rather than a live collection dump.
- `OCR_CACHE` is code-defined as `.imperial_rag/ocr_cache.sqlite3`, but that file does not currently exist in this workspace.

## Keyword FTS5 Internal Tables

SQLite FTS5 creates implementation tables behind `chunks_fts`. These are part of the live keyword database, but application code should treat `chunks_fts` as the public table.

```mermaid
erDiagram
    CHUNKS_FTS {
        TEXT chunk_id
        TEXT text
        TEXT normalized_text
        TEXT metadata
    }

    CHUNKS_FTS_DATA {
        INTEGER id PK
        BLOB block
    }

    CHUNKS_FTS_IDX {
        ANY segid PK
        ANY term PK
        ANY pgno
    }

    CHUNKS_FTS_CONTENT {
        INTEGER id PK
        ANY c0
        ANY c1
        ANY c2
        ANY c3
    }

    CHUNKS_FTS_DOCSIZE {
        INTEGER id PK
        BLOB sz
    }

    CHUNKS_FTS_CONFIG {
        ANY k PK
        ANY v
    }

    CHUNKS_FTS ||--|| CHUNKS_FTS_DATA : "FTS5 storage"
    CHUNKS_FTS ||--|| CHUNKS_FTS_IDX : "FTS5 term index"
    CHUNKS_FTS ||--|| CHUNKS_FTS_CONTENT : "FTS5 content shadow"
    CHUNKS_FTS ||--|| CHUNKS_FTS_DOCSIZE : "FTS5 doc sizes"
    CHUNKS_FTS ||--|| CHUNKS_FTS_CONFIG : "FTS5 config"
```

## Chunk Metadata Payload

The `chunks_fts.metadata` JSON and Qdrant document metadata carry chunk citation fields. In the current generated chunk artifact, every chunk has:

```text
chunk_id
chunk_index
citation_id
duplicate_group_id
file_extension
file_hash
file_id
file_name
file_path
inferred_category
parent_folder
relative_path
source_type
```

Additional metadata can appear for source-specific extraction:

```text
sheet_name
page_number
render_dpi
image_index
embedded_media_name
image_hash
ocr_method
ocr_cached
```

In the current `.imperial_rag/extracted/chunks.jsonl`, only `sheet_name` appears among those optional fields.

## Service Databases

```mermaid
flowchart LR
    app["Imperial RAG app"]
    manifest["SQLite: .imperial_rag/manifest.sqlite3<br/>files"]
    keyword["SQLite FTS5: .imperial_rag/keyword.sqlite3<br/>chunks_fts"]
    qdrant["Qdrant: http://localhost:6333<br/>collection imperial_chunks"]
    phoenix["Phoenix: http://localhost:6006<br/>container-managed storage volume phoenix_data"]

    app --> manifest
    app --> keyword
    app --> qdrant
    app --> phoenix
```

Qdrant and Phoenix were not running on localhost during this check, so their internal live schemas were not inspected.
