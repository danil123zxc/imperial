# Elasticsearch Keyword Search Design

Date: 2026-06-11
Workspace: `/Users/danil/Public/imperial`
Status: Approved in chat. Written spec awaiting user review before implementation planning.

## Context

Imperial RAG currently uses three SQLite-backed local state surfaces:

- `.imperial_rag/manifest.sqlite3` records discovered files, extraction status, chunk counts, duplicate groups, keyword/vector index status, and indexing errors.
- the OCR cache stores OCR text keyed by file hash and image id so repeated ingestion does not rerun expensive OCR.
- `.imperial_rag/keyword.sqlite3` is the keyword-search index.

This design replaces only the keyword-search SQLite database. The manifest and OCR cache remain SQLite because they are small local operational state, not search infrastructure.

The current keyword path is already well isolated:

- ingestion calls `KeywordIndex.replace_all(chunks)`;
- runtime creates a keyword-search object from settings;
- retrieval calls `search_with_scores(query, limit=...)` or `search(query, limit=...)`;
- returned documents carry `_keyword_rank` and `_keyword_score`;
- the existing hybrid path then merges vector and keyword candidates, applies RRF, reranks, and sends reranked evidence to answer generation.

That boundary lets Elasticsearch replace the keyword index without rewriting vector search, RRF, reranking, tracing, or answer generation.

Context7 docs were checked for current Elasticsearch and the official Python client. The Python client supports direct `client.search(...)` calls and a DSL layer; this design uses direct client calls for a small, explicit adapter.

## Goals

- Replace `.imperial_rag/keyword.sqlite3` with a required local Elasticsearch keyword index.
- Keep `manifest.sqlite3` and the OCR cache unchanged.
- Preserve current retrieval behavior and diagnostics as much as possible.
- Preserve current Russian-oriented normalization, stopword filtering, stemming, and bounded relaxed matching.
- Make Elasticsearch part of the normal local service stack.
- Keep default tests offline and fast, with live Elasticsearch tests opt-in.

## Non-Goals

- No replacement of Qdrant vector search.
- No replacement of `manifest.sqlite3`.
- No replacement of the OCR cache.
- No broad search package rewrite that combines Qdrant and Elasticsearch.
- No new answer-generation, reranking, or citation behavior.
- No automatic migration from the old SQLite keyword database. The Elasticsearch index is rebuilt from chunks during ingestion.

## Chosen Approach

Use a keyword-search interface plus an Elasticsearch implementation.

Introduce a small interface or protocol with the existing keyword-search contract:

- `replace_all(documents: list[Document]) -> None`
- `index_documents(documents: list[Document]) -> None`
- `search(query: str, limit: int = 5, k: int | None = None) -> list[Document]`
- `search_with_scores(query: str, limit: int = 5, k: int | None = None) -> list[KeywordHit]`

Implement that contract with `ElasticsearchKeywordIndex`. Retrieval should continue depending only on this contract, not on Elasticsearch details.

This is preferred over rewriting the existing `KeywordIndex` in place because the new service behavior is materially different from a local SQLite file. Honest naming will make operational failures and future tests easier to understand.

## Architecture

Elasticsearch becomes a required local service for keyword search. The repo no longer creates or reads `.imperial_rag/keyword.sqlite3` for keyword retrieval.

Configuration adds:

| Setting | Default | Purpose |
| --- | --- | --- |
| `ELASTICSEARCH_URL` | `http://localhost:9200` | Local Elasticsearch endpoint. |
| `ELASTICSEARCH_INDEX` | `imperial_keyword_chunks` | Keyword chunk index name. |

`Settings` should expose these values as `elasticsearch_url` and `elasticsearch_index`. `keyword_db_path` is no longer active keyword-search configuration. Implementation should remove tests that assert keyword search depends on that path, and may leave the property temporarily only as deprecated compatibility surface if another module still imports it.

`compose.yaml` should add a localhost-only single-node Elasticsearch service. Local startup docs should make Elasticsearch part of the normal stack alongside the existing local services.

## Components

### Keyword Protocol

The protocol is the boundary used by ingestion, runtime, and retrieval. It keeps tests simple and keeps the retrieval stack independent of Elasticsearch client internals.

### ElasticsearchKeywordIndex

This adapter owns:

- client construction;
- health checks where needed;
- index creation and mapping;
- full index replacement during ingestion;
- bulk indexing;
- keyword query construction;
- relaxed query fallback;
- conversion from Elasticsearch hits to LangChain `Document` objects.

### Keyword Helpers

Keep the existing helper behavior:

- `normalize_search_text(...)`;
- `_keyword_query_tokens(...)`;
- stopword filtering;
- lightweight suffix stemming;
- bounded relaxed-token attempts.

These helpers should be moved or kept in a neutral keyword module so both tests and the Elasticsearch adapter can use them without depending on SQLite.

## Index Shape

Store one Elasticsearch document per chunk.

Fields:

| Field | Type | Purpose |
| --- | --- | --- |
| `chunk_id` | keyword | Stable document id and dedupe key. |
| `text` | text, `index: false` | Original chunk text returned as `Document.page_content`; not used for matching. |
| `normalized_text` | text | Normalized searchable text used for keyword matching. |
| `metadata` | object, `enabled: false` | Citation and chunk metadata returned from `_source`; not indexed to avoid mapping drift. |

`normalized_text` should include the same searchable content as today:

- chunk body;
- filename;
- relative path;
- section heading;
- source type.

The Elasticsearch document id should be the current `stable_chunk_id(document)` value.

## Ingestion Flow

The ingestion flow remains corpus-wide and precomputed:

1. scan source documents;
2. refresh the manifest;
3. extract text and OCR where enabled;
4. build chunks;
5. write `chunks.jsonl`;
6. rebuild the Elasticsearch keyword index from the full chunk list;
7. optionally index vectors in Qdrant;
8. update manifest index statuses.

`replace_all(chunks)` should rebuild the Elasticsearch index from the supplied chunk list. If Elasticsearch is unavailable, keyword indexing fails. Because Elasticsearch is required, ingestion should not silently fall back to SQLite.

If there are zero chunks, the Elasticsearch index should still be cleared or recreated so stale keyword hits do not remain.

## Query Flow

Runtime creates `ElasticsearchKeywordIndex(settings)`.

Retrieval continues to call:

```python
hits = keyword_search.search_with_scores(query, limit=settings.keyword_limit)
```

Returned hit documents must include:

- original page content;
- original metadata;
- `_keyword_rank`;
- `_keyword_score`.

`HybridRetriever`, `CandidateMerger`, `RrfCandidateFusion`, `Reranker`, Phoenix retrieval spans, and final evidence behavior should not change.

## Query Behavior

Preserve the current semantics before optimizing:

1. Normalize the user query with the existing helper.
2. Remove low-value stopwords where possible.
3. Search for all meaningful normalized tokens.
4. If there are no hits, try bounded relaxed-token searches.
5. Do not use a broad OR fallback.
6. Sort by Elasticsearch `_score`.
7. Add `_keyword_rank` by hit order and `_keyword_score` from `_score`.

This keeps keyword search useful for exact Russian terms without reintroducing noisy broad matches.

The first implementation should use an explicit `bool.must` query with one `match` clause against `normalized_text` per normalized token. Relaxed searches repeat the same shape with each bounded relaxed token set. A later tuning pass may introduce language-specific analyzers, but analyzer tuning is not required for the migration.

## Required-Service Behavior

Required means:

- local docs and scripts expect Elasticsearch to be running;
- ingestion fails the keyword index step if Elasticsearch is unreachable;
- no SQLite keyword fallback is used;
- `.imperial_rag/keyword.sqlite3` is obsolete and ignored.

At query time, retrieval may still catch Elasticsearch exceptions and mark `keyword_search_status="unavailable"` to preserve existing diagnostic behavior and avoid crashing the entire UI. That is not a fallback to SQLite; it is graceful failure reporting when a required service is down.

## Operations

Add Elasticsearch to `compose.yaml` as a local single-node service bound to localhost. Use a pinned official Elasticsearch image, `discovery.type=single-node`, local data volume, and local-development security settings that do not require credentials on `127.0.0.1:9200`.

The primary local command is:

```bash
docker compose up elasticsearch
```

Add a helper script only if it keeps parity with existing service scripts. If added, the command should be:

```bash
./scripts/start_elasticsearch.sh
```

Update local run instructions so keyword search requires Elasticsearch before ingestion and before runtime queries.

The old `.imperial_rag/keyword.sqlite3` file may remain on disk from earlier runs, but the application should not read it after this migration. It can be manually deleted as generated local state.

## Testing

Default tests should remain offline and deterministic.

Unit tests:

- query token normalization and relaxed token generation;
- Elasticsearch query body construction;
- hit-to-`Document` conversion;
- `_keyword_rank` and `_keyword_score` metadata;
- settings defaults and environment overrides;
- ingestion failure behavior when the keyword index raises.

Retrieval tests:

- continue using fake keyword-search objects;
- assert the existing keyword diagnostics and RRF/rerank behavior still work.

Pipeline tests:

- inject a fake keyword index implementation;
- assert `replace_all(chunks)` is called;
- assert manifest status updates still reflect keyword index success or failure.

Live Elasticsearch tests:

- opt in with `IMPERIAL_RAG_LIVE_ELASTICSEARCH=1`;
- verify Elasticsearch health;
- create or reset a test index;
- index a few documents;
- search exact and relaxed Russian terms;
- clean up the test index.

Normal full pytest should not require Elasticsearch. Live tests should skip with a clear reason unless opted in.

## Acceptance Criteria

- Runtime keyword retrieval uses Elasticsearch, not `.imperial_rag/keyword.sqlite3`.
- Ingestion rebuilds an Elasticsearch keyword index from chunks.
- `manifest.sqlite3` and the OCR cache are unchanged.
- Returned keyword hits preserve `Document` content, citation metadata, `_keyword_rank`, and `_keyword_score`.
- Existing hybrid retrieval stages and diagnostics continue to work.
- Elasticsearch is documented as a required local service.
- Default pytest remains offline.
- Opt-in live Elasticsearch tests prove indexing and search end to end.
- No secrets, generated corpus artifacts, Elasticsearch data directories, Qdrant data, Phoenix traces, or eval outputs are committed.

## Implementation Boundaries

Implement in stages:

1. Add Elasticsearch dependency and settings.
2. Extract or preserve keyword helper functions independent of SQLite.
3. Add the keyword protocol and `ElasticsearchKeywordIndex`.
4. Wire ingestion to rebuild Elasticsearch.
5. Wire runtime to construct Elasticsearch keyword search.
6. Add local service configuration and docs.
7. Add unit and fake-based integration tests.
8. Add opt-in live Elasticsearch tests.
9. Remove or obsolete SQLite keyword-index tests and references.
