# LangChain Elasticsearch Retriever Design

Date: 2026-06-13
Workspace: `/Users/danil/Public/imperial`
Status: Approved in chat. Written spec awaiting user review before implementation planning.

## Context

Imperial RAG already uses maintained LangChain and Elasticsearch building blocks in several places:

- query and ingestion workflows use LangGraph `StateGraph`;
- documents use LangChain `Document`;
- chunking uses `langchain_text_splitters`;
- vector search uses `langchain_qdrant.QdrantVectorStore`;
- keyword indexing uses the official Elasticsearch Python client;
- retrieval currently combines Qdrant vector candidates and Elasticsearch keyword candidates before merge, RRF fusion, reranking, and strict citation answer generation.

The custom code in the keyword path is concentrated in `src/imperial_rag/elasticsearch_keyword.py` and `src/imperial_rag/keyword.py`. It creates the Elasticsearch index, bulk-indexes chunk documents, builds keyword query bodies, converts hits into LangChain `Document` objects, and exposes the repo's existing `search()` and `search_with_scores()` methods.

Current LangChain documentation shows RAG patterns built around retrievers, tools, and LangGraph workflows. Current Elasticsearch documentation shows native `multi_match`, BM25, hybrid search, RRF retrievers, and language analyzers. For this repo, the selected direction is to adopt a LangChain-style Elasticsearch retriever boundary for query-time keyword search while keeping corpus-specific indexing and citation metadata under repo control.

## Goals

- Make query-time Elasticsearch keyword search look like a LangChain retriever internally.
- Reduce bespoke query-time plumbing without changing the public keyword-search contract used by retrieval.
- Preserve the existing hybrid retrieval pipeline:
  - Qdrant vector search;
  - Elasticsearch keyword search;
  - candidate merge;
  - RRF fusion;
  - DashScope rerank or deterministic fallback;
  - strict citation answer generation;
  - citation validation.
- Preserve citation metadata and keyword diagnostics.
- Keep ingestion/index creation explicit and repo-owned for now.
- Keep default tests offline and deterministic.

## Non-Goals

- No replacement of Qdrant vector search.
- No move to Elasticsearch vector or Elasticsearch-native hybrid retrieval in this design.
- No replacement of merge, RRF, rerank, answer generation, or citation validation behavior.
- No model-driven agent tool loop for retrieval.
- No broad analyzer migration in the first implementation.
- No change to local/private Compose service boundaries.
- No generated corpus artifacts, Elasticsearch data, Qdrant data, Phoenix traces, eval outputs, or secrets committed.

## Chosen Approach

Add a LangChain-style retriever wrapper for Elasticsearch keyword search, then keep a small compatibility adapter that exposes the repo's existing keyword-search methods.

The query-time shape becomes:

```text
Elasticsearch index
  -> LangChain Elasticsearch retriever wrapper
  -> KeywordSearch-compatible adapter
  -> existing HybridRetriever
  -> merge/RRF/rerank/answer/citation validation
```

This approach gives the codebase a more standard LangChain boundary while avoiding a risky rewrite of the whole RAG path. It removes some custom query-time orchestration from the Elasticsearch adapter, but it deliberately keeps index shape, source metadata, and citation behavior explicit because those are domain-specific.

## Alternatives Considered

### ES-Native Adapter Simplification

Move more behavior into Elasticsearch mappings and analyzers, then simplify Python code around direct Elasticsearch client calls.

Pros:

- strongest reduction in custom token and stemming logic;
- uses Elasticsearch BM25 and analyzer behavior directly;
- likely improves lexical-search reliability over time.

Cons:

- changes query semantics more than the selected option;
- requires careful analyzer tuning and reindexing;
- less aligned with the user's selected LangChain integration direction.

This remains a good follow-up once the retriever boundary is in place.

### LangChain Retriever Wrapper

Wrap Elasticsearch query-time search in a LangChain-style retriever, then adapt the result back to the current keyword-search interface.

Pros:

- aligns with LangChain patterns already present in the repo;
- small enough to implement and test safely;
- keeps `HybridRetriever` stable;
- preserves corpus-specific indexing and metadata control;
- creates a clearer future path for retriever composition.

Cons:

- does not remove all custom Elasticsearch code;
- score handling may still require a small raw-hit path if the chosen LangChain retriever surface hides scores;
- ingestion and mapping code stay custom.

This is the chosen approach.

### Larger Elasticsearch Retrieval Move

Use Elasticsearch-native hybrid search or RRF and eventually collapse Qdrant plus custom fusion into Elasticsearch.

Pros:

- could remove more custom retrieval code later;
- may centralize lexical and vector retrieval in one service.

Cons:

- changes architecture beyond the keyword adapter;
- risks retrieval-quality regressions;
- requires rethinking vector storage, reranking inputs, diagnostics, and local service assumptions.

This is out of scope for this design.

## Architecture

The first implementation should introduce a query-time retriever wrapper while preserving the current external contract:

```python
search(query: str, limit: int = 5, k: int | None = None) -> list[Document]
search_with_scores(query: str, limit: int = 5, k: int | None = None) -> list[KeywordHit]
```

`HybridRetriever` should continue to depend only on that contract. It should not learn about LangChain retriever internals, Elasticsearch hits, or index mappings.

The Elasticsearch keyword module should become two layers:

1. an indexing layer that creates, clears, replaces, and bulk-indexes the keyword index;
2. a query layer that delegates retrieval to the LangChain-style retriever wrapper and converts documents into the repo's `KeywordHit` shape.

The existing `ElasticsearchKeywordIndex` class may remain as the facade that owns both layers. If implementation introduces an `ElasticsearchKeywordSearch` helper, runtime call sites should still be able to construct the existing facade name unless a deliberate rename is included in the implementation plan.

The query layer may still build a custom query body if the LangChain Elasticsearch retriever supports that as the most reliable way to preserve boosted fields and current matching behavior.

## Components

### ElasticsearchKeywordRetriever

Purpose: provide the LangChain-facing retriever surface for keyword search.

Responsibilities:

- accept a query string and limit;
- query the existing Elasticsearch index;
- return LangChain `Document` objects;
- preserve original chunk text as `page_content`;
- preserve source metadata needed for citations;
- preserve raw score information when the integration surface exposes it.

The first version should favor score-aware retrieval. If the selected LangChain retriever does not expose scores, the wrapper should retain a small Elasticsearch client call internally and still present a retriever-shaped API to the rest of the module.

### ElasticsearchKeywordSearch

Purpose: keep the existing repo contract stable for ingestion/runtime/retrieval consumers.

Responsibilities:

- expose `search()` and `search_with_scores()`;
- call `ElasticsearchKeywordRetriever`;
- resolve `k` versus `limit` the same way the current adapter does;
- convert retriever results to `KeywordHit` values;
- attach `_keyword_rank` and `_keyword_score` metadata.

This compatibility layer keeps `HybridRetriever` and existing tests from needing a broad rewrite.

### KeywordHit Conversion

Purpose: centralize conversion between retrieved documents and current keyword-hit semantics.

Responsibilities:

- copy metadata before mutation;
- add `_keyword_rank` by result order;
- add `_keyword_score` when available;
- never drop citation fields such as `citation_id`, `chunk_id`, `file_name`, `relative_path`, `section_heading`, `source_type`, `sheet_name`, or `page_number`.

## Data Flow

Ingestion stays mostly unchanged:

```text
extracted documents
  -> chunks
  -> stable chunk ids
  -> Elasticsearch documents with text, searchable fields, and metadata
```

Query-time keyword retrieval changes:

```text
question
  -> ElasticsearchKeywordSearch.search_with_scores()
  -> ElasticsearchKeywordRetriever.invoke/search
  -> list[Document]
  -> KeywordHit conversion
  -> HybridRetriever keyword_docs
```

The returned documents must preserve the current shape:

- `Document.page_content` is the stored chunk text;
- `Document.metadata` contains original citation metadata;
- `_keyword_rank` is present on keyword results;
- `_keyword_score` is present when Elasticsearch score data is available.

## Score Handling

Current retrieval uses keyword rank for RRF and the deterministic fallback ranker can use `_keyword_score`. The migration should therefore treat score preservation as important.

The preferred behavior is:

- return Elasticsearch `_score` as `_keyword_score`;
- set `_keyword_rank` from the result order;
- set `keyword_scores_available=True` in diagnostics when all returned keyword results have scores.

If a LangChain retriever hides raw scores:

- keep `_keyword_rank`;
- omit `_keyword_score` or set it only when known;
- preserve runtime behavior rather than failing the query.

Because `search_with_scores()` returns `KeywordHit` values rather than a diagnostics object, score availability should first be represented by whether `_keyword_score` is present on the returned documents. If the implementation adds a top-level diagnostic, `HybridRetriever` should derive `keyword_scores_available` from the returned hits instead of requiring a new mandatory search return type.

The implementation plan should choose the retriever integration that best preserves scores with the least custom code.

## Error Handling

Runtime query behavior should preserve the current graceful failure model:

- Elasticsearch exceptions during query should be caught by `HybridRetriever`;
- diagnostics should mark `keyword_search_status="unavailable"`;
- retrieval should continue with vector candidates if available;
- empty keyword results should produce `keyword_search_status="empty"`;
- score-unavailable behavior should not fail retrieval.

Ingestion behavior remains stricter:

- Elasticsearch is required keyword infrastructure;
- indexing should fail the keyword indexing step if Elasticsearch is unavailable;
- no SQLite keyword fallback should be reintroduced.

## Testing

Default tests should remain offline.

Unit tests:

- fake the retriever output and assert `search()` returns documents;
- fake scored retriever output and assert `search_with_scores()` returns `KeywordHit` values;
- verify `_keyword_rank` and `_keyword_score` metadata conversion;
- verify citation metadata is preserved;
- verify `k` overrides `limit`;
- verify score-unavailable behavior is explicit and non-fatal.

Retrieval tests:

- keep most `HybridRetriever` tests unchanged;
- add one focused test proving `HybridRetriever` can consume the new keyword adapter without depending on LangChain internals.

Elasticsearch adapter tests:

- keep query-body construction tests if the implementation still owns the query body;
- update tests that assume direct client calls from `ElasticsearchKeywordIndex`;
- keep index mapping and bulk-action tests for ingestion behavior.

Live Elasticsearch test:

- keep opt-in behavior behind `IMPERIAL_RAG_LIVE_ELASTICSEARCH=1`;
- create a temporary index;
- index a few chunks;
- retrieve through the new retriever-backed keyword adapter;
- assert citation metadata, rank, and score behavior;
- delete the temporary index at the end.

## Acceptance Criteria

- Query-time keyword search is internally routed through a LangChain-style Elasticsearch retriever wrapper.
- The public keyword-search contract remains compatible with current retrieval code.
- `HybridRetriever` does not need to know about LangChain or Elasticsearch internals.
- Ingestion still owns index creation, replacement, and bulk indexing.
- Returned keyword documents preserve content, citation metadata, `_keyword_rank`, and `_keyword_score` when available.
- Runtime diagnostics preserve current `ok`, `empty`, and `unavailable` keyword statuses.
- Default pytest remains offline and deterministic.
- Opt-in live Elasticsearch tests prove the retriever-backed path works end to end.
- The implementation does not change Qdrant, merge, RRF, rerank, answer generation, citation validation, or private/local service boundaries.
