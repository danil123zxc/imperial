# RAG Accuracy Improvements Design

Date: 2026-06-03
Workspace: `/Users/danil/Public/imperial`
Status: Approved in chat. Written spec awaiting user review before implementation planning.

## Context

The current Imperial RAG system already has the right safety spine: full file manifesting, extraction artifacts, local keyword search, Qdrant vector-store wiring, LangGraph query orchestration, strict evidence-only answering, citation validation, and Phoenix evaluation plumbing.

The live processed state at review time showed:

- 162 files scanned.
- 103 files indexed.
- 970 chunks written to `.imperial_rag/extracted/chunks.jsonl`.
- Keyword FTS populated with 970 chunks.
- Vector indexing skipped for all indexed files in the current processed state.
- 23 failed files, mostly Office lock/temp DOCX files such as `~$...docx` that are manifest-worthy but not real extractable documents.
- 20 PDFs and 3 JPGs with `no_text`, consistent with the current processed state not using OCR for scanned files.

The first optimization target is best answer accuracy. The highest-leverage change is therefore retrieval quality, not a generation rewrite. Strict citation generation should remain unchanged unless retrieval diagnostics prove otherwise.

## Goals

- Improve answer accuracy by selecting better evidence before generation.
- Preserve strict refusal behavior when evidence is absent or weak.
- Preserve citation validation as the final safety gate.
- Make retrieval hyperparameters explicit, configurable, and visible in query diagnostics.
- Support small, precise chunks while still providing enough surrounding context for policy answers.
- Add evaluation coverage that can prove whether retrieval changes helped.

## Non-Goals

- No replacement of the existing manifest, extraction, keyword index, Qdrant, LangGraph, or Phoenix foundations.
- No broad UI redesign.
- No compliance decision engine.
- No graph database.
- No answer generation based on model knowledge outside retrieved evidence.

## Documentation Check

Current LangChain documentation was checked with Context7 before choosing the retrieval design. The relevant documented patterns are:

- Vector-store retrievers can use MMR with `search_type="mmr"` and `search_kwargs` such as `k` and `fetch_k`.
- `ContextualCompressionRetriever` can wrap a base retriever with a reranker/compressor.
- LangChain documents `CohereRerank` and `VoyageAIRerank` as reranking compressors.

Cohere documentation was also checked for reranker selection. `rerank-v3.5` is the primary model choice because it is a dedicated text reranker, supports English and non-English languages including Russian, handles documents and semi-structured inputs, and has a 4096-token context per query-document pair. `rerank-multilingual-v3.0` is the fallback if `rerank-v3.5` is unavailable.

## Architecture

Keep the existing pipeline, but add a dedicated retrieval layer between runtime dependency wiring and the query workflow.

Query flow:

1. Normalize the question.
2. Retrieve a broad semantic candidate pool from Qdrant.
3. Retrieve a broad exact-term candidate pool from the keyword index.
4. Merge and deduplicate candidates by citation id, chunk id, and normalized content.
5. Rerank the merged candidate pool.
6. Expand top hits with neighboring chunks from the same file/source.
7. Cap the final evidence set.
8. Generate a strict cited answer from the final evidence.
9. Validate citations and refuse if citations are invalid or evidence is insufficient.

This keeps generation conservative while improving the evidence set available to it.

## Hyperparameters

Initial defaults:

| Setting | Default |
| --- | ---: |
| `chunk_size` | 400 characters |
| `chunk_overlap` | 50 characters |
| `vector_fetch_k` | 80 |
| `vector_k` | 32 |
| `keyword_limit` | 40 |
| `rerank_input_limit` | 60 deduped chunks |
| `rerank_top_n` | 12 chunks |
| `neighbor_window` | 1 chunk on each side |
| `final_evidence_min` | 18 chunks when at least 18 unique chunks remain after expansion |
| `final_evidence_max` | 24 chunks |
| `mmr_lambda_mult` | 0.4 |
| `primary_reranker` | `cohere:rerank-v3.5` |
| `fallback_reranker` | `cohere:rerank-multilingual-v3.0` |

These are hyperparameters: fixed knobs chosen before running retrieval. They are not learned by the model. They must be read from environment variables with these defaults so eval sweeps can tune them without code edits.

## Chunking

Use `chunk_size=400` and `chunk_overlap=50` as the requested default.

This is smaller than the current `1000/150` default. The advantage is sharper citations and better reranker granularity. The risk is that individual chunks may not contain enough policy context. The retrieval design compensates by:

- retrieving more candidates from both vector and keyword search;
- reranking a larger merged pool;
- expanding top reranked chunks with nearby chunks from the same file/source;
- capping final evidence at a bounded set of 18-24 chunks.

The chunking implementation should continue to preserve citation metadata: file id, relative path, source type, page number, sheet name, image index, chunk index, chunk id, and citation id.

## Retrieval Components

Add a focused retrieval module at `src/imperial_rag/retrieval.py`, rather than packing this logic into `runtime.py` or `workflows.py`.

Components:

- `RetrievalSettings`: typed defaults and environment loading for retrieval hyperparameters.
- `HybridRetriever`: calls Qdrant vector search and keyword search, then returns both candidate lists plus status metadata.
- `CandidateMerger`: deduplicates candidates by citation id, chunk id, and normalized content.
- `Reranker`: uses Cohere `rerank-v3.5` as primary and `rerank-multilingual-v3.0` as fallback.
- `FallbackRanker`: deterministic scoring used when Cohere is not configured or unavailable.
- `NeighborExpander`: includes previous and next chunks from the same file/source for top reranked hits when chunk metadata and neighbor text are present.
- `EvidenceSelector`: caps final evidence to the configured range and preserves citation order.

The existing LangGraph query workflow should call the retrieval layer and continue passing final evidence into the existing strict answer prompt.

## Vector Retrieval

The current processed state has no vector-indexed files. The implementation plan must include a re-ingestion path with vector indexing enabled after Qdrant is running.

Vector retrieval should prefer MMR-style diversity:

- Fetch up to `vector_fetch_k=80` semantic candidates.
- Return `vector_k=32` diverse candidates.
- Use `mmr_lambda_mult=0.4` to balance relevance and diversity.

If the LangChain Qdrant integration supports `as_retriever(search_type="mmr", search_kwargs={...})` in the installed version, use it. If not, use the closest documented Qdrant vector-store method available in the installed package and keep the external retrieval interface stable.

## Keyword Retrieval

Keyword retrieval remains important because the corpus is Russian company policy text with exact department names, filenames, roles, forms, and process terms.

Use `keyword_limit=40`.

Keyword search should keep using normalized Russian-friendly text that includes:

- chunk text;
- file name;
- relative path;
- source type;
- section heading when present;
- sheet name when present.

Implementation must return keyword rank metadata for every keyword candidate. It may also return BM25 scores if the current SQLite FTS query can expose them cleanly. The SQLite storage model does not need to change unless tests show rank-only metadata is not enough for stable fallback ordering.

## Reranking

Primary reranker: Cohere `rerank-v3.5`.

Fallback reranker: Cohere `rerank-multilingual-v3.0`.

Why:

- The corpus and questions are primarily Russian.
- A dedicated reranker compares query and candidate text jointly, which is stronger than relying only on embedding distance or exact term matching.
- The model supports multilingual retrieval and document/semi-structured inputs.
- LangChain exposes Cohere reranking through `CohereRerank` and contextual compression patterns.

Rerank only the top `rerank_input_limit=60` deduped candidates to control cost and latency. Keep `rerank_top_n=12` before neighbor expansion.

When no Cohere key is configured or the reranker fails, use deterministic fallback ranking. The fallback score should combine:

- vector rank plus vector score if the vector-store call exposes one;
- keyword rank plus BM25 score if the SQLite query exposes one;
- exact query term presence in chunk text;
- exact query term presence in file name or path;
- source type boosts for body/table/page content;
- duplicate-group penalties when duplicate chunks compete.

## Neighbor Expansion

Small chunks need context restoration. After reranking:

- For each top reranked chunk, find previous and next chunk with the same `file_id` and `source_type`.
- Include those neighbors when they exist and have not already been selected.
- Keep original top-hit chunks ahead of neighbors.
- Cap final evidence at `final_evidence_max=24`.

Neighbor expansion requires enough metadata to locate adjacent chunks. The current chunk metadata includes `file_id`, `source_type`, and `chunk_index`; implementation should use those fields and load neighbor text from the persisted chunk artifact or keyword index.

## Runtime Diagnostics

Every query result should include retrieval diagnostics in addition to the answer and citations.

Example:

```python
{
    "retrieval": {
        "vector_candidates": 32,
        "keyword_candidates": 40,
        "merged_candidates": 58,
        "rerank_input": 58,
        "reranked_candidates": 12,
        "final_evidence": 21,
        "reranker": "cohere:rerank-v3.5",
        "vector_search_status": "ok",
        "keyword_search_status": "ok",
        "fallbacks": []
    }
}
```

These diagnostics must be present in the query result object and must be included in Phoenix traces when tracing is enabled.

## Error Handling

Retrieval failures should degrade gracefully:

- If Qdrant is unavailable, continue with keyword search and mark `vector_search_status="unavailable"`.
- If the vector index is empty or skipped, continue with keyword search and mark `vector_search_status="empty"` or `vector_search_status="skipped"`.
- If keyword search fails, continue with vector search and mark `keyword_search_status="unavailable"`.
- If Cohere is not configured, use deterministic fallback ranking and record `fallbacks=["reranker_missing_api_key"]`.
- If Cohere fails, use deterministic fallback ranking and record the fallback reason without exposing secrets.
- If reranking returns too few results, backfill from highest-ranked merged candidates.
- If final evidence is empty, use the existing strict refusal text.
- If generation cites unsupported sources, keep the existing citation validator and replace the answer with refusal.

## Evaluation

Evaluation is the acceptance gate for the accuracy work.

Test layers:

- Unit tests for retrieval settings, environment overrides, merge/dedupe, fallback ranking, and neighbor expansion.
- Integration tests with fake vector, keyword, and reranker clients to prove candidate counts and fallback behavior.
- Existing strict citation tests must keep passing.
- Existing Phoenix eval runner should continue to work.
- Expand `evals/questions.jsonl` from 7 cases to 30 Russian questions.

The expanded eval set should include:

- direct policy questions;
- role/job-description questions;
- filename-sensitive questions;
- department/folder-sensitive questions;
- conflict/version questions;
- unsupported external-knowledge questions;
- scanned-document questions once OCR is enabled;
- near-duplicate questions phrased differently.

Initial success criteria:

- Full pytest suite passes.
- Current 7 evals do not regress.
- Expanded local eval pass count improves against the pre-change baseline.
- Unsupported questions still refuse.
- Citation validator still blocks unsupported citations.
- Every query result includes retrieval diagnostics.
- Retrieval diagnostics show the expected shape when both indexes contain enough matches: 32 vector candidates, 40 keyword candidates, 12 reranked chunks, and 18-24 final evidence chunks after neighbor expansion.

## Implementation Boundaries

Implementation must be staged:

1. Add settings and retrieval module with fake-client tests.
2. Wire runtime/query workflow to use the retrieval module.
3. Re-ingest with `chunk_size=400`, `chunk_overlap=50`, and vector indexing enabled.
4. Add reranker integration and deterministic fallback.
5. Add neighbor expansion.
6. Expand evals and compare before/after results.

Do not rewrite extraction, manifesting, citation validation, or the Streamlit UI unless required to surface retrieval diagnostics.
