# Light-Agentic LangGraph RAG Design

Date: 2026-06-13
Workspace: `/Users/danil/Public/imperial`
Status: Approved in chat. Written spec awaiting user review before implementation planning.

## Context

Imperial RAG is already partly LangChain and LangGraph based:

- query state is represented with LangGraph `StateGraph` in `src/imperial_rag/workflows.py`;
- documents use LangChain `Document`;
- Qdrant vector search is wired through LangChain Qdrant integration;
- Elasticsearch keyword search is owned by a small custom adapter in `src/imperial_rag/elasticsearch_keyword.py`;
- query runtime currently wraps retrieval behind `RetrievalService`, then passes retrieved evidence into the graph for answer generation and citation validation.

The current retrieval path is intentionally specialized for this private corpus:

```text
Qdrant vector search + Elasticsearch keyword search
  -> candidate merge
  -> RRF fusion
  -> DashScope rerank or deterministic fallback
  -> strict citation answer generation
  -> citation validation
```

Current LangChain documentation shows Elasticsearch integration surfaces such as `ElasticsearchRetriever` with a custom query body and `ElasticsearchStore` strategies including BM25, vector, sparse, and hybrid retrieval. Current LangGraph documentation shows agentic RAG graphs where a model can decide when to retrieve, rewrite questions, and generate answers. For this repo, the useful direction is not a fully model-driven agent, but a light-agentic graph that adds model-assisted query rewriting while preserving deterministic evidence handling.

## Goals

- Move query orchestration toward a more LangGraph-native shape.
- Add a light-agentic query rewrite step before retrieval.
- Keep retrieval deterministic after rewriting.
- Preserve the current hybrid retrieval behavior, including Elasticsearch keyword metadata boosts, Russian token normalization, page-number matching, Qdrant vector search, merge, RRF, rerank, diagnostics, and strict citation validation.
- Make graph state explicit enough that eval failures can explain whether the original or rewritten query was used.
- Keep default tests offline and deterministic.

## Non-Goals

- No fully agentic tool loop where the model controls the entire evidence path.
- No replacement of Qdrant with Elasticsearch vectors in this design.
- No wholesale replacement of the Elasticsearch adapter with LangChain Elasticsearch integration in this design.
- No change to answer-generation strictness, refusal behavior, citation formatting, or citation validation policy.
- No change to ingestion, chunking, indexing, local Compose services, or generated corpus state.

## Chosen Approach

Use a light-agentic LangGraph query workflow with a best-effort query rewrite node.

The graph should always retrieve evidence. The model may improve the retrieval query, but it should not decide whether retrieval is needed and should not decide which evidence is valid. If rewriting fails, produces an empty value, or produces an invalid value, the graph falls back to the normalized original query.

This approach gives the project a clearer LangGraph-native architecture without sacrificing the repo-specific retrieval behavior that is already tuned for the Imperial corpus.

## Alternatives Considered

### Query-Rewrite Gate

Add a model-assisted rewrite node before deterministic retrieval.

Pros:

- improves retrieval query quality while keeping behavior testable;
- preserves the current evidence pipeline;
- makes rewrite behavior visible in diagnostics;
- small enough for a focused implementation.

Cons:

- does not remove much custom retrieval code by itself;
- requires careful fallback behavior when the model is unavailable.

This is the chosen approach.

### Retriever-as-Tool

Expose the hybrid retriever as a LangChain tool and let the model call it inside the graph.

Pros:

- more recognizable as an agentic LangChain pattern;
- could support later multi-step retrieval.

Cons:

- introduces model-dependent tool-call behavior;
- makes default tests and eval comparisons less stable;
- risks skipping retrieval or over-retrieving unless constrained.

This may be useful later, but it is too much control to give the model in the first architecture step.

### LangChain Retriever Unification

Wrap Qdrant and Elasticsearch behind LangChain retriever objects and compose those retrievers in graph nodes.

Pros:

- reduces custom adapter surface;
- makes future LangChain component reuse easier.

Cons:

- risks losing current Elasticsearch token handling, metadata boosts, page-number search, relaxed fallback search, and diagnostics unless those are rebuilt around the retrievers;
- larger migration surface than the current goal requires.

This should remain a later follow-up after the graph boundary is clearer.

## Architecture

The query workflow should become:

```text
question
  -> normalize_query
  -> rewrite_query
  -> retrieve_candidates
  -> merge_candidates
  -> fuse_candidates
  -> rerank_candidates
  -> generate_answer
  -> validate_citations
```

The new `rewrite_query` node is the only light-agentic step. Every downstream node remains deterministic.

`generate_answer` must use the original user question, not the rewritten retrieval query. The rewritten query exists only to improve search.

## Components

### QueryRewriteNode

Inputs:

- `question`;
- `normalized_query`;
- chat model or rewrite model callable.

Outputs:

- `retrieval_query`;
- rewrite diagnostics.

Behavior:

- asks the model for a concise search query for the private Imperial corpus;
- rejects empty, whitespace-only, or overlong rewrites;
- treats rewrites longer than 300 characters or 50 whitespace-delimited terms as overlong;
- falls back to `normalized_query` on model errors or invalid output;
- records whether rewriting was used, skipped, failed, or fell back.

### RetrievalCandidatesNode

Inputs:

- `retrieval_query`;
- vector search dependency;
- keyword search dependency;
- retrieval settings.

Outputs:

- `vector_candidates`;
- `keyword_candidates`;
- retrieval status diagnostics.

Behavior:

- calls Qdrant vector search with the retrieval query;
- calls Elasticsearch keyword search with the retrieval query;
- preserves current graceful failure behavior for vector and keyword search.

### CandidateMergeNode

Inputs:

- `vector_candidates`;
- `keyword_candidates`.

Outputs:

- `merged_candidates`.

Behavior:

- reuses the current candidate dedupe behavior based on citation id, chunk id, and normalized content.

### RrfFusionNode

Inputs:

- `merged_candidates`;
- RRF settings.

Outputs:

- `fused_candidates`.

Behavior:

- reuses current RRF scoring and stores `_rrf_score` and `_fusion_rank` metadata.

### RerankNode

Inputs:

- `retrieval_query`;
- `fused_candidates`;
- reranker settings.

Outputs:

- final `evidence`;
- rerank diagnostics.

Behavior:

- uses the current DashScope reranker when configured;
- falls back to the deterministic reranker when the primary reranker is unavailable;
- preserves the current rerank input limit and top-N behavior.

### AnswerGenerationNode

Inputs:

- original `question`;
- final `evidence`;
- chat model dependency.

Outputs:

- `answer`;
- formatted citation and source fields.

Behavior:

- uses the original user question for strict answer prompting;
- returns the existing refusal text when there is no evidence.

### CitationValidationNode

Inputs:

- `answer`;
- `evidence`.

Outputs:

- `citations_valid`;
- `invalid_citations`.

Behavior:

- preserves the current citation validation behavior as a hard postcondition.

## Graph State

The graph state should distinguish original user intent from retrieval execution:

| Field | Purpose |
| --- | --- |
| `question` | Original user question. |
| `normalized_query` | Cleaned original question. |
| `retrieval_query` | Rewritten search query, or normalized fallback. |
| `vector_candidates` | Raw Qdrant results. |
| `keyword_candidates` | Raw Elasticsearch results. |
| `merged_candidates` | Deduped vector and keyword candidates. |
| `fused_candidates` | RRF-ranked candidates. |
| `evidence` | Final reranked evidence. |
| `retrieved_documents` | Compatibility alias for final evidence. |
| `answer` | Generated answer or refusal text. |
| `citations` | Formatted citation ids. |
| `sources` | Formatted source display values. |
| `citations_valid` | Citation validation result. |
| `invalid_citations` | Citations not grounded in evidence. |
| `retrieval` | Diagnostics for rewrite, retrieval, fusion, rerank, and fallbacks. |

## Error Handling

Query rewriting is best-effort:

- model unavailable: use `normalized_query`;
- model exception: use `normalized_query`;
- empty rewrite: use `normalized_query`;
- rewrite longer than 300 characters or 50 whitespace-delimited terms: use `normalized_query`;
- invalid response shape: use `normalized_query`.

Retrieval keeps current resilience:

- vector failures set `vector_search_status="unavailable"` and continue with keyword candidates;
- keyword failures set `keyword_search_status="unavailable"` and continue with vector candidates;
- provider mismatch is reported without calling the mismatched vector store;
- no evidence produces the existing refusal answer.

Diagnostics should include:

- `query_rewrite_status`;
- `query_rewrite_used`;
- `retrieval_query`;
- `fallbacks`;
- existing vector, keyword, merge, fusion, rerank, and final evidence counts.

## Testing

Default tests should not require live Elasticsearch, Qdrant, DashScope, or Phoenix.

Unit tests:

- rewrite success populates `retrieval_query`;
- rewrite exception falls back to `normalized_query`;
- empty or overlong rewrite falls back to `normalized_query`;
- answer generation receives the original `question`;
- citation validation still runs after answer generation.

Fake integration tests:

- fake vector and keyword search receive `retrieval_query`;
- fake answer generator receives original `question` and final evidence;
- graph state includes both `question` and `retrieval_query`;
- diagnostics record rewrite status and fallbacks.

Regression tests:

- existing retrieval tests continue to cover merge, RRF, rerank, graceful fallback, and diagnostics;
- workflow tests are updated for the expanded graph state;
- runtime tests verify dependencies are still lazy where appropriate.

Live tests remain opt-in and are not required for this design because ingestion and service wiring do not change.

## Acceptance Criteria

- Query workflow includes a query rewrite node before retrieval.
- The graph always retrieves; the model does not choose to skip retrieval.
- Rewritten query is used only for retrieval.
- Answer generation uses the original user question.
- Invalid or failed rewrites fall back to the normalized query.
- Current hybrid retrieval behavior is preserved.
- Current strict citation/refusal behavior is preserved.
- Diagnostics expose rewrite behavior and the retrieval query used.
- Default pytest remains offline and deterministic.
- No secrets, generated corpus artifacts, Elasticsearch data, Qdrant data, Phoenix traces, or eval outputs are committed.
