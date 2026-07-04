# Phoenix Whole-Pipeline Tracing Improvement Plan

Generated: 2026-06-22 17:02:33 KST

## Goal

Make Phoenix show the whole Imperial RAG story clearly enough to debug bad answers:

```text
documents/ -> ingest/extract -> chunks -> Elasticsearch + Qdrant
question -> vector search + keyword search -> merge -> RRF -> rerank -> strict cited answer
```

This is a plan only. It does not implement code changes.

## Current Repo Reality

- Query traces already have a compact domain tree: `imperial_rag.query`, `retrieval`, `retrieval.vector_search`, `retrieval.keyword_search`, `retrieval.rerank`, `retrieval.final_evidence`, `answer.generate`, `answer.call_model`, and `answer.citation_check`.
- Merge and RRF details are currently summarized on the `retrieval` output as counts and fusion metadata, not visible as their own decision boundary.
- Candidate documents are intentionally opt-in through `IMPERIAL_RAG_TRACE_CANDIDATE_DOCUMENTS` or `IMPERIAL_RAG_TRACE_MODE=retrieval_debug`.
- LangChain/internal spans are suppressed by default with `IMPERIAL_RAG_TRACE_SUPPRESS_INTERNALS=true`; auto-instrumentation is opt-in with `IMPERIAL_RAG_TRACE_AUTO_INSTRUMENT`.
- Ingestion already emits `ingest.corpus`, `ingest.scan_files`, `ingest.extract_files`, `ingest.build_chunks`, `ingest.keyword_index`, and optional `ingest.vector_index`.
- DashScope embedding batch spans already exist as `embedding.dashscope.batch`, so the plan should connect/enrich them rather than inventing another embedding layer.

## Council Verdict

The council mostly agreed on the shape:

- Phoenix should show Imperial-owned domain decisions, not every LangChain/helper detail.
- RRF/fusion deserves explicit visibility because it changes ranking and can explain why a candidate moved or disappeared.
- Merge is useful for debugging dedupe, but it can become noise if promoted to a full default span with document payloads.
- Ingestion should be a batch lineage trace, not a corpus browser.
- The missing piece is a stable trace contract: names, span kinds, cardinality budgets, redaction rules, lineage fields, and tests that keep the trace tree from drifting.

## Recommended Target Shape

### Compact Default Mode

This is the normal Phoenix view:

```text
imperial_rag.query                 CHAIN
  retrieval                        CHAIN
    retrieval.vector_search        RETRIEVER
    retrieval.keyword_search       RETRIEVER
    retrieval.fusion               CHAIN
    retrieval.rerank               RERANKER
    retrieval.final_evidence       RETRIEVER
  answer.generate                  CHAIN
    answer.call_model              LLM
    answer.citation_check          CHAIN
```

`retrieval.fusion` should be a compact merge-plus-RRF decision boundary. It should answer: how many candidates came from each source, how many survived dedupe, what RRF parameters were used, and which top IDs moved into rerank input. It should not include raw candidate text by default.

### Retrieval Debug Mode

When `IMPERIAL_RAG_TRACE_MODE=retrieval_debug`, split the fusion boundary:

```text
retrieval.merge_candidates         CHAIN
retrieval.rrf_fusion               CHAIN or RERANKER
```

Use this mode for forensic debugging: bounded candidate IDs, rank movements, dedupe reasons, source mix, RRF scores, and optional Phoenix document panels through the existing candidate-document gate. Keep raw text controlled by the existing OpenInference hide flags.

### Full Internals Mode

Keep framework spans behind the existing debug escape hatch:

```text
IMPERIAL_RAG_TRACE_SUPPRESS_INTERNALS=false
IMPERIAL_RAG_TRACE_AUTO_INSTRUMENT=true
```

This mode is for framework/library debugging only. It should not become the default user-facing Phoenix tree.

## Trace Contract

Before implementation, define a versioned contract for the Phoenix shape.

Required contract fields:

- `imperial.trace_schema_version`, incremented only when the span contract changes.
- Span names, kinds, and parent-child expectations for compact and debug modes.
- Lineage fields: `imperial.ingest_run_id`, `imperial.corpus_version`, `imperial.index_version`, `imperial.embedding_model`, `imperial.keyword_index`, `imperial.qdrant_collection`.
- Chunk lineage fields where available: `chunk_id`, `citation_id`, `chunk_hash` or `content_sha256`, `document_id`, `source_type`.
- Cardinality limits: top IDs only, bounded duplicate summaries, bounded rank movements, bounded document panels.
- Privacy rules: no raw candidate text in default `output.value`; document content only through existing OpenInference-gated document attributes; full metadata only with `IMPERIAL_RAG_TRACE_FULL_METADATA`.

## Query Trace Improvements

1. Add `retrieval.fusion` in compact mode.
   - Span kind: `CHAIN`.
   - Inputs: query plus counts from vector and keyword candidate lists.
   - Output summary: `vector_candidates`, `keyword_candidates`, `merged_candidates`, `deduped_candidates`, `fused_candidates`, `rerank_input_candidates`, `fusion="rrf"`, `fusion_rrf_k`, source mix, bounded top IDs before/after.
   - No raw document text by default.

2. Keep current retriever/reranker/final-evidence semantics.
   - `retrieval.vector_search` and `retrieval.keyword_search` remain `RETRIEVER`.
   - `retrieval.rerank` remains `RERANKER` with input/output document attributes.
   - `retrieval.final_evidence` remains the default evidence document panel because it explains the final cited answer.

3. Add retrieval-debug split spans.
   - `retrieval.merge_candidates`: dedupe-specific counts, duplicate groups, retained/dropped IDs, source mix.
   - `retrieval.rrf_fusion`: RRF-specific input/output IDs, original ranks, fused ranks, `_rrf_score`, `rrf_k`, and rank deltas.
   - Candidate document panels remain opt-in and bounded.

4. Keep answer spans mostly as-is.
   - `answer.call_model` is already an `LLM` span and should stay Phoenix-native.
   - `answer.citation_check` should remain `CHAIN` unless citation validation becomes a blocking policy/guardrail component.
   - Add only compact summary fields if needed: `answer.refused`, `answer.citations_valid`, `answer.invalid_citation_count`.

## Ingestion Trace Improvements

1. Keep ingestion as a batch build trace.
   - Root: keep `ingest.corpus` or rename carefully to `imperial_rag.ingest` only if tests and docs preserve compatibility.
   - Children: `ingest.scan_files`, `ingest.extract_files`, `ingest.build_chunks`, `ingest.keyword_index`, `ingest.vector_index`.

2. Enrich existing ingestion spans.
   - Scan: total files, supported/unsupported counts, MIME/extension breakdown, duplicate-group counts.
   - Extract: status counts, failed file count, extraction methods, bounded failure classes/messages.
   - Chunks: extracted document count, chunk count, chunk size/overlap, chunk hash/count summaries.
   - Keyword index: Elasticsearch index name, chunk count, replace-all success, indexed count.
   - Vector index: Qdrant collection, chunk count, added ID count, embedding model/dimensions, vector provider metadata.

3. Connect existing embedding spans.
   - Preserve `embedding.dashscope.batch` as the `EMBEDDING` span.
   - Add lineage attributes so embedding batches can be tied to `ingest.vector_index`.
   - Do not trace every chunk as a separate span.

4. Keep document panels out of default ingestion.
   - Default ingestion trace should show IDs, counts, hashes, and failure summaries.
   - Debug mode may show bounded, redacted document/chunk samples.

## Lineage And Freshness

Add enough lineage to answer: "Was this answer produced from the index I just built?"

Recommended fields:

- `imperial.ingest_run_id`: unique per ingest command.
- `imperial.corpus_version`: hash/version of the extracted chunk set or manifest snapshot.
- `imperial.index_version`: hash/version written after keyword/vector index updates.
- `imperial.embedding_model`: current embedding model identifier.
- `imperial.keyword_index`: Elasticsearch index name.
- `imperial.qdrant_collection`: Qdrant collection name.
- `imperial.index_fresh`: boolean or status comparing query-time vector metadata to current provider settings.

Use the same values on ingest and query traces so Phoenix can correlate index freshness without reading local files manually.

## Verification Plan

Focused unit tests:

- Compact query trace order and span kinds.
- `retrieval.fusion` output has merge/RRF counts and bounded IDs but no raw candidate text.
- `retrieval_debug` exposes split merge/RRF diagnostics and still respects document/text hide flags.
- Candidate documents remain absent by default and present only when the existing gates are enabled.
- Ingestion span outputs include lineage, index names, counts, and failure summaries without raw corpus text.
- Existing internal-span suppression remains default.

Live smoke:

```bash
IMPERIAL_RAG_TRACE_RUN_ID=debug_fusion_contract \
uv run python scripts/query.py --trace-phoenix --trace-session-id debug_fusion_contract "test question"
```

Then inspect Phoenix or use `phoenix.client.Client(base_url="http://localhost:6006")` to confirm:

- compact trace has the expected domain tree;
- no duplicate LangChain internals in compact mode;
- `retrieval.fusion` exists and explains merge/RRF;
- final evidence document panel exists;
- raw candidate text is absent by default.

Debug smoke:

```bash
IMPERIAL_RAG_TRACE_MODE=retrieval_debug \
IMPERIAL_RAG_TRACE_RUN_ID=debug_retrieval_split \
uv run python scripts/query.py --trace-phoenix --trace-session-id debug_retrieval_split "test question"
```

Confirm split merge/RRF diagnostics and candidate-doc gating behave exactly as documented.

## Implementation Order

1. Write the trace contract tests first.
2. Add compact `retrieval.fusion`.
3. Add `retrieval_debug` split diagnostics for merge and RRF.
4. Enrich ingestion spans and connect lineage/freshness fields.
5. Update README / `.env.example` with mode guidance.
6. Run focused tests, full pytest, and one live Phoenix smoke.

## Non-Goals

- Do not make Phoenix a full corpus browser.
- Do not enable framework auto-instrumentation by default.
- Do not dump all candidate text into `output.value`.
- Do not change retrieval ranking, reranking, answer generation, or citation validation behavior as part of tracing.

