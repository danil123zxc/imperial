# Chunk-Level Recall@10 Implementation Plan

## Goal

Add a trustworthy chunk-level retrieval recall@10 metric to the Imperial eval stack.

The target contract is that `evals/questions.jsonl` `reference_context_ids` identify gold chunks, not files. The canonical identity for this metric is `metadata.chunk_id`; `metadata.citation_id` may be used only as a legacy fallback for chunks that genuinely have no `chunk_id`; `metadata.file_id` must never count as a chunk recall hit.

The implementation should expose these deterministic output fields across local eval rows, Phoenix annotations, artifact rows, and summary aggregation:

- `chunk_hit_at_10`
- `chunk_recall_at_10`
- `chunk_precision_at_10`
- `retrieved_chunk_ids`
- `matched_context_ids`

`expected_source_hints` remain smoke/debug hints only. They are not recall ground truth.

## Current Verified Baseline

- Current workspace: `/Users/danil/Public/imperial`.
- `git status --short` is already dirty before this plan. Existing user edits include `/Users/danil/Public/imperial/evals/questions.jsonl` and `/Users/danil/Public/imperial/tests/fixtures/eval_corpus_chunks.jsonl`; implementation must preserve them and stage only current-session edits.
- `/Users/danil/Public/imperial/evals/questions.jsonl` currently has 18 rows with non-empty `reference_context_ids`.
- Those rows currently contain 24 total gold ID references and 23 unique IDs. In the current `.imperial_rag/extracted/chunks.jsonl` snapshot, the 23 unique IDs resolve as `metadata.file_id`; zero resolve as `metadata.chunk_id` and zero resolve as `metadata.citation_id`. The dataset is therefore still file-level for recall purposes.
- `/Users/danil/Public/imperial/src/imperial_rag/evals/ragas.py` has `retrieved_context_ids_from_output(output)`, which returns direct `retrieved_context_ids` when present and otherwise deduplicates `metadata.file_id` from returned documents. It does not currently provide chunk identity extraction.
- `/Users/danil/Public/imperial/src/imperial_rag/evals/phoenix_experiment.py` defines `DEFAULT_RETRIEVAL_METRIC_K = 5`. Existing deterministic ID retrieval metrics are therefore not automatically recall@10.
- `/Users/danil/Public/imperial/src/imperial_rag/evals/audit.py` is file-shaped: `CorpusIndex.resolve()` resolves file IDs, audit rows include `resolved_indexed_file_ids` and `candidate_file_ids`, and strict audit currently proves the old file-level contract.
- `/Users/danil/Public/imperial/src/imperial_rag/evals/golden.py` is also file-shaped: `EvidenceCorpus` is keyed by `file_id`, `chunks_for(file_id)` returns every chunk for that file, and evidence packets are generated from file-level references.
- The focused tests that must change include `/Users/danil/Public/imperial/tests/test_ragas_eval.py`, `/Users/danil/Public/imperial/tests/test_evals.py`, `/Users/danil/Public/imperial/tests/test_eval_audit.py`, `/Users/danil/Public/imperial/tests/test_eval_evidence_packets.py`, `/Users/danil/Public/imperial/tests/test_all_evals.py`, and `/Users/danil/Public/imperial/tests/test_retrieval.py`.

## Constraints and Assumptions

- Do not edit `/Users/danil/Public/imperial/evals/questions.jsonl` until a row-by-row migration manifest has been generated and reviewed.
- Do not invent migrated chunk IDs. Every selected chunk ID must come from the active `.imperial_rag/extracted/chunks.jsonl` snapshot or from a consciously regenerated snapshot.
- Record the chunk snapshot before migration with line count and checksum, for example:

```bash
wc -l .imperial_rag/extracted/chunks.jsonl
shasum -a 256 .imperial_rag/extracted/chunks.jsonl
```

- If `.imperial_rag/extracted/chunks.jsonl` is regenerated before implementation or review finishes, regenerate the migration manifest and re-review selected chunks before editing the dataset.
- Keep user-facing citation display separate from retrieval recall identity. User-facing citations may still display `citation_id`, file name, page, section, and locator metadata. Chunk recall should use chunk IDs internally.
- Do not claim `citation_grounding_behavior` or `conflict_behavior` are chunk-level unless the implementation can unambiguously bind displayed citations back to chunk IDs. For the MVP, make retrieval chunk recall correct first and treat chunk-level citation/conflict grounding as follow-up unless the binding is already available in outputs.
- Preserve the existing `id_context_recall` Ragas/Phoenix compatibility path. Ragas still speaks generic context IDs, so after migration it may receive chunk IDs through `retrieved_context_ids` and `reference_context_ids`; however the explicit deterministic artifact fields must use the new `chunk_*_at_10` names.
- `DEFAULT_RETRIEVAL_METRIC_K = 5` should not silently define chunk recall. Add an explicit chunk recall constant or call site, such as `CHUNK_RECALL_METRIC_K = 10`, and use it everywhere chunk metrics are computed.

## Implementation Steps

1. Capture the implementation baseline.

   From `/Users/danil/Public/imperial`, record:

```bash
git status --short
git diff -- /Users/danil/Public/imperial/evals/questions.jsonl /Users/danil/Public/imperial/tests/fixtures/eval_corpus_chunks.jsonl
wc -l .imperial_rag/extracted/chunks.jsonl
shasum -a 256 .imperial_rag/extracted/chunks.jsonl
```

   Treat existing edits in `evals/questions.jsonl` and `tests/fixtures/eval_corpus_chunks.jsonl` as user edits unless proven otherwise.

2. Add chunk identity helpers in `/Users/danil/Public/imperial/src/imperial_rag/evals/ragas.py`.

   Add `retrieved_chunk_ids_from_output(output)` with this behavior:

   - Return cleaned, unique `output["retrieved_chunk_ids"]` if present.
   - Otherwise inspect `output["documents"]` or `output["evidence"]` in ranked order.
   - Extract `metadata.chunk_id` first.
   - Fall back to `metadata.citation_id` only when `chunk_id` is absent.
   - Never fall back to `metadata.file_id`.
   - Deduplicate while preserving retrieval order.

   Keep `retrieved_context_ids_from_output(output)` available for compatibility, but stop using it for chunk recall.

3. Add explicit chunk recall metric logic in `/Users/danil/Public/imperial/src/imperial_rag/evals/phoenix_experiment.py`.

   Add `CHUNK_RECALL_METRIC_K = 10` or pass `k=10` explicitly at every chunk call site. Implement `chunk_recall_metrics(inputs, outputs, reference_outputs=None, *, k=10)`:

   - Gold IDs: cleaned unique `reference_context_ids` from `reference_outputs` or `inputs`.
   - Retrieved IDs: `retrieved_chunk_ids_from_output(outputs)`.
   - Ranked candidate set: first 10 retrieved chunk IDs.
   - Match set: unique ranked retrieved IDs that appear in the gold set.
   - `chunk_hit_at_10`: at least one matched gold chunk in top 10.
   - `chunk_recall_at_10`: `len(matched_gold_chunk_ids) / len(unique_gold_chunk_ids)`.
   - `chunk_precision_at_10`: `len(matched_gold_chunk_ids_in_top_10) / 10`, matching the current precision@k denominator style.
   - Include `retrieved_chunk_ids`, `reference_context_ids`, `matched_context_ids`, and `k` in metadata.
   - Return a skipped/not-applicable result when gold chunk IDs are missing.

   Add a regression test proving that a retrieved document with the right `file_id` but wrong or missing `chunk_id` does not count as a chunk recall hit.

4. Wire retrieved chunk IDs into eval row production.

   Update local and Phoenix row-building paths in `/Users/danil/Public/imperial/src/imperial_rag/evals/phoenix_experiment.py` and `/Users/danil/Public/imperial/src/imperial_rag/evals/ragas_runner.py` so each row carries `retrieved_chunk_ids` alongside any legacy `retrieved_context_ids`.

   For Ragas `id_context_recall` compatibility, feed chunk IDs as generic context IDs only after the dataset migration is complete. Do not let direct file-shaped `retrieved_context_ids` become input to the new chunk metric.

5. Wire chunk recall outputs into all deterministic eval artifacts.

   Update these functions in `/Users/danil/Public/imperial/src/imperial_rag/evals/phoenix_experiment.py`:

   - `run_local_eval`
   - Phoenix evaluator/annotation setup
   - `log_phoenix_eval_annotations`
   - `build_eval_artifact_row`
   - `_deterministic_retrieval_values`
   - `summarize_eval_artifact_rows`

   Required output behavior:

   - Per-row artifacts include `chunk_hit_at_10`, `chunk_recall_at_10`, `chunk_precision_at_10`, `retrieved_chunk_ids`, and `matched_context_ids`.
   - Summaries aggregate chunk recall only over applicable rows with gold chunk IDs.
   - Existing `id_*` fields remain compatible but are not the only source of truth for chunk recall@10.
   - Existing hint-based retrieval fields stay clearly named as hint/source-smoke diagnostics, not true recall.

6. Update audit contract in `/Users/danil/Public/imperial/src/imperial_rag/evals/audit.py`.

   Replace the file-only corpus index with chunk-aware structures, for example:

   - `CorpusChunk` with `chunk_id`, `citation_id`, `file_id`, `relative_path`, `file_name`, `source_type`, `chunk_index`, locator fields, and text.
   - `CorpusIndex.resolve_chunk(context_id)` that resolves `chunk_id` first and `citation_id` only as a legacy fallback.
   - `CorpusIndex.candidate_chunk_ids(hints, old_file_id=None, limit=...)` for migration/backfill suggestions.

   Strict audit should:

   - Fail `reference_context_ids` that resolve only as `file_id`.
   - Fail unresolved chunk IDs unless the row is explicitly a valid refusal/quarantine case.
   - Report chunk-level fields such as `resolved_chunk_ids`, `candidate_chunk_ids`, `candidate_citation_ids`, `candidate_file_ids`, and locator metadata.
   - Keep refusal rows evidence-empty.
   - Keep lane requirements: `cite_answer` rows need at least one gold chunk; `surface_conflict` rows need at least two resolving gold chunks unless quarantined; `refuse_if_not_found` rows should not carry gold chunks.

7. Generate and review the row-by-row migration manifest before dataset edits.

   Add or extend tooling so a manifest can be written under `.imperial_rag/eval-audits/`, for example:

   - `.imperial_rag/eval-audits/chunk-reference-migration.jsonl`
   - `.imperial_rag/eval-audits/chunk-reference-migration.md`

   Each non-refusal row must include:

   - Row `id`
   - `expected_behavior`
   - Old `reference_context_ids`
   - Old file ID or IDs
   - Candidate `chunk_id` values
   - Candidate `citation_id` values
   - Candidate file/source path
   - Chunk index and locator metadata
   - Short evidence quote from each selected chunk
   - Selected chunk IDs
   - Selection status: `selected`, `needs_review`, `blocked`, or `refusal_empty`
   - Notes explaining ambiguous selections

   Review this manifest row by row before editing `/Users/danil/Public/imperial/evals/questions.jsonl`.

8. Migrate `/Users/danil/Public/imperial/evals/questions.jsonl`.

   After manifest review only, replace the 18 non-empty `reference_context_ids` rows with selected `metadata.chunk_id` values from `.imperial_rag/extracted/chunks.jsonl`.

   Requirements:

   - Preserve row IDs, behaviors, suites, tags, reference answers, and existing user edits.
   - Keep refusal rows empty.
   - Do not use `file_id` in `reference_context_ids`.
   - If a row cannot be mapped confidently, mark it in the manifest and decide whether to quarantine it or defer the row rather than inserting a guessed chunk ID.

9. Update `/Users/danil/Public/imperial/src/imperial_rag/evals/golden.py` and evidence packet generation.

   Change `EvidenceCorpus` to resolve selected chunks, not all chunks for a file:

   - Key primary lookup by `chunk_id`.
   - Allow `citation_id` fallback only for legacy chunks without `chunk_id`.
   - Keep `file_id` as metadata in packets, not as the reference identity.
   - `build_evidence_packets()` should include only the selected chunks for each row.
   - `scripts/generate_eval_evidence_packets.py --strict` should fail file-only references and unresolved chunk references.

10. Update `/Users/danil/Public/imperial/tests/fixtures/eval_corpus_chunks.jsonl`.

    Ensure the portable fixture contains every gold chunk ID used by tests and includes enough metadata for audit and evidence-packet tests:

    - `metadata.chunk_id`
    - `metadata.citation_id`
    - `metadata.file_id`
    - `metadata.chunk_index`
    - source path/name metadata
    - locator metadata when available
    - representative text for evidence quotes

11. Update focused tests.

    Required test coverage:

    - `/Users/danil/Public/imperial/tests/test_ragas_eval.py`: `retrieved_chunk_ids_from_output()` extraction order, dedupe, `citation_id` fallback, and no `file_id` fallback.
    - `/Users/danil/Public/imperial/tests/test_evals.py`: `chunk_recall_metrics(..., k=10)`, local eval row fields, Phoenix evaluator/annotation naming, artifact row fields, summary aggregation, and wrong-file/right-file-wrong-chunk misses.
    - `/Users/danil/Public/imperial/tests/test_eval_audit.py`: strict audit resolves chunk IDs, rejects file-only references, reports candidate chunk IDs, and preserves lane/refusal rules.
    - `/Users/danil/Public/imperial/tests/test_eval_evidence_packets.py`: packets include selected chunks only and strict generation fails unresolved/file-only IDs.
    - `/Users/danil/Public/imperial/tests/test_all_evals.py`: top-level eval orchestration forwards or preserves chunk recall fields and keeps `id_context_recall` compatibility.
    - `/Users/danil/Public/imperial/tests/test_retrieval.py`: retrieval outputs used by evals preserve ordered `metadata.chunk_id` through merge/fusion/rerank paths.

12. Runtime/index parity smoke.

    After code and dataset migration, prove the active retrieval path returns ordered documents with `metadata.chunk_id` populated. Prefer a local deterministic test first, then a live smoke if Qdrant and provider configuration are available.

    If live smoke fails because the vector index is stale or missing chunk IDs, reingest and re-index before trusting recall:

```bash
./scripts/start_qdrant.sh
uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial --index-vectors
```

    Do not update the migrated dataset against one chunk snapshot while evaluating against a different indexed snapshot.

## Validation Plan

Run focused tests first:

```bash
uv run python -m pytest tests/test_ragas_eval.py tests/test_evals.py tests/test_eval_audit.py tests/test_eval_evidence_packets.py tests/test_all_evals.py tests/test_retrieval.py -q
```

Run strict audit against the migrated dataset:

```bash
uv run python scripts/audit_eval_rows.py --strict \
  --output-path .imperial_rag/eval-audits/chunk-recall-audit.jsonl \
  --markdown-path .imperial_rag/eval-audits/chunk-recall-audit.md
```

Expected result after migration:

- All non-refusal gold IDs resolve as chunk IDs or approved legacy citation IDs.
- File-only references fail strict audit.
- Refusal rows remain empty.
- Candidate chunk IDs are available for any unresolved or deferred row.

Run strict evidence packet generation:

```bash
uv run python scripts/generate_eval_evidence_packets.py --strict \
  --audit-path .imperial_rag/eval-audits/chunk-recall-audit.jsonl \
  --output-path .imperial_rag/eval-audits/chunk-recall-evidence-packets.jsonl \
  --markdown-path .imperial_rag/eval-audits/chunk-recall-evidence-packets.md
```

Expected result after migration:

- Evidence packets resolve selected chunks only.
- Ready rows include selected chunk evidence.
- Refusal rows remain `refusal_boundary`.
- Strict packet blockers are zero.

Run deterministic/local eval smoke and confirm per-row artifacts include:

- `chunk_hit_at_10`
- `chunk_recall_at_10`
- `chunk_precision_at_10`
- `retrieved_chunk_ids`
- `matched_context_ids`

Run the top-level eval path when local services and provider keys are available:

```bash
uv run python scripts/run_all_evals.py --ragas-metrics id_context_recall --concurrency 5
```

Finish with the repository gate:

```bash
./scripts/check.sh
```

Before handoff, inspect:

```bash
git status --short
git diff -- /Users/danil/Public/imperial/src/imperial_rag/evals/ragas.py \
  /Users/danil/Public/imperial/src/imperial_rag/evals/phoenix_experiment.py \
  /Users/danil/Public/imperial/src/imperial_rag/evals/audit.py \
  /Users/danil/Public/imperial/src/imperial_rag/evals/golden.py \
  /Users/danil/Public/imperial/evals/questions.jsonl \
  /Users/danil/Public/imperial/tests/fixtures/eval_corpus_chunks.jsonl
```

## Rollback or Recovery

- If migration review finds ambiguous chunk selections, do not edit `evals/questions.jsonl`; keep the manifest with `needs_review` or `blocked` status and return the ambiguity for human review.
- If dataset edits have already been made and validation fails, restore only the current-session edits to `/Users/danil/Public/imperial/evals/questions.jsonl` and `/Users/danil/Public/imperial/tests/fixtures/eval_corpus_chunks.jsonl`; do not reset unrelated user changes.
- If the chunk snapshot changes, discard the old migration manifest, regenerate it from the new `.imperial_rag/extracted/chunks.jsonl`, and re-review selected chunks.
- If live retrieval returns documents without `metadata.chunk_id`, treat the vector/index state as stale or incomplete. Reingest/reindex before evaluating chunk recall.
- If Ragas `id_context_recall` compatibility becomes confusing, keep it as a generic wrapper and make `chunk_recall_at_10` the canonical deterministic acceptance field.
- If chunk-level citation/conflict grounding cannot be proven, keep citation/conflict checks at their current source-display behavior and document chunk-level grounding as a follow-up rather than weakening recall semantics.

## Open Questions

- Should `DEFAULT_RETRIEVAL_METRIC_K` remain `5` for legacy hint/id diagnostics while chunk recall uses explicit `10`, or should the default change globally? The safer MVP is an explicit `CHUNK_RECALL_METRIC_K = 10`.
- Should file-level deterministic `id_*` artifact fields remain visible after migration, or should they be renamed/limited to avoid confusion with chunk recall?
- What is the review owner/process for approving the row-by-row migration manifest before editing `evals/questions.jsonl`?
- Are any current rows intentionally golded at a broader-than-chunk level, requiring multiple chunk IDs to represent one old file-level reference?
- Should chunk-level citation/conflict grounding be included in this implementation only if citation-to-chunk binding is already explicit, or deferred as a separate follow-up?
