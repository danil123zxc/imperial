# LLM Council Transcript - Imperial RAG Eval Hardening Final Plan

**Date:** 2026-06-25
**Question:** Check the two eval-hardening plans, take the best decisions from both, and generate one final plan.

## Original Request

The user provided two plans:

- A row-audit/corpus plan: make the existing 37 rows auditable first, fix `imperial-cite-003`, backfill gold IDs, rewrite generic answers, split lanes, then triage missing documents and expand.
- A judge/eval-hardening plan: build a judge calibration runner first, rebuild calibration quality, harden cite rows, add near-miss refusals, lint, then enable `factual_correctness`.

The user explicitly requested `$llm-council`.

## Framed Question

Imperial RAG is a Russian-language local RAG over private internal HR/logistics/warehouse documents. The decision is how to combine two eval-hardening plans into one implementation-ready sequence without editing the dataset yet.

Current repo facts checked live:

- `evals/questions.jsonl` has 37 rows using `expected_behavior`: 27 `cite_answer`, 5 `surface_conflict`, 5 `refuse_if_not_found`.
- 12 rows have `reference_context_ids`.
- Missing gold IDs: 15 cite rows, all 5 conflict rows, and all 5 refusal rows.
- `evals/russian_judge_calibration.jsonl` has 20 rows, all `cite_answer`, with 10 `correct` and 10 `incorrect` labels.
- Phoenix currently supports `faithfulness`, `answer_relevancy`, and `id_context_recall`; `factual_correctness` and `context_recall` are supported in the Ragas helper layer but not in the Phoenix evaluator path.
- `imperial-cite-003` expects "Акт об отсутствии", but its current `reference_context_ids` points to `ПРИКАЗ О табелях...docx`; the raw absence-act `.doc` exists in `documents/`, but no extracted `file_id` contains that source.

## Advisor Responses

### The Contrarian

The final plan should reject Plan 2's ordering. Running judge calibration first is a distraction when the dataset already contains broken evidence contracts. You would be measuring judge behavior against rows that are partially unaudited, under-specified, or pointing at missing corpus state.

Keep Plan 1 as the spine: start with the row audit table, quarantine/fix `imperial-cite-003`, backfill gold `reference_context_ids`, add both IDs to conflict rows, and rewrite vague reference answers into concrete evidence-shaped answers. That is the blocker.

Keep Plan 2's calibration work, but move it after the audit cleanup and before enabling `factual_correctness`. The calibration runner is useful only once the rows represent the intended evaluation lanes. Until then, `factual_correctness` stays off in Phoenix.

Reject adding 5-8 near-miss refusals early. New examples before fixing the existing 37 rows just increase noise. Add them only after refusal/out-of-corpus semantics are explicitly separated from missing-ingestion coverage.

Final order: audit and quarantine, repair gold IDs and answers, split lanes, run judge calibration with hard negatives/conflict/refusal cases, add corpus-aware lint/CI, then enable semantic judge metrics. Missing document coverage belongs last, after ingestion triage.

### The First Principles Thinker

The final plan should start from the eval contract, not the evaluator. The core question is: "For this row, what behavior is objectively testable against indexed private evidence?" Until that is answered, `factual_correctness` is measurement theater.

Keep Plan 1 as Phase 0: row audit first. Produce the table, fix or quarantine `imperial-cite-003`, backfill cite `reference_context_ids`, add both source IDs to conflict rows, rewrite vague reference answers into evidence-shaped Russian answers, and split lanes by behavior. This is the foundation.

Keep Plan 2's judge calibration, but reorder it after the row audit. The judge should not be calibrated against a weak dataset contract. Once rows are cleaned, rebuild calibration to include cite, conflict, and refusal cases with hard negatives. Then run `run_judge_calibration.py` and keep `factual_correctness` disabled until it demonstrates separation from human labels.

Reject adding 5-8 near-miss refusals before cleanup. More examples amplify noise when the existing 37 rows still have missing gold IDs and one known corpus mismatch.

Defer unchunked-document triage until after eval lanes are clean, except high-value missing documents found by the audit.

### The Expansionist

Keep Plan 1's row audit as the spine. Its upside is bigger than cleanup: it turns the eval set into a corpus coverage map, a routing map, and a backlog generator. Use it to expose which failures are retrieval, missing ingestion, bad question design, or judge weakness.

Keep Plan 2's calibration gate, but reorder it behind the audit schema and run it in parallel with row repair. The judge is leverage only after rows have concrete evidence-shaped answers and gold IDs. Running `factual_correctness` first measures noise.

Reject adding new examples before the 37 existing rows are clean. Near-miss refusals are valuable, but only after the current refusal lane is separated from missing-document coverage. Otherwise you create more ambiguity.

Final order: row audit table first; quarantine/fix `imperial-cite-003`; backfill cite IDs; add both source IDs to conflict rows; rewrite generic answers; split lanes into indexed answerability, conflict/version behavior, refusal/out-of-corpus, and known-missing-doc coverage; then rebuild calibration with hard negatives, conflict, and refusal cases; only then enable `factual_correctness` in Phoenix. After that, add 5-8 near-miss refusals and start high-value unchunked document triage.

The expansion opportunity is CI that does more than pass/fail: every eval row should become an addressable product-quality contract tied to corpus coverage, routing, and judge trust.

### The Outsider

Keep Plan 1's row audit as the first real step. Fresh eyes cannot trust any judge, metric, or CI gate until every row says: what document proves this, is it indexed, and what should happen if it is not. Fix `imperial-cite-003`, backfill cite IDs, and add both IDs to conflict rows before adding new examples.

Keep Plan 2's calibration gate, but reorder it after the row audit. Running factual correctness over weak or mispointed rows just measures dataset noise. Calibration should validate the judge against cleaned examples, not compensate for unclear answer keys.

Reject "enable `factual_correctness` in Phoenix" as an early milestone. It is a reward for passing calibration, not part of hardening itself. Also reject adding 5-8 near-miss refusals before the existing refusal/conflict lanes have gold evidence and crisp expected behavior.

Final order: row audit, row fixes and quarantines, gold ID backfill by lane, rewrite generic answers into evidence-shaped Russian answer keys, then judge calibration with hard negatives across cite/conflict/refusal. Only after agreement clears the bar should factual correctness enter Phoenix and CI. New examples come last, targeted at documented coverage gaps.

### The Executor

Keep Plan 1 as the execution backbone. First deliver a row audit: indexed evidence, current IDs, candidate file IDs, answer-key quality, and action. Fix or quarantine `imperial-cite-003` immediately because it poisons downstream scoring.

Keep Plan 2's judge calibration, but reorder it after the audit and before Phoenix `factual_correctness`. Expand calibration with hard negatives, conflict, and refusal pairs. Do not enable semantic judge metrics until calibration clears the bar.

Reject adding new refusal examples before cleanup. Backfill the 15 cite IDs, add both source IDs to conflict rows, rewrite vague reference answers, split lanes by behavior, then add new examples. CI comes after the row contract is stable.

## Peer Reviews

**Anonymization mapping:** A = Outsider, B = Contrarian, C = Executor, D = Expansionist, E = First Principles Thinker.

### Reviewer 1

Strongest: E. It frames the blocker as an eval contract: each row must have objectively testable behavior against indexed evidence before metrics matter. Biggest blind spot: C is correct but too thin and does not specify CI requirements. Common miss: all five under-specify executable acceptance criteria, quarantine rules, lane routing, calibration thresholds, artifact outputs, and current Phoenix metric support.

### Reviewer 2

Strongest: D, with E close behind. D turns the audit into an execution system: evidence contract, corpus coverage, routing map, backlog, eval lanes, calibration gate, and CI. Biggest blind spot: no precise row schema or acceptance criteria. Common miss: add a migration-safe validation step before changing eval meaning, including a current-results snapshot and a JSONL linter.

### Reviewer 3

Strongest: D, with E close behind. D best combines the two plans: audit backbone, calibration after the audit schema exists, lane splitting, CI, and a backlog for missing corpus coverage. Biggest blind spot: no explicit gates for valid rows, calibration pass, acceptable deltas, and quarantine versus fix. Common miss: freeze a baseline before changing rows, save per-row outcomes, use versioned artifacts, pin evaluator prompt/model, and keep a row-edit changelog.

### Reviewer 4

Strongest: E. It anchors the final plan on objectively testable indexed evidence, keeps Plan 1's cleanup order, imports Plan 2's calibration gate, and keeps `factual_correctness` off until labels and hard negatives are trustworthy. Biggest blind spot: missing row schema and acceptance gates. Common miss: add a verification ladder tied to the real eval entrypoints: linter, ID-to-index resolver, one-row Phoenix smoke, calibration threshold, then full eval.

### Reviewer 5

Strongest: E. It anchors the plan on indexed evidence and imports Plan 2's calibration gate without letting metrics outrun the dataset contract. Biggest blind spot: it underuses D's audit-as-map idea. Common miss: capability gating. Phoenix currently lacks `factual_correctness` and `context_recall`, so the final plan must keep them disabled, route them elsewhere, or require a documented upgrade before CI can enforce them. Acceptance counts should require zero cite/conflict rows missing IDs unless quarantined.

## Chairman's Verdict

### Where the Council Agrees

All five advisors reject Plan 2's original ordering. The row audit is the correct first deliverable because the current dataset has broken or incomplete evidence contracts. Judge calibration is important, but it cannot make mispointed, under-specified, or non-indexed rows trustworthy.

The council also agrees that `imperial-cite-003` is not a normal backfill row. It is a corpus-contract bug: the expected absence-act document is not present in the extracted corpus, while the current gold ID points at an unrelated indexed DOCX. That row should be fixed by ingestion/retargeting or quarantined before any metric interprets it.

`factual_correctness` must remain gated. It should not be treated as a simple switch in Phoenix because the current Phoenix evaluator path does not support it, and because the judge/calibration set is still cite-only and too narrow.

### Where the Council Clashes

The only real clash is whether calibration can run in parallel with row repair. The synthesis: build the calibration runner as soon as the audit schema is defined, but do not trust or enforce `factual_correctness` until the row contract is repaired and the calibration set contains hard cite/conflict/refusal cases.

The earlier plan's near-miss refusals survive, but not as Phase 1. They come after the existing 37 rows are cleaned and lanes are split, otherwise they blur refusal behavior with missing-document coverage.

### Blind Spots the Council Caught

The original plans under-specified acceptance criteria. The final plan needs explicit row invariants, lane routing, quarantine semantics, calibration pass thresholds, artifact outputs, model/prompt pinning, and a verification ladder.

The original plans also missed baseline preservation. Before row semantics change, save current per-row eval results so hardening changes can be compared instead of producing an untraceable before/after jump.

### The Recommendation

Use Plan 1 as the spine and Plan 2 as the gated measurement layer.

Final order:

1. Freeze a baseline from the current 37-row dataset.
2. Create the row audit table and classify every row.
3. Repair or quarantine evidence-contract defects, starting with `imperial-cite-003`.
4. Backfill gold file IDs and rewrite evidence-shaped answer keys in one pass per source document.
5. Split eval lanes and define lane-specific metric routing.
6. Build strict corpus-aware lint and ID-resolution checks.
7. Rebuild the calibration set with hard cite/conflict/refusal examples and run the calibration harness.
8. Enable semantic correctness only after the judge passes the gate and the Phoenix path either supports the metric or a separate supported runner is explicitly wired.
9. Add near-miss refusals and targeted missing-document coverage after the existing dataset is clean.

### The One Thing to Do First

Create the row audit artifact. It should include: `id`, `expected_behavior`, lane, current `reference_context_ids`, resolved indexed file(s), candidate file IDs, source document path, indexed status, answer-key quality, action (`keep`, `rewrite`, `quarantine`, `needs_ingestion`), and notes. This is the foundation for both dataset repair and judge validation.

## Final Plan

### Phase 0 - Baseline and Audit Contract

1. Run/save a baseline for the current 37 rows with current supported metrics: deterministic citation/source-hint/retrieval, `faithfulness`, `answer_relevancy`, and `id_context_recall` where applicable.
2. Create the row audit table before editing `evals/questions.jsonl`.
3. Required audit columns: `id`, `expected_behavior`, lane, current `reference_context_ids`, resolved indexed file IDs, candidate file IDs, source path, indexed status, reference-answer quality, expected source hints quality, action, quarantine reason, and backlog category.
4. Acceptance: every row has an explicit action; `imperial-cite-003` is marked `quarantine` or `needs_ingestion` unless retargeted to real indexed evidence.

### Phase 1 - Repair Existing Rows Only

1. Fix `imperial-cite-003` first: either ingest/convert `Акт об отсутствии на рабочем месте.doc`, or retarget the row to the currently indexed DOCX evidence and rewrite the question/reference accordingly. Do not leave it pointing at the wrong file.
2. For the 27 cite rows, make one pass per source document: concrete Russian `reference_answer`, gold file-level `reference_context_ids`, and document-specific `expected_source_hints`.
3. For the 5 conflict rows, include both competing source file IDs and rewrite the answer key to require surfacing the conflict rather than choosing a side.
4. For refusal rows, split true out-of-corpus refusals from known-missing-document coverage. Do not add gold IDs unless the refusal row is explicitly a known-missing-doc lane.
5. Acceptance: zero indexed cite/conflict rows are missing required gold IDs unless quarantined; generic meta-reference answers are gone from cite/conflict rows.

### Phase 2 - Lane Routing and Validation

1. Split lanes explicitly: indexed answerability, conflict/version behavior, refusal/out-of-corpus behavior, and known missing-document coverage.
2. Add a corpus-aware linter/validator that checks required keys, duplicate IDs, valid `expected_behavior`, ID resolution against `.imperial_rag/extracted/chunks.jsonl`, hint substring hits against gold docs, lane-specific ID requirements, quarantine semantics, and unsupported metric names.
3. Keep `expected_source_hints` as smoke hints, not retrieval truth.
4. Verification ladder: row linter, ID-to-index resolver, focused unit tests, one-row Phoenix smoke, then full eval.

### Phase 3 - Judge Calibration

1. Build `scripts/run_judge_calibration.py` to score calibration rows with the same Qwen/Ragas judge intended for factual correctness and compare against `human_label`.
2. Rebuild `evals/russian_judge_calibration.jsonl` so it is not cite-only: include hard/borderline negatives, wrong-role/entity cases, partial-correct answers, false refusals, conflict examples, and refusal examples.
3. Pin and report judge model, prompt/config, run timestamp, row count, accuracy, confusion matrix, and separation between correct/incorrect labels.
4. Gate: `factual_correctness` stays off until the calibration runner clears an explicit pass threshold selected before implementation.

### Phase 4 - Correctness Metrics

1. Do not pretend Phoenix can already enforce `factual_correctness`; current Phoenix code only supports `faithfulness`, `answer_relevancy`, and `id_context_recall`.
2. After calibration passes, either add a supported Phoenix evaluator path for `factual_correctness`/`context_recall` or route those metrics through `scripts/run_ragas_eval.py` with artifacts linked back to row IDs.
3. CI should enforce validator/lint behavior first. Semantic metric enforcement comes only after metric support and calibration are both real.

### Phase 5 - Controlled Expansion

1. Add 5-8 in-domain near-miss refusals only after the 37-row dataset is clean and the refusal lane is unambiguous.
2. Triage unchunked documents after dataset repair. Start with high-value gaps revealed by the audit: absence-act `.doc`, return/defect PDFs, scanned order PDFs, and logistics/schema images.
3. Add new examples from documented coverage gaps: version conflicts, cross-document questions, realistic Russian paraphrases, and known-missing-document coverage.
4. Defer broad bulk ingestion and large synthetic expansion until the eval contract, linter, and judge calibration are stable.

## Definition of Done

- Baseline artifacts saved before row edits.
- Row audit table completed for all 37 rows.
- `imperial-cite-003` fixed or quarantined with a documented reason.
- All indexed cite/conflict rows have resolving file-level `reference_context_ids`.
- Cite/conflict reference answers are concrete and evidence-shaped.
- Eval lanes are explicit and validated.
- Corpus-aware linter and ID resolver pass.
- Calibration runner exists, calibration set covers cite/conflict/refusal hard cases, and the judge clears the chosen threshold.
- `factual_correctness` remains disabled until Phoenix/Ragas routing is actually supported and calibrated.
- New examples are added only after cleanup, starting with near-miss refusals and audit-derived gaps.
