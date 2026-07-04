# LLM Council Transcript — Imperial RAG Evals Pipeline

**Date:** 2026-06-23  
**Question:** What improvements to the Imperial RAG evals pipeline would make it more robust, comprehensive, and aligned with RAGAS best practices?

---

## Framed Question

Imperial RAG is a Russian-language document QA system over internal company HR/logistics documents (warehouse regulations, job descriptions, return procedures). The evals pipeline currently:
- ~20 questions in `evals/questions.jsonl`. Each has: `question`, `expected_behavior` (cite_answer / refuse_if_not_found / surface_conflict), `expected_source_hints` (keyword strings), optional `reference_answer`
- LLM-based metrics via Ragas (Qwen/DashScope): Faithfulness + AnswerRelevancy (default), optional ContextRecall + FactualCorrectness
- Custom deterministic metric: IDContextRecall (checks retrieved file_ids vs reference_context_ids)
- Deterministic: citation_behavior, source_hint_behavior (substring matching), retrieval hit@5/precision@5/NDCG@5 (binary, hint-based)
- Serial execution. Async Ragas scorers bridged via sync wrappers. Two modes: local and Phoenix
- Observed: `reference_context_ids` is empty in all JSONL rows; `surface_conflict` has only 1 example; `reference_answer` is populated in all rows but ContextRecall/FactualCorrectness are off by default; `_phoenix_evaluators` hard-errors on context_recall/factual_correctness

---

## Advisor Responses

### The Contrarian

The evals pipeline has a critical flaw that undermines everything else: the relevance signal is fake.

Binary hit@5 based on keyword substring matching in document text is not relevance — it's a string search dressed up as an evaluation metric. You're computing NDCG over labels that have no actual grounding in whether a document is useful for answering the question. When your precision@5 is 0.4, you don't know if retrieval is bad or if your substring hints are too broad/narrow. The metric is measuring its own inputs, not retrieval quality.

Second problem: 20 questions is not an eval dataset — it's a smoke test. At that scale, a single bad question can swing your Faithfulness score by 5%. You cannot detect regressions, you cannot do ablations, you cannot trust any trend line. Ragas best practices assume hundreds of samples minimum.

Third: `expected_behavior` categories like `surface_conflict` are evaluated how, exactly? If there's no LLM judge or structured rubric for detecting when the system correctly identifies conflicting documents versus hallucinating a conflict, you're annotating behavior you can't actually measure.

Fourth: the async-to-sync bridge for Ragas scorers is a red flag. If those wrappers are running inside a sync event loop or using `asyncio.run()` in a nested context, you'll get intermittent hangs that appear as timeouts rather than errors. Failure modes here are invisible.

The pipeline looks comprehensive on paper but is built on a shaky foundation: too few questions, fake relevance labels, an unmeasured behavioral category, and fragile async plumbing. Fix the data quality and label validity before adding more metrics — more metrics on bad labels is more noise, not more signal.

---

### The First Principles Thinker

The question assumes the current eval structure is the right structure. It isn't.

What problem are we actually solving? You want to know: "does this RAG system give HR staff correct, trustworthy answers about warehouse regulations?" That's the real problem. The current pipeline measures proxies of proxies.

Ragas Faithfulness measures whether the answer is grounded in retrieved context. But if retrieval is broken, Faithfulness scores high on wrong documents. You're measuring coherence within a mistake.

The `expected_behavior` taxonomy (`cite_answer`, `refuse_if_not_found`, `surface_conflict`) is the most valuable signal you have — and it's the least developed. You have ~20 questions total. That's not an eval, that's a smoke test.

The actual gap: You don't know whether real users are getting real answers to real questions. You have no production query distribution. Your 20 synthetic questions almost certainly don't reflect what warehouse workers actually ask.

What would a ground-up design look like?
1. Mine production logs for actual user queries — that's your eval distribution
2. For each behavior class, you need statistical power: minimum 50+ examples per class, not 6-7
3. The `surface_conflict` class is critical for HR documents (policy updates, conflicting regulations) — if you have fewer than 10 conflict cases, you cannot claim the system handles conflicts
4. Binary hit@5 with substring matching is gameable and fragile — you need semantic relevance judgments

The real question you should be asking: What is the failure mode that causes the most harm in this domain? Hallucinated safety regulations? Missing return procedures? Design your eval around that, not around what Ragas happens to measure.

---

### The Expansionist

The evals pipeline is thinking too small. Twenty questions is a prototype, not a pipeline.

Scale the dataset aggressively. With LLM-assisted generation (use the same Qwen endpoint you already have), you can generate 500+ question-answer pairs from your actual documents in an afternoon. Cover edge cases systematically: cross-document synthesis questions, questions where the answer is partially in one doc, adversarial paraphrases of warehouse regulations. The `surface_conflict` category is especially underexplored — conflicts between documents are where the system either shines or fails catastrophically.

Add behavioral coverage that reveals capability ceilings. Right now you're measuring "does it work?" You should be measuring "how far does it go?" Add: multi-hop reasoning questions (answer requires chaining two warehouse policies), temporal questions (which regulation supersedes which), and Russian-specific linguistic stress tests — synonyms, bureaucratic phrasing variants, abbreviations common in HR docs. This surfaces retrieval gaps that keyword substring matching completely misses.

Treat the eval pipeline as a regression harness, not a one-shot run. Wire evals to run on every retrieval config change. Track metric trajectories over time in Phoenix. The NDCG@5 trend line across model/chunking experiments is worth more than any single score.

The biggest undervalued opportunity: your `expected_behavior` taxonomy is a product differentiator. `refuse_if_not_found` and `surface_conflict` are behaviors most RAG systems don't test at all. Expand this taxonomy — add `synthesize_across_docs`, `defer_to_authority` (when two docs conflict, cite the higher-authority one), `acknowledge_ambiguity`. This turns your evals from a quality check into a behavioral specification for the entire system.

The ceiling here isn't the LLM — it's the eval coverage. Remove that ceiling.

---

### The Outsider

You have ~30 questions. ~27 of them are "cite_answer." That's 90% of your dataset doing the same thing. You have 1 question testing conflict detection. One. That means if your system completely breaks at surfacing conflicts, your aggregate score barely moves. This is not a robust evaluation — it's a 30-question exam where 27 questions are the same question in different clothes.

Your `reference_context_ids` field exists in the schema but is populated for zero questions. IDContextRecall — the metric you built a custom scorer for — will silently skip on every single row. You are running infrastructure for a metric that produces no results.

The `expected_source_hints` check does substring matching against document text or metadata. That means a document about "возврат товаров" could satisfy a hint looking for "возврат брака" if those words happen to appear anywhere in the document. You're measuring string co-occurrence, not retrieval quality. An outside reader would expect "source hit" to mean "we retrieved the right document." It doesn't.

The `reference_answer` field is populated for all rows, but `context_recall` and `factual_correctness` — the metrics that actually use it — are optional and off by default. You have ground truth you aren't using.

Three concrete fixes: (1) fill in `reference_context_ids` for at least the cite_answer questions so IDContextRecall actually runs; (2) add at least 5 more `refuse_if_not_found` and `surface_conflict` cases to make those behavior classes statistically meaningful; (3) make `factual_correctness` default-on since you already have `reference_answer` for every row.

---

### The Executor

Three things that will actually move the needle, in order of impact:

**1. Parallelize the eval loop — do it Monday.**
The pipeline runs questions serially. Every question does a live RAG call, then runs Ragas scorers synchronously via async-bridge hacks. At ~3-5s per question + LLM scoring, that's 2-5 minutes minimum, probably more. Wrap `run_target` calls in `asyncio.gather` or a `ThreadPoolExecutor`. Phoenix's `run_experiment` can take an async task function — use that. This is not optional at 20 questions; at 200 it becomes a blocker.

**2. Add `reference_context_ids` to every question in `questions.jsonl` — 2-3 hours of labeling.**
Right now `IDContextRecall` silently skips every question that's missing `reference_context_ids`. That metric is the only one that ties retrieval quality to actual ground-truth document IDs. Without it, hit@5 and precision@5 are computed against fuzzy keyword substring matches in document text — not retrieval quality, approximate string matching. Label the questions with correct `file_id` values. Run `uv run scripts/query.py` on each question, check which documents came back, confirm the right ones, write the IDs.

**3. Stop routing `context_recall` and `factual_correctness` to a separate script — consolidate.**
`_phoenix_evaluators` hard-errors if you pass `context_recall` or `factual_correctness` to Phoenix mode, forcing you to use `run_ragas_eval.py` separately. Both scripts use the same Qwen/DashScope scorers. The split is artificial friction. Add `context_recall` as a Phoenix evaluator directly — `reference_answer` is already passed through `_to_phoenix_dataset_rows` when present. Remove the two-script dance.

What not to do yet: Don't expand to 200+ questions before retrieval relevance scoring is based on real `file_id` ground truth. Scaling bad labels scales bad signal.

---

## Peer Reviews

### Anonymization mapping
- Response A = The Expansionist
- Response B = The Contrarian
- Response C = The Executor
- Response D = The First Principles Thinker
- Response E = The Outsider

### Reviewer 1
1. **Strongest: B (Contrarian)** — identifies the foundational validity problem. More metrics on bad labels is noise, not signal. B also flags the async-sync bridge as an invisible failure mode no other response noticed.
2. **Biggest blind spot: A (Expansionist)** — advocates scaling 500+ questions without questioning label quality. Building a cathedral on sand.
3. **Universal miss:** Russian-language scoring validity. Ragas was benchmarked on English. Using Qwen to judge Russian bureaucratic register is an unvalidated assumption. If the judge scores Russian inconsistently, every metric is suspect.

### Reviewer 2
1. **Strongest: B (Contrarian)** — foundational validity problem plus the async-to-sync bridge as invisible failure mode.
2. **Biggest blind spot: A (Expansionist)** — recommends scaling without noting `reference_context_ids` is unpopulated.
3. **Universal miss:** Russian-language LLM judge calibration. Nobody questioned whether Ragas judges are calibrated for Russian HR/logistics text or bureaucratic register.

### Reviewer 3
1. **Strongest: B (Contrarian)** — connects circular relevance labels to every downstream metric.
2. **Biggest blind spot: A (Expansionist)** — treats `surface_conflict` as underexplored while ignoring that there is currently no mechanism to actually score it.
3. **Universal miss:** Russian-language judge calibration. A hallucinated answer about warehouse safety regulations in correct Russian bureaucratic style could score high on Faithfulness while being factually dangerous.

### Reviewer 4
1. **Strongest: B (Contrarian)** — only one that identifies circular relevance labels.
2. **Biggest blind spot: A (Expansionist)** — LLM-generated QA pairs from Qwen inherit whatever hallucinations the generator has. Scaling bad labels at scale.
3. **Universal miss:** Language-specific evaluation validity. Ragas metrics calibrated on English may not transfer to Russian. Systematic bias, not just noise.

### Reviewer 5
1. **Strongest: B (Contrarian)** — diagnoses root causes rather than prescribing patches on a broken foundation.
2. **Biggest blind spot: A (Expansionist)** — strategically correct, tactically reckless. Multiplies substrate problems identified by B and E.
3. **Universal miss:** Russian-language LLM judge validity. A mis-calibrated judge silently scores garbage as high-quality and corrupts every metric discussion.

---

## Chairman's Verdict

### Where the Council Agrees

Every advisor independently arrived at the same structural diagnosis, just from different entry points:

**The data foundation is broken before any metric runs.** `reference_context_ids` is empty across all rows, which means IDContextRecall — the only metric grounded in actual document identity — produces no output. The metric exists, the infrastructure runs it, and it silently contributes nothing. This is not a configuration oversight; it's a gap that makes the entire retrieval quality signal circular: you're evaluating retrieval using keyword substring matching against the same documents you're retrieving from.

**20 questions is a smoke test, not an eval pipeline.** The `surface_conflict` class has one example. `refuse_if_not_found` is similarly thin. With this distribution, ~90% of your dataset tests `cite_answer`, which means you're measuring whether the happy path works, repeatedly, and calling it coverage.

**`reference_answer` exists for every row but the metrics that consume it are off by default.** `factual_correctness` and `context_recall` both require `reference_answer`. You have ground truth you aren't using. This is the most straightforward fix in the entire pipeline.

**Async-to-sync bridging is a latent reliability risk.** The serial execution model with sync wrappers around async Ragas scorers will produce intermittent, hard-to-diagnose hangs at scale — not clean errors. This is invisible until it breaks.

### Where the Council Clashes

**Scale now vs. fix foundation first.** The Expansionist argues for 500+ LLM-generated questions immediately, treating scale as the primary lever. The Contrarian, the Executor, and implicitly the Outsider all argue the opposite: scaling bad labels produces more noise, not more signal. The Executor is explicit — don't expand until `reference_context_ids` have real ground-truth `file_id` values.

The Expansionist is strategically correct (more questions from real document content would improve coverage) but tactically reckless. LLM-generated QA pairs from Qwen will inherit whatever distributional biases and hallucinations the generator has. If you generate 500 questions before fixing the labeling substrate, you've scaled the problem.

**The right call:** Fix the substrate first, then scale. The Expansionist's instinct about what to build toward is right; the sequencing is wrong.

**Whether behavioral categories are worth expanding.** The Expansionist wants to add `synthesize_across_docs`, `defer_to_authority`, `acknowledge_ambiguity`. The Contrarian notes that `surface_conflict` — already in the schema — has no scoring mechanism. Adding more behavioral classes before you can actually measure the ones you have is administrative overhead, not evaluation signal.

### Blind Spots the Council Caught

**Russian-language LLM judge calibration — every peer review flagged this, no advisor raised it.**

Ragas Faithfulness and AnswerRelevancy were designed and validated on English text. The underlying embeddings and entailment logic have not been benchmarked on Russian bureaucratic register — the specific register of warehouse regulations, job descriptions, and HR policy documents. A hallucinated answer about warehouse safety procedures written in fluent, well-structured Russian bureaucratic prose could score high on Faithfulness and AnswerRelevancy because the judge is detecting stylistic coherence, not factual accuracy. In a domain where a wrong answer about safety regulations has real consequences, a mis-calibrated judge is not just a statistical problem — it's a liability.

Nobody asked: are your Ragas prompt templates in Russian? Does Qwen/DashScope reliably detect entailment in this specific register? Have you spot-checked a sample of scores against human judgment?

This needs to be verified before you trust any Ragas metric output.

### The Recommendation

Do these three things, in this order:

1. **Verify Russian-language judge calibration before trusting any metric.** Take 20 scored outputs — 10 good answers, 10 intentionally degraded ones — and compare Ragas scores to human judgment from a native Russian speaker familiar with HR documents. If Faithfulness and AnswerRelevancy don't correlate with human judgment at r > 0.7, your metrics are decorative. This can be done in a day and gates everything else.

2. **Fill `reference_context_ids` for all existing questions and enable `factual_correctness` by default.** This is 2-3 hours of labeling work. It activates IDContextRecall (your only retrieval metric grounded in real document identity), and it turns on factual correctness scoring against ground truth you already have. These two changes will give you more real signal than any new metric you could add.

3. **Add 5+ questions per underrepresented behavior class (`refuse_if_not_found`, `surface_conflict`) before scaling.** You need statistical floor coverage for the behavior classes that matter most for HR document QA. One `surface_conflict` example is not a test; it's an anecdote. Also consolidate `context_recall`/`factual_correctness` into the Phoenix evaluator path — the two-script split is artificial friction.

Do not scale to 200+ questions until steps 1 and 2 are complete. Scaling before that multiplies noise.

### The One Thing to Do First

Spot-check Russian-language LLM judge calibration: take 10 clearly correct and 10 intentionally wrong RAG outputs on your existing questions, score them with Ragas, and compare against human judgment. If the scores don't separate good from bad answers reliably in Russian, every other improvement you make will be measuring the wrong thing.
