# Source evidence evaluation

## Sources and labels

Freeze existing `extracted/documents/*.json` artifacts with `compare_chunking.py freeze`.
Use `--documents-root` for another extraction directory and `--authority` for another
authority catalog. Authority selection and exact-file deduplication are applied once.
No original documents are reparsed, and no OCR is invoked. The snapshot includes the
retained text and metadata, full text hashes, source IDs, and a manifest hash; creation
is exclusive and every read verifies the hash. Source IDs combine file identity/hash,
source type, original locator, and exact text hash. Multiple extracted elements with
the same legacy locator remain distinct when their text differs.

Chunk metadata now carries `source_spans`. Each span has `source_id`, `text_sha256`,
`start`, `end`, `chunk_start`, and `chunk_end`. Offsets are Python Unicode character
indices, start-inclusive/end-exclusive. A span maps a literal substring of the chunk
to the exact frozen source substring. Table windows map repeated headers and selected
rows separately; inserted separators are credited only when they match an adjacent
source separator. Embedding-only prefixes are not evidence. Existing index adapters
preserve these fields and restore citation text after vector retrieval.

## Reviewing a sidecar

Use `generate_eval_evidence_packets.py --snapshot ... --output-path ...` to create a
private JSONL sidecar. Existing chunk IDs provide navigation candidates only. The
command refuses to overwrite an existing sidecar. Unmatched or ambiguous legacy
references require inspection of the full frozen source; table windows assembled
from disjoint rows may not match as a single substring.

Each question must have exactly one sidecar row containing:

- `id`, `question_hash`, and `snapshot_hash` from the draft.
- `split`: `dev` for the existing 23 questions. Future unseen groups can use `test`.
- `review_status`: change `draft` to `reviewed` only after checking sufficiency.
- `evidence`: required units, each with a unique `evidence_id` and `support_sets`.

Each support set is a list of spans containing `source_id`, `text_sha256`, `start`,
`end`, and `quote`. All spans in a set are required; any complete alternative set
satisfies a unit. Quotes must exactly equal the frozen substring. Select the needed
facts, units, qualifiers, table headers and rows, not entire old chunks by default.
Annotate input facts for calculations. For conflicts, include at least two claim
units and review both sides; the structural validator cannot establish semantic
contradiction. An independently useful alternative passage belongs in another
support set, not another required unit. Draft `candidates` can remain as review aids
and are not included in the assembled benchmark.

The 16 answer and four conflict questions require reviewed evidence. The three
refusal questions require an empty evidence list and reviewed refusal scope. Missing
annotations, empty answerable gold, invalid coordinates, wrong versions, changed
questions, and unreviewed rows stop validation before provider calls. Review status
is a declaration by the annotator, not an automatic semantic check. No command
automatically promotes draft annotations.

The assembled benchmark hash covers questions, reference answers, behaviors, splits,
evidence and snapshot hash. Legacy questions and gold chunk IDs are never rewritten.
Existing questions are development data; re-splitting them does not produce a fresh
held-out test set. Keep paraphrases and overlapping evidence groups together when
adding new test questions.

## Comparisons and packing

The default matrix includes baseline `400:50` and sizes `256`, `512`, `1024` with
0%, 10%, 20% overlap rounded down. Chunk sizes retain the existing
`estimated_token_count-v1` regex-based semantics. Override using
`--configs 400:50,512:51`. Chunk sizing and context-budget tokenization are recorded
separately and must not be confused.

One frozen corpus and benchmark feed every configuration. Embedding model, retrieval
limits, MMR/fusion settings, and reranker remain fixed; the reranked pool contains up
to 100 candidates. Index names use a fresh run UUID, no existing target is deleted,
and no active alias or ingestion pointer is updated. Provider metadata stays beside
each shadow extraction. Shadow indexes are retained for inspection; this workflow
does not promote or automatically delete them.

Queries run with bounded async orchestration (`--concurrency`, default 3) around the
existing synchronous retrieval integrations. Each query is retrieved once per
configuration; all k/budget scores are computed locally from that result. No answer
model or LLM judge is invoked in this stage.

Recall measures full recovery of required units using the union of source intervals,
without counting duplicate coverage twice. Report recall and full-evidence success
at k=1/3/5/10 and budgets 1000/2000/4000. A successful empty retrieval scores zero.
Refusal recall is undefined and excluded from the mean. Invalid mappings, provider
errors and fallbacks are explicit failures; any failed question disqualifies that
configuration from ranking. Run exit status is nonzero if any configuration fails.

### Evidence-based ranking metrics

Full evaluation and new chunk comparisons also report RR, AP and NDCG at
k=1/3/5/10 over the ranked chunks before answer packing. A chunk has binary
relevance when it alone covers every span in at least one complete support set
for any reviewed evidence unit. Partial overlap is insufficient. Evidence spread
across multiple chunks still contributes to union recall, but not to individual
chunk relevance.

For binary relevance `rel_i` at rank `i`, and `R` relevant chunks in the complete
evaluation corpus:

- `RR@k` is `1 / first_relevant_rank`, or zero if there is no hit through k.
- `AP@k = sum(precision@i * rel_i, i=1..k) / R`. The denominator is **R**, not
  `min(R, k)` or the number of retrieved hits; relevant chunks missed by retrieval
  remain in the denominator. AP@k can be below one even with a perfect top k.
- `NDCG@k` divides `sum(rel_i / log2(i+1), i=1..k)` by the same discounted sum
  for `min(R, k)` leading relevant chunks. Relevance gains are binary.

Repeated chunk IDs retain their rank positions but receive credit only once.
Distinct overlapping chunks are separate items, even when they supply the same
fact. Successful empty retrieval and answerable questions with `R=0` score zero.
Refusal scores are undefined and excluded from means. These metrics measure
ordering against reviewed evidence labels; they do not establish exhaustive
semantic relevance judgments over the corpus.

Per-question artifacts and Phoenix evaluator names are `evidence_rr_at_{k}`,
`evidence_ap_at_{k}`, and `evidence_ndcg_at_{k}`. Their arithmetic means are MRR,
MAP and mean NDCG; comparison summaries name them `evidence_mrr_at_{k}`,
`evidence_map_at_{k}` and `evidence_ndcg_at_{k}`. Summaries include
`ranking_questions` (valid, non-refusal contributors) and `zero_relevant_questions`.

The scorer loads the full saved `chunks.jsonl` once and derives relevant IDs for
each question. Live evaluation uses the resolved active extraction root;
comparisons use each configuration's shadow extraction artifact. All corpus
chunks must have unique nonempty chunk IDs and valid snapshot mappings, and every
retrieved chunk must match its corpus ID, text and source spans. A missing, empty,
stale or mismatched corpus fails validation; retrieved hits are never used to
estimate the full denominator. Live corpus validation runs before query/model
calls or experiment creation. No reindexing is performed automatically.

Results record `relevant_chunk_count`, `ranking_corpus_hash`, and
`ranking_metric_version=complete-evidence-ranking-v1`. Live experiment metadata
and comparison manifests/configurations retain the corresponding version/hash.
Comparison replay uses the saved scores after existing result-hash checks and
ranking version/corpus-hash consistency checks, without rerunning retrieval.
Older artifacts without ranking metadata retain their previous evaluators and
summaries; they do not acquire synthetic ranking scores. Already published
experiments are not updated in place.

Budget scores and development configuration selection remain based on evidence
recall and full-evidence success. Complete-chunk relevance depends on chunk
boundaries, so union-based evidence recall remains the primary measure for
cross-chunking comparisons.

The shared context packer uses `tiktoken==0.13.0`, `cl100k_base`. It tries whole chunks
in rank order, skips those that do not fit and continues. It counts the complete
rendered evidence, including labels, separators and repeated text. It performs no
partial truncation, context expansion or additional overlap deduplication. Budget
utilization makes underfilled contexts visible. The budget covers evidence only,
not question/system text, output tokens, or provider chat-template overhead. These
are reproducible proxy tokens, not Qwen billing or model-limit guarantees.

Application budgeting is opt-in using `IMPERIAL_RAG_CONTEXT_TOKEN_BUDGET`; evidence is
packed before generation and citation numbering. Unset preserves prior behavior.

## Full evaluation runner

With local-file input, `run_all_evals.py` requires `--snapshot` and `--annotations`, with
`--questions-path` selecting the matching questions (default `evals/questions.jsonl`).
It validates the snapshot, question hashes and reviewed sidecar before external
setup. Missing inputs, empty datasets and invalid annotations stop the command;
questions alone are insufficient. Annotation preparation and review use the same
workflow described above.

Alternatively, select an existing Phoenix dataset with `--phoenix-dataset-name` or
`--phoenix-dataset-id`, optionally pinned by `--phoenix-dataset-version-id`. It replaces
both local questions and annotations; `--snapshot` remains mandatory. All reviewed
evidence gates run before retrieval/model calls. The same input flags work for
comparison `validate/run`; basic Phoenix/Ragas evaluation accepts unannotated rows.
See [README's Phoenix input schema and commands](../README.md#phoenix-datasets-as-experiment-input).

```bash
uv run python scripts/run_all_evals.py --snapshot .imperial_rag/evidence-eval/snapshot.json --annotations .imperial_rag/evidence-eval/annotations.jsonl
```

The existing async Phoenix runner evaluates live queries once each. It publishes
`evidence_recall_at_{k}` and `full_evidence_success_at_{k}` for k=1/3/5/10, plus
`evidence_recall_budget_{budget}` and `full_evidence_success_budget_{budget}` for
1000/2000/4000 proxy tokens. These reuse the union-coverage scorer and packer.
`ranked_documents` preserves the retrieval order before answer packing and refusal
handling; `documents` remains the answer context. Budget scores describe repacking
the ranked retrieval at each evaluation budget, not necessarily the context used
by the answer model.

Every ranked chunk must map exactly to the supplied snapshot. Missing mappings,
stale text versions, invalid coordinates, retrieval fallbacks/degradation and query
errors produce undefined evidence scores with a failure reason. Any such task makes
the command exit nonzero after the experiment is recorded. Successful empty retrieval
scores zero on answerable questions; refusal evidence scores remain undefined and
are excluded from averages. An older index without `source_spans` must be rebuilt
against the frozen source text before evidence results can be valid; this command
does not rebuild or promote indexes.

Phoenix dataset rows include reviewed evidence, split, benchmark (`dataset_hash`)
and snapshot hashes. ID-dependent retrieval, citation-grounding, conflict and optional
Ragas ID-recall checks are retained under `legacy_*` evaluator names. They are legacy
diagnostics, not evidence-recall substitutes, and may fail after rechunking.
Citation/refusal/source-hint checks and default Ragas faithfulness/answer-relevancy
remain enabled. `--ragas-metrics none` disables judges, not live retrieval or the
answer model. Other callers, including `run_phoenix_eval.py`, keep their prior defaults.

## Artifacts, Phoenix, and answer review

Run artifacts include the benchmark, manifest, per-configuration chunks/results, and
summary. Manifests record hashes, tokenizer/library versions, providers and settings.
Results retain ranked documents, source coordinates, delivered coordinates/IDs,
tokens, utilization, latency and error classes. Hash checks reject modified or
incomplete results when loading them for subsequent stages. Summaries average valid
rows, but only wholly valid configurations can rank. `--split test` reports results
without ranking or automatically shortlisting configurations.

Development ranking uses mean evidence recall at 2000 tokens, then full-evidence
success, then lower delivered-token count, with config ID breaking exact ties.
`answers` evaluates the valid baseline and the top two other valid configurations
using saved retrieval and the same 2000-token packer. It writes private reference-
answer review packets with citation/refusal/source checks and recall of cited
evidence. Legacy chunk-ID checks are not reused against changed chunk boundaries.
Reference-answer correctness still requires review; these checks do not establish
semantic correctness or automatically select a deployment winner.

`phoenix` replays saved results with deterministic code evaluators through the
existing Phoenix runner. For local-file inputs, a binding under the run parent's `phoenix-datasets/` caches
dataset ID and version for the benchmark hash and split. Subsequent configurations
and runs reuse that exact version, verify its contents and endpoint, and never
silently use the latest version. `phoenix.json` records completed publications so
ordinary repeated calls do not duplicate them. After an interrupted network write,
inspect Phoenix before retrying: remote experiment creation and local recording
are not transactional.

For Phoenix inputs, the benchmark and manifest instead retain the original dataset
ID/version, endpoint and example-content hash. Publication reads that pinned version
and selects the saved split in memory using original example IDs; no dataset upload
or modification occurs. Separate answer generation preserves the saved binding.

All snapshots, quotations, annotations, context, answers and evaluation outputs are
private data. Store them under `.imperial_rag/`, keep them out of commits and public
notes, and send Phoenix results only to the configured private diagnostic service.
