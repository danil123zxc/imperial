# LLM Council Transcript - Async Imperial RAG Evals

**Date:** 2026-06-25
**Question:** What is the right implementation strategy to make Imperial RAG evals genuinely asynchronous while preserving CLI usability, metric behavior, tests, and Phoenix compatibility?

## Framed Question

Imperial RAG is a local Python 3.12 RAG app in `/Users/danil/Public/imperial`. The user wants the evals to be asynchronous.

Current repo facts checked live:

- `scripts/run_all_evals.py` is synchronous and calls `scripts/run_phoenix_eval.py::run_phoenix_experiment`.
- `scripts/run_phoenix_eval.py` registers synchronous evaluator functions with `phoenix.client.Client.experiments.run_experiment`.
- `scripts/run_ragas_eval.py::evaluate_ragas_rows` is synchronous. It evaluates sidecar metrics through `imperial_rag.evals.ragas` helpers and reference metrics through `ragas.evaluate(...)`.
- `src/imperial_rag/evals/ragas.py` blocks Ragas awaitables through `_run_coroutine`, which uses `anyio.run(...)` or `loop.run_until_complete(...)`.
- Row evaluators are synchronous `for` loops.
- Tests currently assert that the sync bridge rejects use inside a running event loop.
- Current Ragas docs say `ragas.aevaluate()` is async-first, awaitable, accepts the same parameters as `evaluate()`, supports `batch_size` and `return_executor`, and avoids event-loop patching.

## Advisor Responses

### The Contrarian

The fatal mistake would be "async washing": adding `async def` while Phoenix and sync wrappers still block per row. The right cut is standalone Ragas first, not Phoenix. Add real `*_async` row scorers in `imperial_rag.evals.ragas` that directly await `ascore` / `single_turn_ascore`. Stop calling `_run_coroutine` from helper internals.

Then make `evaluate_ragas_rows_async` schedule eligible rows with bounded concurrency, preserve original row indexes, and sort outputs back into input order. Apply skip and not-applicable decisions before scheduling so concurrency cannot change semantics. Switch reference metrics to `ragas.aevaluate()`. Keep old sync functions as compatibility shells, and make them fail inside an active loop with a "use the async variant" style error. Do not pretend Phoenix is async unless `run_experiment` exposes an async API.

### The First Principles Thinker

The core problem is not "make the scripts async." It is to stop pretending slow LLM judge work is synchronous while keeping the public eval contract stable.

Implement the async boundary in `src/imperial_rag/evals/ragas.py` first. Add native async row scorers that call `ascore` / `single_turn_ascore` directly. Keep sync wrappers only as compatibility shims backed by `anyio.run`, with the current running-loop rejection behavior. Then make `scripts/run_ragas_eval.py` async-first with `evaluate_ragas_rows_async`, bounded gathering, stable input indexes, deterministic output ordering, and unchanged skip semantics. Switch reference metrics from `ragas.evaluate()` to `await ragas.aevaluate(..., batch_size=...)`. Keep Phoenix evaluators sync unless the Phoenix client supports async experiment execution.

### The Expansionist

The bigger prize is a reusable async evaluation engine that can later power smoke runs, nightly runs, calibration gates, and Phoenix-visible experiments without rewriting the metric layer again.

Make `run_ragas_eval.py` async-first now. Add async row scorers, prefer `ascore` / `single_turn_ascore`, and remove hidden event-loop blocking. Preserve sync CLIs as thin wrappers around `anyio.run()`. Row-level sidecar metrics should run through bounded concurrency with stable result slots. Switch reference metrics to `ragas.aevaluate()`. Do not make Phoenix async if `Client.experiments.run_experiment` is sync. Adjacent opportunities include `--concurrency`, `--timeout`, `--batch-size`, and resumable per-row artifacts, but those should not outrun the core contract.

### The Outsider

Make the async boundary boring and explicit: `run_ragas_eval.py` gets an async core first, and every existing CLI keeps working by calling one top-level `anyio.run(...)`.

Refactor `src/imperial_rag/evals/ragas.py` into async primitives: async row scoring, async row evaluation with bounded concurrency, stable row-index reassembly, and unchanged skip semantics. If a metric exposes `ascore` or `single_turn_ascore`, await it. If a legacy scorer is sync-only, isolate it with `anyio.to_thread.run_sync`, not `run_until_complete`. Switch reference metrics to `await ragas.aevaluate(...)`. Leave Phoenix structurally sync unless its client exposes an async experiment API.

### The Executor

Do the standalone Ragas path first. Add `async score_*_row_async(...)` helpers, plus sync wrappers that only run from a non-running loop and preserve the current running-loop rejection behavior.

Then change `scripts/run_ragas_eval.py` to expose `async evaluate_ragas_rows_async(...)` with bounded concurrency and stable ordering. Switch reference metrics from `ragas.evaluate(...)` to `await ragas.aevaluate(...)`, passing the same dataset, metrics, batch size, and exception behavior. Preserve CLI usability by keeping `main()` synchronous. Leave Phoenix mostly alone. Tests should cover async scorer behavior, ordering under concurrent completion, skip semantics, `aevaluate` calls, CLI compatibility, running-loop rejection, and Phoenix output shapes.

## Peer Review Highlights

**Anonymization mapping:** A = Expansionist, B = Contrarian, C = Outsider, D = Executor, E = First Principles Thinker.

- Reviewers converged on **Response C** as the strongest because it had the cleanest async boundary, explicit sync bridge, thread offload for sync-only scorers, and focused parity tests.
- Reviewers consistently called **Response A** the biggest blind spot because it drifted toward timeout/resume/product knobs before locking compatibility.
- Reviewers said all five advisors under-specified failure and artifact semantics: cancellation, per-row timeout/error recording, retry policy, row IDs, run/span correlation, output schema parity, and Ragas version/API guards.
- Reviewers also called out `run_all_evals.py`: keep it Phoenix-first and sync for now, or explicitly add a separate async Ragas mode later. Do not silently change what the default entrypoint means.

## Chairman's Verdict

### Where the Council Agrees

All five advisors agree that the first real async implementation belongs in the standalone Ragas path, not the Phoenix experiment path. They also agree that hidden event-loop bridges inside shared helpers are the core design smell. Async work should be explicit at the metric boundary and bridged only at outer sync entrypoints.

### Where the Council Clashes

The only meaningful clash is scope. The Expansionist wants operational knobs like timeout, resumability, and richer artifacts now. The peer reviewers reject that as premature. The core migration must first prove semantic parity: same row order, same metric keys, same skip semantics, same output shape, and clear errors when async APIs are missing.

### Blind Spots the Council Caught

The initial advisor round under-weighted artifact and correlation stability. Async completion order must not leak into JSONL output order, row IDs, run IDs, Phoenix/OpenInference traces, or metric names. The implementation should add Ragas API guards so upgrades fail clearly if `aevaluate`, `ascore`, or `single_turn_ascore` behavior changes.

### The Recommendation

Implement a clean async core in two layers:

1. In `src/imperial_rag/evals/ragas.py`, add async row scorers and async row evaluators. Await native async Ragas methods directly, offload sync-only scorers with `anyio.to_thread.run_sync`, keep existing sync functions as compatibility wrappers, and preserve running-loop rejection for sync wrappers.
2. In `scripts/run_ragas_eval.py`, add `evaluate_ragas_rows_async(...)` and call `ragas.aevaluate(...)` for reference metrics. Keep `evaluate_ragas_rows(...)` and `main(...)` as synchronous wrappers for current CLI users. Add explicit bounded concurrency and batch-size controls.

Leave Phoenix structurally synchronous until the Phoenix client exposes an async experiment API. `scripts/run_all_evals.py` should remain Phoenix-first and sync in this pass.

### The One Thing to Do First

Add async primitives in `src/imperial_rag/evals/ragas.py` and prove they can run inside an active event loop without touching `_run_coroutine`.
