# Phoenix Trace Readability Council Transcript

Generated: 2026-06-22 02:05:27 Asia/Seoul

## Original Question

Use `llm-council` and `phoenix-tracing` to check the Imperial RAG Phoenix tracing and suggest an improvement plan to make it more human-readable and intuitively understandable using Phoenix best practices.

## Framed Question

In `/Users/danil/Public/imperial`, evaluate Phoenix/OpenInference tracing for a private local Python RAG app using Streamlit, LangGraph/LangChain, Qdrant, Elasticsearch, DashScope/Qwen, and Phoenix. The user wants an improvement plan focused on human-readable, intuitive traces grounded in Phoenix best practices and live repo/runtime truth.

Key context:

- Current local HEAD is `7281db7 fix: suppress noisy internal trace spans` on `codex/phoenix-trace-quality`.
- Local source code now uses a compact domain-first query tree:
  - `imperial_rag.query` as `CHAIN`
  - `retrieval` as `CHAIN`
  - `retrieval.vector_search` and `retrieval.keyword_search` as `RETRIEVER`
  - `retrieval.rerank` as `RERANKER`
  - `retrieval.final_evidence` as `RETRIEVER`
  - `answer.generate` as `CHAIN`
  - `answer.call_model` as `LLM`
  - `answer.citation_check` as `CHAIN`
- Local code defaults `IMPERIAL_RAG_TRACE_AUTO_INSTRUMENT=false`, supports `IMPERIAL_RAG_TRACE_SUPPRESS_INTERNALS=true`, and keeps candidate/full evidence payloads opt-in.
- `phoenix_trace_context()` propagates session, user, metadata, and tags with Phoenix helpers.
- Focused local tests passed: `46 passed, 54 deselected`.
- Docker Compose stack is healthy, with app on `127.0.0.1:8501` and Phoenix on `127.0.0.1:6006`.
- The running app container is stale. Its `/app/src/imperial_rag/tracing.py` lacks `IMPERIAL_RAG_TRACE_SUPPRESS_INTERNALS`, `retrieval.final_evidence`, and `IMPERIAL_RAG_TRACE_AUTO_INSTRUMENT`; its retrieval code still contains `retrieve.merge_candidates` and `retrieve.fuse_candidates`.
- Live Phoenix project `imperial-rag` has 679 spans, mostly old/noisy names. New compact spans exist but are a small minority.
- Newest live Phoenix trace still shows old mixed tree: `imperial_rag.query` as `AGENT`, `LangGraph`, `normalize_query`, `retrieve`, `call_model`, internal retrievers, `retrieve.merge_candidates`, `retrieve.fuse_candidates`, `ChatQwen`, `ChatCompletion`, and `answer.validate_citations`.
- Newest compact Phoenix trace is clean: `imperial_rag.query` -> `retrieval` -> vector/keyword/rerank/final_evidence, plus `answer.generate` -> model call and citation check.
- Phoenix docs from Context7 and the local skill support using `CHAIN`, `RETRIEVER`, `RERANKER`, and `LLM` spans, session/user/metadata/tag helpers, document panels, production batching, and OpenInference hide flags.

## Advisor Responses

### The Contrarian

The fatal flaw is that the plan is treating trace readability as a code-shape problem while the live system proves it is also a deployment and data-hygiene problem.

Right now, Phoenix's dominant user experience is still old noise: `search`, `LangGraph`, `ChatCompletion`, `ElasticsearchRetriever`, `retrieve.merge_candidates`, `retrieve.fuse_candidates`. The clean trace exists, but only as a minority specimen. If you present the current plan as "Phoenix tracing is improved," the user will open Phoenix and mostly see the opposite. That is a trust failure.

The second flaw: the running app is stale. The container lacks the exact toggles and span names the plan relies on. Any improvement plan that does not start with rebuilding/restarting the app image and generating a fresh trace from the live UI is premature.

The third flaw: old traces pollute evaluation. Phoenix is acting like an archaeological layer cake. Unless the plan includes a new Phoenix project name, dataset reset, or explicit filtering by timestamp/version/tag, reviewers will keep judging the system by obsolete spans.

My correction: make "live trace parity" Task 0. Rebuild container, verify env flags inside `/app`, run one query through Streamlit, then inspect Phoenix newest trace. Only after that should you tune span names, document panels, reranker attrs, LLM metadata, or privacy flags. Otherwise the plan optimizes a trace the user cannot reliably see.

### The First Principles Thinker

The real problem is not "Phoenix tracing" in general. It is: can a human open one trace and answer, in order, what question ran, what evidence was gathered, how evidence was narrowed, what model was called, and whether citations were checked?

By that standard, the current code plan is already the right shape. The compact trace is the product: `imperial_rag.query` -> `retrieval` -> vector/keyword/rerank/final evidence -> `answer.generate` -> model call/citation check. That maps to Phoenix's native CHAIN, RETRIEVER, RERANKER, and LLM concepts.

The current failure is deployment and history, not design. Phoenix is mostly showing old traces because the running container is stale and the project contains accumulated noisy data. First action: rebuild/restart the app container from HEAD, then run one known query and verify the newest trace has zero `LangGraph`, `search`, `ChatQwen`, `retrieve.merge_candidates`, and `retrieve.fuse_candidates` top-level noise.

Second action: create a clean Phoenix comparison surface, either a fresh project name or archived old project data. Otherwise every review will keep confusing old evidence with current behavior.

Third action: make the compact tree a contract: a smoke test or trace assertion should fail if framework internals reappear when `IMPERIAL_RAG_TRACE_SUPPRESS_INTERNALS=true`.

Keep auto-instrumentation opt-in. Keep final evidence bounded. Keep session/user/tag propagation. Do not add more spans unless they answer a reader's question.

### The Expansionist

The upside here is bigger than "less noisy traces." If this lands cleanly, Phoenix becomes the user's explanation surface for the whole RAG system.

My plan would lean into that: treat the compact trace as the canonical product experience, then make every span answer a human question. Root: "What did the user ask?" Retrieval: "What evidence was considered?" Rerank: "Why did these sources win?" Answer: "What did the model see and produce?" Citation check: "Did the answer stay grounded?"

The highest-leverage next step is operational, not architectural: rebuild/restart the app container so live Phoenix stops emitting the stale mixed tree. Until the running image matches HEAD, the user will keep seeing two competing mental models: old LangGraph/provider internals versus the new domain-first story.

Then improve the span payloads, not the span count. Add concise, privacy-safe attributes like normalized query, retrieval mode, candidate counts, final evidence count, source labels, rerank top scores, model name, token counts, citation pass/fail, and answer length. Keep full documents and candidate payloads opt-in, but make Phoenix's RETRIEVER and RERANKER document panels useful by default with titles, IDs, source paths, and snippets.

Finally, add a "golden trace" regression test or script that runs one query and asserts the visible tree shape: `imperial_rag.query -> retrieval -> vector/keyword/rerank/final_evidence -> answer.generate -> call_model/citation_check`, with no framework internals unless explicitly enabled. That turns trace readability into a maintained interface, not a lucky screenshot.

### The Outsider

The biggest problem is that "current tracing" means two different things here. The codebase appears to have the improved, compact trace design, but the running app and Phoenix data still mostly show the old noisy behavior. Any improvement plan that ignores the stale container will confuse the user because Phoenix will keep proving the opposite of what the code says.

Start with runtime truth: rebuild/redeploy the app container, confirm `/app/src/imperial_rag/*` contains the new span names and suppression flags, then run one fresh query. Only judge trace quality from traces created after that timestamp.

Second, separate historical noise from current quality. The `imperial-rag` Phoenix project has 679 spans, mostly old names. Either create a fresh Phoenix project for validation or document a cutoff timestamp. Otherwise counts like `search`, `LangGraph`, `ChatCompletion`, and `ElasticsearchRetriever` will make the improved design look ineffective.

Third, the compact trace tree is directionally right. It uses human task names: query, retrieval, vector search, keyword search, rerank, final evidence, answer generation, model call, citation check. That is much more understandable than mixed framework internals. Preserve that shape.

Fourth, improve readability by making each span answer a user question: what was searched, what evidence survived, what was reranked, what model answered, and whether citations passed. Keep documents and full evidence opt-in because this is private.

Final plan: rebuild app, generate fresh traces, reset/filter Phoenix history, then add a trace-quality smoke test that asserts the visible span tree stays compact and domain-first.

### The Executor

Monday plan: stop judging Phoenix by the old project history. The code at `7281db7` already has the right trace shape; the live container is stale and is still generating the noisy mixed tree.

1. Rebuild and restart only the app container, then verify `/app/src/imperial_rag/tracing.py` includes `IMPERIAL_RAG_TRACE_SUPPRESS_INTERNALS`, `IMPERIAL_RAG_TRACE_AUTO_INSTRUMENT`, and that `retrieval.py` emits `retrieval.final_evidence`.

2. Run one fresh query from Streamlit with defaults:
`IMPERIAL_RAG_TRACE_AUTO_INSTRUMENT=false`
`IMPERIAL_RAG_TRACE_SUPPRESS_INTERNALS=true`
Keep candidate docs and full final evidence off unless actively debugging.

3. In Phoenix, validate the newest trace tree is:
`imperial_rag.query` -> `retrieval` -> vector, keyword, rerank, final evidence -> `answer.generate` -> model call, citation check.

4. If that passes, archive or ignore older noisy spans; do not redesign based on the 679-span backlog.

5. Next small improvement: tighten displayed attributes. Make retriever and reranker spans show compact human fields first: query, top-k, source IDs/titles, score summaries, selected evidence count. Keep raw documents behind opt-in flags.

6. After that, add one golden trace test that asserts span names and key OpenInference attributes. This prevents LangChain/LangGraph instrumentation drift from quietly making Phoenix unreadable again.

## Peer Review Responses

Anonymization mapping:

- Response A: Expansionist
- Response B: Contrarian
- Response C: Executor
- Response D: First Principles Thinker
- Response E: Outsider

### Peer Review 1

Strongest: C. It is the most executable: rebuild app, verify container contents, run one fresh Streamlit query, validate the exact tree, ignore old spans, then tune attributes and add a golden trace test. It balances runtime truth with readability work.

Biggest blind spot: A. A has the best product framing, but it underplays Phoenix data hygiene. Rebuilding the app is not enough if the project still visually defaults to old noisy traces. It needs a clean project, cutoff timestamp, or version/tag filtering.

What all five missed: put build/config provenance on the root span: git SHA, Docker image ID/tag, Phoenix project, trace schema version, and the active suppression/auto-instrumentation flags. That would make stale-container problems obvious inside Phoenix itself.

### Peer Review 2

Strongest: E. It best balances runtime truth with trace design: rebuild, confirm flags/span names in the live app, judge only post-cutoff traces, and preserve the compact domain tree.

Biggest blind spot: B. It correctly diagnoses stale deployment/history as the dominant UX problem, but overcorrects. It says little about the trace contract itself: bounded evidence, citation status, model metadata, and readable retriever/reranker attributes.

All five miss: make "current live trace" mechanically attributable. Add image/git SHA, app version, config flags, Phoenix project/session tag, and query run ID to every fresh trace, then verify those exact values in Phoenix. Without that, rebuilds and cutoffs are still partly vibes-based.

### Peer Review 3

Strongest: E. It best balances the real blocker: live parity first, then judge only fresh traces, while preserving the compact trace design and adding a smoke test. It avoids both "just rebuild" and "just redesign."

Biggest blind spot: B. Its deployment/data-hygiene diagnosis is sharp, but it under-specifies the actual trace product contract: which spans stay, which fields make them readable, and how citation/evidence readability is tested.

What all five miss: make stale-image detection permanent. Add a visible app trace/version marker: git SHA, image build time, config flags, trace schema version, and Phoenix project/run tag on every root trace. Then future reviews can immediately tell "old noisy history" from "current code is wrong."

### Peer Review 4

Strongest: D. It best balances the real issue: the compact trace design is already the right shape, while the immediate failure is stale runtime plus polluted Phoenix history. It also makes the trace a human contract, not a span-count exercise.

Biggest blind spot: B. It correctly diagnoses deployment/data hygiene, but overcorrects by implying readability is not a code-shape problem. The compact span contract and payload quality still matter once live parity is restored.

All five miss an explicit machine-verifiable parity gate: stamp container/span metadata with git SHA, image digest, and trace flags; run one known query after a cutoff; query Phoenix programmatically; fail if required spans/attrs are missing, framework noise appears, or privacy/redaction regresses. Manual "inspect newest trace" should become a repeatable smoke test.

### Peer Review 5

Strongest: E. It best balances the real blocker: live runtime truth, stale Docker/Phoenix history, and trace design. It avoids over-optimizing code before proving a fresh trace, uses cutoff/fresh-project hygiene, and keeps the compact human-readable contract.

Biggest blind spot: A. It has the right product instincts, but underweights the immediate operational problem. Adding payload fields and golden tests helps only after proving the running app is actually emitting the new compact spans.

What all five miss: a concrete ownership gate: record image/tag/git SHA and effective env flags as trace attributes or startup diagnostics, so Phoenix itself proves which code/config produced the trace. They also do not explicitly mention privacy regression checks while enriching fields, especially around normalized query, model input/output, source labels, and token metadata.

## Chairman Synthesis

### Where the Council Agrees

The current local source design is mostly right. The compact tree maps well to Phoenix/OpenInference concepts and is more readable than the older framework-shaped trace tree.

The immediate blocker is not another naming refactor. The running app container is stale, and the live Phoenix project is dominated by historical noisy spans. A user opening Phoenix today can still see the old mixed tree and reasonably conclude the tracing is not fixed.

The plan must start with live parity: rebuild, verify container contents, run one fresh query, and judge only the resulting post-cutoff trace.

The compact tree should become a maintained contract rather than a screenshot-level preference.

### Where the Council Clashes

The main tension is emphasis. The Contrarian puts deployment and Phoenix data hygiene first. The Expansionist wants to turn Phoenix into a richer explanation surface. Both are right, but the order matters: first prove the live app emits the compact trace, then enrich the span payloads.

Another tension is whether old Phoenix data should be archived, filtered, or moved to a new project. The council does not require one answer, but it rejects continuing to judge current trace quality from the accumulated `imperial-rag` project without a cutoff or project split.

### Blind Spots the Council Caught

Peer review identified a missing guard: root-span provenance. The traces should carry enough build/config data to prove which code and runtime produced them. Include git SHA, app/image version or digest, trace schema version, Phoenix project, run ID, and effective trace flags.

Peer review also flagged privacy. Human-readable fields are useful, but normalized query, source labels, model inputs/outputs, and token metadata must be checked against `OPENINFERENCE_HIDE_*` behavior.

### The Recommendation

Adopt a runtime-parity-first plan.

1. Rebuild and restart the app container from current HEAD.
2. Verify `/app/src/imperial_rag/tracing.py` and `/app/src/imperial_rag/retrieval.py` contain the compact tracing implementation and suppression flags.
3. Generate one fresh Streamlit trace with default readable/privacy-safe flags.
4. Validate the newest trace programmatically from Phoenix.
5. Use a fresh Phoenix project or cutoff/run tag for future readability review.
6. Add root provenance and a golden trace smoke test.
7. Only then enrich payload fields inside the existing compact tree.

### The One Thing to Do First

Rebuild and restart only the app container, then verify inside `/app` that the running code contains `IMPERIAL_RAG_TRACE_SUPPRESS_INTERNALS`, `IMPERIAL_RAG_TRACE_AUTO_INSTRUMENT`, and `retrieval.final_evidence` before judging Phoenix trace readability again.
