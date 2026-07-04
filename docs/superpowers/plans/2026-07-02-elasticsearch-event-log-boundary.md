# Elasticsearch Event Log Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a corrected final Elasticsearch logging recommendation that preserves the repo's actual split: Elasticsearch is used for the private keyword corpus index and may also be used, opt-in, for reduced operational event analytics.

**Architecture:** Keep runtime behavior unchanged unless a later task explicitly requests docs/code updates. Stderr/Docker `json-file` remains the canonical app log path, Phoenix remains the private trace store, Elasticsearch keyword indexing remains the corpus search path, and Elasticsearch event logs remain optional closed-schema metadata only. The plan corrects the decision artifact, then defines verification gates for any future event-log enablement or remote deployment.

**Tech Stack:** Python 3.12, pytest, Docker Compose, Elasticsearch 8.19/Kibana local stack, Context7 CLI for current Elastic documentation.

---

## File Structure

- Decision artifact target: MUST be resolved before editing. Do not count this plan file itself as the recommendation artifact.
- Default saved output when no existing target is found: `docs/superpowers/reports/2026-07-02-elasticsearch-event-log-boundary-recommendation.md`.
- Reference only: `README.md`, `src/imperial_rag/retrieval/elasticsearch.py`, `src/imperial_rag/observability/eventlog.py`, `src/imperial_rag/observability/logging.py`, `scripts/setup_event_logs.py`, `compose.yaml`, `.env.example`, `tests/test_observability.py`, `tests/test_private_compose_deployment.py`.
- Optional docs target if this becomes a repo documentation change: `README.md`, only if the current README needs a sharper operator note. Current inspected README already documents the main boundary, so no README edit is required by default.

## Task 0: Resolve The Actual Target

**Files:**
- Discover: `docs/`, `README.md`.
- Modify: the resolved recommendation artifact, not this plan file.

- [ ] **Step 1: Search for saved recommendation text**

Run:

```bash
rg -n "condition-compare|Elasticsearch is not for document text|not for document text|never store document text|Phoenix traces" docs README.md
```

Done condition:

1. If the command finds a saved artifact outside this plan, set `TARGET_ARTIFACT` to that exact path and edit it.
2. If only this plan matches, treat the original recommendation as chat-only or external pasted text and create the corrected saved recommendation at `docs/superpowers/reports/2026-07-02-elasticsearch-event-log-boundary-recommendation.md`.
3. Record the chosen `TARGET_ARTIFACT` path in the implementation notes before changing content.

## Task 1: Correct The Recommendation Scope

**Files:**
- Modify: `TARGET_ARTIFACT`.
- Reference: `README.md:3`, `README.md:12`, `README.md:25`, `src/imperial_rag/retrieval/elasticsearch.py:222`.

- [ ] **Step 1: Replace the opening claim**

Use this wording:

```markdown
Yes, but with a precise boundary. In this repo Elasticsearch has two separate roles:

1. It is already the private keyword corpus index for extracted document chunks.
2. It can optionally store reduced operational event metadata for search/analytics.

The rule is not "never store document text in Elasticsearch." The rule is: do not put raw Docker logs, prompts, answers, Phoenix trace payloads/spans/span attributes, document text, paths, raw exception messages, provider responses, or free-form payloads into the operational event-log stream.
```

- [ ] **Step 2: Preserve the existing repo evidence**

Use these local references in the revised artifact:

```markdown
- `README.md:12` says the app builds an Elasticsearch keyword index.
- `src/imperial_rag/retrieval/elasticsearch.py:222` indexes `document.page_content` for keyword search.
- `README.md:260` says searchable event logs are optional, local-only, closed-schema, and do not scrape Docker logs.
- `README.md:274` lists the private fields excluded from event logs.
```

- [ ] **Step 3: Remove the broad "not for document text" wording**

Search the artifact for this kind of sentence:

```text
Elasticsearch is not for document text.
```

Replace it with:

```text
The Elasticsearch event-log stream is not for document text.
```

## Task 2: Replace The Bad External Citation

**Files:**
- Modify: `TARGET_ARTIFACT`.
- Reference: Context7 docs lookup for Elasticsearch 8.19.

- [ ] **Step 1: Remove the incorrect link**

Delete the current citation:

```markdown
[Elastic docs](https://www.elastic.co/guide/en/elasticsearch/reference/8.19/condition-compare.html)
```

That page is about Watcher compare conditions, not the general Elasticsearch search/analytics positioning.

- [ ] **Step 2: Replace it with a safer citation sentence and no unverified URL**

Use this wording:

```markdown
Elasticsearch is a search and analytics engine for centrally indexed data. The privacy boundary here comes from this repo's threat model and closed-schema event-log design, not from an Elasticsearch product limitation.
```

- [ ] **Step 3: Re-run Context7 before publishing any external link**

Run:

```bash
npx ctx7@latest library Elasticsearch "Elasticsearch search analytics near real-time centrally indexed data overview"
npx ctx7@latest docs /websites/elastic_co_guide_en_elasticsearch_reference_8_19 "Elasticsearch search analytics near real-time centrally indexed data overview"
```

Expected: the selected docs should be an Elasticsearch overview/search reference, not Watcher, ILM, aggregation examples, or an unrelated API page. If Context7 returns useful text with a suspicious or mismatched source URL, do not publish the URL. Either omit the external link or manually verify the exact Elastic page title and URL first.

## Task 3: Tighten The Security Caveat

**Files:**
- Modify: `TARGET_ARTIFACT`.
- Reference: `compose.yaml:113`, `compose.yaml:118`, `compose.yaml:121`, `compose.yaml:134`, `README.md:124`, `README.md:126`.

- [ ] **Step 1: Replace "safe because loopback" phrasing**

Use:

```markdown
The checked Compose configuration disables Elasticsearch/Kibana auth and TLS and binds published ports to `127.0.0.1`. That is acceptable only for a trusted local-machine threat model. Loopback reduces exposure, but it is not authentication; shared hosts, tunnels, port forwarding, Docker network changes, or remote deployment require auth/TLS and a new review.
```

- [ ] **Step 2: Call out private corpus index exposure**

Add:

```markdown
Access to local Elasticsearch/Kibana is access to the private corpus keyword index, not only the reduced event-log stream. The keyword index stores extracted document chunk text for retrieval, so RBAC/data-view isolation would need to be redesigned before treating Kibana access as safe for a narrower audience.
```

- [ ] **Step 3: Call out linkage metadata**

Add:

```markdown
Even reduced fields such as `request_id`, `session_id`, `user_hash`, and `phoenix_trace_id` are linkage metadata. They are useful for local operations, but they should stay inside the private operator environment because they can connect Kibana events to private Phoenix trace payloads/spans.
```

## Task 4: State The Event-Log Control Contract

**Files:**
- Modify: `TARGET_ARTIFACT`.
- Reference: `src/imperial_rag/observability/eventlog.py:16`, `src/imperial_rag/observability/eventlog.py:110`, `src/imperial_rag/observability/eventlog.py:191`, `src/imperial_rag/observability/eventlog.py:274`, `src/imperial_rag/observability/logging.py:139`.

- [ ] **Step 1: Replace absolute leakage claims**

Use:

```markdown
The event sink is designed around a closed allowlist plus forbidden-field rejection. That is the privacy boundary, but it still depends on call-site discipline and tests: allowed scalar fields must not be abused to carry private text.
```

- [ ] **Step 2: Mention non-fatal event sink failures**

Add:

```markdown
Event-log schema or delivery failures are non-blocking. Operators should monitor stderr/Docker logs for `imperial_rag.eventlog_schema_rejected` and `imperial_rag.eventlog_delivery_failed` if event logs are enabled.
```

- [ ] **Step 3: Add the producer audit**

Run:

```bash
rg -n "log_event\(|log_failure\(" src scripts
```

Audit every producer before enabling or publishing the recommendation:

1. `scripts/query.py`
2. `src/imperial_rag/app/web.py`
3. `scripts/ingest.py`
4. `scripts/run_all_evals.py`
5. `scripts/run_phoenix_eval.py`
6. `scripts/run_ragas_eval.py`
7. `src/imperial_rag/answering/runtime.py`
8. `src/imperial_rag/cli.py`

Done condition: each producer has a short note proving that allowed scalar fields carry only counts, durations, statuses, enum-like names, IDs, or reduced error types. Add or preserve canary tests for raw question, answer, document path, raw exception message, traceback, citation/source text, and provider-response leakage.

## Task 5: Define Enablement And Rollback

**Files:**
- Modify: `TARGET_ARTIFACT`.
- Reference: `.env.example:115`, `scripts/setup_event_logs.py:10`, `scripts/setup_event_logs.py:41`, `scripts/setup_event_logs.py:64`, `README.md:262`.

- [ ] **Step 1: Add the default state**

Use:

```markdown
Default state: keep `IMPERIAL_RAG_EVENTLOG_ELASTICSEARCH_ENABLED=false`.
```

- [ ] **Step 2: Add the pre-enable locality/auth gate**

Use:

```markdown
Pre-enable gate:

1. Record the target `ELASTICSEARCH_URL`.
2. Confirm the target is the local/private Compose stack or another explicitly reviewed private deployment.
3. Confirm the Compose-published Elasticsearch/Kibana ports remain loopback-bound (`127.0.0.1:9200` and `127.0.0.1:5601`) when using local Compose.
4. Confirm auth/TLS/RBAC state. The current local Compose stack has auth/TLS disabled and is acceptable only for trusted local-machine use.
5. Do not enable event logging against a shared, tunneled, forwarded, or cloud Elasticsearch cluster without a new security review covering auth, TLS, RBAC, data views, retention, and corpus-index isolation.
```

- [ ] **Step 3: Add the enablement and readback sequence**

Use:

```markdown
Enablement sequence:

1. Run `uv run python scripts/setup_event_logs.py`.
2. Confirm the event and eval data streams exist:
   `curl -fsS "http://127.0.0.1:9200/_data_stream/imperial-rag-events-v1?pretty"`
   `curl -fsS "http://127.0.0.1:9200/_data_stream/imperial-rag-eval-summaries-v1?pretty"`
3. Confirm templates use `dynamic: strict`:
   `curl -fsS "http://127.0.0.1:9200/_index_template/imperial-rag-events-template-v1?pretty"`
   `curl -fsS "http://127.0.0.1:9200/_index_template/imperial-rag-eval-summaries-template-v1?pretty"`
4. Confirm ILM retention is attached: 30 days for operational events and 90 days for eval summaries:
   `curl -fsS "http://127.0.0.1:9200/_ilm/policy/imperial-rag-events-delete-30d?pretty"`
   `curl -fsS "http://127.0.0.1:9200/_ilm/policy/imperial-rag-eval-summaries-delete-90d?pretty"`
5. Start the app with `IMPERIAL_RAG_EVENTLOG_ELASTICSEARCH_ENABLED=true`.
6. Emit one sample query/eval/ingest event.
7. Read back one document from each enabled data stream with `_search`, inspect `_source`, and prove only allowed metadata landed. The readback must show no raw question, answer, prompt, message, document text, snippet, citation/source list, filename/path, raw metadata, raw exception message, traceback, credential, or provider API response.
8. Check stderr/Docker logs for `imperial_rag.eventlog_schema_rejected` and `imperial_rag.eventlog_delivery_failed`.
```

- [ ] **Step 4: Add normal rollback**

Use:

```markdown
Normal rollback: disable `IMPERIAL_RAG_EVENTLOG_ELASTICSEARCH_ENABLED`, restart the app, and keep using Docker `json-file` logs as the canonical app log store. Do not backfill Docker logs or Phoenix trace payloads/spans into Elasticsearch.
```

- [ ] **Step 5: Add privacy incident recovery**

Use:

```markdown
Privacy incident recovery if private text lands in event logs:

1. Disable `IMPERIAL_RAG_EVENTLOG_ELASTICSEARCH_ENABLED` and stop/restart writers before collecting more events.
2. Preserve the minimum private evidence needed to understand the incident; do not export broad Kibana screenshots or CSVs.
3. Identify affected data stream(s), backing index names, document IDs, and time windows with Elasticsearch `_search`/`_count`.
4. Delete contaminated documents or delete/recreate the affected event/eval data stream if document-level cleanup is not reliable.
5. Re-run `uv run python scripts/setup_event_logs.py` if streams/templates need recreation.
6. Read back `_count`/`_search` results to prove the leaked event documents are gone.
7. Review and delete any Kibana data views, saved searches, screenshots, CSV exports, or shared links that exposed the leaked fields.
8. Revoke tunnels/port forwards/proxies if exposure crossed the trusted local-machine boundary.
9. Rotate any exposed credentials or linkage secrets/IDs if the incident involved more than reduced local metadata.
10. Treat any corpus-index exposure separately: access to Elasticsearch/Kibana also exposes the private keyword corpus index unless RBAC/data-view isolation is redesigned.
```

## Task 6: Verification Gate

**Files:**
- Test: `tests/test_observability.py`
- Test: `tests/test_private_compose_deployment.py`
- Optional Test: `tests/test_scripts.py`

- [ ] **Step 1: Run and record the focused checks with date/commit/output**

Record:

```text
Recorded: 2026-07-02 02:00:36 KST
Commit: 1f222b7
Worktree note: dirty worktree with unrelated pre-existing files; this plan file was untracked before revision.

uv run python -m pytest tests/test_observability.py tests/test_private_compose_deployment.py -q
Result: 23 passed in 0.02s
```

- [ ] **Step 2: Run and record the broader docs/script check before publishing a repo doc change**

Run:

```bash
uv run python -m pytest tests/test_observability.py tests/test_private_compose_deployment.py tests/test_scripts.py -q
```

Current result:

```text
Recorded: 2026-07-02 02:00:36 KST
Commit: 1f222b7
Result: 49 passed in 0.20s
```

- [ ] **Step 3: Run live checks only if making live-runtime claims**

Run these only if the artifact claims the current running stack state:

```bash
docker --context default compose ps
curl -fsS "http://127.0.0.1:9200/_cluster/health?pretty"
curl -fsS "http://127.0.0.1:5601/api/status"
```

Expected: Elasticsearch and Kibana are reachable only on loopback, and Kibana reports an available Elasticsearch connection.

## Final Decision

Keep the current implementation approach:

- Elasticsearch keyword index: valid for private corpus retrieval.
- Elasticsearch event logs: optional, disabled by default, closed-schema, local-only metadata.
- Docker `json-file`: canonical operational app log store.
- Phoenix: private trace/eval store; do not mirror Phoenix trace payloads/spans/span attributes into event logs. `phoenix_trace_id` remains allowed linkage metadata.
- Remote/shared Elasticsearch or Kibana: out of scope until auth/TLS, credential handling, network exposure, and retention are redesigned and tested.

## Self-Review

- Spec coverage: Covers the review council findings: scope correction, citation replacement, threat-model caveat, closed-schema caveat, enablement/rollback, and verification.
- Placeholder scan: No task uses TBD/TODO/fill-in language.
- Type consistency: No code APIs are introduced; referenced environment variables, event names, and file paths match the inspected repo surface.
