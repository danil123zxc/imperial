# Observability Privacy Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the current Phoenix tracing privacy gaps, remove raw corpus text from fallback retrieval IDs, and document local-only observability boundaries without shipping Elasticsearch app-log indexing on this Phoenix-only branch.

**Architecture:** Immediate work stays inside the current `codex/phoenix-trace-quality` branch: harden OpenInference redaction, make retrieval IDs content-safe, add opt-in HMAC user trace IDs, and document Phoenix/Compose privacy boundaries. Elasticsearch/Kibana app-log indexing remains a future branch with strict schemas, a non-blocking sink, explicit mappings, and retention controls.

**Tech Stack:** Python 3.12, pytest, Streamlit, OpenTelemetry/OpenInference, Phoenix, Docker Compose, Elasticsearch 8.19 for the future app-log branch.

---

## Merged Triage

Current branch, implement now:

- Fix `trace_openinference_step()` so `LLM` span `input.value` is suppressed when `OPENINFERENCE_HIDE_INPUT_MESSAGES=true` or `OPENINFERENCE_HIDE_LLM_PROMPTS=true`.
- Replace every raw `document.page_content` fallback in retrieval identity helpers with a deterministic content fingerprint.
- Add opt-in HMAC user trace IDs with `IMPERIAL_RAG_TRACE_USER_HASH_SECRET`, while preserving the current unsalted SHA-256 fallback when unset.
- Document that Phoenix is a private trace store and can contain raw questions, prompts, answers, and evidence depending on tracing flags.
- Add explicit Compose and README warnings for unauthenticated loopback-only Phoenix, Elasticsearch, and Kibana.
- Add `.env.example` entries for `OPENINFERENCE_HIDE_LLM_PROMPTS` and the HMAC secret.

Current branch, do not implement:

- Do not build Elasticsearch app-log indexing here.
- Do not add failed-login user hashes; that belongs to a future auth/logging design if needed.
- Do not add `IMPERIAL_RAG_LOG_ELASTICSEARCH_ENABLED` yet; that flag belongs with the future ES logging feature.

Future ES app-log branch, required before shipping:

- Closed log schemas by event family; unknown fields rejected before any sink.
- Non-blocking Elasticsearch sink with a queue/background worker or async bulk path; sink errors must not call back into `log_event()`.
- Composable index template with `dynamic: false`, IDs as `keyword`, timestamps as `date`, counters as numeric fields.
- Retention policy or documented cleanup for `imperial_app_logs-*`, defaulting to 30 days.

## File Structure

Immediate branch files:

- Create: `src/imperial_rag/document_ids.py` - small shared helper for content-safe fallback IDs.
- Modify: `src/imperial_rag/tracing.py` - LLM input redaction, HMAC trace user IDs, private-store docstring.
- Modify: `src/imperial_rag/retrieval.py` - safe `_document_key()` and `_retrieval_id()` fallbacks.
- Modify: `src/imperial_rag/workflows.py` - safe legacy `_document_key()` fallback.
- Modify: `src/imperial_rag/elasticsearch_keyword.py` - safe keyword hit `_retrieval_id()` fallback.
- Modify: `tests/test_tracing.py` - redaction and HMAC coverage.
- Modify: `tests/test_retrieval.py` - retrieval ID fallback coverage.
- Modify: `tests/test_workflows.py` - legacy ranking key fallback coverage.
- Modify: `tests/test_elasticsearch_keyword.py` - ES keyword retrieval ID fallback coverage.
- Modify: `tests/test_private_compose_deployment.py` - env, README, and Compose privacy warning coverage.
- Modify: `.env.example` - document prompt redaction and HMAC knobs.
- Modify: `README.md` - document private trace/log boundaries.
- Modify: `compose.yaml` - add local-only/no-auth comments above observability services.

Future ES app-log branch files:

- Create: `src/imperial_rag/log_events.py` - closed event schemas and validators.
- Create: `src/imperial_rag/elasticsearch_app_logs.py` - bounded non-blocking ES sink, index template, retention helper.
- Modify: `src/imperial_rag/observability.py` - validate before sanitizing/logging; wire optional ES sink only when enabled.
- Modify: `src/imperial_rag/config.py` - disabled-by-default app-log ES settings.
- Modify: `compose.yaml`, `.env.example`, `README.md` - app-log settings, local-only warning, retention docs.
- Modify/Create tests: `tests/test_observability.py`, `tests/test_elasticsearch_app_logs.py`, `tests/test_private_compose_deployment.py`.

## Current Branch Tasks

### Task 1: Hide LLM `input.value` When Prompt Hide Flags Are Enabled

**Files:**
- Modify: `tests/test_tracing.py`
- Modify: `src/imperial_rag/tracing.py`

- [ ] **Step 1: Extend the existing LLM message redaction test**

In `tests/test_tracing.py`, update `test_openinference_redaction_env_hides_llm_messages()` after the span exits:

```python
    start_attrs = records[0]["attributes"]
    recorded_span = records[0]["span"]
    assert start_attrs["openinference.span.kind"] == "LLM"
    assert "input.value" not in start_attrs
    assert "input.mime_type" not in start_attrs
    assert "llm.input_messages.0.message.role" not in recorded_span.attributes
    assert "llm.input_messages.0.message.content" not in recorded_span.attributes
    assert "llm.output_messages.0.message.role" not in recorded_span.attributes
    assert "llm.output_messages.0.message.content" not in recorded_span.attributes
    assert recorded_span.attributes["llm.model_name"] == "qwen3.7-plus"
```

- [ ] **Step 2: Add a dedicated `OPENINFERENCE_HIDE_LLM_PROMPTS` test**

Add this test near `test_openinference_redaction_env_hides_llm_messages()`:

```python
def test_openinference_redaction_env_hides_llm_prompt_input_value(monkeypatch) -> None:
    records: list[dict[str, object]] = []

    class FakeSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def set_status(self, status):
            pass

    class FakeSpanContext:
        def __init__(self, span: FakeSpan) -> None:
            self.span = span

        def __enter__(self):
            return self.span

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeTracer:
        def start_as_current_span(self, name, attributes=None):
            span = FakeSpan()
            records.append({"attributes": dict(attributes or {}), "span": span})
            return FakeSpanContext(span)

    monkeypatch.setenv("OPENINFERENCE_HIDE_LLM_PROMPTS", "true")
    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: FakeTracer())

    with tracing_module.trace_llm_step("answer.call_model", "private question") as span:
        span.set_attribute("llm.input_messages.0.message.content", "private question")
        span.set_attribute("llm.model_name", "qwen3.7-plus")

    start_attrs = records[0]["attributes"]
    recorded_span = records[0]["span"]
    assert start_attrs == {"openinference.span.kind": "LLM"}
    assert "llm.input_messages.0.message.content" not in recorded_span.attributes
    assert recorded_span.attributes["llm.model_name"] == "qwen3.7-plus"
```

- [ ] **Step 3: Run the failing focused tests**

Run:

```bash
uv run python -m pytest \
  tests/test_tracing.py::test_openinference_redaction_env_hides_llm_messages \
  tests/test_tracing.py::test_openinference_redaction_env_hides_llm_prompt_input_value \
  -q
```

Expected before implementation: failure because the LLM span still starts with `input.value` and `input.mime_type`.

- [ ] **Step 4: Implement LLM-specific input hiding**

In `src/imperial_rag/tracing.py`, change the input block in `trace_openinference_step()` from:

```python
    if not _hide_inputs():
        span_attributes[_INPUT_VALUE] = input_value
        span_attributes[_INPUT_MIME_TYPE] = _TEXT_MIME_TYPE
```

to:

```python
    if not _hide_span_input(kind):
        span_attributes[_INPUT_VALUE] = input_value
        span_attributes[_INPUT_MIME_TYPE] = _TEXT_MIME_TYPE
```

Add this helper near the existing `_hide_inputs()` helpers:

```python
def _hide_span_input(kind: str) -> bool:
    if _hide_inputs():
        return True
    return str(kind).strip().upper() == "LLM" and _hide_input_messages()
```

- [ ] **Step 5: Verify**

Run:

```bash
uv run python -m pytest tests/test_tracing.py -q
```

Expected: all tracing tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/imperial_rag/tracing.py tests/test_tracing.py
git commit -m "fix: hide llm input values with prompt redaction flags"
```

### Task 2: Replace Raw Content Fallbacks In Retrieval IDs

**Files:**
- Create: `src/imperial_rag/document_ids.py`
- Modify: `src/imperial_rag/retrieval.py`
- Modify: `src/imperial_rag/workflows.py`
- Modify: `src/imperial_rag/elasticsearch_keyword.py`
- Modify: `tests/test_retrieval.py`
- Modify: `tests/test_workflows.py`
- Modify: `tests/test_elasticsearch_keyword.py`

- [ ] **Step 1: Add retrieval fallback tests**

Add `import hashlib` to `tests/test_retrieval.py`.

Add this test near `test_rrf_candidate_fusion_deduplicates_by_retrieval_id_and_merges_metadata()`:

```python
def test_retrieval_id_helpers_hash_content_when_metadata_ids_are_missing() -> None:
    document = Document(page_content="private corpus text", metadata={})
    expected = f"content_sha256:{hashlib.sha256(b'private corpus text').hexdigest()[:12]}"

    assert retrieval_module._document_key(document) == expected
    assert retrieval_module._retrieval_id(document) == expected

    annotated = retrieval_module._annotate_retrieval_documents([document], rank_key="_vector_rank")
    assert annotated[0].metadata["_retrieval_id"] == expected
    assert "private corpus text" not in annotated[0].metadata["_retrieval_id"]
```

- [ ] **Step 2: Add workflow fallback test**

Add these imports to `tests/test_workflows.py`:

```python
import hashlib
import imperial_rag.workflows as workflows_module
```

Add this test near `test_rank_hybrid_candidates_deduplicates_and_boosts_keyword_exact_matches()`:

```python
def test_legacy_workflow_document_key_hashes_content_when_metadata_ids_are_missing() -> None:
    document = Document(page_content="private workflow text", metadata={})
    expected = f"content_sha256:{hashlib.sha256(b'private workflow text').hexdigest()[:12]}"

    assert workflows_module._document_key(document) == expected
    assert "private workflow text" not in workflows_module._document_key(document)
```

- [ ] **Step 3: Add Elasticsearch keyword fallback test**

Add these imports to `tests/test_elasticsearch_keyword.py`:

```python
import hashlib
import imperial_rag.elasticsearch_keyword as elasticsearch_keyword_module
```

Add this test near the other keyword retriever ID assertions:

```python
def test_elasticsearch_retrieval_id_hashes_content_when_ids_and_hit_id_are_missing() -> None:
    document = Document(page_content="private keyword text", metadata={})
    expected = f"content_sha256:{hashlib.sha256(b'private keyword text').hexdigest()[:12]}"

    assert elasticsearch_keyword_module._retrieval_id(document) == expected
    assert "private keyword text" not in elasticsearch_keyword_module._retrieval_id(document)
```

- [ ] **Step 4: Run the failing focused tests**

Run:

```bash
uv run python -m pytest \
  tests/test_retrieval.py::test_retrieval_id_helpers_hash_content_when_metadata_ids_are_missing \
  tests/test_workflows.py::test_legacy_workflow_document_key_hashes_content_when_metadata_ids_are_missing \
  tests/test_elasticsearch_keyword.py::test_elasticsearch_retrieval_id_hashes_content_when_ids_and_hit_id_are_missing \
  -q
```

Expected before implementation: failures because raw page content is still returned.

- [ ] **Step 5: Create the shared helper**

Create `src/imperial_rag/document_ids.py`:

```python
from __future__ import annotations

import hashlib
from typing import Any


def content_fingerprint_id(content: Any, *, length: int = 12) -> str:
    digest = hashlib.sha256(str(content).encode("utf-8")).hexdigest()[:length]
    return f"content_sha256:{digest}"


def first_nonempty_value(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        resolved = str(value).strip()
        if resolved:
            return resolved
    return None


def metadata_or_content_id(*values: Any, content: Any) -> str:
    return first_nonempty_value(*values) or content_fingerprint_id(content)
```

- [ ] **Step 6: Update retrieval ID helpers**

In `src/imperial_rag/retrieval.py`, import the helper:

```python
from imperial_rag.document_ids import metadata_or_content_id
```

Replace `_document_key()` and `_retrieval_id()` with:

```python
def _document_key(document: Document) -> str:
    metadata = document.metadata or {}
    return metadata_or_content_id(metadata.get("citation_id"), metadata.get("chunk_id"), content=document.page_content)


def _retrieval_id(document: Document) -> str:
    metadata = document.metadata or {}
    return metadata_or_content_id(
        metadata.get("_retrieval_id"),
        metadata.get("citation_id"),
        metadata.get("chunk_id"),
        content=document.page_content,
    )
```

- [ ] **Step 7: Update workflow and ES keyword fallbacks**

In `src/imperial_rag/workflows.py`, import the helper:

```python
from imperial_rag.document_ids import metadata_or_content_id
```

Replace `_document_key()` with:

```python
def _document_key(document: Document) -> str:
    metadata = document.metadata or {}
    return metadata_or_content_id(metadata.get("citation_id"), metadata.get("chunk_id"), content=document.page_content)
```

In `src/imperial_rag/elasticsearch_keyword.py`, import the helper:

```python
from imperial_rag.document_ids import metadata_or_content_id
```

Replace `_retrieval_id()` with:

```python
def _retrieval_id(document: Document, *, hit_id: str | None = None) -> str:
    metadata = document.metadata or {}
    return metadata_or_content_id(metadata.get("citation_id"), metadata.get("chunk_id"), hit_id, content=document.page_content)
```

- [ ] **Step 8: Verify**

Run:

```bash
uv run python -m pytest tests/test_retrieval.py tests/test_workflows.py tests/test_elasticsearch_keyword.py -q
```

Expected: tests pass.

- [ ] **Step 9: Commit**

```bash
git add \
  src/imperial_rag/document_ids.py \
  src/imperial_rag/retrieval.py \
  src/imperial_rag/workflows.py \
  src/imperial_rag/elasticsearch_keyword.py \
  tests/test_retrieval.py \
  tests/test_workflows.py \
  tests/test_elasticsearch_keyword.py
git commit -m "fix: avoid raw content fallback retrieval ids"
```

### Task 3: Add Optional HMAC User Trace IDs

**Files:**
- Modify: `src/imperial_rag/tracing.py`
- Modify: `tests/test_tracing.py`
- Modify: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: Add the HMAC test**

In `tests/test_tracing.py`, add `import hmac` beside `import hashlib`.

Add this test near `test_trace_user_id_from_email_is_deterministic_and_pseudonymous()`:

```python
def test_trace_user_id_from_email_uses_local_hmac_secret(monkeypatch) -> None:
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_USER_HASH_SECRET", "local secret")
    expected_hash = hmac.new(
        b"local secret",
        b"user@example.com",
        hashlib.sha256,
    ).hexdigest()[:16]

    user_id = tracing_module.trace_user_id_from_email(" User@Example.COM ")

    assert user_id == f"user_hmac_sha256:{expected_hash}"
    assert "example.com" not in user_id
```

- [ ] **Step 2: Run the failing HMAC test**

Run:

```bash
uv run python -m pytest tests/test_tracing.py::test_trace_user_id_from_email_uses_local_hmac_secret -q
```

Expected before implementation: failure because the function still returns `user_sha256:*`.

- [ ] **Step 3: Implement opt-in HMAC**

In `src/imperial_rag/tracing.py`, add `import hmac` beside `import hashlib`.

Replace `trace_user_id_from_email()` with:

```python
def trace_user_id_from_email(email: str) -> str:
    """Return a pseudonymous Phoenix user ID.

    Set IMPERIAL_RAG_TRACE_USER_HASH_SECRET to use HMAC-SHA256 for stronger
    pseudonymization. Leave it unset to preserve the existing deterministic
    SHA-256 IDs for local trace correlation.
    """

    normalized = str(email).strip().casefold()
    if not normalized:
        return ""
    secret = os.environ.get("IMPERIAL_RAG_TRACE_USER_HASH_SECRET", "").strip()
    if secret:
        digest = hmac.new(secret.encode("utf-8"), normalized.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
        return f"user_hmac_sha256:{digest}"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"user_sha256:{digest}"
```

- [ ] **Step 4: Document the secret**

Add to `.env.example` near the trace settings:

```dotenv
# Optional local secret for HMAC-based Phoenix user correlation. Rotating it breaks historical user correlation.
IMPERIAL_RAG_TRACE_USER_HASH_SECRET=
```

Add to the README common settings list:

```markdown
- `IMPERIAL_RAG_TRACE_USER_HASH_SECRET`: optional local secret for HMAC-based Phoenix user IDs; leave unset to preserve current deterministic local trace correlation.
```

- [ ] **Step 5: Verify**

Run:

```bash
uv run python -m pytest tests/test_tracing.py tests/test_web_app.py -q
```

Expected: tests pass. Existing web app tests may keep stubbing `trace_user_id_from_email()`.

- [ ] **Step 6: Commit**

```bash
git add src/imperial_rag/tracing.py tests/test_tracing.py README.md .env.example
git commit -m "feat: support hmac phoenix user ids"
```

### Task 4: Document Private Observability Boundaries

**Files:**
- Modify: `src/imperial_rag/tracing.py`
- Modify: `compose.yaml`
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `tests/test_private_compose_deployment.py`

- [ ] **Step 1: Add documentation assertions**

In `tests/test_private_compose_deployment.py`, extend `test_env_example_documents_phoenix_privacy_and_batching_knobs()`:

```python
    assert "OPENINFERENCE_HIDE_LLM_PROMPTS=false" in lines
    assert "IMPERIAL_RAG_TRACE_USER_HASH_SECRET=" in lines
```

Extend `test_readme_documents_private_compose_deployment()`:

```python
    assert "unauthenticated by default and are safe only while bound to `127.0.0.1`" in readme
    assert "Phoenix traces are private diagnostic records" in readme
    assert "Future Kibana log views should link to Phoenix only by request/session identifiers" in readme
```

Add this test:

```python
def test_compose_documents_local_only_unauthenticated_observability_services() -> None:
    compose = _read("compose.yaml")

    assert "Phoenix stores private traces and has no auth in this local stack." in compose
    assert "Elasticsearch and Kibana have security disabled for local development." in compose
    assert "Do not rebind these ports or deploy remotely without auth and TLS." in compose
```

- [ ] **Step 2: Run the failing docs tests**

Run:

```bash
uv run python -m pytest tests/test_private_compose_deployment.py -q
```

Expected before documentation updates: failure on the new assertions.

- [ ] **Step 3: Add tracing docstring warning**

In `src/imperial_rag/tracing.py`, replace the `phoenix_trace_context()` docstring with:

```python
    """Propagate private Phoenix trace context to child spans when available.

    Phoenix is the detailed diagnostic store for this local app. Child spans may
    contain raw user queries, prompts, answers, and evidence depending on
    OpenInference and Imperial trace privacy flags.
    """
```

- [ ] **Step 4: Add Compose comments**

In `compose.yaml`, add above `phoenix:`:

```yaml
  # Phoenix stores private traces and has no auth in this local stack.
  # Keep ports loopback-bound unless access control and TLS are added.
```

Add above `elasticsearch:`:

```yaml
  # Elasticsearch and Kibana have security disabled for local development.
  # Safe only while loopback-bound. Do not rebind these ports or deploy remotely without auth and TLS.
```

Add above `kibana:`:

```yaml
  # Kibana is unauthenticated here and can correlate request/session IDs into private Phoenix traces.
```

- [ ] **Step 5: Update `.env.example` prompt privacy controls**

Add the supported LLM prompt flag under the Phoenix/OpenInference privacy controls:

```dotenv
OPENINFERENCE_HIDE_INPUT_MESSAGES=false
OPENINFERENCE_HIDE_LLM_PROMPTS=false
OPENINFERENCE_HIDE_OUTPUT_MESSAGES=false
```

- [ ] **Step 6: Update README private Compose warning**

After the first paragraph under `## Private Compose Deployment`, add:

```markdown
Elasticsearch, Kibana, and Phoenix in this Compose stack are unauthenticated by default and are safe only while bound to `127.0.0.1` on a trusted host. Do not rebind these ports to `0.0.0.0`, publish them through a reverse proxy, or share broad tunnels unless authentication and TLS are enabled.
```

- [ ] **Step 7: Update README Phoenix section**

After the current trace hierarchy paragraph in `### Phoenix`, add:

```markdown
Phoenix traces are private diagnostic records. Depending on `OPENINFERENCE_HIDE_*` and `IMPERIAL_RAG_TRACE_*` flags, spans can include raw user questions, model prompts, model answers, selected evidence text, and document metadata. Treat Phoenix access as access to private corpus-derived data.
```

- [ ] **Step 8: Update README local logs section**

Replace the closing sentence of `### Local Logs` with:

```markdown
Phoenix remains the trace and evaluation system. This v1 logging layer writes sanitized operational events to stderr only; it does not send app logs to Elasticsearch, Sentry, or any other external service. Future Kibana log views should link to Phoenix only by request/session identifiers, and those links must be treated as links into private traces rather than sanitized public records.

If stderr is redirected to a local file, rotate or delete that file according to the machine's privacy requirements. The detached Streamlit helper currently writes to `/tmp/imperial-streamlit-8501.log`, which can be removed with `rm -f /tmp/imperial-streamlit-8501.log` after debugging.
```

- [ ] **Step 9: Verify**

Run:

```bash
uv run python -m pytest tests/test_private_compose_deployment.py tests/test_tracing.py -q
```

Expected: tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/imperial_rag/tracing.py compose.yaml README.md .env.example tests/test_private_compose_deployment.py
git commit -m "docs: clarify private observability boundaries"
```

## Future ES App-Log Branch

Do not execute this section on `codex/phoenix-trace-quality`. Use it as the acceptance contract for the later Elasticsearch app-log branch.

### Future Task 5: Closed Schemas Before Elasticsearch Ingestion

**Files:**
- Create: `src/imperial_rag/log_events.py`
- Modify: `src/imperial_rag/observability.py`
- Modify: `tests/test_observability.py`

- [ ] **Step 1: Add tests that unknown fields are rejected before sanitization**

Add tests proving that `private_blob`, `user_note`, `retrieval_id`, `documents`, `page_content`, `question`, and `answer` never reach stderr or any sink when passed through `log_event()`.

Use this core assertion shape:

```python
def test_log_event_rejects_unknown_fields_without_private_output(capsys) -> None:
    observability.configure_observability(SimpleNamespace(log_level="INFO", log_format="json"))

    observability.log_event(
        "imperial_rag.query",
        operation="query",
        status="success",
        component="cli",
        private_blob="raw question text",
    )

    raw = capsys.readouterr().err
    payload = json.loads(raw)
    assert payload["event"] == "imperial_rag.log_schema_rejected"
    assert payload["rejected_event"] == "imperial_rag.query"
    assert payload["unknown_field_count"] == 1
    assert "private_blob" not in raw
    assert "raw question text" not in raw
```

- [ ] **Step 2: Create closed schemas**

Create `src/imperial_rag/log_events.py` with explicit allowlists per family. Do not allow raw document IDs, source paths, document text, questions, answers, citations, or arbitrary strings.

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class LogSchemaError(ValueError):
    def __init__(self, event: str, unknown_field_count: int) -> None:
        super().__init__(event)
        self.event = event
        self.unknown_field_count = unknown_field_count


@dataclass(frozen=True)
class LogEventSchema:
    fields: frozenset[str]


BASE_FIELDS = frozenset({"operation", "status", "component", "duration_ms", "request_id", "trace_session_id"})
QUERY_FIELDS = frozenset(
    {
        "final_evidence",
        "vector_candidates",
        "keyword_candidates",
        "merged_candidates",
        "rerank_input_candidates",
        "reranked_candidates",
        "reranker",
        "fallback_count",
    }
)
AUTH_FIELDS = frozenset({"auth_action", "user_hash_present"})
FAILURE_FIELDS = frozenset({"exception_type"})

EVENT_SCHEMAS: dict[str, LogEventSchema] = {
    "imperial_rag.query": LogEventSchema(BASE_FIELDS | QUERY_FIELDS),
    "imperial_rag.web_query": LogEventSchema(BASE_FIELDS | QUERY_FIELDS),
    "imperial_rag.auth": LogEventSchema(BASE_FIELDS | AUTH_FIELDS),
    "imperial_rag.failure": LogEventSchema(BASE_FIELDS | FAILURE_FIELDS),
}


def validate_log_event(event: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    schema = EVENT_SCHEMAS.get(event)
    if schema is None:
        raise LogSchemaError(str(event), 0)
    unknown = set(fields) - schema.fields
    if unknown:
        raise LogSchemaError(str(event), len(unknown))
    return {"event": event, **dict(fields)}
```

- [ ] **Step 3: Validate before logging**

In `observability.log_event()`, call `validate_log_event()` before `sanitize_log_fields()`. On schema rejection, emit only:

```python
payload = {
    "event": "imperial_rag.log_schema_rejected",
    "rejected_event": str(event),
    "unknown_field_count": exc.unknown_field_count,
}
```

Do not include unknown field names or values in the rejection log.

- [ ] **Step 4: Verify**

Run:

```bash
uv run python -m pytest tests/test_observability.py tests/test_scripts.py tests/test_web_app.py -q
```

Expected: tests pass and rejected fields never appear in captured logs.

### Future Task 6: Non-Blocking Elasticsearch App-Log Sink

**Files:**
- Create: `src/imperial_rag/elasticsearch_app_logs.py`
- Modify: `src/imperial_rag/observability.py`
- Modify: `src/imperial_rag/config.py`
- Create: `tests/test_elasticsearch_app_logs.py`

- [ ] **Step 1: Add disabled-by-default settings**

In `src/imperial_rag/config.py`, add these helpers near the existing env parsing helpers:

```python
def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if minimum is not None:
        return max(value, minimum)
    return value
```

Then add these fields inside the `Settings` dataclass:

```python
    log_elasticsearch_enabled: bool = field(default_factory=lambda: _env_bool("IMPERIAL_RAG_LOG_ELASTICSEARCH_ENABLED", False))
    log_elasticsearch_index: str = field(default_factory=lambda: os.environ.get("IMPERIAL_RAG_LOG_ELASTICSEARCH_INDEX", "imperial_app_logs"))
    log_elasticsearch_queue_size: int = field(default_factory=lambda: _env_int("IMPERIAL_RAG_LOG_ELASTICSEARCH_QUEUE_SIZE", 1000, minimum=1))
```

- [ ] **Step 2: Implement a bounded queue sink**

The sink must enqueue and return quickly, run a daemon worker, and never call `log_event()` from the worker.

```python
from __future__ import annotations

import queue
import threading
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any


class ElasticsearchAppLogSink:
    def __init__(self, client: Any, *, index: str, queue_size: int = 1000) -> None:
        self._client = client
        self._index = index
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max(queue_size, 1))
        self._thread = threading.Thread(target=self._run, name="imperial-rag-es-logs", daemon=True)
        self._thread.start()

    def emit(self, payload: Mapping[str, Any]) -> bool:
        document = dict(payload)
        document.setdefault("@timestamp", document.get("timestamp") or datetime.now(UTC).isoformat().replace("+00:00", "Z"))
        try:
            self._queue.put_nowait(document)
        except queue.Full:
            return False
        return True

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            document = self._queue.get()
            if document is None:
                return
            try:
                self._client.index(index=self._index, document=document)
            except Exception:
                continue
```

- [ ] **Step 3: Verify**

Run:

```bash
uv run python -m pytest tests/test_elasticsearch_app_logs.py tests/test_observability.py -q
```

Expected: tests pass without live Elasticsearch.

### Future Task 7: Index Template And Retention

**Files:**
- Modify: `src/imperial_rag/elasticsearch_app_logs.py`
- Modify: `README.md`
- Modify: `tests/test_elasticsearch_app_logs.py`

- [ ] **Step 1: Add a composable template with closed mappings**

Context7 was checked for Elasticsearch 8.19 docs. The relevant docs confirm `dynamic: false` mappings and explicit `keyword`, `date`, and numeric field types.

Add:

```python
APP_LOG_INDEX_TEMPLATE = {
    "index_patterns": ["imperial_app_logs*"],
    "priority": 500,
    "template": {
        "settings": {"number_of_shards": 1},
        "mappings": {
            "dynamic": False,
            "properties": {
                "@timestamp": {"type": "date"},
                "timestamp": {"type": "date"},
                "level": {"type": "keyword"},
                "event": {"type": "keyword"},
                "operation": {"type": "keyword"},
                "status": {"type": "keyword"},
                "component": {"type": "keyword"},
                "request_id": {"type": "keyword"},
                "trace_session_id": {"type": "keyword"},
                "duration_ms": {"type": "long"},
                "final_evidence": {"type": "long"},
                "vector_candidates": {"type": "long"},
                "keyword_candidates": {"type": "long"},
                "merged_candidates": {"type": "long"},
                "rerank_input_candidates": {"type": "long"},
                "reranked_candidates": {"type": "long"},
                "fallback_count": {"type": "long"},
                "reranker": {"type": "keyword"},
                "auth_action": {"type": "keyword"},
                "user_hash_present": {"type": "boolean"},
                "exception_type": {"type": "keyword"},
            },
        },
    },
}
```

Install with:

```python
client.indices.put_index_template(name="imperial_app_logs", **APP_LOG_INDEX_TEMPLATE)
```

- [ ] **Step 2: Add retention helper or docs**

Context7 was checked for Elasticsearch 8.19 ILM docs. Use a delete phase with configurable `min_age`; default to 30 days.

```python
APP_LOG_ILM_POLICY = {
    "phases": {
        "delete": {
            "min_age": "30d",
            "actions": {"delete": {}},
        },
    },
}
```

Install with:

```python
client.ilm.put_lifecycle(name="imperial-app-logs-retention", policy=APP_LOG_ILM_POLICY)
```

If ILM is not enabled in local development, document manual cleanup instead:

```bash
curl -X DELETE "http://127.0.0.1:9200/imperial_app_logs*"
```

- [ ] **Step 3: Verify**

Run:

```bash
uv run python -m pytest tests/test_elasticsearch_app_logs.py tests/test_private_compose_deployment.py -q
```

Expected: tests assert `dynamic` is `False`, identifiers are `keyword`, timestamps are `date`, counters are numeric, and retention guidance exists.

### Future Task 8: Compose Scope For ES App Logs

**Files:**
- Modify: `compose.yaml`
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `tests/test_private_compose_deployment.py`

- [ ] **Step 1: Scope ES log indexing to `app` only**

Add these only under the `app` service environment when the future feature ships:

```yaml
      IMPERIAL_RAG_LOG_ELASTICSEARCH_ENABLED: ${IMPERIAL_RAG_LOG_ELASTICSEARCH_ENABLED:-false}
      IMPERIAL_RAG_LOG_ELASTICSEARCH_INDEX: ${IMPERIAL_RAG_LOG_ELASTICSEARCH_INDEX:-imperial_app_logs}
```

Do not put them in `x-imperial-app-base` unless ingestion logging is explicitly accepted.

- [ ] **Step 2: Add disabled defaults**

Add to `.env.example` only in the future branch:

```dotenv
# Optional app-service-only Elasticsearch log indexing. Leave disabled for ad hoc CLI stderr-only logging.
IMPERIAL_RAG_LOG_ELASTICSEARCH_ENABLED=false
IMPERIAL_RAG_LOG_ELASTICSEARCH_INDEX=imperial_app_logs
IMPERIAL_RAG_LOG_ELASTICSEARCH_QUEUE_SIZE=1000
IMPERIAL_RAG_LOG_ELASTICSEARCH_RETENTION_DAYS=30
```

- [ ] **Step 3: Verify**

Run:

```bash
uv run python -m pytest tests/test_private_compose_deployment.py tests/test_observability.py tests/test_elasticsearch_app_logs.py -q
```

Expected: tests pass and future ES app-log indexing remains disabled by default.

## Final Verification For Current Branch

After Tasks 1-4:

```bash
uv run python -m pytest \
  tests/test_tracing.py \
  tests/test_retrieval.py \
  tests/test_workflows.py \
  tests/test_elasticsearch_keyword.py \
  tests/test_web_app.py \
  tests/test_private_compose_deployment.py \
  -q
git diff --check
git status --short
git log --oneline -5
```

Expected:

- All tests pass.
- `git diff --check` reports no whitespace errors.
- Only current-session files are staged/committed.
- No private corpus artifacts, `.env`, `.imperial_rag/`, Phoenix traces, or generated indexes are committed.

## Self-Review

- The existing plan's "retrieval IDs are future-only" item was promoted to current branch because live code has raw `page_content` fallbacks in `retrieval.py`, `workflows.py`, and `elasticsearch_keyword.py`.
- The HMAC env var is `IMPERIAL_RAG_TRACE_USER_HASH_SECRET`, scoped to Phoenix trace user IDs and consistent with the existing trace env namespace.
- Phoenix remains the detailed private store; stderr logs remain the sanitized v1 app log.
- Kibana/Elasticsearch app-log features are intentionally future work and must not be partially enabled in this branch.
- Elasticsearch 8.19 mapping and ILM API shapes were checked with Context7 before writing the future branch requirements.
