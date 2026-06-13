# Phoenix Domain Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Imperial query and Phoenix eval traces domain-first, with clean parent-child spans and native Phoenix LLM message panels for Qwen calls.

**Architecture:** Keep Imperial's existing manual OpenInference tracing boundary in `src/imperial_rag/tracing.py`, add a focused LLM span helper, and make Phoenix auto-instrumentation opt-in so LangGraph/LangChain spans do not dominate the trace tree by default. Runtime, retrieval, and answering create the domain hierarchy explicitly: `imperial_rag.query -> query.workflow -> retrieve -> retrieve.* -> answer.generate -> qwen.chat -> answer.validate_citations`.

**Tech Stack:** Python 3.12, OpenTelemetry, Phoenix `arize-phoenix-otel`, OpenInference semantic attributes, LangGraph, LangChain documents, pytest, uv.

---

## File Structure

- Modify `src/imperial_rag/tracing.py`: add Qwen/LLM tracing helpers, OpenInference message attributes, message hide-flag handling, message truncation, token-count helpers, and opt-in Phoenix auto-instrumentation.
- Modify `src/imperial_rag/runtime.py`: add `query.workflow` under the root query span and wrap provider-owned chat model calls in `qwen.chat`.
- Modify `src/imperial_rag/retrieval.py`: add a `retrieve` grouping span around the existing vector, keyword, merge, fuse, and rerank child spans.
- Modify `src/imperial_rag/web_app.py`: add a focused test-backed guarantee that Streamlit keeps using `create_runtime(settings).query(question)` so it receives the same root trace as CLI and evals. Code may already satisfy this; keep the characterization test.
- Modify `scripts/run_phoenix_eval.py`: add a focused test-backed guarantee that Phoenix experiments reuse the runtime query path per evaluated question. Code may already satisfy this; keep the characterization test.
- Modify `.env.example`: document `IMPERIAL_RAG_TRACE_MESSAGE_CONTENT_CHARS` and `IMPERIAL_RAG_PHOENIX_AUTO_INSTRUMENT`.
- Modify `README.md`: document the domain tracing defaults and the live Phoenix smoke command.
- Modify tests:
  - `tests/test_tracing.py`
  - `tests/test_runtime.py`
  - `tests/test_retrieval.py`
  - `tests/test_scripts.py`
  - `tests/test_web_app.py`
  - `tests/test_evals.py`

Do not stage or modify unrelated dirty files. The current worktree has pre-existing user changes in many files; each task must stage only files it changed for this plan.

---

### Task 1: Add Manual LLM Span Helpers And Auto-Instrumentation Defaults

**Files:**
- Modify: `tests/test_tracing.py`
- Modify: `src/imperial_rag/tracing.py`
- Modify: `.env.example`

- [ ] **Step 1: Write failing tracing tests for manual LLM spans**

Add this helper near the top of `tests/test_tracing.py`, below the imports:

```python
def _install_fake_tracer(monkeypatch):
    records: list[dict[str, object]] = []

    class FakeSpan:
        def __init__(self) -> None:
            self.attributes: dict[str, object] = {}
            self.status = None

        def set_attribute(self, key, value):
            self.attributes[key] = value

        def set_status(self, status):
            self.status = status

        def record_exception(self, exc):
            self.attributes["exception.type"] = type(exc).__name__

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
            records.append({"name": name, "attributes": dict(attributes or {}), "span": span})
            return FakeSpanContext(span)

    monkeypatch.setattr(tracing_module.trace, "get_tracer", lambda name: FakeTracer())
    return records
```

Add these tests to `tests/test_tracing.py`:

```python
def test_trace_llm_step_sets_openinference_message_attributes(monkeypatch) -> None:
    records = _install_fake_tracer(monkeypatch)
    monkeypatch.delenv("OPENINFERENCE_HIDE_INPUT_MESSAGES", raising=False)
    monkeypatch.delenv("OPENINFERENCE_HIDE_OUTPUT_MESSAGES", raising=False)
    monkeypatch.setenv("IMPERIAL_RAG_TRACE_MESSAGE_CONTENT_CHARS", "12")

    with tracing_module.trace_llm_step(
        "qwen.chat",
        [
            {"role": "system", "content": "Use only the provided context."},
            {"role": "user", "content": "Question: Как оформить возврат брака?"},
        ],
        model_name="qwen3.7-plus",
        provider="dashscope",
        invocation_parameters={"temperature": 0},
    ) as span:
        span.set_llm_output_message("Возврат оформляется актом. [S1]")
        span.set_output("Возврат оформляется актом. [S1]")

    assert records[0]["name"] == "qwen.chat"
    assert records[0]["attributes"]["openinference.span.kind"] == "LLM"
    assert records[0]["attributes"]["llm.model_name"] == "qwen3.7-plus"
    assert records[0]["attributes"]["llm.provider"] == "dashscope"
    assert records[0]["attributes"]["llm.invocation_parameters"] == '{"temperature": 0}'
    assert records[0]["attributes"]["llm.input_messages.0.message.role"] == "system"
    assert records[0]["attributes"]["llm.input_messages.0.message.content"] == "Use only the" + "." * 3
    assert records[0]["attributes"]["llm.input_messages.1.message.role"] == "user"
    assert records[0]["attributes"]["llm.input_messages.1.message.content"] == "Question: Как" + "." * 3
    recorded_span = records[0]["span"]
    assert recorded_span.attributes["llm.output_messages.0.message.role"] == "assistant"
    assert recorded_span.attributes["llm.output_messages.0.message.content"] == "Возврат" + "." * 3
    assert recorded_span.attributes["output.value"] == '"Возврат оформляется актом. [S1]"'
    assert recorded_span.attributes["output.mime_type"] == "application/json"
    assert recorded_span.status.status_code is tracing_module.StatusCode.OK
```

```python
def test_trace_llm_step_respects_openinference_message_hide_flags(monkeypatch) -> None:
    records = _install_fake_tracer(monkeypatch)
    monkeypatch.setenv("OPENINFERENCE_HIDE_INPUT_MESSAGES", "true")
    monkeypatch.setenv("OPENINFERENCE_HIDE_OUTPUT_MESSAGES", "true")

    with tracing_module.trace_llm_step(
        "qwen.chat",
        [{"role": "user", "content": "private prompt"}],
        model_name="qwen3.7-plus",
        provider="dashscope",
    ) as span:
        span.set_llm_output_message("private answer")

    assert "llm.input_messages.0.message.content" not in records[0]["attributes"]
    assert "llm.input_messages.0.message.role" not in records[0]["attributes"]
    recorded_span = records[0]["span"]
    assert "llm.output_messages.0.message.content" not in recorded_span.attributes
    assert "llm.output_messages.0.message.role" not in recorded_span.attributes
```

```python
def test_configure_phoenix_tracing_disables_auto_instrumentation_by_default(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    calls: list[dict[str, object]] = []
    provider = object()
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel = types.ModuleType("phoenix.otel")

    def register(**kwargs):
        calls.append(kwargs)
        return provider

    fake_otel.register = register
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.otel", fake_otel)
    monkeypatch.delenv("IMPERIAL_RAG_PHOENIX_AUTO_INSTRUMENT", raising=False)

    settings = Settings(workspace_root=tmp_path)

    assert configure_phoenix_tracing(settings, enabled=True) is provider
    assert calls[0]["auto_instrument"] is False
```

```python
def test_configure_phoenix_tracing_can_opt_into_auto_instrumentation(monkeypatch, tmp_path: Path) -> None:
    _reset_phoenix_tracing_for_tests()
    calls: list[dict[str, object]] = []
    provider = object()
    fake_phoenix = types.ModuleType("phoenix")
    fake_otel = types.ModuleType("phoenix.otel")

    def register(**kwargs):
        calls.append(kwargs)
        return provider

    fake_otel.register = register
    monkeypatch.setitem(sys.modules, "phoenix", fake_phoenix)
    monkeypatch.setitem(sys.modules, "phoenix.otel", fake_otel)
    monkeypatch.setenv("IMPERIAL_RAG_PHOENIX_AUTO_INSTRUMENT", "true")

    settings = Settings(workspace_root=tmp_path)

    assert configure_phoenix_tracing(settings, enabled=True) is provider
    assert calls[0]["auto_instrument"] is True
```

Update the two existing `configure_phoenix_tracing` tests that assert `"auto_instrument": True` so they now expect `"auto_instrument": False` unless the new env var is explicitly set.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/test_tracing.py -q
```

Expected: FAIL with `AttributeError` for `trace_llm_step` and assertion failures for the current `auto_instrument=True` default.

- [ ] **Step 3: Add LLM constants and message helpers**

Edit `src/imperial_rag/tracing.py`. Add these constants near the existing OpenInference constants:

```python
_LLM_MODEL_NAME = "llm.model_name"
_LLM_PROVIDER = "llm.provider"
_LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
_LLM_INPUT_MESSAGES = "llm.input_messages"
_LLM_OUTPUT_MESSAGES = "llm.output_messages"
_LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
_LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
_LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"
_TRACE_MESSAGE_CONTENT_CHARS = 2000
```

Add these methods to `OpenInferenceTraceSpan` after `set_output`:

```python
    def set_llm_output_message(self, content: Any, *, role: str = "assistant", index: int = 0) -> None:
        if _hide_output_messages():
            return
        message = {"role": role, "content": str(content)}
        for key, value in openinference_message_attributes(_LLM_OUTPUT_MESSAGES, [message], start_index=index).items():
            self._span.set_attribute(key, value)

    def set_llm_token_counts(
        self,
        *,
        prompt: int | None = None,
        completion: int | None = None,
        total: int | None = None,
    ) -> None:
        self.set_attribute(_LLM_TOKEN_COUNT_PROMPT, prompt)
        self.set_attribute(_LLM_TOKEN_COUNT_COMPLETION, completion)
        self.set_attribute(_LLM_TOKEN_COUNT_TOTAL, total)
```

Add this helper below `trace_embedding_step`:

```python
@contextmanager
def trace_llm_step(
    name: str,
    messages: Sequence[Any],
    *,
    model_name: str,
    provider: str,
    invocation_parameters: Mapping[str, Any] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[OpenInferenceTraceSpan]:
    span_attributes: dict[str, Any] = {
        _LLM_MODEL_NAME: model_name,
        _LLM_PROVIDER: provider,
    }
    if invocation_parameters is not None:
        span_attributes[_LLM_INVOCATION_PARAMETERS] = invocation_parameters
    span_attributes.update(
        openinference_message_attributes(
            _LLM_INPUT_MESSAGES,
            messages,
            hidden=_hide_input_messages(),
        )
    )
    if attributes:
        span_attributes.update(dict(attributes))

    with trace_openinference_step(
        name,
        _messages_input_value(messages),
        kind="LLM",
        attributes=span_attributes,
    ) as span:
        yield span
```

Add these helpers below `openinference_document_attributes`:

```python
def openinference_message_attributes(
    key_prefix: str,
    messages: Sequence[Any],
    *,
    start_index: int = 0,
    hidden: bool = False,
) -> dict[str, Any]:
    if hidden:
        return {}
    attributes: dict[str, Any] = {}
    content_chars = _env_int(
        "IMPERIAL_RAG_TRACE_MESSAGE_CONTENT_CHARS",
        _TRACE_MESSAGE_CONTENT_CHARS,
        minimum=0,
    )
    for offset, message in enumerate(messages):
        role, content = _message_role_content(message)
        index = start_index + offset
        attributes[f"{key_prefix}.{index}.message.role"] = role
        attributes[f"{key_prefix}.{index}.message.content"] = _compact_text(content, content_chars)
    return attributes
```

```python
def _message_role_content(message: Any) -> tuple[str, str]:
    if isinstance(message, Mapping):
        role = str(message.get("role") or message.get("type") or "user")
        content = str(message.get("content") or "")
        return role, content
    role = str(getattr(message, "role", None) or getattr(message, "type", None) or "user")
    content = str(getattr(message, "content", ""))
    if role == "human":
        role = "user"
    if role == "ai":
        role = "assistant"
    return role, content
```

```python
def _messages_input_value(messages: Sequence[Any]) -> str:
    payload = [
        {"role": role, "content": content}
        for role, content in (_message_role_content(message) for message in messages)
    ]
    return _json_value(payload)
```

- [ ] **Step 4: Add hide flags and opt-in auto-instrumentation**

In `src/imperial_rag/tracing.py`, add these helpers near the existing hide helpers:

```python
def _hide_input_messages() -> bool:
    return (
        _hide_inputs()
        or _env_flag("OPENINFERENCE_HIDE_INPUT_MESSAGES")
        or _env_flag("OPENINFERENCE_HIDE_LLM_PROMPTS")
    )
```

```python
def _hide_output_messages() -> bool:
    return _hide_outputs() or _env_flag("OPENINFERENCE_HIDE_OUTPUT_MESSAGES")
```

```python
def _phoenix_auto_instrument() -> bool:
    return _env_flag("IMPERIAL_RAG_PHOENIX_AUTO_INSTRUMENT")
```

Update `_attribute_hidden`:

```python
def _attribute_hidden(key: str) -> bool:
    if _hide_inputs() and (key == _INPUT_VALUE or key == _INPUT_MIME_TYPE or key.startswith("input.")):
        return True
    if _hide_outputs() and (key == _OUTPUT_VALUE or key == _OUTPUT_MIME_TYPE or key.startswith("output.")):
        return True
    if _hide_input_text() and key.endswith(f".{_DOCUMENT_CONTENT}"):
        return True
    if _hide_input_messages() and key.startswith(f"{_LLM_INPUT_MESSAGES}."):
        return True
    if _hide_output_messages() and key.startswith(f"{_LLM_OUTPUT_MESSAGES}."):
        return True
    return False
```

Update `configure_phoenix_tracing` to pass the opt-in auto-instrument flag:

```python
    _CONFIGURED_PROVIDER = register(
        project_name=resolved_settings.phoenix_project_name,
        endpoint=resolved_settings.phoenix_collector_endpoint,
        auto_instrument=_phoenix_auto_instrument(),
        verbose=False,
    )
```

- [ ] **Step 5: Document new env vars**

Edit `.env.example` and add this Phoenix tracing block near the Phoenix settings:

```dotenv
# Phoenix tracing detail controls
IMPERIAL_RAG_TRACE_MESSAGE_CONTENT_CHARS=2000
IMPERIAL_RAG_PHOENIX_AUTO_INSTRUMENT=0
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
uv run python -m pytest tests/test_tracing.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git status --short
git add src/imperial_rag/tracing.py tests/test_tracing.py .env.example
git commit -m "feat: add manual phoenix llm spans"
```

Expected: commit succeeds with only the three listed files staged.

---

### Task 2: Add Query Workflow Span And Qwen Chat Span

**Files:**
- Modify: `tests/test_runtime.py`
- Modify: `src/imperial_rag/runtime.py`

- [ ] **Step 1: Write failing runtime tests**

Update `test_runtime_query_wraps_workflow_in_agent_span` in `tests/test_runtime.py` so it also captures `query.workflow`:

```python
def test_runtime_query_wraps_workflow_in_agent_span(monkeypatch):
    trace_calls = []

    class FakeTraceSpan:
        def set_output(self, value):
            trace_calls.append({"output": value})

    @contextmanager
    def fake_trace_agent_step(name, input_value, *, attributes=None):
        trace_calls.append({"name": name, "input": input_value, "attributes": attributes, "kind": "AGENT"})
        yield FakeTraceSpan()

    @contextmanager
    def fake_trace_pipeline_step(name, input_value, *, attributes=None):
        trace_calls.append({"name": name, "input": input_value, "attributes": attributes, "kind": "CHAIN"})
        yield FakeTraceSpan()

    class FakeWorkflow:
        def invoke(self, state):
            return {
                "answer": "Оформить акт. [S1]",
                "citations_valid": True,
                "evidence": [object(), object()],
                "retrieval": {
                    "final_evidence": 2,
                    "reranker": "fallback:deterministic",
                    "fallbacks": ["reranker_missing_dashscope_api_key"],
                },
            }

    monkeypatch.setattr("imperial_rag.runtime.trace_agent_step", fake_trace_agent_step)
    monkeypatch.setattr("imperial_rag.runtime.trace_pipeline_step", fake_trace_pipeline_step)
    runtime = Runtime(settings=Settings(), workflow=FakeWorkflow())

    assert runtime.query("Что делать с браком?")["answer"] == "Оформить акт. [S1]"
    assert trace_calls == [
        {
            "name": "imperial_rag.query",
            "input": "Что делать с браком?",
            "attributes": {"runtime.workspace_root": "/Users/danil/Public/imperial"},
            "kind": "AGENT",
        },
        {
            "name": "query.workflow",
            "input": "Что делать с браком?",
            "attributes": {"runtime.workflow": "langgraph"},
            "kind": "CHAIN",
        },
        {
            "output": {
                "answer": "Оформить акт. [S1]",
                "citations_valid": True,
                "evidence_count": 2,
                "retrieval": {
                    "final_evidence": 2,
                    "reranker": "fallback:deterministic",
                    "fallbacks": ["reranker_missing_dashscope_api_key"],
                },
            }
        },
        {
            "output": {
                "answer": "Оформить акт. [S1]",
                "citations_valid": True,
                "evidence_count": 2,
                "retrieval": {
                    "final_evidence": 2,
                    "reranker": "fallback:deterministic",
                    "fallbacks": ["reranker_missing_dashscope_api_key"],
                },
            }
        },
    ]
```

Add this test to `tests/test_runtime.py`:

```python
def test_runtime_provider_generation_traces_qwen_chat(monkeypatch, tmp_path):
    from langchain_core.documents import Document

    trace_calls = []
    docs = [Document(page_content="Возврат брака оформляется актом.", metadata={"citation_id": "return"})]

    class FakeSpan:
        def set_output(self, value):
            trace_calls.append({"output": value})

        def set_llm_output_message(self, content, *, role="assistant", index=0):
            trace_calls.append({"llm_output": content, "role": role, "index": index})

        def set_llm_token_counts(self, *, prompt=None, completion=None, total=None):
            trace_calls.append({"tokens": {"prompt": prompt, "completion": completion, "total": total}})

    @contextmanager
    def fake_trace_llm_step(
        name,
        messages,
        *,
        model_name,
        provider,
        invocation_parameters=None,
        attributes=None,
    ):
        trace_calls.append(
            {
                "name": name,
                "messages": messages,
                "model_name": model_name,
                "provider": provider,
                "invocation_parameters": invocation_parameters,
                "attributes": attributes,
            }
        )
        yield FakeSpan()

    class FakeChatModel:
        def invoke(self, messages):
            trace_calls.append({"chat_messages": messages})
            return type(
                "FakeResponse",
                (),
                {
                    "content": "Возврат брака оформляется актом. [S1]",
                    "usage_metadata": {"input_tokens": 10, "output_tokens": 6, "total_tokens": 16},
                },
            )()

    class FakeWorkflow:
        def __init__(self, retrieve, generate):
            self.retrieve = retrieve
            self.generate = generate

        def invoke(self, state):
            self.retrieve(state["question"])
            answer = self.generate(state["question"], docs)
            return {"answer": answer, "evidence": docs, "citations_valid": True}

    monkeypatch.setattr("imperial_rag.runtime.trace_llm_step", fake_trace_llm_step)
    monkeypatch.setattr("imperial_rag.runtime.ElasticsearchKeywordIndex", lambda settings: object())
    monkeypatch.setattr("imperial_rag.runtime._semantic_search_enabled", lambda: False)
    monkeypatch.setattr("imperial_rag.runtime.build_query_workflow", lambda **kwargs: FakeWorkflow(**kwargs))
    monkeypatch.setattr("imperial_rag.runtime.create_chat_model", lambda: FakeChatModel())
    monkeypatch.setenv("IMPERIAL_RAG_QWEN_CHAT_MODEL", "qwen-test")

    runtime = create_runtime(Settings(workspace_root=tmp_path))
    result = runtime.query("Как оформить возврат брака?")

    assert result["answer"] == "Возврат брака оформляется актом. [S1]"
    assert trace_calls[0]["name"] == "qwen.chat"
    assert trace_calls[0]["model_name"] == "qwen-test"
    assert trace_calls[0]["provider"] == "dashscope"
    assert trace_calls[0]["invocation_parameters"] == {"temperature": 0}
    assert trace_calls[0]["messages"][0]["role"] == "system"
    assert trace_calls[0]["messages"][1]["role"] == "user"
    assert trace_calls[1]["chat_messages"] == trace_calls[0]["messages"]
    assert trace_calls[2] == {"llm_output": "Возврат брака оформляется актом. [S1]", "role": "assistant", "index": 0}
    assert trace_calls[3] == {"output": "Возврат брака оформляется актом. [S1]"}
    assert trace_calls[4] == {"tokens": {"prompt": 10, "completion": 6, "total": 16}}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run python -m pytest tests/test_runtime.py::test_runtime_query_wraps_workflow_in_agent_span tests/test_runtime.py::test_runtime_provider_generation_traces_qwen_chat -q
```

Expected: FAIL because `Runtime.query` does not create `query.workflow` and `runtime.py` does not import or call `trace_llm_step`.

- [ ] **Step 3: Add workflow and LLM imports**

Edit the imports in `src/imperial_rag/runtime.py`:

```python
from imperial_rag.providers import (
    QwenProviderSettings,
    create_chat_model,
    dashscope_configured,
    vector_metadata_matches_config,
)
from imperial_rag.tracing import trace_agent_step, trace_llm_step, trace_pipeline_step
```

- [ ] **Step 4: Add `query.workflow` in `Runtime.query`**

Replace `Runtime.query` in `src/imperial_rag/runtime.py` with:

```python
    def query(self, question: str) -> dict:
        with trace_agent_step(
            "imperial_rag.query",
            question,
            attributes={"runtime.workspace_root": str(self.settings.workspace_root)},
        ) as span:
            with trace_pipeline_step(
                "query.workflow",
                question,
                attributes={"runtime.workflow": "langgraph"},
            ) as workflow_span:
                result = self.query_workflow().invoke({"question": question})
                workflow_span.set_output(_query_trace_output(result))
            span.set_output(_query_trace_output(result))
            return result
```

- [ ] **Step 5: Add Qwen chat tracing in provider-owned generation**

Replace the `generate` closure inside `create_runtime` in `src/imperial_rag/runtime.py` with:

```python
    def generate(question: str, docs):
        messages = build_strict_messages(question, docs)
        provider_settings = QwenProviderSettings.from_env()
        try:
            with trace_llm_step(
                "qwen.chat",
                messages,
                model_name=provider_settings.chat_model,
                provider="dashscope",
                invocation_parameters={"temperature": 0},
            ) as span:
                response = dependencies().chat_model.invoke(messages)
                content = getattr(response, "content", response)
                span.set_llm_output_message(str(content))
                span.set_output(str(content))
                _set_llm_token_counts(span, response)
                return content
        except Exception:
            from imperial_rag.answering import REFUSAL_TEXT

            return REFUSAL_TEXT
```

Add this helper near `_query_trace_output`:

```python
def _set_llm_token_counts(span: Any, response: Any) -> None:
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        metadata = getattr(response, "response_metadata", None)
        if isinstance(metadata, dict):
            usage = metadata.get("token_usage") or metadata.get("usage")
    if not isinstance(usage, dict):
        return
    prompt = usage.get("input_tokens") or usage.get("prompt_tokens")
    completion = usage.get("output_tokens") or usage.get("completion_tokens")
    total = usage.get("total_tokens")
    span.set_llm_token_counts(prompt=prompt, completion=completion, total=total)
```

- [ ] **Step 6: Run focused runtime tests**

Run:

```bash
uv run python -m pytest tests/test_runtime.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git status --short
git add src/imperial_rag/runtime.py tests/test_runtime.py
git commit -m "feat: trace domain query workflow"
```

Expected: commit succeeds with only the two listed files staged.

---

### Task 3: Add Retrieval Grouping Span

**Files:**
- Modify: `tests/test_retrieval.py`
- Modify: `src/imperial_rag/retrieval.py`

- [ ] **Step 1: Update retrieval tracing test**

Update `test_retrieval_service_traces_each_retrieval_step` in `tests/test_retrieval.py` so the expected names include the grouping span:

```python
    assert [record["name"] for record in records] == [
        "retrieve",
        "retrieve.vector_search",
        "retrieve.keyword_search",
        "retrieve.merge_candidates",
        "retrieve.fuse_candidates",
        "retrieve.rerank",
    ]
    assert [record["query"] for record in records] == ["возврат брака"] * 6
    assert records[0]["kind"] == "CHAIN"
    assert records[0]["attributes"] == {
        "retrieval.vector_k": 70,
        "retrieval.keyword_limit": 30,
        "retrieval.rerank_top_n": 1,
    }
    assert records[0]["output"]["final_evidence"] == 1
    assert records[0]["output"]["vector_candidates"] == 1
    assert records[0]["output"]["keyword_candidates"] == 1
```

Then shift the existing child-span assertions down by one index:

```python
    assert records[1]["output"]["status"] == "ok"
    assert records[1]["output"]["count"] == 1
    assert records[1]["output"]["top_documents"][0]["citation_id"] == "v"
    assert records[1]["set_attributes"]["retrieval.documents.0.document.id"] == "v"
    assert records[1]["set_attributes"]["retrieval.documents.0.document.content"] == "vector return"
    assert records[2]["output"]["status"] == "ok"
    assert records[2]["output"]["count"] == 1
    assert records[2]["set_attributes"]["retrieval.documents.0.document.id"] == "k"
    assert records[2]["set_attributes"]["retrieval.documents.0.document.content"] == "Порядок возврата брака"
    assert records[3]["kind"] == "CHAIN"
    assert records[3]["output"]["count"] == 2
    assert records[4]["kind"] == "CHAIN"
    assert records[4]["output"]["fusion"] == "rrf"
    assert records[4]["output"]["count"] == 2
    assert records[5]["kind"] == "RERANKER"
    assert records[5]["attributes"]["reranker.query"] == "возврат брака"
    assert records[5]["attributes"]["reranker.top_k"] == 1
    assert records[5]["set_attributes"]["reranker.model_name"] == "fallback:deterministic"
    assert records[5]["output"]["reranker"] == "fallback:deterministic"
    assert "reranker_missing_dashscope_api_key" in records[5]["output"]["fallbacks"]
    assert records[5]["set_attributes"]["reranker.input_documents.0.document.id"] == "v"
    assert records[5]["set_attributes"]["reranker.input_documents.1.document.id"] == "k"
    assert records[5]["set_attributes"]["reranker.output_documents.0.document.id"] == "k"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run python -m pytest tests/test_retrieval.py::test_retrieval_service_traces_each_retrieval_step -q
```

Expected: FAIL because the first recorded span is currently `retrieve.vector_search`, not `retrieve`.

- [ ] **Step 3: Wrap `RetrievalService.retrieve` in a parent span**

Replace `RetrievalService.retrieve` in `src/imperial_rag/retrieval.py` with:

```python
    def retrieve(self, query: str) -> RetrievalResult:
        with trace_retrieval_step(
            "retrieve",
            query,
            kind="CHAIN",
            attributes={
                "retrieval.vector_k": self.settings.vector_k,
                "retrieval.keyword_limit": self.settings.keyword_limit,
                "retrieval.rerank_top_n": self.settings.rerank_top_n,
            },
        ) as span:
            candidates = self.hybrid.retrieve(query)
            diagnostics = dict(candidates.diagnostics)
            with trace_retrieval_step(
                "retrieve.merge_candidates",
                query,
                kind="CHAIN",
                attributes={
                    "retrieval.vector_candidates": len(candidates.vector_docs),
                    "retrieval.keyword_candidates": len(candidates.keyword_docs),
                },
            ) as merge_span:
                merged = self.merger.merge(candidates.vector_docs, candidates.keyword_docs)
                diagnostics["merged_candidates"] = len(merged)
                _set_documents_span_output(
                    merge_span,
                    merged,
                    vector_candidates=len(candidates.vector_docs),
                    keyword_candidates=len(candidates.keyword_docs),
                )

            with trace_retrieval_step(
                "retrieve.fuse_candidates",
                query,
                kind="CHAIN",
                attributes={
                    "retrieval.fusion": "rrf",
                    "retrieval.fusion_rrf_k": self.settings.rrf_k,
                    "retrieval.input_count": len(merged),
                    "retrieval.rerank_input_limit": self.settings.rerank_input_limit,
                },
            ) as fuse_span:
                fused = self.fusion.fuse(merged, rrf_k=self.settings.rrf_k)
                rerank_input = fused[: self.settings.rerank_input_limit]
                diagnostics["fusion"] = "rrf"
                diagnostics["fusion_rrf_k"] = self.settings.rrf_k
                diagnostics["fused_candidates"] = len(fused)
                diagnostics["rerank_input_candidates"] = len(rerank_input)
                _set_documents_span_output(
                    fuse_span,
                    fused,
                    fusion="rrf",
                    fusion_rrf_k=self.settings.rrf_k,
                    rerank_input_candidates=len(rerank_input),
                )

            with trace_retrieval_step(
                "retrieve.rerank",
                query,
                kind="RERANKER",
                attributes={
                    "reranker.query": query,
                    "reranker.top_k": self.settings.rerank_top_n,
                    "retrieval.rerank_input_limit": self.settings.rerank_input_limit,
                    "retrieval.rerank_top_n": self.settings.rerank_top_n,
                    "retrieval.primary_reranker": self.settings.primary_reranker,
                },
            ) as rerank_span:
                rerank_span.set_reranker_input_documents(rerank_input)
                reranked = self.reranker.rerank(query, rerank_input, diagnostics)
                rerank_span.set_attribute("reranker.model_name", diagnostics.get("reranker"))
                _set_documents_span_output(
                    rerank_span,
                    reranked,
                    reranker=diagnostics.get("reranker"),
                    rerank_input=diagnostics.get("rerank_input"),
                    reranked_candidates=diagnostics.get("reranked_candidates"),
                    fallbacks=diagnostics.get("fallbacks", []),
                )
                rerank_span.set_reranker_output_documents(reranked)

            evidence = reranked
            diagnostics["final_evidence"] = len(evidence)
            result = RetrievalResult(
                evidence=evidence,
                vector_docs=candidates.vector_docs,
                keyword_docs=candidates.keyword_docs,
                diagnostics=diagnostics,
            )
            span.set_output(
                {
                    "vector_candidates": len(candidates.vector_docs),
                    "keyword_candidates": len(candidates.keyword_docs),
                    "merged_candidates": diagnostics.get("merged_candidates"),
                    "fused_candidates": diagnostics.get("fused_candidates"),
                    "rerank_input_candidates": diagnostics.get("rerank_input_candidates"),
                    "reranked_candidates": diagnostics.get("reranked_candidates"),
                    "final_evidence": diagnostics.get("final_evidence"),
                    "reranker": diagnostics.get("reranker"),
                    "fallbacks": diagnostics.get("fallbacks", []),
                }
            )
            return result
```

- [ ] **Step 4: Run focused retrieval tests**

Run:

```bash
uv run python -m pytest tests/test_retrieval.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
git status --short
git add src/imperial_rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: group retrieval trace spans"
```

Expected: commit succeeds with only the two listed files staged.

---

### Task 4: Preserve UI And Eval Routing Through Runtime Query

**Files:**
- Modify: `tests/test_scripts.py`
- Modify: `tests/test_web_app.py`
- Modify: `tests/test_evals.py`
- Modify only if tests fail: `scripts/query.py`
- Modify only if tests fail: `src/imperial_rag/web_app.py`
- Modify only if tests fail: `scripts/run_phoenix_eval.py`

- [ ] **Step 1: Add CLI runtime routing test**

Add this test to `tests/test_scripts.py`:

```python
def test_query_script_uses_create_runtime_query_path(monkeypatch):
    module = _load_script("scripts/query.py", "query_script_runtime_path")
    calls = []

    class FakeRuntime:
        def query(self, question):
            calls.append(("query", question))
            return {"answer": "ok", "sources": []}

    runtime_module = types.ModuleType("imperial_rag.runtime")
    runtime_module.create_runtime = lambda settings: calls.append(("create_runtime", settings)) or FakeRuntime()
    monkeypatch.setitem(sys.modules, "imperial_rag.runtime", runtime_module)

    settings = object()

    assert module._query(settings=settings, question="Как оформить возврат брака?") == {"answer": "ok", "sources": []}
    assert calls == [("create_runtime", settings), ("query", "Как оформить возврат брака?")]
```

- [ ] **Step 2: Add Streamlit runtime routing test**

Add this test to `tests/test_web_app.py`:

```python
def test_query_runtime_uses_create_runtime_query_path(monkeypatch):
    from imperial_rag import web_app

    calls = []

    class FakeRuntime:
        def query(self, question):
            calls.append(("query", question))
            return {"answer": "ok", "sources": []}

    runtime_module = types.ModuleType("imperial_rag.runtime")
    runtime_module.create_runtime = lambda settings: calls.append(("create_runtime", settings)) or FakeRuntime()
    monkeypatch.setitem(sys.modules, "imperial_rag.runtime", runtime_module)

    settings = object()

    assert web_app.query_runtime(settings, "Как оформить возврат брака?") == {"answer": "ok", "sources": []}
    assert calls == [("create_runtime", settings), ("query", "Как оформить возврат брака?")]
```

- [ ] **Step 3: Add Phoenix eval runtime reuse test**

Add this test to `tests/test_evals.py`:

```python
def test_phoenix_experiment_reuses_runtime_query_path(monkeypatch):
    module = _load_eval_runner()
    captured: dict[str, object] = {}
    runtime_calls = []

    class FakeDatasets:
        def create_dataset(self, **kwargs):
            return {"dataset_id": "dataset-1"}

    class FakeExperiments:
        def run_experiment(self, **kwargs):
            captured["task"] = kwargs["task"]
            captured["evaluators"] = kwargs["evaluators"]
            captured["experiment_name"] = kwargs["experiment_name"]
            return SimpleNamespace(id="experiment-1")

    class FakeClient:
        def __init__(self, **kwargs):
            self.datasets = FakeDatasets()
            self.experiments = FakeExperiments()

    class FakeRuntime:
        def query(self, question: str) -> dict[str, object]:
            runtime_calls.append(question)
            return {"answer": f"Ответ на {question}", "citations": ["[S1] body"], "evidence": []}

    fake_client_module = types.ModuleType("phoenix.client")
    fake_client_module.Client = FakeClient
    monkeypatch.setitem(sys.modules, "phoenix.client", fake_client_module)
    monkeypatch.setattr(module, "build_runtime", lambda settings=None: FakeRuntime())
    monkeypatch.setattr(module, "_get_ragas_faithfulness_scorer", lambda: object())

    module.run_phoenix_experiment(
        examples=[
            {
                "question": "Что делать с браком?",
                "expected_behavior": "cite_answer",
                "expected_source_hints": ["брак"],
            }
        ],
        settings=SimpleNamespace(phoenix_client_endpoint="http://localhost:6006"),
        dataset_name="imperial-rag-gold-questions",
        experiment_name="imperial-rag-citation-grounding",
        ragas_metric_names=[],
    )

    output = captured["task"]({"question": "Что делать с браком?"})

    assert output["answer"] == "Ответ на Что делать с браком?"
    assert runtime_calls == ["Что делать с браком?"]
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run python -m pytest tests/test_scripts.py::test_query_script_uses_create_runtime_query_path tests/test_web_app.py::test_query_runtime_uses_create_runtime_query_path tests/test_evals.py::test_phoenix_experiment_reuses_runtime_query_path -q
```

Expected: PASS if the current routing already satisfies the design. If any test fails, continue to Step 5.

- [ ] **Step 5: Fix routing only if the focused routing tests fail**

If the CLI test fails, replace `_query` in `scripts/query.py` with:

```python
def _query(settings: Any, question: str) -> dict[str, Any]:
    try:
        from imperial_rag.runtime import create_runtime
    except (ImportError, AttributeError):
        create_runtime = None
    if create_runtime is not None:
        return _coerce_result(create_runtime(settings).query(question))

    try:
        from imperial_rag.runtime import Runtime
    except (ImportError, AttributeError):
        Runtime = None

    if Runtime is not None:
        runtime = Runtime(settings=settings)
        return _coerce_result(runtime.query(question))

    from imperial_rag.runtime import build_live_query_workflow

    workflow = build_live_query_workflow(settings)
    return _coerce_result(workflow.invoke({"question": question}))
```

If the Streamlit test fails, replace `query_runtime` in `src/imperial_rag/web_app.py` with:

```python
def query_runtime(settings: Any, question: str) -> dict[str, Any]:
    try:
        from imperial_rag.runtime import create_runtime
    except (ImportError, AttributeError):
        create_runtime = None
    if create_runtime is not None:
        return _coerce_result(create_runtime(settings).query(question))

    try:
        from imperial_rag.runtime import Runtime
    except (ImportError, AttributeError):
        Runtime = None

    if Runtime is not None:
        return _coerce_result(Runtime(settings=settings).query(question))

    from imperial_rag.runtime import build_live_query_workflow

    return _coerce_result(build_live_query_workflow(settings).invoke({"question": question}))
```

If the eval test fails, update `run_phoenix_experiment` in `scripts/run_phoenix_eval.py` so it builds one runtime before `bound_target` and calls `run_target(inputs, runtime=runtime)`:

```python
    runtime = build_runtime(settings=settings)

    def bound_target(inputs: dict[str, Any]) -> dict[str, Any]:
        return run_target(inputs, runtime=runtime)
```

- [ ] **Step 6: Run focused tests again**

Run:

```bash
uv run python -m pytest tests/test_scripts.py::test_query_script_uses_create_runtime_query_path tests/test_web_app.py::test_query_runtime_uses_create_runtime_query_path tests/test_evals.py::test_phoenix_experiment_reuses_runtime_query_path -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```bash
git status --short
git add tests/test_scripts.py tests/test_web_app.py tests/test_evals.py
git add scripts/query.py src/imperial_rag/web_app.py scripts/run_phoenix_eval.py
git commit -m "test: lock query tracing runtime path"
```

Expected: commit succeeds. If `scripts/query.py`, `src/imperial_rag/web_app.py`, or `scripts/run_phoenix_eval.py` did not change, Git leaves them unstaged and commits only the test files.

---

### Task 5: Document Domain Tracing Defaults

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add README tracing section**

Add this section near the Phoenix/evaluation documentation in `README.md`:

```markdown
### Phoenix query traces

Imperial emits domain-first Phoenix traces for query and eval runs. The default trace tree is:

`imperial_rag.query -> query.workflow -> retrieve -> retrieve.* -> answer.generate -> qwen.chat -> answer.validate_citations`

The manual spans are the supported debugging surface. Framework auto-instrumentation is opt-in with:

```bash
IMPERIAL_RAG_PHOENIX_AUTO_INSTRUMENT=1
```

Prompt messages in the `qwen.chat` span are truncated to `IMPERIAL_RAG_TRACE_MESSAGE_CONTENT_CHARS`
characters per message, defaulting to `2000`. Use OpenInference hide flags such as
`OPENINFERENCE_HIDE_INPUTS=true`, `OPENINFERENCE_HIDE_OUTPUTS=true`,
`OPENINFERENCE_HIDE_INPUT_MESSAGES=true`, and `OPENINFERENCE_HIDE_OUTPUT_MESSAGES=true` when traces
should avoid private prompt or answer text.

Run a live traced query with:

```bash
uv run python scripts/query.py "Как оформить возврат брака?" --trace-phoenix
```

Then inspect `http://127.0.0.1:6006` in the `imperial-rag` Phoenix project.
```

- [ ] **Step 2: Run docs sanity check**

Run:

```bash
rg -n "Phoenix query traces|IMPERIAL_RAG_TRACE_MESSAGE_CONTENT_CHARS|IMPERIAL_RAG_PHOENIX_AUTO_INSTRUMENT" README.md .env.example
```

Expected: all three terms are found in both `README.md` and `.env.example`.

- [ ] **Step 3: Commit**

Run:

```bash
git status --short
git add README.md
git commit -m "docs: describe phoenix domain traces"
```

Expected: commit succeeds with only `README.md` staged.

---

### Task 6: Run Focused Suite And Live Smoke

**Files:**
- No source edits in this task.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
uv run python -m pytest tests/test_tracing.py tests/test_runtime.py tests/test_workflows.py tests/test_retrieval.py tests/test_evals.py tests/test_web_app.py -q
```

Expected: PASS.

- [ ] **Step 2: Check Phoenix and app stack status**

Run:

```bash
docker compose ps
curl -fsS http://127.0.0.1:6006/ >/dev/null
```

Expected: Phoenix is reachable. If Phoenix is not reachable, run:

```bash
docker compose up -d phoenix
curl -fsS http://127.0.0.1:6006/ >/dev/null
```

Expected: Phoenix becomes reachable.

- [ ] **Step 3: Run one traced query**

Run:

```bash
uv run python scripts/query.py "Как оформить возврат брака?" --trace-phoenix --trace-session-id domain-trace-smoke
```

Expected: the command prints an answer or the existing refusal text. It should not fail because of tracing.

- [ ] **Step 4: Inspect Phoenix manually**

Open `http://127.0.0.1:6006/projects` and inspect the `imperial-rag` project. The newest trace for session `domain-trace-smoke` should show:

- `imperial_rag.query` as the root `AGENT` span.
- `query.workflow` as a child `CHAIN` span.
- `retrieve` as a child grouping span containing vector, keyword, merge, fuse, and rerank spans.
- `answer.generate` as a child span containing `qwen.chat`.
- `qwen.chat` with `llm.model_name`, prompt messages, assistant output message, and token counts when available.

- [ ] **Step 5: Inspect final diff and status**

Run:

```bash
git status --short
git log --oneline -5
```

Expected: only unrelated pre-existing dirty files remain. The task commits from this plan appear at the top of the log.
