# Imperial RAG — Codebase Review

**Date:** 2026-06-19  
**Branch:** `codex/phoenix-trace-quality`  
**Scope:** Full `src/imperial_rag/` codebase (all Python source files)  
**Focus:** Correctness bugs, missing guards, custom code that should be replaced with frameworks/libraries

---

## Summary

10 findings across correctness bugs, silent failure modes, dead configuration, and repeated custom utilities that established libraries cover. No code was changed; all findings are observations only.

---

## Findings

### 1 · `runtime.py:144` — LLM exceptions swallowed silently; user receives refusal text instead of error

**Severity: HIGH (correctness)**

`generate()` inside `create_runtime` catches every `Exception` from the chat model and returns `{"answer": REFUSAL_TEXT, ...}` with no re-raise and no user-visible error signal.

```python
# runtime.py
try:
    response = dependencies().chat_model.invoke(build_strict_messages(question, docs))
except Exception as exc:
    return {
        "answer": REFUSAL_TEXT,   # same as "no evidence found"
        "trace_attributes": {"answer.model_status": "error", ...},
    }
```

The caller in `workflows.py` extracts `str(generated["answer"])`, which is the refusal string. The web UI renders it identically to a legitimate "I could not find this clearly in the indexed documents" response. Network timeouts, rate-limit errors, expired API keys, and token-limit overflows all produce silent refusals. The `answer.model_status: "error"` tag is written only to Phoenix traces — invisible to end users and to any caller that doesn't inspect the trace.

**Fix direction:** Surface the error to the caller (raise, or include an `"error"` key in the returned dict) so the web app can display a meaningful error message.

---

### 2 · `config.py:8` — Hardcoded developer path as default workspace root

**Severity: HIGH (correctness)**

```python
DEFAULT_WORKSPACE_ROOT = Path("/Users/danil/Public/imperial")

class Settings:
    workspace_root: Path = field(
        default_factory=lambda: Path(os.environ.get("IMPERIAL_RAG_WORKSPACE_ROOT", DEFAULT_WORKSPACE_ROOT))
    )
```

When `IMPERIAL_RAG_WORKSPACE_ROOT` is unset, every `Settings()` instance uses `/Users/danil/Public/imperial`. On any other machine or CI environment, `scan_files` calls `documents_root.rglob("*")` on a path that does not exist and returns zero files — ingestion silently processes nothing. The manifest, Qdrant, and Elasticsearch layers then operate on empty data with no error.

**Fix direction:** Replace with `Path.cwd()` or derive from `Path(__file__).resolve().parents[3]` (the project root relative to `src/imperial_rag/config.py`).

---

### 3 · `pipeline.py:90` — `assign_duplicate_groups` called twice

**Severity: MEDIUM (correctness / efficiency)**

```python
# manifest.py — scan_files already assigns duplicate groups
def scan_files(documents_root: Path) -> list[FileRecord]:
    records = [...]
    return assign_duplicate_groups(records)   # ← groups assigned here

# pipeline.py — wraps scan_files result in another assign_duplicate_groups call
records = deps["assign_duplicate_groups"](deps["scan_files"](Path(settings.documents_root)))
```

`scan_files` returns records with `duplicate_group_id` already set. `pipeline._run` then passes that list through `assign_duplicate_groups` a second time, creating fresh `FileRecord` objects with the same values. The second pass is idempotent today only because SHA-256 hashes don't change. If `scan_files` is refactored to drop the internal call (to separate concerns), `pipeline.py` keeps working; if `pipeline.py` is refactored to drop the outer call, duplicates stop being grouped. The API contract is ambiguous and the redundant work scales linearly with corpus size.

**Fix direction:** Remove the `assign_duplicate_groups` call from inside `scan_files`, making it a pure file-listing function. Keep the single explicit call in `pipeline._run`.

---

### 4 · `manifest.py:104` — `ManifestStore` SQLite connection is never closed

**Severity: MEDIUM (resource leak)**

`ManifestStore.__init__` opens a persistent `sqlite3` connection that must be closed by calling `.close()`. No `__enter__`/`__exit__` context manager is implemented. `pipeline._run` creates a `ManifestStore` and never calls `.close()`:

```python
# pipeline.py
manifest_store = deps["ManifestStore"](Path(settings.manifest_db_path))
manifest_store.replace_records(records)
# ...many operations...
# no manifest_store.close()
```

In long-running processes (the Streamlit server), repeated ingestion runs accumulate open SQLite handles. In WAL mode, unclosed connections prevent checkpoint compaction. On Windows, leaked handles block file-level operations from other processes.

**Fix direction:** Implement `__enter__` / `__exit__` on `ManifestStore` so callers can use `with ManifestStore(...) as store:`. This also makes usage consistent with `AuthStore`, which already opens per-call connections.

---

### 5 · `config.py:16` — `_log_format_from_env` always returns `"json"`; env var and Settings field are dead

**Severity: LOW (dead code)**

```python
def _log_format_from_env() -> str:
    raw = os.environ.get("IMPERIAL_RAG_LOG_FORMAT", "json").strip().casefold()
    return raw if raw == "json" else "json"   # both branches return "json"
```

Setting `IMPERIAL_RAG_LOG_FORMAT=text` has no effect. The `Settings.log_format` field is also never read anywhere in `observability.py` or elsewhere in the codebase. The env var, the helper, and the dataclass field are all inert.

**Fix direction:** Either implement non-JSON log formatting (e.g. plain text for local development) or remove `_log_format_from_env`, `Settings.log_format`, and the env var from documentation.

---

### 6 · `retrieval.py:21`, `providers.py:41`, `tracing.py:446` — Three diverging copies of `_env_*` helpers

**Severity: MEDIUM (reuse / diverging behavior)**

The same pattern — read env var, strip, default, cast — is re-implemented independently in three modules:

| Module | Functions |
|---|---|
| `retrieval.py:21–38` | `_env_int`, `_env_float`, `_env_str` |
| `providers.py:41–73` | `_env_str`, `_env_optional_str`, `_env_optional_int`, `_env_bool`, `_env_optional_bool` |
| `tracing.py:446–460` | `_env_flag`, `_env_int` (with `minimum=` guard and `ValueError` fallback) |

The implementations have already diverged. `providers._env_str` strips the value; `retrieval._env_str` does not. `tracing._env_int` falls back to `default` on `ValueError`; `retrieval._env_int` raises. A misconfigured env var (e.g. `IMPERIAL_RAG_CHUNK_SIZE=" 5 "`) parses correctly via `tracing` but raises in `retrieval`.

`pydantic-settings` (`BaseSettings`) would replace all of this with declarative, typed, validated settings fields and eliminates the per-field boilerplate entirely.

---

### 7 · `workflows.py:85` and `retrieval.py:221` — `_document_key` and `_content_key` duplicated verbatim

**Severity: MEDIUM (reuse)**

Both functions are defined identically in two modules:

```python
# workflows.py lines 85, 90
def _document_key(document: Document) -> str:
    metadata = document.metadata or {}
    return metadata_or_content_id(metadata.get("citation_id"), metadata.get("chunk_id"), content=document.page_content)

def _content_key(document: Document) -> str:
    return " ".join(document.page_content.split()).casefold()

# retrieval.py lines 221, 247 — identical bodies
```

`workflows.py` already imports `CandidateMerger` from `retrieval` but not these two helpers. If `_document_key` is updated in `retrieval.py` (e.g. to prefer a different metadata field), `workflows.rank_hybrid_candidates` silently diverges, producing different de-duplication behavior from `CandidateMerger.merge`.

**Fix direction:** Export `_document_key` and `_content_key` from `retrieval.py` (or `document_ids.py`) and import them in `workflows.py`.

---

### 8 · `retrieval.py:255` — `_searchable_text` duplicates `keyword.searchable_document_text`

**Severity: LOW (reuse)**

`retrieval._searchable_text` and `keyword.searchable_document_text` join the same five fields in the same order; the only difference is a `.casefold()` call at the end. `workflows.py:113–121` inlines the same join a third time without casefold.

```python
# keyword.py — canonical definition
def searchable_document_text(document: Document) -> str:
    return " ".join([document.page_content, file_name, relative_path, section_heading, source_type])

# retrieval.py — same, plus casefold
def _searchable_text(document: Document) -> str:
    return " ".join([...same five fields...]).casefold()
```

A new metadata field (e.g. `embedded_media_name`) added to one copy silently misses the other two, causing inconsistent scoring.

**Fix direction:** `retrieval._searchable_text` should call `keyword.searchable_document_text(document).casefold()`.

---

### 9 · `tracing.py:244` — `trace_user_id_from_email(None)` returns a hashed user ID instead of `""`

**Severity: LOW (latent bug)**

```python
def trace_user_id_from_email(email: str) -> str:
    normalized = str(email).strip().casefold()   # str(None) == "none"
    if not normalized:                            # "none" is truthy — guard skipped
        return ""
    # ... hashes "none" and returns "user_sha256:<digest>"
```

If `None` is passed (possible from any caller that doesn't statically enforce `str`), the string `"none"` is hashed and returned as a real pseudonymous user ID, attributing trace data to a phantom user. The web app path is safe today (`current_user.email` is a `NOT NULL` SQLite column), but the function's signature gives no protection for other callers.

**Fix direction:** Guard at the top: `if not email or not isinstance(email, str): return ""`.

---

### 10 · `keyword.py:10` — Custom Russian stemmer; `"найт"` implementation detail leaks into stopwords

**Severity: MEDIUM (framework preference / correctness)**

```python
_ENDING_RE = re.compile(r"(иями|ями|ами|ого|его|...")
def stem_token(token: str) -> str:
    token = token.casefold().replace("ё", "е")
    while len(token) > 4:
        shortened = _ENDING_RE.sub("", token)
        if shortened == token:
            break
        token = shortened
    return token
```

This is a hand-rolled iterative suffix stripper with no morphological awareness. Known problem: `stem_token("найти")` → `"найт"`, which is then listed in `_QUERY_STOPWORDS`:

```python
_QUERY_STOPWORDS = frozenset({"найт", ...})
```

`"найт"` is not a Russian word — it is the output of this specific stemmer applied to "найти" (to find). If the stemmer regex changes, the stopword entry either stops matching anything or incorrectly matches real tokens. The stopword list is coupled to the stemmer implementation.

`nltk.stem.SnowballStemmer("russian")` or `pymorphy3` provide proper Russian stemming. Per the project's stated preference for libraries over custom code, this module is a candidate for replacement.

---

## Secondary Observations (no individual severity rating)

These are smaller issues consistent with the reuse/framework theme:

| Location | Observation |
|---|---|
| `providers.py:41–73` | 5 `_env_*` functions → replace with `pydantic-settings BaseSettings` |
| `manifest.py:100` | Raw `sqlite3` with hand-written `_to_row`/`_from_row` and no schema migrations → `SQLAlchemy Core` or `SQLModel` |
| `runtime.py:23,38` | `_NoopVectorSearch` and `_ProviderMismatchVectorSearch` are custom null-object classes for LangChain interfaces → use `langchain_core.retrievers.BaseRetriever` subclass or `InMemoryVectorStore` |
| `retrieval.py:338,346` | `CandidateMerger()` instantiated twice inside `RrfCandidateFusion.fuse`; the class is stateless — one instance is enough |
| `retrieval.py:236–243` | `_annotate_retrieval_documents` creates an intermediate `Document` object just to pass to `_retrieval_id`, which only reads `.metadata` — the intermediate object can be eliminated |
| `runtime.py:116–127` | `dict` used as single-slot cache (`{"value": ...}`) instead of a `nonlocal` variable or `functools.cached_property` |
| `indexing.py:76–87` | `make_qdrant_store` constructs `Settings()` twice — once to read env, once to pass selected fields forward |
