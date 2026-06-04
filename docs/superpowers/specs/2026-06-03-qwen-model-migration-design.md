# Qwen Model Migration Design

Date: 2026-06-03
Workspace: `/Users/danil/Public/imperial`
Status: Approved in chat, then updated after plan review. Written spec awaiting user review before implementation planning.

## Context

Imperial RAG currently has four paid/model-backed surfaces:

- answer generation uses OpenAI chat models in runtime and workflow fallbacks;
- OCR uses OpenAI vision chat for image/PDF/DOCX embedded-image text extraction;
- vector indexing and semantic search use OpenAI embeddings through Qdrant;
- reranker settings name Cohere models, while the live runtime still mostly uses deterministic candidate ranking.

The user wants all model-backed behavior moved to hosted Qwen through Alibaba DashScope, including LLM, reranker, embeddings, and OCR/vision.

This checkout is a git repository. The implementation workflow must capture `git status --short` before edits, stage only files touched by the migration, and commit the finished plan or implementation checkpoint according to the repository guidelines.

## Documentation Check

Current documentation was checked with Context7 before choosing the design:

- Official Qwen docs show Qwen model usage in LangChain/LlamaIndex contexts and confirm Qwen covers text generation and multimodal/vision use cases.
- DashScope Python SDK docs show `TextEmbedding`, `TextReRank`, and `MultiModalConversation` APIs.
- LangChain docs show a Qwen chat integration with `from langchain_qwq import ChatQwen` and `DASHSCOPE_API_KEY`.
- Current Qwen Cloud docs recommend `qwen3.7-max` for highest-capability text generation, `text-embedding-v4` as the latest text embedding model, and `qwen3-rerank` for reranking.
- Current Qwen Cloud OCR docs provide the dedicated `qwen-vl-ocr` model family, including `qwen-vl-ocr-2025-11-20` for OCR and structured document extraction.
- Current Qwen Cloud docs also show region-specific DashScope/OpenAI-compatible base URLs. Beijing, Singapore, and US endpoints are different, and Singapore may require a workspace-qualified endpoint. The provider config must not hard-code a single global URL.
- `text-embedding-v4` supports configurable dimensions, with 1024 as the default and 1536/2048 positioned for higher retrieval precision.
- `qwen3-rerank` is the text reranker for RAG, while `qwen3-vl-rerank` is the multimodal reranker. This migration only needs text reranking for extracted chunks.
- Qwen OCR can be called through DashScope `MultiModalConversation` and supports OCR options such as `text_recognition`, `multi_lan`, and higher-precision document tasks. The migration must choose the exact task and parse the OCR response explicitly.
- The installed environment already contains `langchain_community.embeddings.dashscope.DashScopeEmbeddings`, `langchain_community.document_compressors.dashscope_rerank.DashScopeRerank`, and `langchain_community.chat_models.tongyi.ChatTongyi`.
- Local import inspection before this update showed `dashscope` and `langchain_qwq` are not currently installed, even though the deprecated `langchain_community` DashScope/Tongyi integrations are importable. Dependency verification must check actual imports after `uv sync`, not just dependency names in `pyproject.toml`.
- Local inspection warned that `langchain-community` is being sunset, so new code should prefer the documented standalone `langchain_qwq` chat dependency where available and use community integrations only where no standalone replacement is confirmed.

## Goals

- Replace default OpenAI and Cohere model providers with hosted Qwen/DashScope.
- Use maintained LangChain integrations where possible.
- Add `langchain-qwq` as the primary Qwen chat dependency.
- Keep model names and provider behavior configurable through environment variables.
- Preserve the strict evidence-only answer prompt and citation validator.
- Require a clean vector reindex after changing embedding providers.
- Keep normal tests offline by using fakes and import-level assertions.

## Non-Goals

- No local Hugging Face model hosting.
- No UI redesign.
- No change to the source document corpus.
- No broad rewrite of manifesting, extraction, chunking, Qdrant storage, or Phoenix tracing.
- No silent fallback to OpenAI or Cohere for default runtime behavior.
- No live DashScope calls in the default pytest suite.

## Recommended Approach

Use a LangChain-first DashScope/Qwen provider layer.

Primary integrations:

- Answer LLM: `langchain_qwq.ChatQwen`.
- Embeddings: `DashScopeEmbeddings` from LangChain integration packages available in this environment, unless implementation confirms a newer standalone package.
- Reranker: `DashScopeRerank` from LangChain integration packages available in this environment, unless implementation confirms a newer standalone package.
- OCR/vision: DashScope SDK `MultiModalConversation`, unless implementation proves the selected LangChain Qwen chat integration reliably supports the needed image payload format.

Compatibility fallback:

- `ChatTongyi` may be kept as a local compatibility fallback while introducing `langchain-qwq`.
- OpenAI/Cohere should not remain as configured defaults. Any legacy fallback should be opt-in and clearly named, not automatic.

## Architecture

Add a focused provider module rather than spreading provider choices across `runtime.py`, `ocr.py`, `indexing.py`, `retrieval.py`, and scripts.

Proposed module:

- `src/imperial_rag/providers.py`

Responsibilities:

- read Qwen/DashScope model settings;
- check `DASHSCOPE_API_KEY`;
- configure DashScope/OpenAI-compatible endpoint settings without leaking credentials;
- create chat, embedding, reranker, and vision/OCR clients;
- provide lightweight protocols or wrappers for test fakes;
- expose clear capability checks for scripts and runtime.

Existing modules should call provider helpers instead of importing provider SDKs directly:

- `indexing.py` asks the provider module for embeddings when no test embedding object is supplied.
- `runtime.py` asks the provider module for chat and semantic-search readiness.
- `ocr.py` uses the provider module or a dedicated Qwen vision client wrapper.
- `retrieval.py` uses a DashScope reranker class behind the same rerank interface as the deterministic fallback.
- `scripts/ingest.py` and `pipeline.py` use provider capability checks before OCR or vector indexing.

## Configuration

Use these environment variables:

| Variable | Purpose |
| --- | --- |
| `DASHSCOPE_API_KEY` | DashScope credential used by Qwen chat, embeddings, rerank, and vision. |
| `IMPERIAL_RAG_DASHSCOPE_REGION` | Human-readable provider region such as `beijing`, `singapore`, or `us`, used only for diagnostics and config validation. |
| `IMPERIAL_RAG_DASHSCOPE_BASE_URL` | Base URL for SDK-style DashScope calls, such as `https://dashscope.aliyuncs.com/api/v1` or the workspace-qualified Singapore endpoint. |
| `IMPERIAL_RAG_DASHSCOPE_COMPAT_BASE_URL` | OpenAI-compatible base URL for chat, embeddings, or OCR paths that use the compatible API. |
| `IMPERIAL_RAG_QWEN_CHAT_MODEL` | Hosted Qwen chat model for answer generation. |
| `IMPERIAL_RAG_QWEN_VISION_MODEL` | Qwen vision/multimodal model for OCR. |
| `IMPERIAL_RAG_QWEN_OCR_TASK` | OCR task passed to DashScope when using built-in OCR options. Default to multilingual text extraction for Russian/English corpus text. |
| `IMPERIAL_RAG_QWEN_OCR_MIN_PIXELS` | Optional OCR image upscaling threshold. |
| `IMPERIAL_RAG_QWEN_OCR_MAX_PIXELS` | Optional OCR image downscaling threshold to control OCR cost and payload size. |
| `IMPERIAL_RAG_QWEN_OCR_ENABLE_ROTATE` | Optional OCR auto-rotation toggle, defaulting to provider default unless tests require a fixed value. |
| `IMPERIAL_RAG_QWEN_EMBEDDING_MODEL` | DashScope text embedding model for Qdrant. |
| `IMPERIAL_RAG_QWEN_EMBEDDING_DIMENSIONS` | Optional embedding dimension when the selected integration supports it. |
| `IMPERIAL_RAG_QWEN_RERANK_MODEL` | DashScope text reranker model. |
| `IMPERIAL_RAG_ALLOW_LEGACY_OPENAI` | Optional escape hatch for old OpenAI behavior during debugging only. |
| `IMPERIAL_RAG_ALLOW_LEGACY_COHERE` | Optional escape hatch for old Cohere reranking during debugging only. |

Initial best-quality provider defaults:

- chat model: `qwen3.7-max`;
- embedding model: `text-embedding-v4`;
- embedding dimensions: `2048` for this high-precision local RAG corpus if the selected integration exposes dimension control cleanly; otherwise leave unset and record that the provider default is 1024 dimensions;
- rerank model: `qwen3-rerank`;
- OCR/vision model: `qwen-vl-ocr-2025-11-20`, with `qwen-vl-ocr-latest` as the moving alias fallback;
- OCR task: `multi_lan` unless implementation testing proves a more precise built-in task preserves Russian and English text better for this corpus.

## Data Flow

Ingestion with `--enable-ocr`:

1. Check `DASHSCOPE_API_KEY`.
2. Create a Qwen vision OCR client.
3. Configure the selected DashScope base URL before the first call.
4. Send a local image as a base64 data URL or provider-supported local-file payload.
5. Preserve the current instruction when using prompt-driven OCR: extract all visible Russian and English text verbatim, without summarizing.
6. When using built-in OCR options, set the configured OCR task and pixel/rotation options explicitly.
7. Parse the DashScope OCR response into a plain text string, preserving provider errors as warnings without exposing secrets.
8. Store OCR output in the existing OCR cache and extraction artifacts.

Ingestion with `--index-vectors`:

1. Check `DASHSCOPE_API_KEY`.
2. Create DashScope/Qwen embeddings.
3. Add chunks to the configured Qdrant collection with stable chunk ids.
4. Verify the target Qdrant collection is empty, newly created, or already marked with the same embedding provider, model, dimensions, and distance metric.
5. Record the Qwen embedding model and dimensions in manifest/index status where the current schema supports it.

Query runtime:

1. Keyword retrieval continues to work locally.
2. Semantic retrieval is enabled only when DashScope embeddings are configured.
3. Semantic retrieval refuses to use a Qdrant collection whose stored embedding provider, model, or dimensions do not match the configured Qwen embedding settings.
4. Candidate merge/dedupe stays provider-independent.
5. Reranking uses DashScope `qwen3-rerank` when configured.
6. Reranking falls back to deterministic ranking if DashScope rerank is missing or fails.
7. Answer generation uses `ChatQwen` with the existing strict citation prompt.
8. Citation validation remains the final guardrail and refuses unsupported answers.

## Vector Reindexing

Changing from OpenAI embeddings to DashScope embeddings changes vector semantics and may change dimensions.

The migration must require a clean vector reindex:

- clear/recreate the existing Qdrant collection, or
- use a new collection name such as `imperial_chunks_qwen`.

Do not mix OpenAI and Qwen embeddings in one collection. The implementation plan should include a clear operator command for rebuilding vectors after Qdrant is running.

The implementation should add an explicit vector-collection guard:

- store provider metadata such as `provider=dashscope`, `model=text-embedding-v4`, `dimensions=<resolved>`, and `distance=<resolved>` in local index status and, if practical, Qdrant collection metadata;
- fail fast before indexing into a non-empty collection with mismatched provider metadata;
- disable semantic retrieval and record a clear diagnostic if query-time metadata does not match the current config;
- prefer a new default collection name such as `imperial_chunks_qwen` for the first migration to avoid accidental reuse of OpenAI vectors.

## Error Handling

Missing or invalid `DASHSCOPE_API_KEY` should be explicit:

- `--enable-ocr` without a key skips OCR and reports that Qwen OCR is unavailable.
- `--index-vectors` without a key fails fast with a clear message.
- Query runtime without a key can still use keyword retrieval, but Qwen vector retrieval, reranking, and generation are unavailable.
- If answer generation cannot run, return the existing refusal text.
- If DashScope rerank fails, use deterministic fallback ranking and record diagnostics.
- If Qdrant vector collection metadata is missing or mismatched, skip vector search, keep keyword retrieval available, and report `vector_search_status="provider_mismatch"` or a similarly explicit diagnostic.
- If Qwen OCR fails for one embedded image or page, keep the current behavior of recording a warning and continuing extraction where possible.
- Provider errors must not expose secrets in logs, exceptions, diagnostics, or Phoenix traces.

## Testing

Normal tests should stay offline and deterministic.

Required tests:

- dependency test: `pyproject.toml` includes `langchain-qwq` and `dashscope`;
- dependency import test: after `uv sync --extra dev`, `dashscope` and `langchain_qwq` import successfully and expose the classes/functions the provider factories use;
- config tests: Qwen/DashScope model, region, base URL, OCR task, OCR image option, embedding dimension, and reranker env defaults and overrides are read correctly;
- provider tests: chat, embeddings, rerank, and OCR factories select Qwen/DashScope classes and respect missing keys;
- indexing tests: default Qdrant embeddings come from Qwen/DashScope, while injected test embeddings still work;
- indexing tests: vector indexing refuses mismatched existing collection metadata and records Qwen provider/model/dimension metadata on success;
- runtime tests: mismatched vector metadata disables semantic search without breaking keyword-only retrieval;
- runtime tests: semantic-search readiness uses `DASHSCOPE_API_KEY`, not OpenAI/Azure keys;
- OCR tests: Qwen vision payload preserves the verbatim extraction instruction or configured OCR task, base64 image payload, pixel/rotation options, and response parsing;
- retrieval tests: DashScope rerank path and deterministic fallback both work;
- script tests: `ingest.py --index-vectors` fails clearly without `DASHSCOPE_API_KEY`.

Optional live test:

- add an opt-in DashScope smoke test guarded by an environment variable such as `IMPERIAL_RAG_LIVE_DASHSCOPE=1`.

## Dependency Changes

Add:

- `langchain-qwq`;
- `dashscope`.

After `uv sync --extra dev`, verify with an import-level command that `dashscope`, `langchain_qwq.ChatQwen`, and the selected embedding/rerank classes import in this checkout. If `langchain_qwq.ChatQwen` is unavailable or incompatible, the implementation plan should switch chat generation to DashScope SDK `Generation` or OpenAI-compatible `ChatOpenAI` with DashScope base URL behind an explicitly named Qwen provider wrapper, rather than silently falling back to OpenAI.

Review after implementation:

- remove `langchain-openai` if no code path imports it by default;
- remove `langchain-cohere` if no legacy reranker path remains;
- keep `langchain-community` only if DashScope embeddings or reranking still require it.

Because this project uses `uv`, dependency changes should be made through `pyproject.toml` and then synchronized with `uv sync --extra dev`.

## Acceptance Criteria

- No default production code path imports `ChatOpenAI` or `OpenAIEmbeddings`.
- No default production code path requires a Cohere key.
- `DASHSCOPE_API_KEY` is the only required provider key for hosted Qwen behavior.
- Region/base URL settings are explicit and work for the selected Alibaba Cloud region.
- Ingestion can run OCR through Qwen-VL when enabled.
- OCR implementation uses the documented DashScope/OpenAI-compatible payload shape and parses provider responses deterministically.
- Vector indexing uses DashScope/Qwen embeddings.
- Vector indexing and query-time semantic retrieval refuse to mix OpenAI and Qwen embedding collections.
- Query runtime uses Qwen chat for answer generation.
- Reranking uses DashScope rerank when available and deterministic fallback otherwise.
- Keyword-only degraded mode remains clear and safe.
- Full pytest suite passes with no live DashScope dependency.
- The design remains compatible with Phoenix tracing and current strict citation validation.

## Implementation Boundaries

The implementation should be planned as one focused migration:

1. Add dependency/config/provider tests.
2. Add provider module and Qwen/DashScope factories.
3. Add region/base URL and OCR option configuration.
4. Replace embedding wiring in indexing and semantic-search gating.
5. Add vector provider metadata and mismatch guards.
6. Replace runtime answer model wiring.
7. Replace OCR/vision wiring with an explicitly documented DashScope payload and parser.
8. Add DashScope reranker integration and fallback diagnostics.
9. Update scripts and tests.
10. Run the full suite.
11. Rebuild Qdrant vectors with the Qwen embedding model in a clean or new collection.

The next step after user review is to invoke the writing-plans skill and produce a detailed implementation plan.
