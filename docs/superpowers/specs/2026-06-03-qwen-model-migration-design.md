# Qwen Model Migration Design

Date: 2026-06-03
Workspace: `/Users/danil/Public/imperial`
Status: Approved in chat. Written spec awaiting user review before implementation planning.

## Context

Imperial RAG currently has four paid/model-backed surfaces:

- answer generation uses OpenAI chat models in runtime and workflow fallbacks;
- OCR uses OpenAI vision chat for image/PDF/DOCX embedded-image text extraction;
- vector indexing and semantic search use OpenAI embeddings through Qdrant;
- reranker settings name Cohere models, while the live runtime still mostly uses deterministic candidate ranking.

The user wants all model-backed behavior moved to hosted Qwen through Alibaba DashScope, including LLM, reranker, embeddings, and OCR/vision.

This checkout is not a git repository, so the design document cannot be committed in the current workspace. The implementation workflow should still use direct file scans and test commands for verification.

## Documentation Check

Current documentation was checked with Context7 before choosing the design:

- Official Qwen docs show Qwen model usage in LangChain/LlamaIndex contexts and confirm Qwen covers text generation and multimodal/vision use cases.
- DashScope Python SDK docs show `TextEmbedding`, `TextReRank`, and `MultiModalConversation` APIs.
- LangChain docs show a Qwen chat integration with `from langchain_qwq import ChatQwen` and `DASHSCOPE_API_KEY`.
- Current Qwen Cloud docs recommend `qwen3.7-max` for highest-capability text generation, `text-embedding-v4` as the latest text embedding model, and `qwen3-rerank` for reranking.
- Current Qwen Cloud OCR docs provide the dedicated `qwen-vl-ocr` model family, including `qwen-vl-ocr-2025-11-20` for OCR and structured document extraction.
- The installed environment already contains `langchain_community.embeddings.dashscope.DashScopeEmbeddings`, `langchain_community.document_compressors.dashscope_rerank.DashScopeRerank`, and `langchain_community.chat_models.tongyi.ChatTongyi`.
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
| `IMPERIAL_RAG_QWEN_CHAT_MODEL` | Hosted Qwen chat model for answer generation. |
| `IMPERIAL_RAG_QWEN_VISION_MODEL` | Qwen vision/multimodal model for OCR. |
| `IMPERIAL_RAG_QWEN_EMBEDDING_MODEL` | DashScope text embedding model for Qdrant. |
| `IMPERIAL_RAG_QWEN_EMBEDDING_DIMENSIONS` | Optional embedding dimension when the selected integration supports it. |
| `IMPERIAL_RAG_QWEN_RERANK_MODEL` | DashScope text reranker model. |
| `IMPERIAL_RAG_ALLOW_LEGACY_OPENAI` | Optional escape hatch for old OpenAI behavior during debugging only. |
| `IMPERIAL_RAG_ALLOW_LEGACY_COHERE` | Optional escape hatch for old Cohere reranking during debugging only. |

Initial best-quality provider defaults:

- chat model: `qwen3.7-max`;
- embedding model: `text-embedding-v4`;
- embedding dimensions: unset by default so the LangChain integration can use the provider default; use `2048` only if the chosen integration exposes dimension control cleanly;
- rerank model: `qwen3-rerank`;
- OCR/vision model: `qwen-vl-ocr-2025-11-20`, with `qwen-vl-ocr` as the moving alias fallback.

## Data Flow

Ingestion with `--enable-ocr`:

1. Check `DASHSCOPE_API_KEY`.
2. Create a Qwen vision OCR client.
3. Preserve the current instruction: extract all visible Russian and English text verbatim, without summarizing.
4. Store OCR output in the existing OCR cache and extraction artifacts.

Ingestion with `--index-vectors`:

1. Check `DASHSCOPE_API_KEY`.
2. Create DashScope/Qwen embeddings.
3. Add chunks to the configured Qdrant collection with stable chunk ids.
4. Record the Qwen embedding model in manifest/index status where the current schema supports it.

Query runtime:

1. Keyword retrieval continues to work locally.
2. Semantic retrieval is enabled only when DashScope embeddings are configured.
3. Candidate merge/dedupe stays provider-independent.
4. Reranking uses DashScope rerank when configured.
5. Reranking falls back to deterministic ranking if DashScope rerank is missing or fails.
6. Answer generation uses `ChatQwen` with the existing strict citation prompt.
7. Citation validation remains the final guardrail and refuses unsupported answers.

## Vector Reindexing

Changing from OpenAI embeddings to DashScope embeddings changes vector semantics and may change dimensions.

The migration must require a clean vector reindex:

- clear/recreate the existing Qdrant collection, or
- use a new collection name such as `imperial_chunks_qwen`.

Do not mix OpenAI and Qwen embeddings in one collection. The implementation plan should include a clear operator command for rebuilding vectors after Qdrant is running.

## Error Handling

Missing or invalid `DASHSCOPE_API_KEY` should be explicit:

- `--enable-ocr` without a key skips OCR and reports that Qwen OCR is unavailable.
- `--index-vectors` without a key fails fast with a clear message.
- Query runtime without a key can still use keyword retrieval, but Qwen vector retrieval, reranking, and generation are unavailable.
- If answer generation cannot run, return the existing refusal text.
- If DashScope rerank fails, use deterministic fallback ranking and record diagnostics.
- If Qwen OCR fails for one embedded image or page, keep the current behavior of recording a warning and continuing extraction where possible.
- Provider errors must not expose secrets in logs, exceptions, diagnostics, or Phoenix traces.

## Testing

Normal tests should stay offline and deterministic.

Required tests:

- dependency test: `pyproject.toml` includes `langchain-qwq` and `dashscope`;
- config tests: Qwen/DashScope model env defaults and overrides are read correctly;
- provider tests: chat, embeddings, rerank, and OCR factories select Qwen/DashScope classes and respect missing keys;
- indexing tests: default Qdrant embeddings come from Qwen/DashScope, while injected test embeddings still work;
- runtime tests: semantic-search readiness uses `DASHSCOPE_API_KEY`, not OpenAI/Azure keys;
- OCR tests: Qwen vision payload preserves the verbatim extraction instruction;
- retrieval tests: DashScope rerank path and deterministic fallback both work;
- script tests: `ingest.py --index-vectors` fails clearly without `DASHSCOPE_API_KEY`.

Optional live test:

- add an opt-in DashScope smoke test guarded by an environment variable such as `IMPERIAL_RAG_LIVE_DASHSCOPE=1`.

## Dependency Changes

Add:

- `langchain-qwq`;
- `dashscope`.

Review after implementation:

- remove `langchain-openai` if no code path imports it by default;
- remove `langchain-cohere` if no legacy reranker path remains;
- keep `langchain-community` only if DashScope embeddings or reranking still require it.

Because this project uses `uv`, dependency changes should be made through `pyproject.toml` and then synchronized with `uv sync --extra dev`.

## Acceptance Criteria

- No default production code path imports `ChatOpenAI` or `OpenAIEmbeddings`.
- No default production code path requires a Cohere key.
- `DASHSCOPE_API_KEY` is the only required provider key for hosted Qwen behavior.
- Ingestion can run OCR through Qwen-VL when enabled.
- Vector indexing uses DashScope/Qwen embeddings.
- Query runtime uses Qwen chat for answer generation.
- Reranking uses DashScope rerank when available and deterministic fallback otherwise.
- Keyword-only degraded mode remains clear and safe.
- Full pytest suite passes with no live DashScope dependency.
- The design remains compatible with Phoenix tracing and current strict citation validation.

## Implementation Boundaries

The implementation should be planned as one focused migration:

1. Add dependency/config/provider tests.
2. Add provider module and Qwen/DashScope factories.
3. Replace embedding wiring in indexing and semantic-search gating.
4. Replace runtime answer model wiring.
5. Replace OCR/vision wiring.
6. Add DashScope reranker integration and fallback diagnostics.
7. Update scripts and tests.
8. Run the full suite.
9. Rebuild Qdrant vectors with the Qwen embedding model.

The next step after user review is to invoke the writing-plans skill and produce a detailed implementation plan.
