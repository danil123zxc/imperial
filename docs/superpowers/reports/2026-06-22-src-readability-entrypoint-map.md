# Imperial `src/` Readability Entrypoint Map

Date: 2026-06-22
Scope: import and entrypoint audit for the lifecycle-oriented `src/imperial_rag` restructure.

## Entrypoints That Must Keep Working

- Streamlit UI: `uv run python -m streamlit run src/imperial_rag/web_app/__main__.py --server.address 127.0.0.1 --server.port 8501`
- Query CLI: `uv run python scripts/query.py "question text"`
- Ingest CLI: `uv run python scripts/ingest.py --workspace-root /Users/danil/Public/imperial`
- Phoenix eval CLI: `uv run python scripts/run_phoenix_eval.py`
- Ragas eval CLI: `uv run python scripts/run_ragas_eval.py`
- Full regression suite: `uv run python -m pytest -q`

## Compatibility Imports

These existing import paths are treated as compatibility surfaces for this migration:

- `imperial_rag.answering`
- `imperial_rag.auth`
- `imperial_rag.chunking`
- `imperial_rag.elasticsearch_keyword`
- `imperial_rag.extraction`
- `imperial_rag.indexing`
- `imperial_rag.keyword`
- `imperial_rag.manifest`
- `imperial_rag.observability`
- `imperial_rag.ocr`
- `imperial_rag.pipeline`
- `imperial_rag.providers`
- `imperial_rag.ragas_eval`
- `imperial_rag.retrieval`
- `imperial_rag.runtime`
- `imperial_rag.tracing`
- `imperial_rag.web_app`
- `imperial_rag.workflows`

## Monkeypatch-Sensitive Surfaces

The current tests and local probes patch old import paths, especially:

- `imperial_rag.indexing.QdrantClient`, `QdrantVectorStore`, and `create_embeddings`
- `imperial_rag.runtime.build_query_workflow`, `RetrievalService`, `ElasticsearchKeywordIndex`, `make_qdrant_store`, and `create_chat_model`
- `imperial_rag.retrieval.trace_retrieval_step`
- `imperial_rag.tracing._collector_endpoint_reachable` and `trace.get_tracer`
- `imperial_rag.workflows.ChatOpenAI`
- `imperial_rag.web_app` constants and helper functions
- `imperial_rag.providers` provider factory helpers

Compatibility wrappers should keep public imports stable. Tests that patch private implementation globals may move to the new implementation module paths when needed.

## Superseded Plan Note

`docs/superpowers/specs/2026-06-03-readability-module-structure-design.md` and
`docs/superpowers/plans/2026-06-03-readability-module-structure.md` are superseded for current implementation work.
They predate the current Elasticsearch keyword module, Qdrant provider metadata flow, and Phoenix trace-shape changes.
