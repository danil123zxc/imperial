# Local RAG System Design

Date: 2026-06-02
Workspace: `/Users/danil/Public/imperial`
Corpus root: `/Users/danil/Public/imperial/documents`

## Status

Supersession note: `2026-06-03-phoenix-observability-design.md` supersedes the LangSmith tracing and evaluation decisions in this document. The RAG architecture remains otherwise unchanged.

Design revised in chat. Implementation planning has not started. Awaiting user review of this written spec.

## Context

The workspace is a document corpus, not an application repository. The corpus contains Russian-language company materials: regulations, orders, job descriptions, forms, warehouse/logistics/sales/HR/accounting instructions, scanned orders, diagrams, and spreadsheets.

Initial corpus scan found:

- 162 files total.
- About 130 likely content files.
- Main formats: DOCX, PDF, XLSX, JPG, DOC, RTF, RAR, temp/system/lock files.
- All checked PDFs had zero native extractable text, so PDF OCR is required.
- DOCX files contain substantial text and tables, and may also contain embedded images that must be OCR'd.
- The corpus includes duplicates and version variants. These must be marked, not removed.

## Goals

Build a local/private RAG system for this corpus, phased as:

1. Local web chat for strict cited policy Q&A.
2. Document catalog/search using the same manifest and index.
3. Compliance workflows after retrieval quality is proven.

The first usable version is a local web chat where users ask questions and receive answers grounded only in the indexed documents, with citations.

## Decisions

- Runtime: local/private machine or server.
- Source files stay local.
- AI APIs are allowed for OCR, embeddings, reranking, and answer generation.
- Interface: local web chat.
- Answer mode: strict citation mode.
- Vector database: Qdrant running locally.
- Keyword search: local BM25 or full-text index for exact Russian terms.
- LangChain ecosystem: use LangChain, LangGraph, and LangSmith integrations wherever practical.
- Tracing and evaluation: use LangSmith.
- Archives such as RAR are scanned into the manifest but not extracted in v1.
- Every file under `documents/` gets a manifest record, including `~$...`, `.~lock...`, `Thumbs.db`, `.tmp`, archives, duplicates, and unknown file types.

## Non-Goals For V1

- No cloud-hosted employee app.
- No final compliance decision engine.
- No graph database.
- No archive extraction.
- No silent file skipping.
- No answer based on general model knowledge when documents do not support it.
- No bespoke RAG framework when a maintained LangChain, LangGraph, or LangSmith integration covers the need.

## LangChain Ecosystem Policy

Prefer maintained LangChain ecosystem integrations over custom functions, classes, or adapters whenever the fit is reasonable.

Use LangChain for:

- document loader abstractions where they support the needed format;
- text splitters and document representations;
- Qdrant vector store integration;
- embedding model integration;
- retriever composition;
- model calls;
- output parsing and structured answer validation where useful.

Use LangGraph for:

- explicit ingestion and indexing workflow orchestration;
- query-time RAG workflow orchestration;
- retryable steps with clear state transitions;
- future compliance workflows after the RAG base is stable.

Use LangSmith for:

- tracing ingestion, retrieval, reranking, and answer generation;
- debugging failed or low-quality answers;
- RAG evaluation datasets;
- citation-grounding and refusal-behavior evaluation;
- regression tracking over the Russian gold question set.

Custom code is allowed only for corpus-specific behavior that is not well covered by maintained integrations, such as the exact file manifest schema, source-path citation formatting, DOCX embedded-image extraction coordination, archive manifest policy, and local UI behavior.

LangGraph is an orchestration framework in this design, not a graph database. The v1 database choice remains Qdrant plus local manifest/keyword storage.

## Architecture

The system has five layers:

1. File discovery and manifest.
2. Extraction and OCR.
3. Chunking and indexing.
4. Retrieval and answer generation.
5. Local web chat and ingestion status UI.

The system creates a processed workspace separate from the source documents. The processed workspace stores the manifest, extracted text, OCR text, chunks, embeddings, indexing status, and logs.

LangGraph should orchestrate the ingestion/indexing workflow and the query-time RAG workflow. LangChain integrations should provide the document, embedding, vector store, retriever, LLM, and output parsing pieces where available.

Qdrant stores semantic vectors and citation payload metadata, ideally through the LangChain Qdrant integration rather than a custom Qdrant adapter. A local keyword index handles exact Russian terms, filenames, headings, roles, and process names. Retrieval combines both signals before answer generation.

## File Manifest

The manifest is the accountability layer. It records every discovered file, whether or not the file becomes searchable.

Each manifest record should include:

- stable file id;
- absolute path;
- relative path under `documents/`;
- filename;
- extension;
- size;
- content hash;
- modified timestamp;
- parent folder path;
- inferred department/folder category;
- duplicate group id when hashes match;
- extraction status;
- extraction method attempted;
- extraction warnings/errors;
- chunk count;
- embedding/indexing status;
- last indexed timestamp.

Recommended statuses:

- `indexed`: text extracted and indexed.
- `manifest_only`: file recorded but not extracted, such as archive files in v1.
- `no_text`: readable file but no text found.
- `unsupported`: no safe extractor available.
- `failed`: extraction attempted and failed.
- `pending`: discovered but not processed yet.

## Extraction Rules

Scan every file under `documents/`. Do not skip temp/system/lock files. Do not remove duplicates.

Extraction by type:

- DOCX: extract paragraphs, headings, tables, and embedded images.
- DOCX embedded images: OCR each image and attach OCR text to the parent DOCX with image references.
- PDF: OCR page by page, because scanned PDFs are expected.
- JPG/PNG: OCR as standalone image documents.
- XLSX: extract each sheet into structured row/cell text.
- DOC/RTF: extract with available local conversion/parsing tools if safe.
- RAR/archive files: record path, size, hash, and status only; do not unpack or extract in v1.
- Unknown/system files: record in manifest; extract only if a safe parser exists.

OCR output must be persisted. It should not be used only transiently. Stored OCR text allows inspection, correction, reindexing, and cost control.

## Chunking

Chunks should preserve citation context rather than using only blind fixed-size splitting.

Chunk sources:

- document sections/headings;
- paragraphs grouped into coherent blocks;
- DOCX table rows or table sections;
- PDF page OCR text;
- standalone image OCR text;
- DOCX embedded image OCR text;
- XLSX sheet sections or row ranges.

Each chunk should include:

- chunk id;
- parent file id;
- source type: body, table, sheet, page, image, embedded_image;
- chunk text;
- section heading when available;
- page number when available;
- sheet name and row range when available;
- table index when available;
- image index when available;
- source path;
- duplicate group id;
- extraction method.

Chunk size should be large enough to preserve policy context but small enough for precise citation. A practical starting point is roughly 700-1200 tokens with small overlap for narrative text; table, page, sheet, and image chunks should use their natural boundaries first.

## Qdrant Vector Store

Use Qdrant locally as the vector database. Context7 documentation check on 2026-06-02 confirmed Qdrant supports local Docker usage, Python client connection to `http://localhost:6333`, and payload filtering.

Design requirements:

- Store one vector point per searchable chunk.
- Use chunk id as the point id.
- Store citation metadata as Qdrant payload.
- Keep enough payload fields for filtering by file path, department, source type, file hash, duplicate group, and extraction status.
- Do not expose Qdrant outside the local/private environment.

Qdrant is responsible for semantic retrieval, not for replacing the manifest or keyword index.

## Keyword Search

Use a local keyword index alongside Qdrant.

Reason: Russian company policy questions often depend on exact words, filenames, role names, dates, forms, and process terms. Exact matches like `возврат брака`, `табель`, `водитель-экспедитор`, `приказ`, and document titles should strongly influence retrieval.

Keyword search should index:

- chunk text;
- file names;
- folder names;
- headings;
- sheet names;
- extracted table labels;
- OCR text.

## Retrieval And Answering

Question flow:

1. Normalize the question.
2. Retrieve candidates from Qdrant vector search.
3. Retrieve candidates from keyword search.
4. Merge and rerank candidates.
5. Build an evidence-only prompt from top chunks.
6. Generate answer in strict citation mode.
7. Return answer with citations and source list.

The query-time flow should be represented as a LangGraph workflow with explicit state for the original question, normalized query, vector candidates, keyword candidates, merged/reranked evidence, answer draft, citation checks, and final response.

Strict citation behavior:

- Every meaningful claim must cite retrieved source chunks.
- If evidence is weak, answer: "I could not find this clearly in the indexed documents."
- If documents conflict, show both cited sources and state that the documents disagree.
- If relevant-looking files exist but were not extractable, mention that some files are present but not searchable.
- Do not fill gaps with general model knowledge.

Retrieval should favor exact matches in filenames, headings, and source titles when the question uses specific company terms.

## Local Web Chat

The v1 interface should be simple:

- question input;
- answer panel;
- citation list;
- source file paths;
- per-citation reference such as page, table, sheet, or image;
- ingestion status summary.

The chat should make uncertainty visible. It should be easy to see when an answer is based on strong citations, conflicting citations, or no sufficient evidence.

## Error Handling And Auditability

Ingestion must never fail silently.

Track:

- total files scanned;
- files indexed;
- files OCR'd;
- files recorded as manifest-only;
- files with extraction failures;
- files with no readable text;
- unsupported files;
- duplicate groups;
- chunk counts;
- embedding/indexing failures;
- exact error messages.

The UI should include an ingestion status view or panel so the user can inspect what the system did with the corpus.

## Testing Strategy

Test ingestion first, then retrieval and answer grounding.

Core tests:

- manifest includes every file under `documents/`;
- archives are recorded but not extracted;
- temp/system/lock files are recorded;
- duplicates are marked but not removed;
- DOCX extraction includes body text, tables, and embedded-image OCR;
- PDFs and JPGs go through OCR;
- XLSX sheets become structured text;
- chunk metadata includes path and page/table/sheet/image references;
- Qdrant points receive vectors with correct payload metadata;
- keyword search finds exact Russian terms;
- answer service refuses unsupported questions;
- answer service cites every meaningful claim;
- conflicting versions are surfaced instead of merged silently;
- LangSmith captures traces for representative ingestion and query runs;
- LangSmith evaluations catch unsupported-answer hallucinations and missing citations.

Evaluation set examples:

- "Как оформить возврат брака из магазина?"
- "Какие обязанности у водителя-экспедитора?"
- "Что делать при отсутствии сотрудника на рабочем месте?"
- "Какие правила по табелям и мотивационным листам?"
- "Кто отвечает за приемку товара на складе?"

## Implementation Defaults

Use these defaults for the next implementation plan unless the user changes them during review:

- Python-oriented document processing and RAG service, because the corpus requires DOCX, XLSX, PDF, image, and OCR handling.
- LangChain integrations first for loaders, text splitters, embeddings, Qdrant vector store access, retrievers, model calls, output parsing, and reusable RAG utilities.
- LangGraph for ingestion/indexing orchestration and query-time RAG orchestration.
- LangSmith for tracing, debugging, evaluation datasets, and regression checks.
- Qdrant local Docker for semantic vector search, reachable only inside the local/private environment.
- Local keyword index owned by the application for exact Russian terms, filenames, headings, and OCR text.
- Environment-configured AI API calls for OCR, embeddings, reranking, and answer generation. Credentials must not be stored in source files.
- Persisted extraction artifacts, OCR text, chunks, manifests, and logs so reindexing does not require repeating every expensive extraction step.
- Local web chat served from the same local/private system as the backend.
