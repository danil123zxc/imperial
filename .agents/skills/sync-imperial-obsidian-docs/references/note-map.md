# Imperial RAG Note Map

Vault: `Second brain`

Folder: `1. Projects/Imperial RAG/`

Use the IDs below with `obsidian_docs.py`. Path matches are candidate signals; read the diff and update only notes whose durable content changed.

| ID | Note | Owns | Typical change signals |
| --- | --- | --- | --- |
| `index` | `Imperial RAG.md` | Newcomer reading path, project map, concise snapshot, source-of-truth rules | Note links/titles, headline architecture, baseline snapshot |
| `brief` | `01 - Project Brief` | Problem, goal, users, success criteria, non-goals, provider boundary | Fundamental product or safety boundary |
| `architecture` | `02 - Architecture` | Package/service topology, component ownership, cross-system flow, trust boundary | `src/imperial_rag/**` ownership moves, `compose.yaml`, major dependencies |
| `ingestion` | `03 - Data and Ingestion` | Manifest, extraction, OCR, chunking, dedupe, authority, indexing, shadow/promotion | `ingestion/**`, `indexing/**`, `scripts/ingest.py`, `scripts/promote_ingestion.py` |
| `retrieval` | `04 - Retrieval and Answering` | Query normalization, search legs, merge, RRF, rerank, evidence, answer/citation behavior | `retrieval/**`, `answering/**`, query settings |
| `observability` | `05 - Observability and Evaluation` | Logs, event streams, Phoenix traces, privacy, eval datasets/metrics/gates | `observability/**`, `evals/**`, eval scripts, trace settings |
| `decisions` | `06 - Architecture Decisions` | Durable decisions and tradeoffs | A deliberate technology, security, data, orchestration, or rollout decision |
| `current-state` | `07 - Current State` | Implemented capabilities, dated generated/runtime snapshot, known gaps | Any shipped feature; live reingestion/runtime verification |
| `roadmap` | `08 - Roadmap and Planned Directions` | Approved future work, completed-direction reconciliation | Plans approved, completed, rejected, or reprioritized |
| `runbook` | `09 - Runbook` | Install, checks, services, ingest/query/eval/promotion commands, diagnosis | `scripts/**`, `compose.yaml`, `.env.example`, operator-facing behavior |
| `pipeline-schema` | `10 - Pipeline Schema` | Mermaid ingestion/query flows, data shapes, metadata, gates | Pipeline stages, artifacts, retrieval diagnostics, promotion flow |
| `database-schema` | `11 - Database Schema` | SQLite tables, Elasticsearch/Qdrant/Phoenix surfaces, relationships | Storage schemas, index mappings, aliases, app-owned databases |

## Cross-Cutting Rules

- New or moved implementation modules usually affect `architecture` even when behavior is unchanged.
- A code capability belongs in `current-state`; a generated count belongs there only with a verification date.
- Pipeline behavior should agree across `ingestion` or `retrieval`, `pipeline-schema`, and `runbook`.
- Storage ownership should agree across `architecture`, `database-schema`, and `current-state`.
- Completed roadmap work must move to current state or be marked implemented; do not leave it phrased as future work.
- Update `index` when the recommended reading order, headline snapshot, or note graph changes.

## Newcomer Coverage Checklist

The note set should let a new maintainer answer:

1. What problem does Imperial RAG solve, and what must remain private?
2. Where do ingestion, retrieval, answering, UI, evaluation, and service integrations live?
3. How does a source file become a cited answer?
4. Which databases, indexes, files, and service volumes own which state?
5. Why are Elasticsearch, Qdrant, LangGraph, Phoenix, and Qwen/DashScope used?
6. How are candidate ingestion runs validated and promoted or rolled back?
7. How are retrieval quality, citations, privacy, and regressions evaluated?
8. How does an operator install, run, inspect, troubleshoot, and verify the system?
9. Which statements describe current code, current generated state, current runtime, or future plans?
