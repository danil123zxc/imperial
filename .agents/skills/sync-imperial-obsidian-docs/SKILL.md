---
name: sync-imperial-obsidian-docs
description: Keep the Imperial RAG Obsidian project notes aligned with the live repository. Use after any Imperial code, configuration, schema, operational-script, pipeline, strategy, or architecture change; when a code-changing task reaches its documentation gate; or when auditing and refreshing the Imperial RAG notes in the Second brain vault.
---

# Sync Imperial Obsidian Docs

Treat `Second brain/1. Projects/Imperial RAG/` as the detailed human documentation layer and `README.md` as the concise repo/operator guide. Use the installed `obsidian-cli` skill and run all vault operations through the bundled adapter.

## Workflow

1. Capture the task baseline with `git status --short` before editing code.
2. When implementation is stable, inspect the session diff and changed paths. Read [references/note-map.md](references/note-map.md) and select candidate notes by meaning, not filenames alone.
3. Run the vault gate:

   ```bash
   uv run python .agents/skills/sync-imperial-obsidian-docs/scripts/obsidian_docs.py check
   ```

4. For each affected note, read the current content and copy the reported SHA-256:

   ```bash
   uv run python .agents/skills/sync-imperial-obsidian-docs/scripts/obsidian_docs.py read --note retrieval
   ```

5. Make the smallest coherent edit. Preserve frontmatter, curated explanations, ADR history, links, and unaffected sections. Write the complete revised Markdown on stdin and pass the hash from step 4:

   ```bash
   uv run python .agents/skills/sync-imperial-obsidian-docs/scripts/obsidian_docs.py write \
     --note retrieval --expect-sha256 <sha256> <<'MARKDOWN'
   <complete revised note>
   MARKDOWN
   ```

6. Read every changed note back through the adapter. Check headings, Mermaid, wikilinks, commands, and factual claims against the final repo diff.
7. Run final code checks. If a check causes another code change, repeat the documentation-impact assessment.
8. In the handoff, list updated note titles or state `Obsidian docs: no durable documentation impact` with a short reason.

## Update Standard

- Update durable facts that help a new maintainer understand purpose, ownership, data flow, schemas, retrieval strategy, evaluation, privacy, operations, failure modes, or deliberate tradeoffs.
- Treat a test-only or behavior-preserving change as a no-op unless it changes a documented invariant, module map, command, or troubleshooting path.
- Distinguish four kinds of truth explicitly: implemented code capability, generated-state snapshot, currently running services, and planned direction.
- Add or refresh `updated: YYYY-MM-DD` only on notes actually reviewed. Date snapshot values and keep them only when verified from live artifacts or service queries.
- Append an ADR only for a deliberate architectural decision with a meaningful tradeoff. Never rewrite old ADR history to make it look current.
- Keep the index concise and keep its newcomer reading path and note links accurate.

## Privacy And Safety

- Never copy secrets, credentials, private document text, extracted chunks, raw questions or answers, prompts, Phoenix payloads, eval outputs, auth rows, chat transcripts, filenames, or private paths beyond the already documented workspace/service paths.
- Use aggregates, schemas, field names, component names, and sanitized commands. Treat `documents/`, `.imperial_rag/`, Phoenix, and eval artifacts as private evidence sources.
- Never write an arbitrary vault path. The adapter allows only the registered Imperial notes and uses an optimistic SHA-256 guard to avoid overwriting concurrent edits.
- If Obsidian is closed or the vault gate fails, do not invent a successful sync. The code commit may proceed, but the final response must prominently say `Obsidian docs: pending`, include the CLI error, and name the candidate notes.

## No-Op Examples

- Renaming a private local variable with no ownership or behavior change.
- Adding coverage for already documented behavior.
- Reformatting or deduplicating an internal helper without changing the newcomer-facing module map.

## Update Examples

- Changing hybrid fusion or evidence selection: update Retrieval and Answering, Pipeline Schema, and Current State.
- Adding a SQLite table or service-owned data surface: update Database Schema and Current State; update Architecture if ownership changes.
- Adding a shadow ingestion or promotion gate: update Data and Ingestion, Pipeline Schema, Runbook, Current State, and an ADR when the tradeoff is deliberate.
