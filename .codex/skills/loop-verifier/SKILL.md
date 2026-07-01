---
name: loop-verifier
description: Independently verify loop-produced Imperial RAG changes before any commit or PR.
user_invocable: true
---

# Loop Verifier Skill - Imperial RAG

You are the checker in a maker/checker split. Reject unless evidence is strong.

## Inputs

- The implementer's summary and diff.
- Original issue, CI failure, or human request.
- `loop-constraints.md`, `docs/safety.md`, and the allowed file scope.
- Test or verifier commands relevant to the change.

## Checklist

All must pass for `APPROVE`:

1. Scope is limited to the stated problem.
2. No denylist paths changed.
3. No secrets, document text, generated corpus content, chat history, Phoenix payloads, or eval outputs were copied into committed files.
4. The change addresses the stated target.
5. Tests or equivalent checks were run and results are reported.
6. No tests were disabled or weakened.

## Default Commands

- Source changes: `./scripts/check.sh`
- Eval dataset changes: `uv run python scripts/audit_eval_rows.py --strict --output-path <tmp>`
- Ingestion promotion changes: `uv run python scripts/check_ingestion_promotion.py --baseline-root <path> --shadow-root <path>`
- Trace-shape changes: `uv run python scripts/validate_phoenix_trace.py --run-id <id> --json`

## Verdict

Use exactly one:

```markdown
## Verdict: APPROVE | REJECT | ESCALATE_HUMAN

### Evidence
- Tests:
- Scope:
- Privacy:

### Notes
- ...
```

If you cannot run the relevant check, use `ESCALATE_HUMAN`.
