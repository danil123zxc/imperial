from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from _bootstrap import ensure_src_on_path as _ensure_src_on_path

_ensure_src_on_path(__file__)

from imperial_rag.evals.audit import (  # noqa: E402
    build_eval_audit_report,
    write_jsonl,
    write_markdown_table,
)


DEFAULT_QUESTIONS_PATH = Path("evals/questions.jsonl")
DEFAULT_CHUNKS_PATH = Path(".imperial_rag/extracted/chunks.jsonl")
DEFAULT_DOCUMENTS_ROOT = Path("documents")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Imperial RAG eval rows against the extracted corpus index.")
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--documents-root", type=Path, default=DEFAULT_DOCUMENTS_ROOT)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--markdown-path", type=Path)
    parser.add_argument("--findings-path", type=Path)
    parser.add_argument(
        "--phoenix-metrics",
        default="",
        help="Comma-separated Phoenix evaluator metric names to validate against the supported Phoenix path.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when validation errors are present.")
    args = parser.parse_args(argv)

    report = build_eval_audit_report(
        questions_path=args.questions_path,
        chunks_path=args.chunks_path,
        documents_root=args.documents_root,
        phoenix_metric_names=_metric_names(args.phoenix_metrics),
    )
    audit_rows = report.audit_rows
    findings = report.findings

    write_jsonl(args.output_path, audit_rows)
    if args.markdown_path:
        write_markdown_table(args.markdown_path, audit_rows)
    if args.findings_path:
        write_jsonl(args.findings_path, findings)

    action_counts = Counter(row["action"] for row in audit_rows)
    error_count = sum(1 for finding in findings if finding.get("severity") == "error")
    warning_count = sum(1 for finding in findings if finding.get("severity") == "warning")
    print(f"audit_rows={len(audit_rows)}")
    print("audit_actions=" + ",".join(f"{key}:{action_counts[key]}" for key in sorted(action_counts)))
    print(f"audit_findings_errors={error_count}")
    print(f"audit_findings_warnings={warning_count}")
    print(f"audit_output={args.output_path}")
    if args.markdown_path:
        print(f"audit_markdown={args.markdown_path}")
    if args.findings_path:
        print(f"audit_findings={args.findings_path}")
    return 1 if args.strict and error_count else 0


def _metric_names(raw_metrics: str) -> list[str]:
    if raw_metrics.strip().casefold() in {"", "none"}:
        return []
    return [metric.strip() for metric in raw_metrics.split(",") if metric.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
