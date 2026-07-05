from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from _bootstrap import ensure_src_on_path as _ensure_src_on_path

_ensure_src_on_path(__file__)

from imperial_rag.evals.audit import (  # noqa: E402
    build_eval_audit_report,
    write_jsonl as write_audit_jsonl,
)
from imperial_rag.evals.golden import (  # noqa: E402
    build_evidence_packets,
    load_evidence_corpus,
    write_jsonl,
)


DEFAULT_QUESTIONS_PATH = Path("evals/questions.jsonl")
DEFAULT_CHUNKS_PATH = Path(".imperial_rag/extracted/chunks.jsonl")
DEFAULT_DOCUMENTS_ROOT = Path("documents")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate document-grounded review packets for Imperial RAG eval golden answers."
    )
    parser.add_argument("--questions-path", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--documents-root", type=Path, default=DEFAULT_DOCUMENTS_ROOT)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--markdown-path", type=Path)
    parser.add_argument("--audit-path", type=Path)
    parser.add_argument("--findings-path", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when eval contract errors are present.")
    args = parser.parse_args(argv)

    report = build_eval_audit_report(
        questions_path=args.questions_path,
        chunks_path=args.chunks_path,
        documents_root=args.documents_root,
    )
    findings = report.findings
    packets = build_evidence_packets(
        report.rows,
        corpus=load_evidence_corpus(args.chunks_path),
        audit_rows=report.audit_rows,
    )

    write_jsonl(args.output_path, packets)
    if args.markdown_path:
        write_markdown_packets(args.markdown_path, packets)
    if args.audit_path:
        write_audit_jsonl(args.audit_path, report.audit_rows)
    if args.findings_path:
        write_audit_jsonl(args.findings_path, findings)

    status_counts = Counter(str(packet.get("gold_status") or "") for packet in packets)
    error_count = sum(1 for finding in findings if finding.get("severity") == "error")
    packet_blockers = _strict_packet_blockers(packets)
    print(f"evidence_packets={len(packets)}")
    print("gold_statuses=" + ",".join(f"{key}:{status_counts[key]}" for key in sorted(status_counts)))
    print(f"eval_contract_errors={error_count}")
    print(f"strict_packet_blockers={len(packet_blockers)}")
    print(f"evidence_packets_output={args.output_path}")
    if args.markdown_path:
        print(f"evidence_packets_markdown={args.markdown_path}")
    return 1 if args.strict and (error_count or packet_blockers) else 0


def write_markdown_packets(path: Path, packets: Iterable[Mapping[str, Any]]) -> None:
    lines: list[str] = ["# Imperial Eval Evidence Packets", ""]
    for packet in packets:
        lines.extend(_markdown_packet(packet))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _markdown_packet(packet: Mapping[str, Any]) -> list[str]:
    evidence = list(packet.get("evidence") or [])
    notes = [str(note) for note in packet.get("review_notes") or []]
    lines = [
        f"## {packet.get('id')}",
        "",
        f"- status: `{packet.get('gold_status')}`",
        f"- behavior: `{packet.get('expected_behavior')}`",
        f"- lane: `{packet.get('lane')}`",
        f"- question: {packet.get('question')}",
        f"- current answer: {packet.get('current_reference_answer')}",
        f"- reference_context_ids: `{', '.join(packet.get('current_reference_context_ids') or [])}`",
        "",
        "### Review Notes",
    ]
    if notes:
        lines.extend(f"- {note}" for note in notes)
    else:
        lines.append("- none")
    lines.extend(["", "### Evidence"])
    if evidence:
        for chunk in evidence:
            lines.extend(
                [
                    f"- `{chunk.get('chunk_id')}`",
                    f"  - citation: `{chunk.get('citation_id')}`",
                    f"  - source: `{chunk.get('relative_path')}`",
                    f"  - text: {_single_line(chunk.get('text'))}",
                ]
            )
    else:
        lines.append("- none")
    lines.append("")
    return lines


def _single_line(value: Any, *, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _strict_packet_blockers(packets: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    for packet in packets:
        status = str(packet.get("gold_status") or "")
        if status in {"ready_for_review", "refusal_boundary"}:
            continue
        if status == "quarantined" and _is_explicit_dataset_quarantine(packet):
            continue
        blockers.append(
            {
                "id": str(packet.get("id") or ""),
                "gold_status": status,
            }
        )
    return blockers


def _is_explicit_dataset_quarantine(packet: Mapping[str, Any]) -> bool:
    audit = packet.get("audit")
    if not isinstance(audit, Mapping):
        return False
    notes = [str(note) for note in audit.get("notes") or []]
    return "row is quarantined by explicit dataset metadata" in notes


if __name__ == "__main__":
    raise SystemExit(main())
