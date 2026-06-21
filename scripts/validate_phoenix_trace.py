from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REQUIRED_SPANS: dict[str, str] = {
    "imperial_rag.query": "CHAIN",
    "retrieval": "CHAIN",
    "retrieval.vector_search": "RETRIEVER",
    "retrieval.keyword_search": "RETRIEVER",
    "retrieval.rerank": "RERANKER",
    "retrieval.final_evidence": "RETRIEVER",
    "answer.generate": "CHAIN",
    "answer.call_model": "LLM",
    "answer.citation_check": "CHAIN",
}
FORBIDDEN_SPAN_NAMES = {
    "LangGraph",
    "normalize_query",
    "retrieve",
    "call_model",
    "search",
    "ChatQwen",
    "ChatCompletion",
    "ElasticsearchRetriever",
    "retrieve.merge_candidates",
    "retrieve.fuse_candidates",
    "answer.validate_citations",
}
ROOT_PROVENANCE_ATTRIBUTES = (
    "imperial.trace_run_id",
    "imperial.git_sha",
    "imperial.trace_mode",
    "imperial.trace_suppress_internals",
    "imperial.trace_auto_instrument",
    "imperial.phoenix_project",
)
RETRIEVAL_DOCUMENT_SPANS = (
    "retrieval.vector_search",
    "retrieval.keyword_search",
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _ensure_src_on_path()

    from imperial_rag.config import Settings
    from imperial_rag.env import load_project_env

    load_project_env()
    settings = Settings()
    project_name = args.project_name or settings.phoenix_project_name
    base_url = args.base_url or settings.phoenix_client_endpoint
    records = fetch_latest_span_records(project_name=project_name, base_url=base_url, run_id=args.run_id)
    errors = validate_span_records(
        records,
        expected_run_id=args.run_id,
        require_retrieval_documents=args.require_retrieval_documents,
    )
    if args.json:
        print(json.dumps({"ok": not errors, "errors": errors, "span_count": len(records)}, ensure_ascii=False))
    elif errors:
        print("phoenix_trace_validation=failed", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    else:
        print(f"phoenix_trace_validation=passed project={project_name} spans={len(records)}")
    return 1 if errors else 0


def validate_span_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_run_id: str | None = None,
    require_retrieval_documents: bool = False,
) -> list[str]:
    normalized = [_normalize_record(record) for record in records]
    errors: list[str] = []
    records_by_name = {record["name"]: record for record in normalized if record["name"]}
    for name, expected_kind in REQUIRED_SPANS.items():
        record = records_by_name.get(name)
        if record is None:
            errors.append(f"missing required span: {name}")
            continue
        actual_kind = str(record["span_kind"] or "").upper()
        if actual_kind != expected_kind:
            errors.append(f"span {name} has kind {actual_kind or '<missing>'}, expected {expected_kind}")
    for record in normalized:
        name = str(record["name"])
        if name in FORBIDDEN_SPAN_NAMES:
            errors.append(f"forbidden stale/internal span present: {name}")
    root = records_by_name.get("imperial_rag.query")
    if root is None:
        return errors
    root_attrs = root["attributes"]
    for key in ROOT_PROVENANCE_ATTRIBUTES:
        if key not in root_attrs:
            errors.append(f"root span missing provenance attribute: {key}")
    if expected_run_id and root_attrs.get("imperial.trace_run_id") != expected_run_id:
        errors.append(
            "root span run id mismatch: "
            f"expected {expected_run_id}, got {root_attrs.get('imperial.trace_run_id')}"
        )
    if root_attrs.get("imperial.trace_suppress_internals") is not True:
        errors.append("root span must have imperial.trace_suppress_internals=true")
    if require_retrieval_documents:
        for name in RETRIEVAL_DOCUMENT_SPANS:
            record = records_by_name.get(name)
            if record is None:
                continue
            if not _has_retrieval_document_id_and_content(record["attributes"]):
                errors.append(f"span {name} missing retrieval document id/content attributes")
    return errors


def fetch_latest_span_records(*, project_name: str, base_url: str, run_id: str | None = None) -> list[dict[str, Any]]:
    try:
        from phoenix.client import Client
    except ImportError as exc:
        raise SystemExit("Phoenix client is not installed; install arize-phoenix-client.") from exc

    client = Client(base_url=base_url)
    spans = client.spans.get_spans_dataframe(project_name=project_name)
    records = _records_from_dataframe(spans)
    return _select_trace_records(records, run_id=run_id)


def _select_trace_records(records: list[dict[str, Any]], *, run_id: str | None) -> list[dict[str, Any]]:
    normalized = [_normalize_record(record) for record in records]
    if run_id:
        matches = [record for record in normalized if record["attributes"].get("imperial.trace_run_id") == run_id]
        trace_ids = {_trace_id(record) for record in matches if _trace_id(record)}
        if trace_ids:
            return [record for record in normalized if _trace_id(record) in trace_ids]
        return matches

    roots = [record for record in normalized if record["name"] == "imperial_rag.query"]
    if not roots:
        return normalized
    latest_root = max(roots, key=_sort_time)
    trace_id = _trace_id(latest_root)
    if not trace_id:
        return normalized
    return [record for record in normalized if _trace_id(record) == trace_id]


def _records_from_dataframe(value: Any) -> list[dict[str, Any]]:
    if hasattr(value, "to_dict"):
        return list(value.to_dict("records"))
    return list(value)


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    attrs = _record_attributes(record)
    span_kind = (
        record.get("span_kind")
        or record.get("openinference.span.kind")
        or attrs.get("openinference.span.kind")
        or attrs.get("span_kind")
        or ""
    )
    return {
        **dict(record),
        "name": str(record.get("name") or record.get("span_name") or ""),
        "span_kind": str(span_kind).upper(),
        "attributes": attrs,
    }


def _coerce_attributes(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _record_attributes(record: Mapping[str, Any]) -> dict[str, Any]:
    attrs = _coerce_attributes(
        record.get("attributes")
        or record.get("span_attributes")
        or record.get("attributes_json")
        or {}
    )
    for key, value in record.items():
        if not str(key).startswith("attributes.") or _is_missing(value):
            continue
        suffix = str(key)[len("attributes.") :]
        if isinstance(value, Mapping):
            attrs[suffix] = dict(value)
            for child_key, child_value in value.items():
                if not _is_missing(child_value):
                    attrs[f"{suffix}.{child_key}"] = child_value
            continue
        attrs[suffix] = value
    return attrs


def _has_retrieval_document_id_and_content(attrs: Mapping[str, Any]) -> bool:
    has_id = any(
        key.startswith("retrieval.documents.") and key.endswith(".document.id")
        for key in attrs
    )
    has_content = any(
        key.startswith("retrieval.documents.") and key.endswith(".document.content")
        for key in attrs
    )
    if has_id and has_content:
        return True

    documents = _coerce_documents(attrs.get("retrieval.documents"))
    return any(_document_has_id(document) and _document_has_content(document) for document in documents)


def _coerce_documents(value: Any) -> list[Any]:
    if isinstance(value, str) and value.strip():
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)) else []


def _document_has_id(document: Any) -> bool:
    if not isinstance(document, Mapping):
        return False
    nested = document.get("document")
    if isinstance(nested, Mapping):
        return bool(nested.get("id") or nested.get("document.id"))
    return bool(document.get("document.id") or document.get("id"))


def _document_has_content(document: Any) -> bool:
    if not isinstance(document, Mapping):
        return False
    nested = document.get("document")
    if isinstance(nested, Mapping):
        return bool(nested.get("content") or nested.get("document.content"))
    return bool(document.get("document.content") or document.get("content"))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, float) and value != value


def _trace_id(record: Mapping[str, Any]) -> str:
    for key in ("trace_id", "context.trace_id", "trace.id"):
        value = record.get(key)
        if value:
            return str(value)
    context = record.get("context")
    if isinstance(context, Mapping) and context.get("trace_id"):
        return str(context["trace_id"])
    return ""


def _sort_time(record: Mapping[str, Any]) -> str:
    for key in ("start_time", "start_time_iso", "end_time", "timestamp"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return ""


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the latest Imperial RAG Phoenix trace shape.")
    parser.add_argument("--project-name", help="Phoenix project name; defaults to Settings().phoenix_project_name.")
    parser.add_argument("--base-url", help="Phoenix client URL; defaults to Settings().phoenix_client_endpoint.")
    parser.add_argument("--run-id", help="Validate the trace containing this imperial.trace_run_id.")
    parser.add_argument(
        "--require-retrieval-documents",
        action="store_true",
        help="Require vector and keyword retriever spans to include retrieved document id/content attributes.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


if __name__ == "__main__":
    raise SystemExit(main())
