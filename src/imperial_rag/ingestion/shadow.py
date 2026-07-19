from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from qdrant_client import models

from imperial_rag.ingestion.promotion import PromotionGateResult, check_promotion_gates


def promote_shadow_run(
    settings: Any,
    run_id: str,
    *,
    questions_path: Path,
    elasticsearch_client: Any | None = None,
    qdrant_client: Any | None = None,
) -> PromotionGateResult:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", run_id):
        raise ValueError("invalid shadow run ID")
    run_root = Path(settings.processed_root) / "shadow-runs" / run_id
    shadow_root = run_root / "extracted"
    run_manifest = json.loads((run_root / "shadow-run.json").read_text(encoding="utf-8"))
    keyword_index = str(run_manifest["keyword_index"])
    qdrant_collection = str(run_manifest["qdrant_collection"])
    keyword_alias = f"{settings.elasticsearch_index}__active"
    qdrant_alias = f"{settings.qdrant_collection}__active"

    result = check_promotion_gates(
        _active_baseline_root(settings),
        shadow_root,
        questions_path=questions_path,
        expected_keyword_index=keyword_index,
        expected_qdrant_collection=qdrant_collection,
    )
    if not result.passed:
        return result

    chunks = _count_jsonl_rows(shadow_root / "chunks.jsonl")
    es = elasticsearch_client or _elasticsearch_client(settings.elasticsearch_url)
    qdrant = qdrant_client or _qdrant_client(settings.qdrant_url)
    es_count = int(es.count(index=keyword_index)["count"])
    qdrant_count = int(qdrant.count(collection_name=qdrant_collection, exact=True).count)
    count_errors = []
    if es_count != chunks:
        count_errors.append(f"Elasticsearch document count mismatch: {es_count} != {chunks}")
    if qdrant_count != chunks:
        count_errors.append(f"Qdrant point count mismatch: {qdrant_count} != {chunks}")
    if count_errors:
        return PromotionGateResult(False, [*result.errors, *count_errors], {**result.summary, "chunk_rows": chunks})

    previous_es = _swap_elasticsearch_alias(es, keyword_alias, keyword_index)
    try:
        previous_qdrant = _swap_qdrant_alias(qdrant, qdrant_alias, qdrant_collection)
    except Exception:
        if previous_es:
            _swap_elasticsearch_alias(es, keyword_alias, previous_es)
        else:
            _remove_elasticsearch_alias(es, keyword_alias)
        raise

    try:
        _write_active_pointer(
            Path(settings.processed_root) / "active-ingestion.json",
            {
                "schema_version": "imperial-active-ingestion-v1",
                "shadow_run_id": run_id,
                "artifact_root": str(shadow_root),
                "manifest_db_path": str(run_root / "manifest.sqlite3"),
                "keyword_alias": keyword_alias,
                "keyword_index": keyword_index,
                "qdrant_alias": qdrant_alias,
                "qdrant_collection": qdrant_collection,
            },
        )
    except Exception:
        if previous_es:
            _swap_elasticsearch_alias(es, keyword_alias, previous_es)
        else:
            _remove_elasticsearch_alias(es, keyword_alias)
        if previous_qdrant:
            _swap_qdrant_alias(qdrant, qdrant_alias, previous_qdrant)
        else:
            _remove_qdrant_alias(qdrant, qdrant_alias)
        raise
    return PromotionGateResult(True, [], {**result.summary, "chunk_rows": chunks, "promoted": True})


def _swap_elasticsearch_alias(client: Any, alias: str, target: str) -> str | None:
    if client.indices.exists(index=alias) and not client.indices.exists_alias(name=alias):
        raise RuntimeError(f"Elasticsearch alias name conflicts with a physical index: {alias}")
    previous: str | None = None
    actions: list[dict[str, Any]] = []
    if client.indices.exists_alias(name=alias):
        aliases = client.indices.get_alias(name=alias)
        previous = next(iter(aliases), None)
        actions.append({"remove": {"index": "*", "alias": alias}})
    actions.append({"add": {"index": target, "alias": alias}})
    client.indices.update_aliases(actions=actions)
    return previous


def _remove_elasticsearch_alias(client: Any, alias: str) -> None:
    if client.indices.exists_alias(name=alias):
        client.indices.update_aliases(actions=[{"remove": {"index": "*", "alias": alias}}])


def _swap_qdrant_alias(client: Any, alias: str, target: str) -> str | None:
    aliases = {item.alias_name: item.collection_name for item in client.get_aliases().aliases}
    operations: list[Any] = []
    if alias in aliases:
        operations.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    operations.append(
        models.CreateAliasOperation(
            create_alias=models.CreateAlias(collection_name=target, alias_name=alias)
        )
    )
    client.update_collection_aliases(change_aliases_operations=operations)
    return aliases.get(alias)


def _remove_qdrant_alias(client: Any, alias: str) -> None:
    aliases = {item.alias_name for item in client.get_aliases().aliases}
    if alias not in aliases:
        return
    client.update_collection_aliases(
        change_aliases_operations=[
            models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias))
        ]
    )


def _write_active_pointer(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _active_baseline_root(settings: Any) -> Path:
    pointer = Path(settings.processed_root) / "active-ingestion.json"
    if not pointer.exists():
        return Path(settings.extraction_root)
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    return Path(payload["artifact_root"])


def _elasticsearch_client(url: str) -> Any:
    from elasticsearch import Elasticsearch

    return Elasticsearch(url)


def _qdrant_client(url: str) -> Any:
    from qdrant_client import QdrantClient

    return QdrantClient(url=url)
