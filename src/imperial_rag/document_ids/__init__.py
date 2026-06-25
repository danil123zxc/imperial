from __future__ import annotations

import hashlib
from typing import Any


def content_fingerprint_id(content: Any) -> str:
    normalized = str(content).strip()
    if not normalized or normalized == "None":
        raise ValueError(f"Cannot fingerprint empty or None content: {content!r}")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
    return f"content_sha256:{digest}"


def first_nonempty_value(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        resolved = str(value).strip()
        if resolved:
            return resolved
    return None


def metadata_or_content_id(*values: Any, content: Any) -> str:
    if hit := first_nonempty_value(*values):
        return hit
    normalized = str(content).strip() if content is not None else ""
    if not normalized or normalized == "None":
        raise ValueError("No usable ID: metadata values exhausted and content is empty or None")
    return content_fingerprint_id(content)


def document_key(document: Any) -> str:
    metadata = dict(getattr(document, "metadata", {}) or {})
    return metadata_or_content_id(metadata.get("citation_id"), metadata.get("chunk_id"), content=getattr(document, "page_content", ""))


def content_key(document: Any) -> str:
    return " ".join(str(getattr(document, "page_content", "")).split()).casefold()
